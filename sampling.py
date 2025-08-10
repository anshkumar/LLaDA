import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Dict, Any
from llada_model import LLaDAForMaskedLM
from dataclasses import dataclass
import logging


@dataclass
class SamplingConfig:
    """Configuration for LLaDA sampling"""
    # Sampling method: "fixed_length", "semi_autoregressive_origin", "semi_autoregressive_padding"
    sampling_method: str = "fixed_length"
    
    # Remasking strategy: "random" or "low_confidence"
    remasking_strategy: str = "low_confidence"
    
    # Generation parameters
    max_length: int = 512
    num_iterations: int = 10
    remasking_ratio: float = 0.8  # Proportion of predictions to remask
    
    # Special tokens
    mask_token_id: int = 126336
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    # Generation settings
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Semi-autoregressive specific
    initial_length: int = 32  # Initial length for semi-autoregressive methods
    length_increment: int = 16  # How much to extend each step


class LLaDASampler:
    """LLaDA sampling implementation with three different methods"""
    
    def __init__(self, model: LLaDAForMaskedLM, config: SamplingConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        assert config.sampling_method in [
            "fixed_length", "semi_autoregressive_origin", "semi_autoregressive_padding"
        ], f"Unknown sampling method: {config.sampling_method}"
        
        assert config.remasking_strategy in [
            "random", "low_confidence"
        ], f"Unknown remasking strategy: {config.remasking_strategy}"
    
    def sample(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the specified sampling method
        
        Args:
            prompt_ids: Input prompt token IDs of shape (batch_size, prompt_length)
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated token IDs of shape (batch_size, total_length)
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_length - prompt_ids.shape[1]
        
        if self.config.sampling_method == "fixed_length":
            return self._fixed_length_sampling(prompt_ids, max_new_tokens)
        elif self.config.sampling_method == "semi_autoregressive_origin":
            return self._semi_autoregressive_origin_sampling(prompt_ids, max_new_tokens)
        elif self.config.sampling_method == "semi_autoregressive_padding":
            return self._semi_autoregressive_padding_sampling(prompt_ids, max_new_tokens)
        else:
            raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")
    
    def _fixed_length_sampling(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int
    ) -> torch.Tensor:
        """
        Fixed-length sampling: Start with all positions masked, iteratively unmask
        """
        batch_size, prompt_length = prompt_ids.shape
        total_length = prompt_length + max_new_tokens
        
        # Initialize sequence with prompt + masked tokens
        sequence = torch.full(
            (batch_size, total_length),
            self.config.mask_token_id,
            device=self.device,
            dtype=torch.long
        )
        sequence[:, :prompt_length] = prompt_ids
        
        # Track which positions are still masked (excluding prompt)
        masked_positions = torch.ones(
            (batch_size, total_length),
            device=self.device,
            dtype=torch.bool
        )
        masked_positions[:, :prompt_length] = False  # Don't mask prompt
        
        self.logger.info(f"Starting fixed-length sampling for {max_new_tokens} tokens")
        
        for iteration in range(self.config.num_iterations):
            if not masked_positions.any():
                break
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(input_ids=sequence)
                logits = outputs.logits
            
            # Sample from logits
            predictions = self._sample_from_logits(logits, masked_positions)
            
            # Update sequence with predictions
            sequence[masked_positions] = predictions[masked_positions]
            
            # Determine which predictions to remask
            if iteration < self.config.num_iterations - 1:  # Don't remask on final iteration
                remask_positions = self._get_remask_positions(
                    logits, predictions, masked_positions
                )
                
                # Apply remasking
                sequence[remask_positions] = self.config.mask_token_id
                masked_positions = remask_positions
            
            self.logger.debug(f"Iteration {iteration + 1}: {masked_positions.sum().item()} tokens still masked")
        
        return sequence
    
    def _semi_autoregressive_origin_sampling(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int
    ) -> torch.Tensor:
        """
        Semi-autoregressive origin: Start with short length, gradually extend
        """
        batch_size, prompt_length = prompt_ids.shape
        current_length = prompt_length + self.config.initial_length
        max_total_length = prompt_length + max_new_tokens
        
        # Initialize with prompt + initial masked tokens
        sequence = torch.full(
            (batch_size, current_length),
            self.config.mask_token_id,
            device=self.device,
            dtype=torch.long
        )
        sequence[:, :prompt_length] = prompt_ids
        
        self.logger.info(f"Starting semi-autoregressive origin sampling")
        
        while current_length < max_total_length:
            # Generate for current length
            masked_positions = torch.ones(
                (batch_size, current_length),
                device=self.device,
                dtype=torch.bool
            )
            masked_positions[:, :prompt_length] = False  # Don't mask prompt
            
            # Iterative refinement for current length
            for iteration in range(self.config.num_iterations):
                if not masked_positions.any():
                    break
                
                with torch.no_grad():
                    outputs = self.model(input_ids=sequence)
                    logits = outputs.logits
                
                predictions = self._sample_from_logits(logits, masked_positions)
                sequence[masked_positions] = predictions[masked_positions]
                
                if iteration < self.config.num_iterations - 1:
                    remask_positions = self._get_remask_positions(
                        logits, predictions, masked_positions
                    )
                    sequence[remask_positions] = self.config.mask_token_id
                    masked_positions = remask_positions
            
            # Check if generation is complete (EOS generated)
            if self._check_eos_generated(sequence, prompt_length):
                break
            
            # Extend sequence length
            next_length = min(
                current_length + self.config.length_increment,
                max_total_length
            )
            
            if next_length > current_length:
                # Extend sequence with masked tokens
                extension = torch.full(
                    (batch_size, next_length - current_length),
                    self.config.mask_token_id,
                    device=self.device,
                    dtype=torch.long
                )
                sequence = torch.cat([sequence, extension], dim=1)
                current_length = next_length
            else:
                break
        
        return sequence
    
    def _semi_autoregressive_padding_sampling(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int
    ) -> torch.Tensor:
        """
        Semi-autoregressive padding: Start with full length, gradually unmask from left to right
        """
        batch_size, prompt_length = prompt_ids.shape
        total_length = prompt_length + max_new_tokens
        
        # Initialize full sequence with masked tokens
        sequence = torch.full(
            (batch_size, total_length),
            self.config.mask_token_id,
            device=self.device,
            dtype=torch.long
        )
        sequence[:, :prompt_length] = prompt_ids
        
        # Current generation position (starts after prompt)
        current_pos = prompt_length
        
        self.logger.info(f"Starting semi-autoregressive padding sampling")
        
        while current_pos < total_length:
            # Determine current window to generate
            window_end = min(
                current_pos + self.config.initial_length,
                total_length
            )
            
            # Mask positions for current window
            masked_positions = torch.zeros(
                (batch_size, total_length),
                device=self.device,
                dtype=torch.bool
            )
            masked_positions[:, current_pos:window_end] = True
            
            # Iterative refinement for current window
            for iteration in range(self.config.num_iterations):
                if not masked_positions.any():
                    break
                
                with torch.no_grad():
                    outputs = self.model(input_ids=sequence)
                    logits = outputs.logits
                
                predictions = self._sample_from_logits(logits, masked_positions)
                sequence[masked_positions] = predictions[masked_positions]
                
                if iteration < self.config.num_iterations - 1:
                    remask_positions = self._get_remask_positions(
                        logits, predictions, masked_positions
                    )
                    sequence[remask_positions] = self.config.mask_token_id
                    masked_positions = remask_positions
            
            # Check if EOS was generated in current window
            if self._check_eos_generated(sequence[:, current_pos:window_end], 0):
                # Find EOS position and truncate
                for batch_idx in range(batch_size):
                    eos_positions = (sequence[batch_idx, current_pos:] == self.config.eos_token_id).nonzero()
                    if len(eos_positions) > 0:
                        eos_pos = current_pos + eos_positions[0].item() + 1
                        sequence[batch_idx, eos_pos:] = self.config.pad_token_id
                break
            
            # Move to next window
            current_pos = window_end
        
        return sequence
    
    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        masked_positions: torch.Tensor
    ) -> torch.Tensor:
        """Sample tokens from logits with temperature, top-p, and top-k filtering"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Apply temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Apply top-k filtering
        if self.config.top_k > 0:
            top_k = min(self.config.top_k, vocab_size)
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            
            # Set logits for non-top-k tokens to negative infinity
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_filtered
        
        # Apply top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set filtered logits to negative infinity
            sorted_logits[sorted_indices_to_remove] = float('-inf')
            
            # Scatter back to original order
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, sorted_indices, sorted_logits)
            logits = logits_filtered
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        predictions = torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
        predictions = predictions.view(batch_size, seq_len)
        
        return predictions
    
    def _get_remask_positions(
        self,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        masked_positions: torch.Tensor
    ) -> torch.Tensor:
        """Determine which positions to remask based on the remasking strategy"""
        if self.config.remasking_strategy == "random":
            return self._random_remasking(masked_positions)
        elif self.config.remasking_strategy == "low_confidence":
            return self._low_confidence_remasking(logits, predictions, masked_positions)
        else:
            raise ValueError(f"Unknown remasking strategy: {self.config.remasking_strategy}")
    
    def _random_remasking(self, masked_positions: torch.Tensor) -> torch.Tensor:
        """Randomly select positions to remask"""
        batch_size, seq_len = masked_positions.shape
        
        # Count currently masked positions
        num_masked = masked_positions.sum(dim=1)
        num_to_remask = (num_masked * self.config.remasking_ratio).long()
        
        # Create new mask
        new_masked_positions = torch.zeros_like(masked_positions)
        
        for batch_idx in range(batch_size):
            if num_to_remask[batch_idx] > 0:
                # Get currently masked positions for this batch
                masked_indices = masked_positions[batch_idx].nonzero().squeeze(-1)
                
                if len(masked_indices) > 0:
                    # Randomly select positions to keep masked
                    n_select = min(num_to_remask[batch_idx].item(), len(masked_indices))
                    selected_indices = torch.randperm(len(masked_indices))[:n_select]
                    positions_to_remask = masked_indices[selected_indices]
                    
                    new_masked_positions[batch_idx, positions_to_remask] = True
        
        return new_masked_positions
    
    def _low_confidence_remasking(
        self,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        masked_positions: torch.Tensor
    ) -> torch.Tensor:
        """Select positions with lowest confidence to remask"""
        batch_size, seq_len = masked_positions.shape
        
        # Calculate confidence scores (probability of predicted token)
        probs = F.softmax(logits, dim=-1)
        confidence_scores = torch.gather(probs, -1, predictions.unsqueeze(-1)).squeeze(-1)
        
        # Only consider currently masked positions
        confidence_scores = confidence_scores.masked_fill(~masked_positions, float('inf'))
        
        # Count positions to remask
        num_masked = masked_positions.sum(dim=1)
        num_to_remask = (num_masked * self.config.remasking_ratio).long()
        
        # Create new mask
        new_masked_positions = torch.zeros_like(masked_positions)
        
        for batch_idx in range(batch_size):
            if num_to_remask[batch_idx] > 0:
                # Get confidence scores for this batch
                batch_confidence = confidence_scores[batch_idx]
                
                # Find positions with lowest confidence
                _, lowest_conf_indices = torch.topk(
                    batch_confidence,
                    k=min(num_to_remask[batch_idx].item(), masked_positions[batch_idx].sum().item()),
                    largest=False  # Get lowest confidence
                )
                
                new_masked_positions[batch_idx, lowest_conf_indices] = True
        
        return new_masked_positions
    
    def _check_eos_generated(self, sequence: torch.Tensor, start_pos: int = 0) -> bool:
        """Check if EOS token has been generated"""
        return (sequence[:, start_pos:] == self.config.eos_token_id).any()
    
    def generate(
        self,
        input_text: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        High-level generation interface
        
        Args:
            input_text: Input text string (requires tokenizer)
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            num_return_sequences: Number of sequences to return
            
        Returns:
            List of generated token sequences
        """
        if input_ids is None:
            if input_text is None:
                raise ValueError("Either input_text or input_ids must be provided")
            # Note: You would need a tokenizer here in practice
            raise NotImplementedError("Text input requires a tokenizer")
        
        # Expand for multiple return sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        # Ensure input is on correct device
        input_ids = input_ids.to(self.device)
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.sample(input_ids, max_new_tokens)
        
        # Split back into individual sequences
        if num_return_sequences > 1:
            sequences = [generated_ids[i:i+1] for i in range(num_return_sequences)]
        else:
            sequences = [generated_ids]
        
        return sequences


def create_sampler(
    model: LLaDAForMaskedLM,
    sampling_method: str = "fixed_length",
    remasking_strategy: str = "low_confidence",
    **kwargs
) -> LLaDASampler:
    """
    Convenience function to create a sampler with specified configuration
    """
    config = SamplingConfig(
        sampling_method=sampling_method,
        remasking_strategy=remasking_strategy,
        **kwargs
    )
    return LLaDASampler(model, config)


# Example usage and testing functions
def test_sampling_methods():
    """Test all three sampling methods"""
    from transformers import LlamaConfig
    
    # Create a small test model
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=1024,
    )
    
    model = LLaDAForMaskedLM(config)
    
    # Test each sampling method
    methods = ["fixed_length", "semi_autoregressive_origin", "semi_autoregressive_padding"]
    remasking_strategies = ["random", "low_confidence"]
    
    # Create dummy prompt
    prompt_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # Dummy prompt
    
    for method in methods:
        for strategy in remasking_strategies:
            print(f"\nTesting {method} with {strategy} remasking:")
            
            sampler = create_sampler(
                model,
                sampling_method=method,
                remasking_strategy=strategy,
                max_length=50,
                num_iterations=5
            )
            
            try:
                results = sampler.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=20,
                    num_return_sequences=1
                )
                print(f"✓ Generated sequence shape: {results[0].shape}")
            except Exception as e:
                print(f"✗ Error: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_sampling_methods() 