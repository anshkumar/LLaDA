import torch
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import logging


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class TokenizerHelper:
    """Helper class for tokenization tasks"""
    
    def __init__(self, tokenizer_name: str = "meta-llama/Llama-2-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Special tokens for LLaDA
        self.mask_token_id = 126336
        self.bos_token_id = self.tokenizer.bos_token_id or 1
        self.eos_token_id = self.tokenizer.eos_token_id or 2
        self.pad_token_id = self.tokenizer.pad_token_id or 0
    
    def encode_conversation(
        self,
        conversation: List[Dict[str, str]],
        max_length: int = 4096
    ) -> Dict[str, Any]:
        """
        Encode a conversation for SFT training
        
        Args:
            conversation: List of conversation turns with 'from' and 'value' keys
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and prompt_length
        """
        # Build conversation text
        conversation_text = ""
        prompt_text = ""
        
        for turn in conversation:
            role = turn.get('from', turn.get('role', ''))
            content = turn.get('value', turn.get('content', ''))
            
            if role in ['human', 'user']:
                turn_text = f"<|user|>\n{content}<|endoftext|>"
                conversation_text += turn_text
                prompt_text += turn_text
            elif role in ['gpt', 'assistant']:
                turn_text = f"<|assistant|>\n{content}<|endoftext|>"
                conversation_text += turn_text
                break  # Only include first assistant response
        
        # Tokenize
        input_ids = self.tokenizer.encode(conversation_text, add_special_tokens=True)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        
        # Truncate if necessary
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            # Adjust prompt length if needed
            prompt_length = min(len(prompt_ids), max_length)
        else:
            prompt_length = len(prompt_ids)
        
        return {
            'input_ids': input_ids,
            'prompt_length': prompt_length,
            'conversation_text': conversation_text,
            'prompt_text': prompt_text
        }
    
    def encode_instruction_response(
        self,
        instruction: str,
        response: str,
        system: Optional[str] = None,
        max_length: int = 4096
    ) -> Dict[str, Any]:
        """
        Encode instruction-response pair for SFT training
        
        Args:
            instruction: User instruction
            response: Assistant response
            system: Optional system message
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and prompt_length
        """
        # Build prompt
        if system:
            prompt_text = f"<|system|>\n{system}<|endoftext|><|user|>\n{instruction}<|endoftext|><|assistant|>\n"
        else:
            prompt_text = f"<|user|>\n{instruction}<|endoftext|><|assistant|>\n"
        
        full_text = prompt_text + response + "<|endoftext|>"
        
        # Tokenize
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        
        # Truncate if necessary
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            prompt_length = min(len(prompt_ids), max_length)
        else:
            prompt_length = len(prompt_ids)
        
        return {
            'input_ids': input_ids,
            'prompt_length': prompt_length,
            'full_text': full_text,
            'prompt_text': prompt_text
        }
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class DatasetConverter:
    """Convert various dataset formats to LLaDA format"""
    
    def __init__(self, tokenizer_helper: TokenizerHelper):
        self.tokenizer = tokenizer_helper
        self.logger = logging.getLogger(__name__)
    
    def convert_sharegpt_to_llada(
        self,
        input_file: str,
        output_file: str,
        max_length: int = 4096,
        max_examples: Optional[int] = None
    ):
        """Convert ShareGPT format to LLaDA SFT format"""
        self.logger.info(f"Converting ShareGPT data from {input_file} to {output_file}")
        
        examples = []
        count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_examples and count >= max_examples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    if 'conversations' in data:
                        result = self.tokenizer.encode_conversation(
                            data['conversations'], max_length
                        )
                        examples.append(result)
                        count += 1
                        
                        if count % 1000 == 0:
                            self.logger.info(f"Processed {count} examples")
                            
                except Exception as e:
                    self.logger.warning(f"Error processing line {count}: {e}")
                    continue
        
        # Save converted data
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f)
                f.write('\n')
        
        self.logger.info(f"Converted {len(examples)} examples to {output_file}")
    
    def convert_alpaca_to_llada(
        self,
        input_file: str,
        output_file: str,
        max_length: int = 4096,
        max_examples: Optional[int] = None
    ):
        """Convert Alpaca format to LLaDA SFT format"""
        self.logger.info(f"Converting Alpaca data from {input_file} to {output_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        count = 0
        
        for item in data:
            if max_examples and count >= max_examples:
                break
            
            try:
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                # Combine instruction and input
                if input_text:
                    full_instruction = f"{instruction}\n\n{input_text}"
                else:
                    full_instruction = instruction
                
                result = self.tokenizer.encode_instruction_response(
                    full_instruction, output_text, max_length=max_length
                )
                examples.append(result)
                count += 1
                
                if count % 1000 == 0:
                    self.logger.info(f"Processed {count} examples")
                    
            except Exception as e:
                self.logger.warning(f"Error processing item {count}: {e}")
                continue
        
        # Save converted data
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f)
                f.write('\n')
        
        self.logger.info(f"Converted {len(examples)} examples to {output_file}")
    
    def prepare_pretraining_data(
        self,
        input_files: List[str],
        output_file: str,
        max_length: int = 4096,
        chunk_size: int = 10000
    ):
        """Prepare text data for pre-training"""
        self.logger.info(f"Preparing pre-training data from {len(input_files)} files")
        
        all_examples = []
        
        for input_file in input_files:
            self.logger.info(f"Processing {input_file}")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                if input_file.endswith('.jsonl'):
                    # JSONL format
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            text = data.get('text', '')
                            if text:
                                # Tokenize and chunk
                                token_ids = self.tokenizer.tokenizer.encode(
                                    text, add_special_tokens=True
                                )
                                
                                # Split into chunks
                                for i in range(0, len(token_ids), max_length):
                                    chunk = token_ids[i:i + max_length]
                                    if len(chunk) >= 32:  # Minimum chunk size
                                        all_examples.append({'input_ids': chunk})
                        except Exception as e:
                            continue
                else:
                    # Plain text format
                    text = f.read()
                    token_ids = self.tokenizer.tokenizer.encode(
                        text, add_special_tokens=True
                    )
                    
                    # Split into chunks
                    for i in range(0, len(token_ids), max_length):
                        chunk = token_ids[i:i + max_length]
                        if len(chunk) >= 32:
                            all_examples.append({'input_ids': chunk})
        
        # Save examples in chunks
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        for i in range(0, len(all_examples), chunk_size):
            chunk = all_examples[i:i + chunk_size]
            chunk_file = f"{output_file}.{i // chunk_size:04d}.jsonl"
            
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for example in chunk:
                    json.dump(example, f)
                    f.write('\n')
        
        self.logger.info(f"Prepared {len(all_examples)} examples in chunks")


class ModelAnalyzer:
    """Analyze LLaDA model behavior and performance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_attention_patterns(
        self,
        model,
        input_ids: torch.Tensor,
        layer_idx: int = -1,
        head_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """Analyze attention patterns in LLaDA model"""
        model.eval()
        
        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                output_attentions=True
            )
        
        # Get attention weights
        attentions = outputs.attentions[layer_idx]  # (batch, heads, seq_len, seq_len)
        attention_head = attentions[0, head_idx].cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_head,
            cmap='Blues',
            cbar=True,
            square=True
        )
        plt.title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Attention pattern saved to {save_path}")
        
        plt.show()
        
        return attention_head
    
    def analyze_masking_behavior(
        self,
        model,
        sampler,
        input_ids: torch.Tensor,
        num_iterations: int = 10,
        save_path: Optional[str] = None
    ):
        """Analyze how masking evolves during sampling"""
        model.eval()
        
        # Track masking over iterations
        masking_history = []
        confidence_history = []
        
        # Initialize with all positions masked (except prompt)
        batch_size, prompt_length = input_ids.shape
        total_length = prompt_length + 50  # Generate 50 tokens
        
        sequence = torch.full(
            (batch_size, total_length),
            sampler.config.mask_token_id,
            device=input_ids.device,
            dtype=torch.long
        )
        sequence[:, :prompt_length] = input_ids
        
        masked_positions = torch.ones(
            (batch_size, total_length),
            device=input_ids.device,
            dtype=torch.bool
        )
        masked_positions[:, :prompt_length] = False
        
        for iteration in range(num_iterations):
            if not masked_positions.any():
                break
            
            with torch.no_grad():
                outputs = model(input_ids=sequence)
                logits = outputs.logits
            
            # Calculate confidence
            probs = torch.softmax(logits, dim=-1)
            max_probs, predictions = torch.max(probs, dim=-1)
            
            # Update sequence
            sequence[masked_positions] = predictions[masked_positions]
            
            # Track statistics
            masking_ratio = masked_positions.float().mean().item()
            avg_confidence = max_probs[masked_positions].mean().item()
            
            masking_history.append(masking_ratio)
            confidence_history.append(avg_confidence)
            
            # Get new masking for next iteration
            if iteration < num_iterations - 1:
                masked_positions = sampler._get_remask_positions(
                    logits, predictions, masked_positions
                )
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Masking ratio over iterations
        ax1.plot(masking_history, 'b-o')
        ax1.set_title('Masking Ratio Over Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Masking Ratio')
        ax1.grid(True)
        
        # Confidence over iterations
        ax2.plot(confidence_history, 'r-o')
        ax2.set_title('Average Confidence Over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Average Confidence')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Masking analysis saved to {save_path}")
        
        plt.show()
        
        return {
            'masking_history': masking_history,
            'confidence_history': confidence_history,
            'final_sequence': sequence
        }
    
    def compare_sampling_methods(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        num_samples: int = 5
    ):
        """Compare different sampling methods"""
        from sampling import create_sampler
        
        methods = [
            "fixed_length",
            "semi_autoregressive_origin", 
            "semi_autoregressive_padding"
        ]
        strategies = ["random", "low_confidence"]
        
        results = {}
        
        for method in methods:
            for strategy in strategies:
                key = f"{method}_{strategy}"
                self.logger.info(f"Testing {key}")
                
                sampler = create_sampler(
                    model,
                    sampling_method=method,
                    remasking_strategy=strategy,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    num_iterations=10
                )
                
                samples = []
                for _ in range(num_samples):
                    try:
                        generated = sampler.generate(
                            input_ids=input_ids,
                            max_new_tokens=max_new_tokens,
                            num_return_sequences=1
                        )
                        samples.append(generated[0])
                    except Exception as e:
                        self.logger.warning(f"Error in {key}: {e}")
                        continue
                
                results[key] = {
                    'samples': samples,
                    'method': method,
                    'strategy': strategy
                }
        
        return results


def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            
            # Simple forward pass for perplexity calculation
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()


def save_model_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    output_dir: str,
    config: Dict[str, Any]
):
    """Save a complete model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    
    # Save training state
    training_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step
    }
    training_path = os.path.join(output_dir, "training_state.bin")
    torch.save(training_state, training_path)
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.getLogger(__name__).info(f"Checkpoint saved to {output_dir}")


def load_model_checkpoint(
    model,
    checkpoint_dir: str,
    load_optimizer: bool = True,
    load_scheduler: bool = True
) -> Tuple[Optional[Any], Optional[Any], int]:
    """Load a model checkpoint"""
    logger = logging.getLogger(__name__)
    
    # Load model state
    model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {model_path}")
    else:
        logger.warning(f"Model weights not found at {model_path}")
    
    optimizer = None
    scheduler = None
    step = 0
    
    # Load training state
    training_path = os.path.join(checkpoint_dir, "training_state.bin")
    if os.path.exists(training_path) and (load_optimizer or load_scheduler):
        training_state = torch.load(training_path, map_location="cpu")
        
        if load_optimizer and "optimizer" in training_state:
            optimizer = training_state["optimizer"]
        
        if load_scheduler and "scheduler" in training_state:
            scheduler = training_state["scheduler"]
        
        step = training_state.get("step", 0)
        logger.info(f"Loaded training state from {training_path}")
    
    return optimizer, scheduler, step 