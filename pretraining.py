import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, get_linear_schedule_with_warmup
from llada_model import LLaDAForMaskedLM
import os
import json
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional
import wandb
from dataclasses import dataclass


@dataclass
class PretrainingConfig:
    """Configuration for pre-training"""
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "./llada_pretrained"
    data_path: str = "./pretraining_data"
    
    # Training hyperparameters
    learning_rate: float = 5e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_steps: int = 100000
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Model hyperparameters
    max_length: int = 4096
    mask_token_id: int = 126336
    eps: float = 1e-3
    random_length_prob: float = 0.01
    
    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 5000
    eval_steps: int = 1000
    save_total_limit: int = 3
    
    # Mixed precision
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Wandb
    wandb_project: str = "llada_pretraining"
    wandb_run_name: Optional[str] = None


def forward_process(input_ids: torch.Tensor, eps: float = 1e-3) -> tuple:
    """
    Forward diffusion process for LLaDA
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
        eps: Minimum masking probability
    
    Returns:
        noisy_batch: Input with masked tokens
        masked_indices: Boolean tensor indicating masked positions
        p_mask: Masking probabilities for each position
    """
    b, l = input_ids.shape
    device = input_ids.device
    
    # Sample random timesteps for each sequence in the batch
    t = torch.rand(b, device=device)
    
    # Calculate masking probability: p_mask = (1 - eps) * t + eps
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)  # Shape: (b, l)
    
    # Determine which tokens to mask
    masked_indices = torch.rand((b, l), device=device) < p_mask
    
    # Apply masking (126336 is the [MASK] token)
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    
    return noisy_batch, masked_indices, p_mask


class PretrainingDataset(Dataset):
    """Dataset for LLaDA pre-training"""
    
    def __init__(self, data_path: str, max_length: int = 4096):
        self.data_path = data_path
        self.max_length = max_length
        self.data_files = []
        
        # Collect all data files
        if os.path.isdir(data_path):
            for file in os.listdir(data_path):
                if file.endswith('.jsonl') or file.endswith('.json'):
                    self.data_files.append(os.path.join(data_path, file))
        else:
            self.data_files = [data_path]
        
        # Load and prepare data
        self.examples = []
        self._load_data()
    
    def _load_data(self):
        """Load data from files"""
        for file_path in self.data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    for line in f:
                        data = json.loads(line.strip())
                        if 'input_ids' in data:
                            input_ids = data['input_ids']
                            if len(input_ids) <= self.max_length:
                                self.examples.append(torch.tensor(input_ids, dtype=torch.long))
                        elif 'text' in data:
                            # If text is provided, you would need a tokenizer here
                            # For now, assume input_ids are provided
                            pass
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'input_ids' in item:
                                input_ids = item['input_ids']
                                if len(input_ids) <= self.max_length:
                                    self.examples.append(torch.tensor(input_ids, dtype=torch.long))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx]}


def collate_fn(batch, max_length: int = 4096, pad_token_id: int = 0):
    """Collate function for pre-training data"""
    input_ids = [item["input_ids"] for item in batch]
    
    # Pad sequences to the same length
    batch_max_len = min(max_length, max(len(seq) for seq in input_ids))
    
    padded_input_ids = []
    for seq in input_ids:
        if len(seq) > batch_max_len:
            # Truncate if too long
            padded_seq = seq[:batch_max_len]
        else:
            # Pad if too short
            padding_length = batch_max_len - len(seq)
            padded_seq = torch.cat([seq, torch.full((padding_length,), pad_token_id, dtype=torch.long)])
        padded_input_ids.append(padded_seq)
    
    return {
        "input_ids": torch.stack(padded_input_ids)
    }


class LLaDAPretrainer:
    """LLaDA pre-training trainer"""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_model()
        self.setup_optimizer()
        
        # Initialize wandb if configured
        if config.wandb_project:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_model(self):
        """Setup LLaDA model"""
        # Load base config from LLaMA
        if os.path.exists(self.config.model_name_or_path):
            model_config = LlamaConfig.from_pretrained(self.config.model_name_or_path)
        else:
            # Default config if path doesn't exist
            model_config = LlamaConfig(
                vocab_size=32000,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                max_position_embeddings=4096,
                rms_norm_eps=1e-6,
            )
        
        # Initialize LLaDA model
        self.model = LLaDAForMaskedLM(model_config)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # AdamW optimizer
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps
        )
    
    def compute_loss(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute LLaDA pre-training loss
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
        
        Returns:
            Dictionary containing loss and metrics
        """
        # Apply random length sampling (1% of the time)
        if torch.rand(1).item() < self.config.random_length_prob:
            random_length = torch.randint(1, input_ids.shape[1] + 1, (1,)).item()
            input_ids = input_ids[:, :random_length]
        
        # Forward diffusion process
        noisy_batch, masked_indices, p_mask = forward_process(
            input_ids, eps=self.config.eps
        )
        
        # Forward pass through model
        outputs = self.model(input_ids=noisy_batch)
        logits = outputs.logits
        
        # Compute loss only on masked tokens
        if masked_indices.sum() == 0:
            # If no tokens are masked, return zero loss
            return {
                "loss": torch.tensor(0.0, device=self.device, requires_grad=True),
                "num_masked_tokens": torch.tensor(0),
                "perplexity": torch.tensor(float('inf'))
            }
        
        # Get predictions and targets for masked tokens only
        masked_logits = logits[masked_indices]
        masked_targets = input_ids[masked_indices]
        masked_p_mask = p_mask[masked_indices]
        
        # Compute cross-entropy loss
        token_loss = F.cross_entropy(
            masked_logits, 
            masked_targets, 
            reduction='none'
        )
        
        # Weight by inverse masking probability as in the paper
        weighted_token_loss = token_loss / masked_p_mask
        
        # Average loss as in the paper: sum over tokens, divide by total sequence length
        loss = weighted_token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        
        # Compute metrics
        with torch.no_grad():
            num_masked_tokens = masked_indices.sum()
            perplexity = torch.exp(token_loss.mean()) if len(token_loss) > 0 else torch.tensor(float('inf'))
        
        return {
            "loss": loss,
            "num_masked_tokens": num_masked_tokens,
            "perplexity": perplexity
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        
        # Compute loss
        loss_dict = self.compute_loss(input_ids)
        loss = loss_dict["loss"]
        
        # Backward pass
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        loss.backward()
        
        # Convert to float for logging
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        return metrics
    
    def train(self):
        """Main training loop"""
        # Setup data
        dataset = PretrainingDataset(
            self.config.data_path, 
            max_length=self.config.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=lambda batch: collate_fn(batch, self.config.max_length),
            pin_memory=True
        )
        
        self.logger.info(f"Starting training with {len(dataset)} examples")
        
        global_step = 0
        total_loss = 0.0
        total_tokens = 0
        
        # Create infinite dataloader
        def infinite_dataloader():
            while True:
                for batch in dataloader:
                    yield batch
        
        data_iter = infinite_dataloader()
        
        # Training loop
        progress_bar = tqdm(range(self.config.max_steps), desc="Training")
        
        for step in progress_bar:
            batch = next(data_iter)
            
            # Training step
            metrics = self.train_step(batch)
            
            total_loss += metrics["loss"] * self.config.gradient_accumulation_steps
            total_tokens += metrics["num_masked_tokens"]
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    learning_rate = self.scheduler.get_last_lr()[0]
                    
                    log_metrics = {
                        "train/loss": avg_loss,
                        "train/perplexity": metrics["perplexity"],
                        "train/learning_rate": learning_rate,
                        "train/global_step": global_step,
                        "train/num_masked_tokens": metrics["num_masked_tokens"]
                    }
                    
                    # Log to wandb
                    if self.config.wandb_project:
                        wandb.log(log_metrics, step=global_step)
                    
                    # Log to console
                    self.logger.info(
                        f"Step {global_step}: loss={avg_loss:.4f}, "
                        f"ppl={metrics['perplexity']:.2f}, lr={learning_rate:.2e}"
                    )
                    
                    total_loss = 0.0
                    total_tokens = 0
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "ppl": f"{metrics['perplexity']:.2f}"
                })
        
        # Save final checkpoint
        self.save_checkpoint(global_step, final=True)
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint"""
        output_dir = os.path.join(
            self.config.output_dir, 
            f"checkpoint-{step}" if not final else "final"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save config
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save training state
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step
        }
        torch.save(training_state, os.path.join(output_dir, "training_state.bin"))
        
        self.logger.info(f"Checkpoint saved to {output_dir}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLaDA Pre-training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./llada_pretrained", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum training steps")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PretrainingConfig(**config_dict)
    else:
        config = PretrainingConfig()
    
    # Override with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_steps:
        config.max_steps = args.max_steps
    
    # Start training
    trainer = LLaDAPretrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 