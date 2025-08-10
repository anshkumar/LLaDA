import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, get_linear_schedule_with_warmup
from llada_model import LLaDAForMaskedLM
from pretraining import forward_process, PretrainingConfig
import os
import json
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional, List
import wandb
from dataclasses import dataclass


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning"""
    model_name_or_path: str = "./llada_pretrained"
    output_dir: str = "./llada_sft"
    data_path: str = "./sft_data"
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    max_steps: int = 10000
    warmup_steps: int = 500
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Model hyperparameters
    max_length: int = 4096
    mask_token_id: int = 126336
    eps: float = 1e-3
    
    # Special tokens (adjust according to your tokenizer)
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    # Logging and saving
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Mixed precision
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Wandb
    wandb_project: str = "llada_sft"
    wandb_run_name: Optional[str] = None


class SFTDataset(Dataset):
    """Dataset for LLaDA supervised fine-tuning"""
    
    def __init__(self, data_path: str, max_length: int = 4096):
        self.data_path = data_path
        self.max_length = max_length
        self.examples = []
        self._load_data()
    
    def _load_data(self):
        """Load SFT data with input_ids and prompt_lengths"""
        if os.path.isfile(self.data_path):
            data_files = [self.data_path]
        else:
            data_files = [
                os.path.join(self.data_path, f) 
                for f in os.listdir(self.data_path) 
                if f.endswith('.jsonl') or f.endswith('.json')
            ]
        
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    for line in f:
                        data = json.loads(line.strip())
                        self._process_example(data)
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            self._process_example(item)
                    else:
                        self._process_example(data)
    
    def _process_example(self, data: Dict[str, Any]):
        """Process a single example"""
        if 'input_ids' in data and 'prompt_length' in data:
            # Direct format: input_ids and prompt_length provided
            input_ids = data['input_ids']
            prompt_length = data['prompt_length']
            
            if len(input_ids) <= self.max_length:
                self.examples.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'prompt_length': prompt_length
                })
        
        elif 'conversations' in data:
            # Conversation format - need to construct input_ids and find prompt_length
            self._process_conversation(data['conversations'])
        
        elif 'instruction' in data and 'response' in data:
            # Instruction-response format
            self._process_instruction_response(data)
    
    def _process_conversation(self, conversations: List[Dict[str, str]]):
        """Process conversation format"""
        # This is a simplified implementation
        # You would need a proper tokenizer for real implementation
        # For now, assume the conversation is already tokenized
        
        # Find where the assistant response starts (this is the prompt length)
        prompt_tokens = []
        response_tokens = []
        
        for turn in conversations:
            if turn['from'] == 'human' or turn['from'] == 'user':
                # Add user tokens to prompt
                if 'input_ids' in turn:
                    prompt_tokens.extend(turn['input_ids'])
            elif turn['from'] == 'gpt' or turn['from'] == 'assistant':
                # Assistant response
                if 'input_ids' in turn:
                    response_tokens.extend(turn['input_ids'])
                break  # Only process first assistant response
        
        if prompt_tokens and response_tokens:
            input_ids = prompt_tokens + response_tokens
            prompt_length = len(prompt_tokens)
            
            if len(input_ids) <= self.max_length:
                self.examples.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'prompt_length': prompt_length
                })
    
    def _process_instruction_response(self, data: Dict[str, str]):
        """Process instruction-response format"""
        # This would require a tokenizer in practice
        # For now, assume tokenized versions are provided
        if 'instruction_ids' in data and 'response_ids' in data:
            instruction_ids = data['instruction_ids']
            response_ids = data['response_ids']
            
            input_ids = instruction_ids + response_ids
            prompt_length = len(instruction_ids)
            
            if len(input_ids) <= self.max_length:
                self.examples.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'prompt_length': prompt_length
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def sft_collate_fn(batch: List[Dict[str, Any]], max_length: int = 4096, pad_token_id: int = 0):
    """Collate function for SFT data"""
    input_ids = [item["input_ids"] for item in batch]
    prompt_lengths = [item["prompt_length"] for item in batch]
    
    # Pad sequences to the same length
    batch_max_len = min(max_length, max(len(seq) for seq in input_ids))
    
    padded_input_ids = []
    for seq in input_ids:
        if len(seq) > batch_max_len:
            # Truncate if too long
            padded_seq = seq[:batch_max_len]
        else:
            # Pad if too short (pad with EOS tokens as shown in paper)
            padding_length = batch_max_len - len(seq)
            # Use EOS token for padding as shown in the paper example
            eos_token_id = 2  # Adjust according to your tokenizer
            padded_seq = torch.cat([seq, torch.full((padding_length,), eos_token_id, dtype=torch.long)])
        padded_input_ids.append(padded_seq)
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long)
    }


class LLaDASFTTrainer:
    """LLaDA supervised fine-tuning trainer"""
    
    def __init__(self, config: SFTConfig):
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
        # Load pretrained model
        if os.path.exists(os.path.join(self.config.model_name_or_path, "pytorch_model.bin")):
            # Load config
            config_path = os.path.join(self.config.model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                model_config = LlamaConfig(**saved_config)
            else:
                model_config = LlamaConfig()
            
            # Initialize model and load weights
            self.model = LLaDAForMaskedLM(model_config)
            state_dict = torch.load(
                os.path.join(self.config.model_name_or_path, "pytorch_model.bin"),
                map_location="cpu"
            )
            self.model.load_state_dict(state_dict)
            self.logger.info(f"Loaded pretrained model from {self.config.model_name_or_path}")
        else:
            # Initialize from scratch if no pretrained model
            model_config = LlamaConfig()
            self.model = LLaDAForMaskedLM(model_config)
            self.logger.info("Initialized model from scratch")
        
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
    
    def compute_sft_loss(self, input_ids: torch.Tensor, prompt_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute LLaDA SFT loss
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            prompt_lengths: Length of prompt for each sequence in batch
        
        Returns:
            Dictionary containing loss and metrics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Apply forward diffusion process
        noisy_batch, _, p_mask = forward_process(input_ids, eps=self.config.eps)
        
        # Do not add noise to the prompt (key difference from pre-training)
        token_positions = torch.arange(seq_len, device=device).expand(batch_size, seq_len)
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        
        # Restore original tokens for prompt positions
        noisy_batch[prompt_mask] = input_ids[prompt_mask]
        
        # Calculate answer lengths (including padded tokens)
        prompt_mask_int = prompt_mask.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_mask_int), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, seq_len)
        
        # Find masked positions
        masked_indices = (noisy_batch == self.config.mask_token_id)
        
        # Forward pass through model
        outputs = self.model(input_ids=noisy_batch)
        logits = outputs.logits
        
        # Compute loss only on masked tokens
        if masked_indices.sum() == 0:
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "num_masked_tokens": torch.tensor(0),
                "perplexity": torch.tensor(float('inf'))
            }
        
        # Get predictions and targets for masked tokens
        masked_logits = logits[masked_indices]
        masked_targets = input_ids[masked_indices]
        masked_p_mask = p_mask[masked_indices]
        masked_answer_lengths = answer_lengths[masked_indices]
        
        # Compute cross-entropy loss
        token_loss = F.cross_entropy(
            masked_logits,
            masked_targets,
            reduction='none'
        )
        
        # Weight by inverse masking probability
        weighted_token_loss = token_loss / masked_p_mask
        
        # Normalize by answer length and batch size as in the paper
        ce_loss = torch.sum(weighted_token_loss / masked_answer_lengths) / batch_size
        
        # Compute metrics
        with torch.no_grad():
            num_masked_tokens = masked_indices.sum()
            perplexity = torch.exp(token_loss.mean()) if len(token_loss) > 0 else torch.tensor(float('inf'))
            
            # Additional metrics for SFT
            num_prompt_tokens = prompt_mask.sum()
            num_response_tokens = masked_indices.sum()
        
        return {
            "loss": ce_loss,
            "num_masked_tokens": num_masked_tokens,
            "num_prompt_tokens": num_prompt_tokens,
            "num_response_tokens": num_response_tokens,
            "perplexity": perplexity
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        prompt_lengths = batch["prompt_lengths"].to(self.device)
        
        # Compute loss
        loss_dict = self.compute_sft_loss(input_ids, prompt_lengths)
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
        dataset = SFTDataset(
            self.config.data_path,
            max_length=self.config.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=lambda batch: sft_collate_fn(batch, self.config.max_length, self.config.pad_token_id),
            pin_memory=True
        )
        
        self.logger.info(f"Starting SFT training with {len(dataset)} examples")
        
        global_step = 0
        total_loss = 0.0
        
        # Create infinite dataloader
        def infinite_dataloader():
            while True:
                for batch in dataloader:
                    yield batch
        
        data_iter = infinite_dataloader()
        
        # Training loop
        progress_bar = tqdm(range(self.config.max_steps), desc="SFT Training")
        
        for step in progress_bar:
            batch = next(data_iter)
            
            # Training step
            metrics = self.train_step(batch)
            
            total_loss += metrics["loss"] * self.config.gradient_accumulation_steps
            
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
                        "train/num_masked_tokens": metrics["num_masked_tokens"],
                        "train/num_prompt_tokens": metrics["num_prompt_tokens"],
                        "train/num_response_tokens": metrics["num_response_tokens"]
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
        self.logger.info("SFT training completed!")
    
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
    """Main SFT training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLaDA Supervised Fine-tuning")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model_name_or_path", type=str, default="./llada_pretrained", help="Path to pretrained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to SFT data")
    parser.add_argument("--output_dir", type=str, default="./llada_sft", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = SFTConfig(**config_dict)
    else:
        config = SFTConfig()
    
    # Override with command line arguments
    if args.model_name_or_path:
        config.model_name_or_path = args.model_name_or_path
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
    trainer = LLaDASFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 