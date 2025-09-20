import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision
from torch.nn.parallel import DistributedDataParallel as DDP
import functools
import torch.distributed as dist
import os
import json
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional, List
import wandb
from huggingface_hub import HfApi

# Import our modules
from llada_model import LLaDAForMaskedLM
from pretraining import forward_process
from tts_config import TTSConfig
from tts_dataset import create_tts_datasets, tts_data_collator, AlternatingDistributedSampler
from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR

# SNAC Hierarchical Structure Constants
SNAC_AUDIO_TOKEN_START = 128266  # 128256 + 10 (base + special tokens)
SNAC_CODEBOOK_SIZE = 4096

def get_valid_snac_token_range(position_mod_7: int) -> tuple:
    """
    Get valid token ID range for dataset's linear position mapping
    
    The dataset uses linear position-based mapping where each position (0-6)
    gets its own 4096-token range, rather than the hierarchical SNAC structure.
    This matches how the dataset was preprocessed.
    
    Args:
        position_mod_7: Position within 7-token frame (0-6)
        
    Returns:
        tuple: (start_token_id, end_token_id) for valid range
    """
    base = SNAC_AUDIO_TOKEN_START
    
    # Linear position mapping: each position gets its own 4096-token range
    start_token = base + position_mod_7 * SNAC_CODEBOOK_SIZE
    end_token = start_token + SNAC_CODEBOOK_SIZE
    
    return (start_token, end_token)

def validate_snac_token(token_id: int, position_mod_7: int) -> bool:
    """Check if token_id is valid for the given SNAC position"""
    valid_range = get_valid_snac_token_range(position_mod_7)
    return valid_range[0] <= token_id < valid_range[1]

class LLaDATTSTrainer:
    """LLaDA trainer specifically for TTS with mixed text and audio data"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.use_data_parallel = False  # Initialize before setup_distributed
        self.use_wandb = config.use_wandb  # Store wandb status
        
        # Apply Liger Kernel optimizations FIRST, before any model operations
        if self.config.use_liger_kernel:
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama(
                    rope=True,
                    rms_norm=True,
                    swiglu=True,
                    cross_entropy=False,  # Disable regular cross_entropy when using fused version
                    fused_linear_cross_entropy=True  # Use fused version for better performance
                )
                print("🚀 Applied Liger Kernel optimizations (20% speedup + 60% memory reduction)")
            except ImportError:
                print("⚠️  Liger Kernel not installed. Install with: pip install liger-kernel")
                print("   Missing out on 20% speedup + 60% memory reduction")
            except Exception as e:
                print(f"⚠️  Failed to apply Liger Kernel optimizations: {e}")
        
        self.setup_logging()
        self.setup_distributed()
        self.setup_tokenizer()
        self.setup_model()
        self.setup_datasets()  # Move before optimizer to calculate max_steps first
        self.setup_optimizer()
        
        # Tracking for different data types
        self.text_step = 0
        self.audio_step = 0
        self.global_step = 0
        self.start_epoch = 0
        
        # HuggingFace Hub API
        self.api = HfApi()
        
        # Initialize wandb if configured
        if self.use_wandb and config.wandb_project and self.is_main_process():
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
    
    def setup_distributed(self):
        """Setup distributed training"""
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Check if we're in a distributed environment
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                # Running with torchrun or similar distributed launcher
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl')
                
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
                torch.cuda.set_device(self.local_rank)
                self.use_data_parallel = False
                
                print(f"🔧 Distributed training initialized: rank {self.rank}/{self.world_size}, local_rank {self.local_rank}")
            else:
                # Single-node multi-GPU without torchrun - just use GPU 0 and warn
                self.world_size = 1
                self.rank = 0
                self.local_rank = 0
                self.use_data_parallel = False
                
                print(f"⚠️  Multiple GPUs detected ({torch.cuda.device_count()}) but not using distributed launcher")
                print(f"💡 For multi-GPU training, use: python run_distributed_tts.py --config your_config.yaml --gpus {torch.cuda.device_count()}")
                print(f"🔧 Running on single GPU (GPU 0) for now")
        else:
            # Single GPU or CPU
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.use_data_parallel = False
            
            print(f"🔧 Single device training: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
    
    def is_main_process(self):
        """Check if this is the main process"""
        return self.rank == 0
    
    def setup_tokenizer(self):
        """Setup tokenizer with custom tokens"""
        self.logger.info("Setting up tokenizer with custom tokens")
        
        # Load base tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        
        # Add custom tokens for TTS
        new_tokens = self.config.get_new_token_names()
        self.logger.info(f"Adding {len(new_tokens)} custom tokens for TTS")
        
        num_added = self.tokenizer.add_tokens(new_tokens)
        self.logger.info(f"Successfully added {num_added} new tokens")
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.config.pad_token_id = self.tokenizer.eos_token_id
    
    def setup_model(self):
        """Setup LLaDA model with resized embeddings"""
        self.logger.info("Setting up LLaDA model")
        
        # Configure BitsAndBytes if enabled
        bnb_config = None
        if self.config.use_bnb_quantization:
            try:
                from transformers import BitsAndBytesConfig
                
                if self.config.bnb_4bit:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                    self.logger.info("🔧 Using 4-bit BitsAndBytes quantization")
                else:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.logger.info("🔧 Using 8-bit BitsAndBytes quantization")
                    
            except ImportError:
                self.logger.warning("⚠️ BitsAndBytes not installed. Install with: pip install bitsandbytes")
                self.logger.warning("   Continuing without quantization...")
                bnb_config = None
        
        # Try loading as AutoModel first, then LlamaForCausalLM
        self.logger.info("Attempting to load with AutoModelForCausalLM...")
        
        # For distributed training, we need to be more careful with device mapping
        if bnb_config and self.world_size > 1:
            # In distributed setup, load on current device only
            device_map = {"": self.local_rank}
            self.logger.info(f"Loading quantized model on device {self.local_rank} for distributed training")
        elif bnb_config:
            # Single GPU setup can use auto device mapping
            device_map = "auto"
            self.logger.info("Loading quantized model with auto device mapping")
        else:
            # No quantization, load on CPU first
            device_map = "cpu"
        
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float32 if not bnb_config else torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        self.logger.info("Successfully loaded with AutoModelForCausalLM")
        
        # Get and validate config
        model_config = hf_model.config
        self.logger.info(f"Loaded model config: {type(model_config)}")
        self.logger.info(f"Model architecture: {model_config.architectures if hasattr(model_config, 'architectures') else 'unknown'}")
        
        # Create the LLaDA model structure but then swap in the quantized base model
        self.logger.info("Creating LLaDA model shell...")
        self.model = LLaDAForMaskedLM(model_config)
        
        self.logger.info("Swapping in base model...")
        self.model.model = hf_model.model
        
        self.logger.info("Copying LM head...")
        self.model.lm_head = hf_model.lm_head

        # Clean up HF model to save memory
        # del hf_model
        # torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.logger.info(f"Successfully adapted model to LLaDA structure.")
                
        # Resize token embeddings for custom tokens
        original_vocab_size = self.model.config.vocab_size
        new_vocab_size = len(self.tokenizer)
        
        if new_vocab_size > original_vocab_size:
            self.logger.info(f"Resizing embeddings from {original_vocab_size} to {new_vocab_size}")
            self.model.resize_token_embeddings(new_vocab_size)
        
        # Move model to device with appropriate precision
        if bnb_config:
            # Model is already on the correct device with quantization
            self.logger.info(f"✅ Model with quantization on device {self.device}")
        else:
            # Determine the appropriate dtype
            if getattr(self.config, 'fp16', True):
                model_dtype = torch.float16
                precision_info = "FP16"
            else:
                model_dtype = torch.float32
                precision_info = "FP32"
            
            self.model = self.model.to(device=self.device, dtype=model_dtype)
            self.logger.info(f"✅ Model moved to device {self.device} with {precision_info} precision")

        # If not using FSDP but in a distributed environment, use DDP
        if not self.config.fsdp and self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            self.logger.info("🚀 Wrapped model with DistributedDataParallel (DDP)")

        # Enable gradient checkpointing AFTER wrapping
        if getattr(self.config, 'gradient_checkpointing', True):
            # DDP requires accessing the original model via .module
            model_to_checkpoint = self.model.module if isinstance(self.model, DDP) else self.model
            model_to_checkpoint.gradient_checkpointing_enable()
            self.logger.info("✅ Enabled gradient checkpointing")
        else:
            self.logger.info("⚠️ Gradient checkpointing disabled")

        self.logger.info(f"Model setup complete. Parameters on rank {self.rank}: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def compute_position_aware_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                                  masked_indices: torch.Tensor, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute position-aware loss that enforces SNAC hierarchical constraints (Vectorized)
        
        This function ensures that the model only predicts valid SNAC tokens for each 
        position within the 7-token hierarchical frame structure.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            masked_indices: Boolean mask indicating which positions to compute loss on
            input_ids: Original input sequence for finding audio regions
            
        Returns:
            dict: Loss statistics and debugging information
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Constants for token detection
        START_OF_SPEECH = 128257
        END_OF_SPEECH = 128258

        # Find start and end of speech for the whole batch
        start_speech_indices = (input_ids == START_OF_SPEECH).nonzero(as_tuple=True)
        end_speech_indices = (input_ids == END_OF_SPEECH).nonzero(as_tuple=True)

        # Create a mask for valid audio regions to compute loss on
        audio_region_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # This part still requires a loop, but it's over the batch size, not every token.
        for batch_idx in range(batch_size):
            # Find relevant start/end markers for this specific batch item
            starts = start_speech_indices[1][start_speech_indices[0] == batch_idx]
            ends = end_speech_indices[1][end_speech_indices[0] == batch_idx]
            if len(starts) > 0 and len(ends) > 0:
                # Use first start and end marker
                audio_region_mask[batch_idx, starts[0] + 1:ends[0]] = True

        # Combine with the original masked_indices
        final_loss_mask = audio_region_mask & masked_indices
        
        # If no audio tokens are masked, return zero loss
        if final_loss_mask.sum() == 0:
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "num_position_aware_tokens": 0,
            }

        # --- Vectorized Position Calculation ---
        # Create a tensor representing the position of each token in the sequence
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Calculate the position within each 7-token frame
        # We need to find the start of the audio segment for each batch item to do this accurately
        # We assume the first START_OF_SPEECH token is the reference.
        first_start_positions = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        # Find the first occurrence of START_OF_SPEECH in each row using a vectorized approach
        b_indices, s_indices = start_speech_indices
        if b_indices.numel() > 0:
           # Use scatter_reduce to find the minimum sequence index (first occurrence) for each batch index.
           max_seq_len_val = seq_len + 1 
           min_s_indices = torch.full((batch_size,), max_seq_len_val, dtype=torch.long, device=device)
           min_s_indices.scatter_reduce_(0, b_indices, s_indices.long(), reduce="amin", include_self=False)
           
           # Update the positions where a start token was actually found
           found_mask = min_s_indices != max_seq_len_val
           first_start_positions[found_mask] = min_s_indices[found_mask]

        # Calculate relative position from the start of speech token
        relative_positions = positions - (first_start_positions[:, None] + 1)
        position_in_frame = relative_positions % 7
        
        # Only consider positions within the valid audio regions
        position_in_frame[~audio_region_mask] = -1 # Mark invalid positions

        # --- Vectorized Logit Masking ---
        # Create a mask for the entire vocabulary based on position
        # Shape: [batch_size, seq_len, 7, vocab_size]
        
        # Create position-aware mask for the vocabulary
        pos_in_frame_expanded = position_in_frame.unsqueeze(-1)
        valid_codebook_indices = (pos_in_frame_expanded >= 0) * (pos_in_frame_expanded < 7)

        # Create ranges for valid tokens for each position
        base = SNAC_AUDIO_TOKEN_START
        pos_indices = torch.arange(7, device=device)
        start_tokens = base + pos_indices * SNAC_CODEBOOK_SIZE
        end_tokens = start_tokens + SNAC_CODEBOOK_SIZE

        # Get the valid ranges for each token in the input
        # Shape: [batch_size, seq_len, 2]
        valid_ranges = torch.stack([
            torch.gather(start_tokens.unsqueeze(0).expand(batch_size, -1), 1, position_in_frame.clamp(0, 6)),
            torch.gather(end_tokens.unsqueeze(0).expand(batch_size, -1), 1, position_in_frame.clamp(0, 6))
        ], dim=-1)

        # Create a full vocabulary mask
        vocab_indices = torch.arange(vocab_size, device=device).view(1, 1, -1)
        
        # The mask is True where the token is *invalid*
        logit_mask = ~((vocab_indices >= valid_ranges[..., 0:1]) & (vocab_indices < valid_ranges[..., 1:2]))
        logit_mask[~final_loss_mask] = False # Don't mask logits that are not part of the loss

        # Apply the mask to the logits
        masked_logits = logits.masked_fill(logit_mask, float('-inf'))

        # --- Loss Calculation ---
        # Select only the logits and targets where we need to compute the loss
        loss_logits = masked_logits[final_loss_mask]
        loss_targets = targets[final_loss_mask]

        if loss_logits.numel() == 0:
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "num_position_aware_tokens": 0,
            }

        loss = F.cross_entropy(loss_logits, loss_targets)

        return {
            "loss": loss,
            "num_position_aware_tokens": loss_logits.size(0),
        }

    def _calculate_metrics(self, loss_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Calculates metrics in a no_grad context to avoid FSDP state errors."""
        with torch.no_grad():
            loss_type = loss_outputs.get("loss_type", "unknown")
            loss = loss_outputs["loss"]
            masked_indices = loss_outputs["masked_indices"]
            input_ids = loss_outputs["input_ids"]
            logits = loss_outputs["logits"]
            
            num_masked_tokens = masked_indices.sum()
            if num_masked_tokens == 0:
                return {
                    "loss": loss.item(), "num_masked_tokens": 0, "perplexity": float('inf'), "loss_type": loss_type
                }

            perplexity = torch.exp(loss)
            
            metrics = {
                "loss": loss.item(),
                "num_masked_tokens": num_masked_tokens.item(),
                "perplexity": perplexity.item(),
                "loss_type": loss_type
            }

            # Add masking analysis and prediction sampling
            from tts_forward_process import analyze_tts_masking
            masking_analysis = analyze_tts_masking(input_ids, masked_indices=masked_indices)
            metrics.update({
                "audio_mask_percentage": masking_analysis["audio_mask_rate"],
                "text_mask_percentage": masking_analysis["text_mask_rate"],
                "correctly_targeted_masking": masking_analysis["correctly_targeted"]
            })

            masked_logits = logits[masked_indices]
            masked_targets = input_ids[masked_indices]
            masked_predictions = torch.argmax(masked_logits, dim=-1)
            
            sample_size = min(50, len(masked_targets))
            metrics.update({
                "sample_targets": masked_targets[:sample_size].cpu().tolist(),
                "sample_predictions": masked_predictions[:sample_size].cpu().tolist(),
            })

            return metrics

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
        
        # Set initial momentum based on config
        initial_beta1 = getattr(self.config, 'initial_momentum', 0.9)
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(initial_beta1, 0.95),
            eps=1e-8
        )
        
        # Store original learning rate for momentum decay compensation
        self.original_lr = self.config.learning_rate
        self.original_beta1 = initial_beta1
        
        # Learning rate scheduler
        total_steps = self.steps_per_epoch * self.config.epochs
        
        if self.config.lr_scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.lr_scheduler_type == "constant":
            self.scheduler = LambdaLR(self.optimizer, lambda step: 1.0)
        else:
            # Default to linear warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
    
    def setup_datasets(self):
        """Setup datasets and dataloader"""
        self.logger.info("Setting up datasets")
        
        # Create datasets
        self.text_dataset, self.tts_dataset, self.combined_dataset = create_tts_datasets(
            self.config, self.tokenizer
        )
        
        # Calculate warmup steps for scheduler
        dataset_size = len(self.combined_dataset)
        self.warmup_steps = self.config.calculate_warmup_steps(dataset_size)
        self.steps_per_epoch = dataset_size // (self.config.batch_size * self.config.gradient_accumulation_steps * self.world_size)
        if self.steps_per_epoch == 0:
            self.steps_per_epoch = 1
        self.logger.info(f"Training setup: {self.config.epochs} epochs, {self.steps_per_epoch} steps per epoch, {self.warmup_steps} warmup steps")
        
        # Log dataset information
        if self.config.ratio == 0.0:
            self.logger.info(f"TTS-only training with {len(self.tts_dataset)} TTS examples")
        else:
            self.logger.info(f"Mixed training: {len(self.text_dataset)} text + {len(self.tts_dataset)} TTS examples")
        
        # Create dataloader with custom sampler
        if self.world_size > 1:
            if self.config.ratio > 0.0:
                # Use alternating sampler for mixed training
                sampler = AlternatingDistributedSampler(
                    self.combined_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                )
            else:
                # Use regular distributed sampler for TTS-only
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(
                    self.combined_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                )
        else:
            sampler = None
        
        self.dataloader = DataLoader(
            self.combined_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            collate_fn=lambda batch: tts_data_collator(batch, self.config.pad_token_id),
            drop_last=True,
            num_workers=self.config.num_workers,  # Set to 0 for FSDP compatibility
            pin_memory=True if torch.cuda.is_available() else False,
        )
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], training_progress: float = None) -> Dict[str, torch.Tensor]:
        """
        Compute loss for mixed text/TTS data
        """
        input_ids = batch["input_ids"].to(self.device)
        # Note: attention_mask removed - not needed for FlashAttention-only LLaDA
        data_types = batch.get("data_types", [])
        
        # Determine training approach based on training_mode config
        if self.config.training_mode == "pretraining":
            # Always use pretraining loss regardless of data type
            return self._compute_pretraining_loss(input_ids, training_progress)
        elif self.config.training_mode == "sft":
            # Use SFT loss for TTS data, pretraining loss for text data
            is_tts_batch = any(dt == 'tts' for dt in data_types)
            
            if is_tts_batch and "prompt_lengths" in batch:
                # TTS data - use SFT-style training with prompt masking
                prompt_lengths = batch["prompt_lengths"].to(self.device)
                return self._compute_sft_loss(input_ids, prompt_lengths, training_progress)
            else:
                # Text data - use standard pre-training
                return self._compute_pretraining_loss(input_ids, training_progress)
        else:
            raise ValueError(f"Unknown training_mode: {self.config.training_mode}. Use 'pretraining' or 'sft'")
    
    def _compute_pretraining_loss(self, input_ids: torch.Tensor, training_progress: float = None) -> Dict[str, torch.Tensor]:
        """Compute pre-training loss with TTS-aware masking (only mask audio tokens)"""
        # Import TTS-aware forward process
        from tts_forward_process import tts_forward_process, analyze_tts_masking
        
        # Apply TTS-aware forward diffusion process (only mask audio tokens)
        noisy_batch, masked_indices, p_mask = tts_forward_process(
            input_ids, 
            eps=self.config.eps,
            training_progress=training_progress,
            use_linear_schedule=getattr(self.config, 'use_linear_masking_schedule', True),
            use_curriculum_learning=getattr(self.config, 'use_curriculum_learning', False),
            curriculum_target_progress=getattr(self.config, 'curriculum_target_progress', 0.8)
        )
        
        # Forward pass through model
        outputs = self.model(input_ids=noisy_batch)
        logits = outputs.logits
        
        # Compute loss only on masked tokens
        if masked_indices.sum() == 0:
            return {
                "loss": torch.tensor(0.0, device=self.device, requires_grad=True),
                "logits": logits,
                "masked_indices": masked_indices,
                "input_ids": input_ids,
                "loss_type": "pretraining"
            }
        
        # Get predictions and targets for masked tokens only
        masked_logits = logits[masked_indices]
        masked_targets = input_ids[masked_indices]
        masked_p_mask = p_mask[masked_indices]
        
        # Choose loss computation method based on configuration
        if getattr(self.config, 'use_position_aware_loss', False):
            # Use position-aware loss for SNAC hierarchical constraints
            position_loss_result = self.compute_position_aware_loss(logits, input_ids, masked_indices, input_ids)
            loss = position_loss_result["loss"]
            
            self.logger.debug(f"Using position-aware loss with SNAC constraints")
        else:
            # Standard cross-entropy loss
            token_loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')
            
            # Weight by inverse masking probability (optional)
            if self.config.use_weighted_loss:
                weighted_token_loss = token_loss / masked_p_mask
                self.logger.debug(f"Using weighted loss (1/p_mask)")
            else:
                weighted_token_loss = token_loss
                self.logger.debug(f"Using unweighted loss")
            
            # Average loss
            loss = weighted_token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        
        # Compute metrics
        with torch.no_grad():
            num_masked_tokens = masked_indices.sum()
            perplexity = torch.exp(loss)
        
        # Collect additional debugging info
        debug_info = {}
        
        # Add TTS masking analysis
        if num_masked_tokens > 0:
            masking_analysis = analyze_tts_masking(input_ids, masked_indices=masked_indices)
            debug_info.update({
                "masking_analysis": masking_analysis,
                "audio_mask_percentage": masking_analysis["audio_mask_rate"],
                "text_mask_percentage": masking_analysis["text_mask_rate"],
                "correctly_targeted_masking": masking_analysis["correctly_targeted"]
            })
        
        # Sample predictions for logging (more comprehensive sampling)
        with torch.no_grad():
            if num_masked_tokens > 0:
                # Get predictions
                masked_predictions = torch.argmax(masked_logits, dim=-1)
                
                # Sample more tokens for better analysis (up to 50 tokens)
                sample_size = min(50, len(masked_targets))
                debug_info.update({
                    "sample_targets": masked_targets[:sample_size].cpu().tolist(),
                    "sample_predictions": masked_predictions[:sample_size].cpu().tolist(),
                    "sample_probabilities": torch.softmax(masked_logits[:sample_size], dim=-1).max(dim=-1)[0].cpu().tolist(),
                    "sample_p_mask": masked_p_mask[:sample_size].cpu().tolist()
                })
                
                # Token type analysis (hardcoded SNAC constants)
                TOKENISER_LENGTH = 128256
                vocab_size = TOKENISER_LENGTH  # Use hardcoded base vocab size
                audio_token_start = vocab_size  # Audio tokens start after base vocab
                
                # Count token types in targets
                text_tokens = (masked_targets < vocab_size).sum().item()
                custom_tokens = (masked_targets >= vocab_size).sum().item()
                
                # Count token types in predictions  
                pred_text_tokens = (masked_predictions < vocab_size).sum().item()
                pred_custom_tokens = (masked_predictions >= vocab_size).sum().item()
                
                debug_info.update({
                    "target_text_tokens": text_tokens,
                    "target_custom_tokens": custom_tokens,
                    "pred_text_tokens": pred_text_tokens, 
                    "pred_custom_tokens": pred_custom_tokens,
                    "text_token_ratio": text_tokens / num_masked_tokens.item() if num_masked_tokens > 0 else 0,
                    "custom_token_ratio": custom_tokens / num_masked_tokens.item() if num_masked_tokens > 0 else 0
                })

        return {
            "loss": loss,
            "logits": logits,
            "masked_indices": masked_indices,
            "input_ids": input_ids,
            "loss_type": "pretraining"
        }
    
    def _compute_sft_loss(self, input_ids: torch.Tensor, prompt_lengths: torch.Tensor, training_progress: float = None) -> Dict[str, torch.Tensor]:
        """Compute SFT loss for TTS data with TTS-aware masking"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Import TTS-aware forward process
        from tts_forward_process import tts_forward_process, analyze_tts_masking
        
        # Apply TTS-aware forward diffusion process (only mask audio tokens)
        noisy_batch, masked_indices, p_mask = tts_forward_process(
            input_ids, 
            eps=self.config.eps,
            training_progress=training_progress,
            use_linear_schedule=getattr(self.config, 'use_linear_masking_schedule', True),
            use_curriculum_learning=getattr(self.config, 'use_curriculum_learning', False),
            curriculum_target_progress=getattr(self.config, 'curriculum_target_progress', 0.8)
        )
        
        # For SFT, we DON'T need to restore prompt tokens because TTS-aware masking
        # already ensures that only audio tokens are masked (never text/prompt tokens)
        
        # Calculate answer lengths using TTS token boundaries (hardcoded SNAC constants)
        TOKENISER_LENGTH = 128256
        START_OF_SPEECH = TOKENISER_LENGTH + 1  # 128257
        END_OF_SPEECH = TOKENISER_LENGTH + 2    # 128258
        
        answer_lengths = torch.zeros((batch_size, seq_len), device=device)
        
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx]
            
            # Find START_OF_SPEECH and END_OF_SPEECH positions
            start_speech_positions = (sequence == START_OF_SPEECH).nonzero(as_tuple=True)[0]
            end_speech_positions = (sequence == END_OF_SPEECH).nonzero(as_tuple=True)[0]
            
            if len(start_speech_positions) > 0 and len(end_speech_positions) > 0:
                start_pos = start_speech_positions[0].item()
                end_pos = end_speech_positions[0].item()
                
                # Answer length is the audio region length
                audio_length = max(1, end_pos - start_pos - 1)  # Avoid division by zero
                answer_lengths[batch_idx, :] = audio_length
        
        # masked_indices already computed by tts_forward_process
        
        # Forward pass through model
        outputs = self.model(input_ids=noisy_batch)
        logits = outputs.logits
        
        # Compute loss only on masked tokens
        if masked_indices.sum() == 0:
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "logits": logits,
                "masked_indices": masked_indices,
                "input_ids": input_ids,
                "loss_type": "sft"
            }
        
        # Get predictions and targets for masked tokens
        masked_logits = logits[masked_indices]
        masked_targets = input_ids[masked_indices]
        masked_p_mask = p_mask[masked_indices]
        masked_answer_lengths = answer_lengths[masked_indices]
        
        debug_info = {}
        # Choose loss computation method based on configuration
        if getattr(self.config, 'use_position_aware_loss', False):
            # Use position-aware loss for SNAC hierarchical constraints
            position_loss_result = self.compute_position_aware_loss(logits, input_ids, masked_indices, input_ids)
            ce_loss = position_loss_result["loss"]
            
            self.logger.debug(f"Using position-aware SFT loss with SNAC constraints")
        else:
            # Standard cross-entropy loss
            token_loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')
            
            # Weight by inverse masking probability (optional)
            if self.config.use_weighted_loss:
                weighted_token_loss = token_loss / masked_p_mask
            else:
                weighted_token_loss = token_loss
            
            # Normalize by answer length and batch size
            ce_loss = torch.sum(weighted_token_loss / masked_answer_lengths) / batch_size
        
        # Compute metrics
        with torch.no_grad():
            num_masked_tokens = masked_indices.sum()
            perplexity = torch.exp(ce_loss)
        
        # Add TTS masking analysis and prediction sampling for SFT too
        if num_masked_tokens > 0:
            masking_analysis = analyze_tts_masking(input_ids, masked_indices=masked_indices)
            debug_info.update({
                "audio_mask_percentage": masking_analysis["audio_mask_rate"],
                "text_mask_percentage": masking_analysis["text_mask_rate"],
                "correctly_targeted_masking": masking_analysis["correctly_targeted"]
            })
            
            # Sample predictions for logging (same as pretraining)
            with torch.no_grad():
                # Get predictions
                masked_predictions = torch.argmax(masked_logits, dim=-1)
                
                # Sample up to 50 tokens for analysis
                sample_size = min(50, len(masked_targets))
                debug_info.update({
                    "sample_targets": masked_targets[:sample_size].cpu().tolist(),
                    "sample_predictions": masked_predictions[:sample_size].cpu().tolist(),
                    "sample_probabilities": torch.softmax(masked_logits[:sample_size], dim=-1).max(dim=-1)[0].cpu().tolist(),
                    "sample_p_mask": masked_p_mask[:sample_size].cpu().tolist()
                })
                
                # Token type analysis (hardcoded SNAC constants)
                TOKENISER_LENGTH = 128256
                vocab_size = TOKENISER_LENGTH  # Use hardcoded base vocab size
                
                # Count token types in targets
                text_tokens = (masked_targets < vocab_size).sum().item()
                custom_tokens = (masked_targets >= vocab_size).sum().item()
                
                # Count token types in predictions  
                pred_text_tokens = (masked_predictions < vocab_size).sum().item()
                pred_custom_tokens = (masked_predictions >= vocab_size).sum().item()
                
                debug_info.update({
                    "target_text_tokens": text_tokens,
                    "target_custom_tokens": custom_tokens,
                    "pred_text_tokens": pred_text_tokens, 
                    "pred_custom_tokens": pred_custom_tokens,
                    "text_token_ratio": text_tokens / num_masked_tokens.item() if num_masked_tokens > 0 else 0,
                    "custom_token_ratio": custom_tokens / num_masked_tokens.item() if num_masked_tokens > 0 else 0
                })
        
        return {
            "loss": ce_loss,
            "logits": logits,
            "masked_indices": masked_indices,
            "input_ids": input_ids,
            "loss_type": "sft"
        }
    
    def update_momentum_and_lr(self, global_step: int, total_steps: int):
        """Update momentum and learning rate for momentum decay optimization"""
        if not getattr(self.config, 'use_momentum_decay', False):
            return
            
        # Calculate training progress
        tau = min(global_step / max(total_steps, 1), 1.0)
        
        # Calculate new momentum (β_t)
        beta0 = self.original_beta1
        final_beta = getattr(self.config, 'final_momentum', 0.5)
        beta_t = final_beta + (beta0 - final_beta) * (1 - tau)
        
        # Calculate compensated learning rate
        lr_compensation = (1 - beta0) / (1 - beta_t)
        new_lr = self.original_lr * lr_compensation
        
        # Update optimizer parameters
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            param_group['betas'] = (beta_t, param_group['betas'][1])
        
        # Store for logging
        self._current_momentum = beta_t
        self._current_lr_compensation = lr_compensation

    def train_step(self, batch: Dict[str, torch.Tensor], global_step: int = 0, total_steps: int = 1) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Update momentum and learning rate for diffusion optimization
        self.update_momentum_and_lr(global_step, total_steps)
        
        # Calculate training progress (0.0 to 1.0)
        training_progress = min(global_step / max(total_steps, 1), 1.0)
        
        # Store for logging
        self._last_training_progress = training_progress
        
        # Compute loss with training progress for linear masking schedule
        loss_outputs = self.compute_loss(batch, training_progress)
        loss = loss_outputs["loss"]
        
        # Backward pass
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        loss.backward()
        
        # Calculate metrics for logging
        metrics = self._calculate_metrics(loss_outputs)
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, Any], global_step: int, epoch: int = None):
        """Log metrics based on data type"""
        if not self.is_main_process():
            return
        
        loss_type = metrics.get("loss_type", "unknown")
        
        # For TTS-only training (ratio = 0), always log as audio
        if self.config.ratio == 0.0:
            wandb_metrics = {
                "audio_loss": metrics["loss"],
                "audio_perplexity": metrics["perplexity"],
                "audio_step": self.audio_step,
                "global_step": global_step,
                "learning_rate": self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate
            }
            
            # Add optimization metrics
            if hasattr(self, '_last_training_progress'):
                if getattr(self.config, 'use_curriculum_learning', False):
                    # Curriculum learning metrics
                    wandb_metrics["curriculum_gamma"] = min(self._last_training_progress / getattr(self.config, 'curriculum_target_progress', 0.8), 1.0)
                elif getattr(self.config, 'use_linear_masking_schedule', True):
                    # Linear masking schedule
                    target_mask_pct = self.config.eps + (1.0 - self.config.eps) * self._last_training_progress
                    wandb_metrics["target_masking_percentage"] = target_mask_pct * 100
            
            # Add momentum decay metrics
            if hasattr(self, '_current_momentum'):
                wandb_metrics["current_momentum"] = self._current_momentum
                wandb_metrics["lr_compensation_factor"] = self._current_lr_compensation
                wandb_metrics["effective_lr"] = self.optimizer.param_groups[0]['lr']
            if epoch is not None:
                wandb_metrics["epoch"] = epoch
            
            # Add debugging metrics for pretraining
            if loss_type == "pretraining":
                debug_metrics = {}
                for key in ["target_text_tokens", "target_custom_tokens", "pred_text_tokens", "pred_custom_tokens",
                           "text_token_ratio", "custom_token_ratio", "audio_mask_percentage", "text_mask_percentage", 
                           "correctly_targeted_masking", "position_aware_tokens", "valid_target_ratio", "avg_position_accuracy"]:
                    if key in metrics:
                        debug_metrics[f"debug_{key}"] = metrics[key]
                wandb_metrics.update(debug_metrics)
                
                # Add position-specific metrics if available
                if "position_stats" in metrics:
                    for pos, stats in metrics["position_stats"].items():
                        wandb_metrics[f"position_{pos}_accuracy"] = stats["accuracy"]
                        wandb_metrics[f"position_{pos}_loss"] = stats["avg_loss"]
            
            if self.use_wandb:
                wandb.log(wandb_metrics, step=global_step)
            
            # Enhanced console logging with token analysis
            console_msg = (
                f"TTS Step {self.audio_step}: loss={metrics['loss']:.4f}, "
                f"ppl={metrics['perplexity']:.2f}, masked_tokens={metrics['num_masked_tokens']}"
            )
            
            # Add optimization progress info
            if hasattr(self, '_last_training_progress'):
                if getattr(self.config, 'use_curriculum_learning', False):
                    gamma = min(self._last_training_progress / getattr(self.config, 'curriculum_target_progress', 0.8), 1.0)
                    console_msg += f", curriculum_γ={gamma:.2f}"
                elif getattr(self.config, 'use_linear_masking_schedule', True):
                    target_mask_pct = self.config.eps + (1.0 - self.config.eps) * self._last_training_progress
                    console_msg += f", target_mask={target_mask_pct:.1%}"
            
            # Add momentum decay info
            if hasattr(self, '_current_momentum'):
                console_msg += f", β={self._current_momentum:.3f}"
            
            if loss_type == "pretraining" and "text_token_ratio" in metrics:
                console_msg += (
                    f", text_ratio={metrics['text_token_ratio']:.2f}, "
                    f"custom_ratio={metrics['custom_token_ratio']:.2f}"
                )
                if "audio_mask_percentage" in metrics:
                    console_msg += f", audio_masked={metrics['audio_mask_percentage']:.1f}%"
                if "correctly_targeted_masking" in metrics:
                    console_msg += f", correct_masking={'✅' if metrics['correctly_targeted_masking'] else '❌'}"
                
                # Add position-aware metrics
                if "avg_position_accuracy" in metrics:
                    console_msg += f", pos_acc={metrics['avg_position_accuracy']:.3f}"
                if "valid_target_ratio" in metrics:
                    console_msg += f", valid_targets={metrics['valid_target_ratio']:.3f}"
            
            self.logger.info(console_msg)
            
            # Detailed prediction logging at configurable frequency
            if self.audio_step % self.config.prediction_logging_steps == 0 and "sample_targets" in metrics:
                self._log_prediction_analysis(metrics, global_step)
            
            self.audio_step += 1
        else:
            # Mixed training - track steps for different data types
            cycle_length = int(self.config.ratio * 10) + 1  # Convert ratio to cycle length
            
            if (global_step % cycle_length) == 0:  # TTS batch
                wandb_metrics = {
                    "audio_loss": metrics["loss"],
                    "audio_perplexity": metrics["perplexity"],
                    "audio_step": self.audio_step,
                    "global_step": global_step,
                    "learning_rate": self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate
                }
                
                # Add SFT masking metrics
                if loss_type == "sft":
                    for key in ["audio_mask_percentage", "text_mask_percentage", "correctly_targeted_masking"]:
                        if key in metrics:
                            wandb_metrics[f"debug_{key}"] = metrics[key]
                
                if self.use_wandb:
                    wandb.log(wandb_metrics, step=global_step)
                
                console_msg = (
                    f"Audio Step {self.audio_step}: loss={metrics['loss']:.4f}, "
                    f"ppl={metrics['perplexity']:.2f}, masked_tokens={metrics['num_masked_tokens']}"
                )
                
                # Add SFT masking info to console
                if loss_type == "sft":
                    if "audio_mask_percentage" in metrics:
                        console_msg += f", audio_masked={metrics['audio_mask_percentage']:.1f}%"
                    if "correctly_targeted_masking" in metrics:
                        console_msg += f", correct_masking={'✅' if metrics['correctly_targeted_masking'] else '❌'}"
                
                self.logger.info(console_msg)
                
                # Detailed prediction logging for SFT
                if self.audio_step % self.config.prediction_logging_steps == 0 and "sample_targets" in metrics:
                    self._log_prediction_analysis(metrics, global_step)
                
                self.audio_step += 1
            else:  # Text batch
                wandb_metrics = {
                    "text_loss": metrics["loss"],
                    "text_perplexity": metrics["perplexity"],
                    "text_step": self.text_step,
                    "global_step": global_step,
                    "learning_rate": self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate
                }
                if self.use_wandb:
                    wandb.log(wandb_metrics, step=global_step)
                
                self.logger.info(
                    f"Text Step {self.text_step}: loss={metrics['loss']:.4f}, "
                    f"ppl={metrics['perplexity']:.2f}, masked_tokens={metrics['num_masked_tokens']}"
                )
                self.text_step += 1
    
    def _log_prediction_analysis(self, metrics: Dict[str, Any], global_step: int):
        """Log detailed prediction vs ground truth analysis to WandB"""
        if not self.use_wandb:
            return
            
        try:
            import wandb
            
            # Create prediction analysis table
            predictions_data = []
            
            targets = metrics.get("sample_targets", [])
            predictions = metrics.get("sample_predictions", [])
            probabilities = metrics.get("sample_probabilities", [])
            p_masks = metrics.get("sample_p_mask", [])
            
            # Use hardcoded SNAC constants
            TOKENISER_LENGTH = 128256
            vocab_size = TOKENISER_LENGTH
            
            for i, (target, pred, prob, p_mask) in enumerate(zip(targets, predictions, probabilities, p_masks)):
                # Decode tokens to text if possible
                try:
                    target_text = self.tokenizer.decode([target]) if target < len(self.tokenizer) else f"<CUSTOM:{target}>"
                    pred_text = self.tokenizer.decode([pred]) if pred < len(self.tokenizer) else f"<CUSTOM:{pred}>"
                except:
                    target_text = f"<UNK:{target}>"
                    pred_text = f"<UNK:{pred}>"
                
                token_type = "text" if target < vocab_size else "custom"
                pred_type = "text" if pred < vocab_size else "custom"
                is_correct = target == pred
                
                # Calculate token distance for custom tokens
                token_distance = abs(target - pred) if token_type == "custom" and pred_type == "custom" else None
                
                predictions_data.append({
                    "index": i,
                    "target_id": target,
                    "prediction_id": pred,
                    "target_text": target_text,
                    "prediction_text": pred_text,
                    "probability": prob,
                    "p_mask": p_mask,
                    "target_type": token_type,
                    "prediction_type": pred_type,
                    "is_correct": is_correct,
                    "type_match": token_type == pred_type,
                    "token_distance": token_distance if token_distance else 0
                })
            
            # Create WandB table
            table = wandb.Table(
                columns=["index", "target_id", "prediction_id", "target_text", "prediction_text", 
                        "probability", "p_mask", "target_type", "prediction_type", "is_correct", "type_match", "token_distance"],
                data=[[row[col] for col in ["index", "target_id", "prediction_id", "target_text", "prediction_text", 
                      "probability", "p_mask", "target_type", "prediction_type", "is_correct", "type_match", "token_distance"]] 
                      for row in predictions_data]
            )
            
            # Enhanced statistics
            accuracy = sum(1 for row in predictions_data if row["is_correct"]) / len(predictions_data) if predictions_data else 0
            type_accuracy = sum(1 for row in predictions_data if row["type_match"]) / len(predictions_data) if predictions_data else 0
            
            # Separate analysis for text vs custom tokens
            text_predictions = [row for row in predictions_data if row["target_type"] == "text"]
            custom_predictions = [row for row in predictions_data if row["target_type"] == "custom"]
            
            text_accuracy = sum(1 for row in text_predictions if row["is_correct"]) / len(text_predictions) if text_predictions else 0
            custom_accuracy = sum(1 for row in custom_predictions if row["is_correct"]) / len(custom_predictions) if custom_predictions else 0
            
            # Average token distance for custom tokens
            custom_distances = [row["token_distance"] for row in custom_predictions if row["token_distance"] is not None]
            avg_custom_distance = sum(custom_distances) / len(custom_distances) if custom_distances else 0
            
            # Log comprehensive metrics
            wandb.log({
                "predictions_table": table,
                "prediction_accuracy": accuracy,
                "token_type_accuracy": type_accuracy,
                "text_token_accuracy": text_accuracy,
                "custom_token_accuracy": custom_accuracy,
                "avg_prediction_confidence": sum(row["probability"] for row in predictions_data) / len(predictions_data) if predictions_data else 0,
                "avg_custom_token_distance": avg_custom_distance,
                "num_text_predictions": len(text_predictions),
                "num_custom_predictions": len(custom_predictions),
                "total_predictions_analyzed": len(predictions_data)
            }, step=global_step)
            
            # Enhanced logging
            self.logger.info(f"📊 Prediction Analysis (Step {global_step}):")
            self.logger.info(f"   Overall Accuracy: {accuracy:.3f} ({sum(1 for row in predictions_data if row['is_correct'])}/{len(predictions_data)})")
            self.logger.info(f"   Type Accuracy: {type_accuracy:.3f}")
            self.logger.info(f"   Text Token Accuracy: {text_accuracy:.3f} ({len(text_predictions)} samples)")
            self.logger.info(f"   Custom Token Accuracy: {custom_accuracy:.3f} ({len(custom_predictions)} samples)")
            if avg_custom_distance > 0:
                self.logger.info(f"   Avg Custom Token Distance: {avg_custom_distance:.1f}")
            
            # Save detailed predictions to file for inspection
            self._save_predictions_to_file(predictions_data, global_step)
            
        except Exception as e:
            self.logger.warning(f"Failed to log prediction analysis: {e}")
    
    def _save_predictions_to_file(self, predictions_data: list, global_step: int):
        """Save detailed predictions to a text file for manual inspection"""
        if not self.is_main_process():
            return
            
        try:
            # Create predictions directory
            predictions_dir = os.path.join(self.config.output_dir, "predictions")
            os.makedirs(predictions_dir, exist_ok=True)
            
            # Write predictions to file
            file_path = os.path.join(predictions_dir, f"predictions_step_{global_step}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Prediction Analysis - Step {global_step}\n")
                f.write("=" * 60 + "\n\n")
                
                # Group by token type
                text_preds = [row for row in predictions_data if row["target_type"] == "text"]
                custom_preds = [row for row in predictions_data if row["target_type"] == "custom"]
                
                f.write(f"TEXT TOKEN PREDICTIONS ({len(text_preds)} samples):\n")
                f.write("-" * 40 + "\n")
                for row in text_preds:
                    status = "✅" if row["is_correct"] else "❌"
                    f.write(f"{status} Target: '{row['target_text']}' ({row['target_id']}) -> Pred: '{row['prediction_text']}' ({row['prediction_id']}) [conf: {row['probability']:.3f}]\n")
                
                f.write(f"\nCUSTOM TOKEN PREDICTIONS ({len(custom_preds)} samples):\n")
                f.write("-" * 40 + "\n")
                for row in custom_preds:
                    status = "✅" if row["is_correct"] else "❌"
                    distance = f" [dist: {row['token_distance']}]" if row['token_distance'] > 0 else ""
                    f.write(f"{status} Target: {row['target_id']} -> Pred: {row['prediction_id']} [conf: {row['probability']:.3f}]{distance}\n")
                
                # Summary
                f.write(f"\nSUMMARY:\n")
                f.write(f"Overall Accuracy: {sum(1 for row in predictions_data if row['is_correct'])}/{len(predictions_data)} = {sum(1 for row in predictions_data if row['is_correct'])/len(predictions_data):.3f}\n")
                f.write(f"Text Accuracy: {sum(1 for row in text_preds if row['is_correct'])}/{len(text_preds)} = {sum(1 for row in text_preds if row['is_correct'])/len(text_preds):.3f}\n" if text_preds else "Text Accuracy: N/A (no text tokens)\n")
                f.write(f"Custom Accuracy: {sum(1 for row in custom_preds if row['is_correct'])}/{len(custom_preds)} = {sum(1 for row in custom_preds if row['is_correct'])/len(custom_preds):.3f}\n" if custom_preds else "Custom Accuracy: N/A (no custom tokens)\n")
                
            self.logger.info(f"💾 Saved detailed predictions to {file_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save predictions to file: {e}")
    
    def save_checkpoint(self, output_dir: str, is_interrupt: bool = False):
        """Save model checkpoint"""
        if not self.is_main_process():
            return

        os.makedirs(output_dir, exist_ok=True)
        
        # Determine the model to save from (DDP wraps the original model in a .module attribute)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        if isinstance(self.model, FSDP):
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = self.model.state_dict()
            torch.save(cpu_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        self.tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step,
            "epoch": self.start_epoch,
            "text_step": self.text_step,
            "audio_step": self.audio_step
        }
        torch.save(training_state, os.path.join(output_dir, "training_state.bin"))
        
        if is_interrupt:
            self.logger.info(f"Interruption checkpoint saved to {output_dir}")
        else:
            self.logger.info(f"Model saved to {output_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_dir):
            self.logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return
            
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model

        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            model_to_load.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"Loaded model state from {model_path}")

        training_state_path = os.path.join(checkpoint_dir, "training_state.bin")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            self.optimizer.load_state_dict(training_state["optimizer"])
            self.scheduler.load_state_dict(training_state["scheduler"])
            self.global_step = training_state.get("step", 0)
            self.start_epoch = training_state.get("epoch", 0)
            self.text_step = training_state.get("text_step", 0)
            self.audio_step = training_state.get("audio_step", 0)
            self.logger.info(f"Loaded training state from {training_state_path}. Resuming from step {self.global_step} at epoch {self.start_epoch}")

    def train(self):
        """Main training loop using epochs"""
        self.logger.info(f"Starting TTS training with {len(self.combined_dataset)} examples for {self.config.epochs} epochs")
        self.logger.info(f"🎯 Training mode: {self.config.training_mode.upper()}")
        
        total_steps = self.steps_per_epoch * self.config.epochs
        steps_per_save = int(self.config.save_epochs * self.steps_per_epoch) if self.config.save_epochs > 0 else 0
        
        # Log enabled optimization strategies
        optimizations = []
        if getattr(self.config, 'use_curriculum_learning', False):
            optimizations.append(f"📚 Curriculum Learning (CLTS) → focus on harder timesteps over {self.config.curriculum_target_progress:.0%} of training")
        elif getattr(self.config, 'use_linear_masking_schedule', True):
            optimizations.append(f"📈 Linear masking: 1% → 100% over {total_steps} steps")
        
        if getattr(self.config, 'use_momentum_decay', False):
            optimizations.append(f"⚡ Momentum decay: β={self.config.initial_momentum:.1f} → {self.config.final_momentum:.1f} with LR compensation")
        
        for opt in optimizations:
            self.logger.info(opt)
        
        # Training loop by epochs
        for epoch in range(self.start_epoch, self.config.epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.epochs}")
            
            # Set epoch for distributed sampler
            if hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)
            
            # Initialize progress bar for this epoch
            progress_bar = tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch + 1}/{self.config.epochs}", 
                disable=not self.is_main_process(),
                total=len(self.dataloader)
            )
            
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop for this epoch
            for step, batch in enumerate(progress_bar):
                # Training step with linear masking schedule
                metrics = self.train_step(batch, self.global_step, total_steps)
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm and self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    epoch_loss += metrics['loss']
                    num_batches += 1
                    
                    if self.global_step > 0 and steps_per_save > 0 and self.global_step % steps_per_save == 0:
                        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-step-{self.global_step}")
                        self.logger.info(f"Saving checkpoint at step {self.global_step} to {checkpoint_dir}")
                        self.save_checkpoint(checkpoint_dir)

                    # Log metrics periodically
                    if self.global_step % 10 == 0:
                        if self.is_main_process():
                            # Print loss immediately
                            loss_type = "audio" if self.config.ratio == 0.0 else metrics.get('loss_type', 'unknown')
                            current_step = self.audio_step if self.config.ratio == 0.0 else (self.text_step if metrics.get('loss_type') == 'text' else self.audio_step)
                            ppl = torch.exp(torch.tensor(metrics['loss'])).item()
                            self.logger.info(f"Epoch {epoch + 1}, Step {current_step}: loss={metrics['loss']:.4f}, ppl={ppl:.2f}, masked_tokens={metrics.get('num_masked_tokens', 'N/A')}")
                            
                            self.log_metrics(metrics, self.global_step, epoch)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "type": metrics.get('loss_type', 'unknown'),
                    "epoch_avg": f"{epoch_loss / max(num_batches, 1):.4f}"
                })

            # Calculate epoch average loss
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.logger.info(f"Epoch {epoch + 1} completed! Average loss: {avg_epoch_loss:.4f}")
            
            progress_bar.close()
        
        # Save final checkpoint
        if self.is_main_process():
            final_checkpoint_dir = os.path.join(self.config.output_dir, "final_checkpoint")
            self.save_checkpoint(final_checkpoint_dir)
            self.logger.info(f"Training completed! Final model saved to {final_checkpoint_dir}")
            self.logger.info(f"Total training steps: {self.global_step}")
