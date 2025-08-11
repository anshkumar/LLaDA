import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
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


class LLaDATTSTrainer:
    """LLaDA trainer specifically for TTS with mixed text and audio data"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.setup_logging()
        self.setup_distributed()
        self.setup_tokenizer()
        self.setup_model()
        self.setup_datasets()  # Move before optimizer to calculate max_steps first
        self.setup_optimizer()
        
        # Tracking for different data types
        self.text_step = 0
        self.audio_step = 0
        
        # HuggingFace Hub API
        self.api = HfApi()
        
        # Initialize wandb if configured
        if config.wandb_project and self.is_main_process():
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
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
        
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
        
        # Load or create model config
        from transformers import LlamaConfig
        
        if os.path.exists(self.config.model_name_or_path):
            # Try to load from local checkpoint
            try:
                if os.path.exists(os.path.join(self.config.model_name_or_path, "config.json")):
                    # LLaDA checkpoint
                    model_config = LlamaConfig.from_pretrained(self.config.model_name_or_path)
                    self.model = LLaDAForMaskedLM(model_config)
                    
                    # Try to load model weights
                    model_path = os.path.join(self.config.model_name_or_path, "pytorch_model.bin")
                    if os.path.exists(model_path):
                        state_dict = torch.load(model_path, map_location="cpu")
                        self.model.load_state_dict(state_dict, strict=False)
                        self.logger.info(f"Loaded model weights from {model_path}")
                else:
                    # Try loading as HuggingFace model and convert
                    from transformers import LlamaForCausalLM, AutoModelForCausalLM
                    
                    # Load HuggingFace model
                    hf_model = None
                    try:
                        hf_model = AutoModelForCausalLM.from_pretrained(
                            self.config.model_name_or_path,
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            trust_remote_code=True
                        )
                    except Exception as e1:
                        try:
                            hf_model = LlamaForCausalLM.from_pretrained(
                                self.config.model_name_or_path,
                                torch_dtype=torch.float32,
                                device_map="cpu",
                                trust_remote_code=True
                            )
                        except Exception as e2:
                            self.logger.error(f"Failed to load local model. AutoModel: {e1}, Llama: {e2}")
                            raise e2
                    
                    model_config = hf_model.config
                    
                    # Ensure proper config type
                    if not isinstance(model_config, LlamaConfig):
                        llama_config = LlamaConfig(
                            vocab_size=getattr(model_config, 'vocab_size', 32000),
                            hidden_size=getattr(model_config, 'hidden_size', 4096),
                            intermediate_size=getattr(model_config, 'intermediate_size', 11008),
                            num_hidden_layers=getattr(model_config, 'num_hidden_layers', 32),
                            num_attention_heads=getattr(model_config, 'num_attention_heads', 32),
                            num_key_value_heads=getattr(model_config, 'num_key_value_heads', 32),
                            max_position_embeddings=getattr(model_config, 'max_position_embeddings', 4096),
                            rms_norm_eps=getattr(model_config, 'rms_norm_eps', 1e-6),
                            rope_theta=getattr(model_config, 'rope_theta', 10000.0),
                            attention_bias=getattr(model_config, 'attention_bias', False),
                            tie_word_embeddings=getattr(model_config, 'tie_word_embeddings', False),
                        )
                        model_config = llama_config
                    
                    # Create LLaDA model
                    self.model = LLaDAForMaskedLM(model_config)
                    
                    # Copy weights from HuggingFace model
                    try:
                        self.model.model.load_state_dict(hf_model.model.state_dict(), strict=False)
                        self.model.lm_head.load_state_dict(hf_model.lm_head.state_dict(), strict=False)
                    except Exception as e:
                        self.logger.warning(f"Error copying weights: {e}")
                        self._copy_weights_layer_by_layer(hf_model.model, self.model.model)
                        if hasattr(hf_model, 'lm_head') and hasattr(self.model, 'lm_head'):
                            self.model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
                    
                    del hf_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    self.logger.info(f"Converted HuggingFace model from {self.config.model_name_or_path}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load model from {self.config.model_name_or_path}: {e}")
                import traceback
                self.logger.warning(f"Full error traceback: {traceback.format_exc()}")
                self.logger.info("Initializing model from scratch")
                model_config = LlamaConfig()
                self.model = LLaDAForMaskedLM(model_config)
        else:
            # Try to download from HuggingFace Hub
            try:
                from transformers import LlamaForCausalLM, AutoModelForCausalLM
                
                # Try loading as AutoModel first, then LlamaForCausalLM
                hf_model = None
                try:
                    self.logger.info("Attempting to load with AutoModelForCausalLM...")
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_name_or_path,
                        torch_dtype=torch.float32,  # Use float32 to avoid conversion issues
                        device_map="cpu",  # Load on CPU first
                        trust_remote_code=True
                    )
                    self.logger.info("Successfully loaded with AutoModelForCausalLM")
                except Exception as e1:
                    self.logger.warning(f"AutoModelForCausalLM failed: {e1}")
                    try:
                        self.logger.info("Attempting to load with LlamaForCausalLM...")
                        hf_model = LlamaForCausalLM.from_pretrained(
                            self.config.model_name_or_path,
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            trust_remote_code=True
                        )
                        self.logger.info("Successfully loaded with LlamaForCausalLM")
                    except Exception as e2:
                        self.logger.error(f"Both loading methods failed. AutoModel error: {e1}, Llama error: {e2}")
                        raise e2
                
                if hf_model is None:
                    raise ValueError("Failed to load model with any method")
                
                # Get and validate config
                model_config = hf_model.config
                self.logger.info(f"Loaded model config: {type(model_config)}")
                self.logger.info(f"Model architecture: {model_config.architectures if hasattr(model_config, 'architectures') else 'unknown'}")
                
                # Create LLaDA model with proper config handling
                try:
                    # Ensure we have a proper LlamaConfig
                    if not isinstance(model_config, LlamaConfig):
                        self.logger.info("Converting config to LlamaConfig...")
                        # Convert to LlamaConfig if it's not already
                        llama_config = LlamaConfig(
                            vocab_size=getattr(model_config, 'vocab_size', 32000),
                            hidden_size=getattr(model_config, 'hidden_size', 4096),
                            intermediate_size=getattr(model_config, 'intermediate_size', 11008),
                            num_hidden_layers=getattr(model_config, 'num_hidden_layers', 32),
                            num_attention_heads=getattr(model_config, 'num_attention_heads', 32),
                            num_key_value_heads=getattr(model_config, 'num_key_value_heads', 32),
                            max_position_embeddings=getattr(model_config, 'max_position_embeddings', 4096),
                            rms_norm_eps=getattr(model_config, 'rms_norm_eps', 1e-6),
                            rope_theta=getattr(model_config, 'rope_theta', 10000.0),
                            attention_bias=getattr(model_config, 'attention_bias', False),
                            tie_word_embeddings=getattr(model_config, 'tie_word_embeddings', False),
                        )
                        model_config = llama_config
                    
                    self.model = LLaDAForMaskedLM(model_config)
                    self.logger.info("Successfully created LLaDA model")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create LLaDA model: {e}")
                    raise e
                
                # Copy weights from HuggingFace model more carefully
                self.logger.info("Copying model weights...")
                
                # Copy the base model weights
                try:
                    self.model.model.load_state_dict(hf_model.model.state_dict(), strict=False)
                    self.logger.info("Successfully copied base model weights")
                except Exception as e:
                    self.logger.warning(f"Error copying base model weights: {e}")
                    # Try copying layer by layer
                    self._copy_weights_layer_by_layer(hf_model.model, self.model.model)
                
                # Copy the language model head
                try:
                    self.model.lm_head.load_state_dict(hf_model.lm_head.state_dict(), strict=False)
                    self.logger.info("Successfully copied lm_head weights")
                except Exception as e:
                    self.logger.warning(f"Error copying lm_head weights: {e}")
                    # Manual copy
                    if hasattr(hf_model, 'lm_head') and hasattr(self.model, 'lm_head'):
                        self.model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
                
                # Clean up HF model to save memory
                del hf_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                self.logger.info(f"Downloaded and converted model from {self.config.model_name_or_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to download model {self.config.model_name_or_path}: {e}")
                import traceback
                self.logger.warning(f"Full error traceback: {traceback.format_exc()}")
                self.logger.info("Initializing model from scratch")
                model_config = LlamaConfig()
                self.model = LLaDAForMaskedLM(model_config)
        
        # Resize token embeddings for custom tokens
        original_vocab_size = self.model.config.vocab_size
        new_vocab_size = len(self.tokenizer)
        
        if new_vocab_size > original_vocab_size:
            self.logger.info(f"Resizing embeddings from {original_vocab_size} to {new_vocab_size}")
            self.model.resize_token_embeddings(new_vocab_size)
        
        # Move to device
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        self.logger.info("✅ Enabled gradient checkpointing")
        
        # Move to device with mixed precision
        self.model = self.model.to(device=self.device, dtype=torch.bfloat16)
        self.logger.info("✅ Using bfloat16 mixed precision")
        
        # Setup FSDP if enabled and multiple GPUs
        if self.config.fsdp and self.world_size > 1:
            from torch.distributed.fsdp import MixedPrecision, CPUOffload, ShardingStrategy
            from torch.distributed.fsdp.wrap import ModuleWrapPolicy
            from llada_model import LLaDADecoderLayer
            
            self.logger.info("Setting up memory-optimized FSDP")
            self.model = FSDP(
                self.model,
                auto_wrap_policy=ModuleWrapPolicy({LLaDADecoderLayer}),
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                device_id=self.local_rank,
                cpu_offload=CPUOffload(offload_params=False),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                sync_module_states=True,
                limit_all_gathers=True,  # Memory optimization
            )
        
        self.logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _copy_weights_layer_by_layer(self, source_model, target_model):
        """Copy weights layer by layer as fallback"""
        try:
            # Copy embeddings
            if hasattr(source_model, 'embed_tokens') and hasattr(target_model, 'embed_tokens'):
                target_model.embed_tokens.weight.data.copy_(source_model.embed_tokens.weight.data)
                self.logger.info("Copied embedding weights")
            
            # Copy layers
            if hasattr(source_model, 'layers') and hasattr(target_model, 'layers'):
                for i, (source_layer, target_layer) in enumerate(zip(source_model.layers, target_model.layers)):
                    try:
                        target_layer.load_state_dict(source_layer.state_dict(), strict=False)
                    except:
                        # Copy individual components
                        self._copy_layer_components(source_layer, target_layer)
                self.logger.info(f"Copied {len(source_model.layers)} layer weights")
            
            # Copy norm
            if hasattr(source_model, 'norm') and hasattr(target_model, 'norm'):
                target_model.norm.load_state_dict(source_model.norm.state_dict(), strict=False)
                self.logger.info("Copied norm weights")
                
        except Exception as e:
            self.logger.warning(f"Error in layer-by-layer copying: {e}")
    
    def _copy_layer_components(self, source_layer, target_layer):
        """Copy individual layer components"""
        try:
            # Copy attention weights (but not the attention mask logic)
            if hasattr(source_layer, 'self_attn') and hasattr(target_layer, 'self_attn'):
                for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(source_layer.self_attn, attr) and hasattr(target_layer.self_attn, attr):
                        getattr(target_layer.self_attn, attr).load_state_dict(
                            getattr(source_layer.self_attn, attr).state_dict(), strict=False
                        )
            
            # Copy MLP weights
            if hasattr(source_layer, 'mlp') and hasattr(target_layer, 'mlp'):
                target_layer.mlp.load_state_dict(source_layer.mlp.state_dict(), strict=False)
            
            # Copy layer norms
            for attr in ['input_layernorm', 'post_attention_layernorm']:
                if hasattr(source_layer, attr) and hasattr(target_layer, attr):
                    getattr(target_layer, attr).load_state_dict(
                        getattr(source_layer, attr).state_dict(), strict=False
                    )
                    
        except Exception as e:
            self.logger.warning(f"Error copying layer components: {e}")
    
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
        if self.config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.lr_scheduler_type == "constant":
            from torch.optim.lr_scheduler import LambdaLR
            self.scheduler = LambdaLR(self.optimizer, lambda epoch: 1.0)
        else:
            # Default to linear warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps
            )
    
    def setup_datasets(self):
        """Setup datasets and dataloader"""
        self.logger.info("Setting up datasets")
        
        # Create datasets
        self.text_dataset, self.tts_dataset, self.combined_dataset = create_tts_datasets(
            self.config, self.tokenizer
        )
        
        # Calculate max_steps if not already set
        if self.config.max_steps is None:
            dataset_size = len(self.combined_dataset)
            self.config.max_steps = self.config.calculate_max_steps(dataset_size)
            self.logger.info(f"Calculated max_steps: {self.config.max_steps} (from {self.config.epochs} epochs over {dataset_size} examples)")
        
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
            num_workers=0,  # Set to 0 for FSDP compatibility
            pin_memory=True if torch.cuda.is_available() else False,
        )
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for mixed text/TTS data
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        data_types = batch.get("data_types", [])
        
        # Determine if this is a TTS batch
        is_tts_batch = any(dt == 'tts' for dt in data_types)
        
        if is_tts_batch and "prompt_lengths" in batch:
            # TTS data - use SFT-style training with prompt masking
            prompt_lengths = batch["prompt_lengths"].to(self.device)
            return self._compute_sft_loss(input_ids, prompt_lengths, attention_mask)
        else:
            # Text data - use standard pre-training
            return self._compute_pretraining_loss(input_ids, attention_mask)
    
    def _compute_pretraining_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute pre-training loss for text data"""
        # Apply forward diffusion process
        noisy_batch, masked_indices, p_mask = forward_process(input_ids, eps=self.config.eps)
        
        # Forward pass through model
        outputs = self.model(input_ids=noisy_batch, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute loss only on masked tokens
        if masked_indices.sum() == 0:
            return {
                "loss": torch.tensor(0.0, device=self.device, requires_grad=True),
                "num_masked_tokens": torch.tensor(0),
                "perplexity": torch.tensor(float('inf')),
                "loss_type": "pretraining"
            }
        
        # Get predictions and targets for masked tokens only
        masked_logits = logits[masked_indices]
        masked_targets = input_ids[masked_indices]
        masked_p_mask = p_mask[masked_indices]
        
        # Compute cross-entropy loss
        token_loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')
        
        # Weight by inverse masking probability
        weighted_token_loss = token_loss / masked_p_mask
        
        # Average loss
        loss = weighted_token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        
        # Compute metrics
        with torch.no_grad():
            num_masked_tokens = masked_indices.sum()
            perplexity = torch.exp(token_loss.mean()) if len(token_loss) > 0 else torch.tensor(float('inf'))
        
        return {
            "loss": loss,
            "num_masked_tokens": num_masked_tokens,
            "perplexity": perplexity,
            "loss_type": "pretraining"
        }
    
    def _compute_sft_loss(self, input_ids: torch.Tensor, prompt_lengths: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute SFT loss for TTS data"""
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
        outputs = self.model(input_ids=noisy_batch, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute loss only on masked tokens
        if masked_indices.sum() == 0:
            return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "num_masked_tokens": torch.tensor(0),
                "perplexity": torch.tensor(float('inf')),
                "loss_type": "sft"
            }
        
        # Get predictions and targets for masked tokens
        masked_logits = logits[masked_indices]
        masked_targets = input_ids[masked_indices]
        masked_p_mask = p_mask[masked_indices]
        masked_answer_lengths = answer_lengths[masked_indices]
        
        # Compute cross-entropy loss
        token_loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')
        
        # Weight by inverse masking probability
        weighted_token_loss = token_loss / masked_p_mask
        
        # Normalize by answer length and batch size
        ce_loss = torch.sum(weighted_token_loss / masked_answer_lengths) / batch_size
        
        # Compute metrics
        with torch.no_grad():
            num_masked_tokens = masked_indices.sum()
            perplexity = torch.exp(token_loss.mean()) if len(token_loss) > 0 else torch.tensor(float('inf'))
        
        return {
            "loss": ce_loss,
            "num_masked_tokens": num_masked_tokens,
            "perplexity": perplexity,
            "loss_type": "sft"
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Compute loss
        loss_dict = self.compute_loss(batch)
        loss = loss_dict["loss"]
        
        # Backward pass
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        loss.backward()
        
        # Convert to float for logging
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, Any], global_step: int):
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
            if self.config.wandb_project:
                wandb.log(wandb_metrics, step=global_step)
            
            self.logger.info(
                f"TTS Step {self.audio_step}: loss={metrics['loss']:.4f}, "
                f"ppl={metrics['perplexity']:.2f}, masked_tokens={metrics['num_masked_tokens']}"
            )
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
                if self.config.wandb_project:
                    wandb.log(wandb_metrics, step=global_step)
                
                self.logger.info(
                    f"Audio Step {self.audio_step}: loss={metrics['loss']:.4f}, "
                    f"ppl={metrics['perplexity']:.2f}, masked_tokens={metrics['num_masked_tokens']}"
                )
                self.audio_step += 1
            else:  # Text batch
                wandb_metrics = {
                    "text_loss": metrics["loss"],
                    "text_perplexity": metrics["perplexity"],
                    "text_step": self.text_step,
                    "global_step": global_step,
                    "learning_rate": self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate
                }
                if self.config.wandb_project:
                    wandb.log(wandb_metrics, step=global_step)
                
                self.logger.info(
                    f"Text Step {self.text_step}: loss={metrics['loss']:.4f}, "
                    f"ppl={metrics['perplexity']:.2f}, masked_tokens={metrics['num_masked_tokens']}"
                )
                self.text_step += 1
    
    def save_model(self, output_dir: str, step: int):
        """Save model checkpoint"""
        if not self.is_main_process():
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(self.model, FSDP):
            # FSDP model saving
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = self.model.state_dict()
            
            # Save model state
            torch.save(cpu_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            # Regular model saving
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save training state
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step,
            "text_step": self.text_step,
            "audio_step": self.audio_step
        }
        torch.save(training_state, os.path.join(output_dir, "training_state.bin"))
        
        self.logger.info(f"Model saved to {output_dir}")
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting TTS training with {len(self.combined_dataset)} examples")
        
        global_step = 0
        
        # Create infinite dataloader
        def infinite_dataloader():
            while True:
                if hasattr(self.dataloader.sampler, 'set_epoch'):
                    self.dataloader.sampler.set_epoch(global_step // len(self.dataloader))
                for batch in self.dataloader:
                    yield batch
        
        data_iter = infinite_dataloader()
        
        # Training loop
        progress_bar = tqdm(range(self.config.max_steps), desc="TTS Training", disable=not self.is_main_process())
        
        for step in progress_bar:
            batch = next(data_iter)
            
            # Training step
            metrics = self.train_step(batch)
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    self.log_metrics(metrics, global_step)
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{global_step}")
                    self.logger.info(f"Saving checkpoint at step {global_step} to {checkpoint_dir}")
                    try:
                        self.save_model(checkpoint_dir, global_step)
                        self.logger.info(f"Successfully saved checkpoint-{global_step}")
                    except Exception as e:
                        self.logger.error(f"Failed to save checkpoint at step {global_step}: {e}")
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "type": metrics.get('loss_type', 'unknown')
                })
        
        # Save final checkpoint
        final_dir = os.path.join(self.config.output_dir, "final")
        self.save_model(final_dir, global_step)
        
        self.logger.info("TTS training completed!")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLaDA TTS Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--max_steps", type=int, help="Override max steps")
    
    args = parser.parse_args()
    
    # Load config
    config = TTSConfig.from_yaml(args.config)
    
    # Override with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_steps:
        config.max_steps = args.max_steps
    
    # Start training
    trainer = LLaDATTSTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 