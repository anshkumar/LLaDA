import yaml
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pretraining import PretrainingConfig
from sft_training import SFTConfig


@dataclass
class TTSConfig:
    """Configuration for TTS-specific LLaDA training"""
    # Model and tokenizer
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "./llada_tts"
    
    # Datasets
    text_dataset: str = "text_qa_dataset"
    tts_dataset: str = "tts_dataset"
    
    # Custom tokens for TTS
    num_audio_tokens: int = 7 * 4096  # 7 codebooks * 4096 vocab size
    num_special_tokens: int = 10  # Additional special tokens
    audio_token_prefix: str = "<audio_token_"
    special_token_prefix: str = "<special_token_"
    
    # Training parameters
    learning_rate: float = 5e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    warmup_epochs: float = 0.1  # Warmup for first 10% of training
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    save_epochs: int = 1  # Save checkpoint every N epochs
    
    # TTS-specific parameters
    max_length: int = 4096
    mask_token_id: int = 126336
    eps: float = 1e-3
    ratio: float = 0.5  # Ratio of text to audio training (0.0 = TTS only)
    lr_scheduler_type: str = "cosine"  # Learning rate scheduler type
    
    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 2000
    save_total_limit: int = 3
    
    # Mixed precision and distributed training
    fp16: bool = True
    fsdp: bool = True
    number_processes: int = 1
    
    # Wandb
    wandb_project: str = "llada_tts"
    wandb_run_name: Optional[str] = None
    
    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    @classmethod
    def from_yaml(cls, config_file: str) -> 'TTSConfig':
        """Load configuration from YAML file"""
        with open(config_file, "r") as file:
            config_dict = yaml.safe_load(file)
        
        # Map YAML keys to our config
        mapped_config = {
            'text_dataset': config_dict.get('text_QA_dataset'),  # Can be None
            'tts_dataset': config_dict.get('TTS_dataset', 'tts_dataset'),
            'model_name_or_path': config_dict.get('model_name', 'meta-llama/Llama-2-7b-hf'),
            'tokenizer_name': config_dict.get('tokenizer_name'),  # Can be None, will use model_name
            'wandb_run_name': config_dict.get('run_name'),
            'wandb_project': config_dict.get('project_name', 'llada_tts'),
            'output_dir': config_dict.get('save_folder', './llada_tts'),
            'batch_size': config_dict.get('batch_size', 4),
            'learning_rate': config_dict.get('learning_rate', 5e-4),
            'number_processes': config_dict.get('number_processes', 1),
            'ratio': config_dict.get('ratio', 0.5),
            'save_steps': config_dict.get('save_steps', 2000),
            'pad_token_id': config_dict.get('pad_token', 0),
            'epochs': config_dict.get('epochs', 1),
            'save_epochs': config_dict.get('save_epochs', 1),  # Save every N epochs
            'warmup_epochs': config_dict.get('warmup_epochs', 0.1),
            'lr_scheduler_type': config_dict.get('lr_scheduler_type', 'cosine'),
        }
        
        # If no tokenizer specified, use model name
        if mapped_config['tokenizer_name'] is None:
            mapped_config['tokenizer_name'] = mapped_config['model_name_or_path']
        
        # If no text dataset, set ratio to 0 (TTS only)
        if mapped_config['text_dataset'] is None:
            mapped_config['ratio'] = 0.0
            mapped_config['text_dataset'] = mapped_config['tts_dataset']  # Use TTS dataset as fallback
        
        # Remove None values
        mapped_config = {k: v for k, v in mapped_config.items() if v is not None}
        
        return cls(**mapped_config)
    
    def get_total_new_tokens(self) -> int:
        """Get total number of new tokens to add"""
        return self.num_audio_tokens + self.num_special_tokens
    
    def get_new_token_names(self) -> List[str]:
        """Generate list of new token names"""
        tokens = []
        
        # Audio tokens
        for i in range(self.num_audio_tokens):
            tokens.append(f"{self.audio_token_prefix}{i}>")
        
        # Special tokens
        for i in range(self.num_special_tokens):
            tokens.append(f"{self.special_token_prefix}{i}>")
        
        return tokens
    
    def calculate_warmup_steps(self, dataset_size: int) -> int:
        """Calculate warmup steps based on dataset size and warmup epochs"""
        steps_per_epoch = dataset_size // (self.batch_size * self.gradient_accumulation_steps * self.number_processes)
        if steps_per_epoch == 0:  # Handle small datasets
            steps_per_epoch = 1
        return int(self.warmup_epochs * steps_per_epoch)
    
    def to_pretraining_config(self) -> PretrainingConfig:
        """Convert to PretrainingConfig for pre-training phase"""
        return PretrainingConfig(
            model_name_or_path=self.model_name_or_path,
            output_dir=f"{self.output_dir}/pretrained",
            data_path=self.text_dataset,  # Use text dataset for pre-training
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            max_length=self.max_length,
            mask_token_id=self.mask_token_id,
            eps=self.eps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            wandb_project=self.wandb_project,
            wandb_run_name=f"{self.wandb_run_name}_pretrain" if self.wandb_run_name else None
        )
    
    def to_sft_config(self) -> SFTConfig:
        """Convert to SFTConfig for fine-tuning phase"""
        return SFTConfig(
            model_name_or_path=f"{self.output_dir}/pretrained",
            output_dir=f"{self.output_dir}/sft",
            data_path=self.tts_dataset,
            learning_rate=self.learning_rate * 0.2,  # Lower LR for fine-tuning
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_steps=self.max_steps // 2,  # Fewer steps for fine-tuning
            warmup_steps=self.warmup_steps // 2,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            max_length=self.max_length,
            mask_token_id=self.mask_token_id,
            eps=self.eps,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            wandb_project=self.wandb_project,
            wandb_run_name=f"{self.wandb_run_name}_sft" if self.wandb_run_name else None
        )


def create_sample_tts_config():
    """Create a sample TTS configuration file"""
    config = {
        'text_QA_dataset': 'your_text_dataset',
        'TTS_dataset': 'your_tts_dataset',
        'model_name': 'meta-llama/Llama-2-7b-hf',
        'tokenizer_name': 'meta-llama/Llama-2-7b-hf',
        'run_name': 'llada_tts_experiment',
        'project_name': 'llada_tts',
        'save_folder': './llada_tts_output',
        'epochs': 3,
        'batch_size': 4,
        'save_steps': 2000,
        'pad_token': 0,
        'number_processes': 1,
        'learning_rate': 5e-4,
        'ratio': 0.5,
        # TTS-specific settings
        'num_audio_tokens': 7 * 4096,
        'num_special_tokens': 10,
        'max_length': 4096,
        'mask_token_id': 126336,
        'eps': 1e-3,
    }
    
    with open('tts_config.yaml', 'w') as f:
        yaml.dump(config, f, indent=2)
    
    print("Created sample TTS config: tts_config.yaml") 