#!/usr/bin/env python3
"""
LLaDA TTS Training Script

This script provides TTS-specific training for LLaDA with custom audio tokens,
similar to the original transformer TTS training script.

Usage:
    python train_llada_tts.py --config config.yaml
    
Config file should contain:
    text_QA_dataset: "path/to/text/dataset"
    TTS_dataset: "path/to/tts/dataset"
    model_name: "meta-llama/Llama-2-7b-hf"
    tokenizer_name: "meta-llama/Llama-2-7b-hf"
    run_name: "llada_tts_experiment"
    project_name: "llada_tts"
    save_folder: "./llada_tts_output"
    batch_size: 4
    learning_rate: 5e-4
    ratio: 0.5
    number_processes: 1
    save_steps: 2000
    pad_token: 0
"""

import os
import sys
import argparse
import logging
import torch
from transformers import set_seed

# Import our TTS modules
from tts_config import TTSConfig, create_sample_tts_config
from tts_training import LLaDATTSTrainer


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=getattr(logging, level.upper()),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('tts_training.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="LLaDA TTS Training")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--create-sample-config", action="store_true", help="Create sample config file")
    
    # Override options
    parser.add_argument("--text_dataset", type=str, help="Override text dataset")
    parser.add_argument("--tts_dataset", type=str, help="Override TTS dataset")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--save_epochs", type=int, help="Override save frequency (epochs)")
    parser.add_argument("--ratio", type=float, help="Override text/TTS ratio")
    parser.add_argument("--wandb_project", type=str, help="Override wandb project")
    parser.add_argument("--wandb_run_name", type=str, help="Override wandb run name")
    
    # Training options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Setup logging and seed
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Allow async CUDA ops
    
    logger = logging.getLogger(__name__)
    
    # Clear CUDA cache at start if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 Cleared CUDA cache at startup")
    
    # Create sample config if requested
    if args.create_sample_config:
        create_sample_tts_config()
        logger.info("Sample config created: tts_config.yaml")
        return
    
    # Load config
    if not args.config:
        logger.error("Config file is required. Use --config path/to/config.yaml")
        logger.info("Or use --create-sample-config to create a sample config file")
        return
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return
    
    logger.info(f"Loading config from {args.config}")
    config = TTSConfig.from_yaml(args.config)
    
    # Override with command line arguments
    if args.text_dataset:
        config.text_dataset = args.text_dataset
    if args.tts_dataset:
        config.tts_dataset = args.tts_dataset
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.epochs:
        config.epochs = args.epochs
    if args.save_epochs:
        config.save_epochs = args.save_epochs
    if args.ratio:
        config.ratio = args.ratio
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    
    # Log configuration
    logger.info("Training Configuration:")
    if config.ratio == 0.0:
        logger.info(f"  Training Mode: TTS-Only")
        logger.info(f"  TTS Dataset: {config.tts_dataset}")
    else:
        logger.info(f"  Training Mode: Mixed (Text + TTS)")
        logger.info(f"  Text Dataset: {config.text_dataset}")
        logger.info(f"  TTS Dataset: {config.tts_dataset}")
        logger.info(f"  Text/TTS Ratio: {config.ratio}")
    
    logger.info(f"  Model: {config.model_name_or_path}")
    logger.info(f"  Tokenizer: {config.tokenizer_name}")
    logger.info(f"  Output Dir: {config.output_dir}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  LR Scheduler: {config.lr_scheduler_type}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Save Every: {config.save_epochs} epochs")
    logger.info(f"  Pad Token ID: {config.pad_token_id}")
    logger.info(f"  Audio Tokens: {config.num_audio_tokens}")
    logger.info(f"  Special Tokens: {config.num_special_tokens}")
    logger.info(f"  FSDP: {config.fsdp}")
    logger.info(f"  Processes: {config.number_processes}")
    logger.info(f"  Save Steps: {config.save_steps}")
    logger.info(f"  Wandb Project: {config.wandb_project}")
    logger.info(f"  Wandb Run: {config.wandb_run_name}")
    
    # Check if datasets exist
    if not os.path.exists(config.text_dataset) and not config.text_dataset.startswith('http'):
        logger.warning(f"Text dataset path may not exist: {config.text_dataset}")
    
    if not os.path.exists(config.tts_dataset) and not config.tts_dataset.startswith('http'):
        logger.warning(f"TTS dataset path may not exist: {config.tts_dataset}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config to output directory
    import yaml
    config_save_path = os.path.join(config.output_dir, "training_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config.__dict__, f, indent=2)
    logger.info(f"Saved training config to {config_save_path}")
    
    # Start training
    logger.info("Initializing LLaDA TTS trainer...")
    try:
        trainer = LLaDATTSTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            # TODO: Add checkpoint loading logic
        
        logger.info("Starting TTS training...")
        trainer.train()
        
        logger.info("TTS training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 