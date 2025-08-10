#!/usr/bin/env python3
"""
Main training script for LLaDA (Large Language and Diffusion Alignment)

This script provides a unified interface for both pre-training and supervised fine-tuning.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import set_seed

# Import our modules
from pretraining import LLaDAPretrainer, PretrainingConfig
from sft_training import LLaDASFTTrainer, SFTConfig
from llada_model import LLaDAForMaskedLM
from sampling import create_sampler


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def load_config(config_path: str, config_type: str) -> Dict[str, Any]:
    """Load configuration from file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Validate config type
    if config_type not in config_dict:
        raise ValueError(f"Config type '{config_type}' not found in {config_path}")
    
    return config_dict[config_type]


def create_sample_configs():
    """Create sample configuration files"""
    configs = {
        "pretraining": {
            "model_name_or_path": "meta-llama/Llama-2-7b-hf",
            "output_dir": "./llada_pretrained",
            "data_path": "./pretraining_data",
            "learning_rate": 5e-4,
            "batch_size": 8,
            "gradient_accumulation_steps": 8,
            "max_steps": 100000,
            "warmup_steps": 2000,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "max_length": 4096,
            "mask_token_id": 126336,
            "eps": 1e-3,
            "random_length_prob": 0.01,
            "logging_steps": 100,
            "save_steps": 5000,
            "eval_steps": 1000,
            "save_total_limit": 3,
            "fp16": True,
            "dataloader_num_workers": 4,
            "wandb_project": "llada_pretraining",
            "wandb_run_name": None
        },
        "sft": {
            "model_name_or_path": "./llada_pretrained",
            "output_dir": "./llada_sft",
            "data_path": "./sft_data",
            "learning_rate": 1e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 16,
            "max_steps": 10000,
            "warmup_steps": 500,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "max_length": 4096,
            "mask_token_id": 126336,
            "eps": 1e-3,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "logging_steps": 50,
            "save_steps": 1000,
            "eval_steps": 500,
            "save_total_limit": 3,
            "fp16": True,
            "dataloader_num_workers": 4,
            "wandb_project": "llada_sft",
            "wandb_run_name": None
        }
    }
    
    # Create config files
    for config_type, config_data in configs.items():
        config_file = f"{config_type}_config.json"
        with open(config_file, 'w') as f:
            json.dump({config_type: config_data}, f, indent=2)
        print(f"Created sample config: {config_file}")


def run_pretraining(args):
    """Run pre-training"""
    logger = logging.getLogger(__name__)
    logger.info("Starting LLaDA pre-training")
    
    # Load or create config
    if args.config:
        config_dict = load_config(args.config, "pretraining")
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
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(config.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Start training
    trainer = LLaDAPretrainer(config)
    trainer.train()
    
    logger.info("Pre-training completed successfully!")


def run_sft(args):
    """Run supervised fine-tuning"""
    logger = logging.getLogger(__name__)
    logger.info("Starting LLaDA supervised fine-tuning")
    
    # Load or create config
    if args.config:
        config_dict = load_config(args.config, "sft")
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
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.wandb_run_name = args.wandb_run_name
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(config.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Start training
    trainer = LLaDASFTTrainer(config)
    trainer.train()
    
    logger.info("SFT completed successfully!")


def run_inference(args):
    """Run inference with trained model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting LLaDA inference")
    
    # Load model
    model_path = args.model_name_or_path or args.output_dir
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config_dict = json.load(f)
        
        # Try to extract model config (might be nested)
        if 'model_config' in model_config_dict:
            from transformers import LlamaConfig
            model_config = LlamaConfig(**model_config_dict['model_config'])
        else:
            # Assume it's a direct config
            from transformers import LlamaConfig
            model_config = LlamaConfig()
    else:
        from transformers import LlamaConfig
        model_config = LlamaConfig()
    
    # Load model
    model = LLaDAForMaskedLM(model_config)
    
    model_weights_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_weights_path):
        state_dict = torch.load(model_weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_weights_path}")
    else:
        logger.warning("No model weights found, using randomly initialized model")
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create sampler
    sampler = create_sampler(
        model,
        sampling_method=args.sampling_method,
        remasking_strategy=args.remasking_strategy,
        max_length=args.max_length,
        num_iterations=args.num_iterations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Interactive inference
    if args.interactive:
        print("Starting interactive inference. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\nPrompt: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                # For demo purposes, create dummy input_ids
                # In practice, you'd use a tokenizer
                prompt_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)  # Dummy
                
                results = sampler.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.num_return_sequences
                )
                
                print(f"\nGenerated {len(results)} sequences:")
                for i, result in enumerate(results):
                    print(f"Sequence {i+1}: {result.tolist()}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error during generation: {e}")
    
    # Batch inference from file
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        results = []
        for prompt in prompts:
            # Create dummy input_ids (in practice, use tokenizer)
            prompt_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
            
            generated = sampler.generate(
                input_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_return_sequences
            )
            
            results.append({
                'prompt': prompt,
                'generated': [seq.tolist() for seq in generated]
            })
        
        # Save results
        output_file = args.output_file or "inference_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Inference results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="LLaDA Training and Inference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    # Create sample configs command
    create_config_parser = subparsers.add_parser("create-configs", help="Create sample configuration files")
    
    # Pre-training command
    pretrain_parser = subparsers.add_parser("pretrain", help="Run pre-training")
    pretrain_parser.add_argument("--config", type=str, help="Path to config file")
    pretrain_parser.add_argument("--data_path", type=str, help="Path to training data")
    pretrain_parser.add_argument("--output_dir", type=str, help="Output directory")
    pretrain_parser.add_argument("--batch_size", type=int, help="Batch size")
    pretrain_parser.add_argument("--learning_rate", type=float, help="Learning rate")
    pretrain_parser.add_argument("--max_steps", type=int, help="Maximum training steps")
    pretrain_parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    pretrain_parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")
    
    # SFT command
    sft_parser = subparsers.add_parser("sft", help="Run supervised fine-tuning")
    sft_parser.add_argument("--config", type=str, help="Path to config file")
    sft_parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model")
    sft_parser.add_argument("--data_path", type=str, help="Path to SFT data")
    sft_parser.add_argument("--output_dir", type=str, help="Output directory")
    sft_parser.add_argument("--batch_size", type=int, help="Batch size")
    sft_parser.add_argument("--learning_rate", type=float, help="Learning rate")
    sft_parser.add_argument("--max_steps", type=int, help="Maximum training steps")
    sft_parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    sft_parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument("--model_name_or_path", type=str, help="Path to trained model")
    inference_parser.add_argument("--output_dir", type=str, help="Model directory (alternative to model_name_or_path)")
    inference_parser.add_argument("--sampling_method", type=str, default="fixed_length",
                                choices=["fixed_length", "semi_autoregressive_origin", "semi_autoregressive_padding"],
                                help="Sampling method")
    inference_parser.add_argument("--remasking_strategy", type=str, default="low_confidence",
                                choices=["random", "low_confidence"],
                                help="Remasking strategy")
    inference_parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    inference_parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate")
    inference_parser.add_argument("--num_iterations", type=int, default=10, help="Number of sampling iterations")
    inference_parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return")
    inference_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    inference_parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    inference_parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    inference_parser.add_argument("--interactive", action="store_true", help="Interactive inference mode")
    inference_parser.add_argument("--input_file", type=str, help="Input file with prompts")
    inference_parser.add_argument("--output_file", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging and seed
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting LLaDA {args.command}")
    
    # Run command
    if args.command == "create-configs":
        create_sample_configs()
    elif args.command == "pretrain":
        run_pretraining(args)
    elif args.command == "sft":
        run_sft(args)
    elif args.command == "inference":
        run_inference(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 