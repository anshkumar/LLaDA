#!/usr/bin/env python3
"""
Example usage script for LLaDA training and inference

This script demonstrates how to:
1. Prepare data for training
2. Train a LLaDA model
3. Run inference with different sampling methods
4. Analyze model behavior
"""

import os
import torch
import json
import numpy as np
from transformers import LlamaConfig

# Import LLaDA modules
from llada_model import LLaDAForMaskedLM
from pretraining import LLaDAPretrainer, PretrainingConfig
from sft_training import LLaDASFTTrainer, SFTConfig
from sampling import create_sampler, SamplingConfig
from utils import TokenizerHelper, DatasetConverter, ModelAnalyzer


def create_dummy_data():
    """Create dummy data for demonstration"""
    print("Creating dummy training data...")
    
    # Create directories
    os.makedirs("./data/pretraining", exist_ok=True)
    os.makedirs("./data/sft", exist_ok=True)
    
    # Create dummy pre-training data
    pretraining_data = []
    for i in range(1000):
        # Random token sequences
        length = np.random.randint(100, 512)
        tokens = np.random.randint(1, 32000, size=length).tolist()
        pretraining_data.append({"input_ids": tokens})
    
    with open("./data/pretraining/dummy_data.jsonl", "w") as f:
        for item in pretraining_data:
            f.write(json.dumps(item) + "\n")
    
    # Create dummy SFT data
    sft_data = []
    for i in range(500):
        # Create prompt and response
        prompt_length = np.random.randint(20, 100)
        response_length = np.random.randint(50, 200)
        
        prompt_tokens = np.random.randint(1, 32000, size=prompt_length).tolist()
        response_tokens = np.random.randint(1, 32000, size=response_length).tolist()
        
        input_ids = prompt_tokens + response_tokens
        
        sft_data.append({
            "input_ids": input_ids,
            "prompt_length": prompt_length
        })
    
    with open("./data/sft/dummy_data.jsonl", "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")
    
    print("Dummy data created successfully!")


def demo_pretraining():
    """Demonstrate pre-training"""
    print("\n" + "="*50)
    print("DEMO: Pre-training LLaDA")
    print("="*50)
    
    # Configure pre-training
    config = PretrainingConfig(
        data_path="./data/pretraining",
        output_dir="./models/llada_pretrained_demo",
        batch_size=2,  # Small for demo
        max_steps=100,  # Very short training
        learning_rate=5e-4,
        logging_steps=20,
        save_steps=50,
        max_length=256,  # Shorter sequences for demo
        wandb_project=None  # Disable wandb for demo
    )
    
    print(f"Training configuration:")
    print(f"- Data path: {config.data_path}")
    print(f"- Output dir: {config.output_dir}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Max steps: {config.max_steps}")
    
    # Create trainer and run
    trainer = LLaDAPretrainer(config)
    
    print("\nStarting pre-training...")
    try:
        trainer.train()
        print("✓ Pre-training completed successfully!")
    except Exception as e:
        print(f"✗ Pre-training failed: {e}")
        return False
    
    return True


def demo_sft():
    """Demonstrate supervised fine-tuning"""
    print("\n" + "="*50)
    print("DEMO: Supervised Fine-tuning")
    print("="*50)
    
    # Configure SFT
    config = SFTConfig(
        model_name_or_path="./models/llada_pretrained_demo",
        data_path="./data/sft",
        output_dir="./models/llada_sft_demo",
        batch_size=2,
        max_steps=50,  # Very short training
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=25,
        max_length=256,
        wandb_project=None  # Disable wandb for demo
    )
    
    print(f"SFT configuration:")
    print(f"- Model path: {config.model_name_or_path}")
    print(f"- Data path: {config.data_path}")
    print(f"- Output dir: {config.output_dir}")
    print(f"- Max steps: {config.max_steps}")
    
    # Create trainer and run
    trainer = LLaDASFTTrainer(config)
    
    print("\nStarting SFT...")
    try:
        trainer.train()
        print("✓ SFT completed successfully!")
    except Exception as e:
        print(f"✗ SFT failed: {e}")
        return False
    
    return True


def demo_inference():
    """Demonstrate inference with different sampling methods"""
    print("\n" + "="*50)
    print("DEMO: Inference with Different Sampling Methods")
    print("="*50)
    
    # Load model
    try:
        model_path = "./models/llada_sft_demo"
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            model_config = LlamaConfig()  # Use default config for demo
        else:
            model_config = LlamaConfig(
                vocab_size=32000,
                hidden_size=512,  # Smaller for demo
                intermediate_size=1024,
                num_hidden_layers=4,
                num_attention_heads=8,
                max_position_embeddings=1024,
            )
        
        model = LLaDAForMaskedLM(model_config)
        
        # Try to load weights
        model_weights_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_weights_path):
            state_dict = torch.load(model_weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print("✓ Loaded trained model weights")
        else:
            print("! Using randomly initialized model (no trained weights found)")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("Creating a small demo model instead...")
        
        # Create small demo model
        model_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
        )
        model = LLaDAForMaskedLM(model_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    
    # Create dummy prompt
    prompt_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    print(f"Input prompt: {prompt_ids.tolist()}")
    
    # Test different sampling methods
    methods = ["fixed_length", "semi_autoregressive_padding"]
    strategies = ["random", "low_confidence"]
    
    results = {}
    
    for method in methods:
        for strategy in strategies:
            key = f"{method}_{strategy}"
            print(f"\nTesting {key}...")
            
            try:
                sampler = create_sampler(
                    model,
                    sampling_method=method,
                    remasking_strategy=strategy,
                    max_length=64,  # Short for demo
                    num_iterations=5,
                    temperature=1.0
                )
                
                generated = sampler.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=20,
                    num_return_sequences=1
                )
                
                result_tokens = generated[0].cpu().tolist()
                results[key] = result_tokens
                print(f"✓ Generated: {result_tokens}")
                
            except Exception as e:
                print(f"✗ Error in {key}: {e}")
                results[key] = None
    
    # Compare results
    print(f"\n{'Method':<30} {'Strategy':<15} {'Success':<10}")
    print("-" * 55)
    for key, result in results.items():
        method, strategy = key.split('_', 1)
        success = "✓" if result is not None else "✗"
        print(f"{method:<30} {strategy:<15} {success:<10}")
    
    return results


def demo_analysis():
    """Demonstrate model analysis"""
    print("\n" + "="*50)
    print("DEMO: Model Analysis")
    print("="*50)
    
    try:
        # Create a small model for analysis
        model_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
        )
        model = LLaDAForMaskedLM(model_config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Create analyzer
        analyzer = ModelAnalyzer()
        
        # Create dummy input
        input_ids = torch.randint(1, 100, (1, 32), device=device)
        print(f"Analyzing with input shape: {input_ids.shape}")
        
        # Analyze attention patterns (without visualization to avoid display issues)
        print("\n1. Analyzing attention patterns...")
        try:
            attention_matrix = analyzer.analyze_attention_patterns(
                model, input_ids, layer_idx=0, head_idx=0, save_path=None
            )
            print(f"✓ Attention matrix shape: {attention_matrix.shape}")
            print(f"✓ Attention statistics: mean={attention_matrix.mean():.4f}, std={attention_matrix.std():.4f}")
        except Exception as e:
            print(f"✗ Attention analysis failed: {e}")
        
        # Compare sampling methods
        print("\n2. Comparing sampling methods...")
        try:
            comparison_results = analyzer.compare_sampling_methods(
                model, input_ids, max_new_tokens=20, num_samples=2
            )
            
            print(f"✓ Compared {len(comparison_results)} method combinations")
            for key, data in comparison_results.items():
                num_samples = len(data['samples'])
                print(f"  - {key}: {num_samples} successful samples")
                
        except Exception as e:
            print(f"✗ Sampling comparison failed: {e}")
        
        print("✓ Analysis completed!")
        
    except Exception as e:
        print(f"✗ Analysis demo failed: {e}")


def demo_data_conversion():
    """Demonstrate data conversion utilities"""
    print("\n" + "="*50)
    print("DEMO: Data Conversion")
    print("="*50)
    
    try:
        # Create tokenizer helper (using a simple config to avoid download)
        print("Creating tokenizer helper...")
        
        # Create dummy conversion examples
        print("\n1. Creating sample conversation data...")
        
        # Sample ShareGPT format
        sharegpt_data = [
            {
                "conversations": [
                    {"from": "user", "value": "What is 2+2?"},
                    {"from": "assistant", "value": "2+2 equals 4."}
                ]
            },
            {
                "conversations": [
                    {"from": "user", "value": "Tell me about Python."},
                    {"from": "assistant", "value": "Python is a programming language."}
                ]
            }
        ]
        
        os.makedirs("./data/examples", exist_ok=True)
        
        with open("./data/examples/sample_sharegpt.jsonl", "w") as f:
            for item in sharegpt_data:
                f.write(json.dumps(item) + "\n")
        
        # Sample Alpaca format
        alpaca_data = [
            {
                "instruction": "What is the capital of France?",
                "input": "",
                "output": "The capital of France is Paris."
            },
            {
                "instruction": "Translate the following to Spanish",
                "input": "Hello world",
                "output": "Hola mundo"
            }
        ]
        
        with open("./data/examples/sample_alpaca.json", "w") as f:
            json.dump(alpaca_data, f, indent=2)
        
        print("✓ Sample data created!")
        print("✓ Data conversion utilities are available in utils.py")
        print("  - Use TokenizerHelper for encoding conversations")
        print("  - Use DatasetConverter for format conversion")
        
    except Exception as e:
        print(f"✗ Data conversion demo failed: {e}")


def main():
    """Run all demos"""
    print("LLaDA Training and Inference Demo")
    print("=" * 60)
    
    # Create dummy data
    create_dummy_data()
    
    # Demo data conversion
    demo_data_conversion()
    
    # Demo pre-training
    pretraining_success = demo_pretraining()
    
    # Demo SFT (only if pre-training succeeded)
    if pretraining_success:
        sft_success = demo_sft()
    else:
        print("\nSkipping SFT demo due to pre-training failure")
        sft_success = False
    
    # Demo inference (works with or without trained model)
    demo_inference()
    
    # Demo analysis
    demo_analysis()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("1. Replace dummy data with real training data")
    print("2. Adjust configurations for your use case")
    print("3. Run full training with: python train_llada.py")
    print("4. Use the trained model for your applications")
    
    print(f"\nFiles created during demo:")
    print(f"- ./data/pretraining/dummy_data.jsonl")
    print(f"- ./data/sft/dummy_data.jsonl")
    print(f"- ./data/examples/sample_*.json*")
    if pretraining_success:
        print(f"- ./models/llada_pretrained_demo/")
    if sft_success:
        print(f"- ./models/llada_sft_demo/")


if __name__ == "__main__":
    main() 