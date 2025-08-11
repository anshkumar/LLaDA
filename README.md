# LLaDA: Large Language and Diffusion Alignment

This repository contains a complete implementation of LLaDA (Large Language and Diffusion Alignment), a novel approach that combines diffusion models with transformer architectures for text generation.

## Overview

LLaDA modifies the traditional autoregressive language model by:
1. **Removing causal masking** from the self-attention mechanism
2. **Using a diffusion-based training process** with random masking
3. **Implementing specialized sampling methods** for generation

## Architecture

- **Base Model**: Modified LLaMA architecture without causal attention
- **Mask Token**: Uses token ID 126336 as the special mask token
- **Training Process**: Diffusion-style forward process with varying masking probabilities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LLaDA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install with development dependencies:
```bash
pip install -e .
```

## Quick Start

### Text Generation (Standard LLaDA)

#### 1. Create Configuration Files

Generate sample configuration files:
```bash
python train_llada.py create-configs
```

This creates `pretraining_config.json` and `sft_config.json` with default settings.

#### 2. Pre-training

```bash
python train_llada.py pretrain \
  --data_path ./pretraining_data \
  --output_dir ./llada_pretrained \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --epochs 3
```

#### 3. Supervised Fine-tuning

```bash
python train_llada.py sft \
  --model_name_or_path ./llada_pretrained \
  --data_path ./sft_data \
  --output_dir ./llada_sft \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --epochs 2
```

#### 4. Inference

```bash
python train_llada.py inference \
  --model_name_or_path ./llada_sft \
  --sampling_method fixed_length \
  --remasking_strategy low_confidence \
  --interactive
```

### Text-to-Speech (TTS) Training

LLaDA supports specialized TTS training with custom audio tokens for autoregressive speech synthesis.

#### 1. Create TTS Configuration

Generate a sample TTS configuration:
```bash
python train_llada_tts.py --create-sample-config
```

This creates `sample_tts_config.yaml` with optimized settings.

#### 2. TTS Training

Train LLaDA for TTS with your dataset:
```bash
python train_llada_tts.py --config sample_tts_config.yaml
```

#### 3. Mixed Text + TTS Training

For joint text and speech modeling, specify both datasets in your config:
```yaml
# Both datasets for mixed training
text_QA_dataset: "/path/to/text/dataset" 
TTS_dataset: "/path/to/tts/dataset"
ratio: 0.5  # 50% text, 50% TTS

# Or TTS-only training (default if text_QA_dataset not provided)
TTS_dataset: "/path/to/tts/dataset"
# ratio automatically set to 0.0 for TTS-only
```

#### 4. TTS Configuration Example

```yaml
# Dataset configuration
TTS_dataset: "/workspace/combined_tts_dataset_pretrain"
# text_QA_dataset: "/path/to/text/dataset"  # Optional for mixed training

# Model configuration  
model_name: "meta-llama/Llama-3.2-3B-Instruct"
tokenizer_name: "meta-llama/Llama-3.2-3B-Instruct"

# Training arguments - Optimized for H100 80GB
epochs: 10
batch_size: 8
gradient_accumulation_steps: 1
number_processes: 4
save_epochs: 1  # Save every epoch
learning_rate: 5.0e-5
lr_scheduler_type: "linear"
max_grad_norm: 1.0
max_length: 2048

# Paths and logging
save_folder: "checkpoints_llada_tts"
project_name: "llada-tts-experiment" 
run_name: "llada-tts-run-1"
```

#### 5. TTS Features

- **🎵 Audio Token Support**: Automatically adds 28,682 custom audio tokens (7 codebooks × 4096 vocab + 10 special tokens)
- **📊 Epoch-Based Training**: Clean epoch-based training instead of step-based
- **💾 Smart Checkpointing**: Save after each epoch with `checkpoint-epoch-N/` naming
- **⚡ Memory Optimized**: FSDP, gradient checkpointing, mixed precision for large models
- **📈 Advanced Logging**: Separate loss tracking for text vs audio with Weights & Biases
- **🔄 Mixed Training**: Support for joint text and TTS training with configurable ratios

## Data Format

### Pre-training Data

Pre-training data should be in JSONL format with tokenized sequences:

```json
{"input_ids": [1, 2, 3, ..., 1000]}
{"input_ids": [1, 2, 3, ..., 1500]}
```

### SFT Data

SFT data should include both input sequences and prompt lengths:

```json
{"input_ids": [1, 2, 3, ..., 1000], "prompt_length": 50}
{"input_ids": [1, 2, 3, ..., 800], "prompt_length": 30}
```

Alternatively, use conversation format:
```json
{
  "conversations": [
    {"from": "user", "value": "What is the capital of France?"},
    {"from": "assistant", "value": "The capital of France is Paris."}
  ]
}
```

### TTS Data

TTS datasets should be in HuggingFace datasets format with the following structure:

#### For Local Datasets (saved with `save_to_disk`)
```
/path/to/tts_dataset/
├── data-00000-of-00001.arrow
├── dataset_info.json
├── metadata.json
└── state.json
```

#### Required Fields
Each example should contain:
```json
{
  "input_ids": [1, 2, 3, ..., 1000],           # Tokenized sequence including audio tokens
  "attention_mask": [1, 1, 1, ..., 1],         # Attention mask
  "prompt_lengths": 50,                        # Length of text prompt (before audio)
  "data_type": "tts"                           # Data type identifier
}
```

#### Audio Token Format
Audio tokens follow the pattern:
- **Audio tokens**: `<audio_token_0>` to `<audio_token_28671>` (7 codebooks × 4096)
- **Special tokens**: `<special_token_0>` to `<special_token_9>`

Example sequence:
```
"Hello world <audio_token_1234> <audio_token_567> ... <special_token_0>"
```

#### Mixed Text + TTS Dataset
For combined training, the dataset loader automatically handles:
- Text examples (data_type: "text")
- TTS examples (data_type: "tts") 
- Proper batching based on configured ratio

## Training Process

### Pre-training

The pre-training follows the paper's methodology:

1. **Forward Process**: Random masking with probability `p_mask = (1 - eps) * t + eps`
2. **Loss Computation**: Cross-entropy loss weighted by inverse masking probability
3. **Random Length**: 1% of sequences use random lengths from [1, 4096]

Key code from the paper:
```python
def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask
```

### Supervised Fine-tuning

SFT modifies the pre-training process by:

1. **Prompt Preservation**: Never mask tokens in the prompt
2. **Answer-only Loss**: Only compute loss on response tokens
3. **Length Normalization**: Normalize loss by answer length

## Sampling Methods

LLaDA supports three sampling methods:

### 1. Fixed-Length Sampling
- Start with all positions masked
- Iteratively unmask tokens
- Fixed output length

### 2. Semi-Autoregressive Origin
- Start with short sequence
- Gradually extend length
- Check for EOS tokens

### 3. Semi-Autoregressive Padding  
- Start with full-length sequence
- Unmask from left to right
- Window-based generation

### Remasking Strategies

- **Random Remasking**: Randomly select tokens to remask
- **Low-Confidence Remasking**: Remask tokens with lowest confidence scores

## Configuration

### Pre-training Config
```json
{
  "pretraining": {
    "model_name_or_path": "meta-llama/Llama-2-7b-hf",
    "output_dir": "./llada_pretrained",
    "data_path": "./pretraining_data",
    "learning_rate": 5e-4,
    "batch_size": 8,
    "gradient_accumulation_steps": 8,
    "epochs": 3,
    "warmup_epochs": 0.1,
    "max_length": 4096,
    "mask_token_id": 126336,
    "eps": 1e-3,
    "random_length_prob": 0.01
  }
}
```

### SFT Config
```json
{
  "sft": {
    "model_name_or_path": "./llada_pretrained",
    "output_dir": "./llada_sft",
    "data_path": "./sft_data",
    "learning_rate": 1e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 16,
    "epochs": 2,
    "save_epochs": 1
  }
}
```

### TTS Config
```yaml
# Dataset configuration
TTS_dataset: "/workspace/combined_tts_dataset_pretrain"
# text_QA_dataset: "/path/to/text/dataset"  # Optional for mixed training

voice_type: "all"

# Model configuration
model_name: "meta-llama/Llama-3.2-3B-Instruct"
tokenizer_name: "meta-llama/Llama-3.2-3B-Instruct"

# Training arguments - Memory optimized for H100 80GB
epochs: 10
batch_size: 8
gradient_accumulation_steps: 1
number_processes: 4
pad_token: 128263
save_epochs: 1  # Save checkpoint every N epochs
learning_rate: 5.0e-5
lr_scheduler_type: "linear"
max_grad_norm: 1.0
max_length: 2048

# Naming and paths
save_folder: "checkpoints_llada_tts"
project_name: "llada-tts-experiment"
run_name: "llada-tts-run-1"

# TTS-specific settings (optional, will use defaults)
# num_audio_tokens: 28672  # 7 * 4096
# num_special_tokens: 10
# mask_token_id: 126336
# eps: 1e-3
```

### TTS Command Line Options
```bash
# Override config values from command line
python train_llada_tts.py \
  --config sample_tts_config.yaml \
  --epochs 5 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --save_epochs 2 \
  --output_dir ./custom_checkpoints
```

## Model Analysis

Use the built-in analysis tools:

```python
from utils import ModelAnalyzer, TokenizerHelper
from llada_model import LLaDAForMaskedLM
from sampling import create_sampler

# Load model
model = LLaDAForMaskedLM.from_pretrained("./llada_sft")
analyzer = ModelAnalyzer()

# Analyze attention patterns
analyzer.analyze_attention_patterns(model, input_ids)

# Compare sampling methods
results = analyzer.compare_sampling_methods(model, input_ids)
```

## Data Conversion

Convert existing datasets to LLaDA format:

```python
from utils import TokenizerHelper, DatasetConverter

tokenizer = TokenizerHelper("meta-llama/Llama-2-7b-hf")
converter = DatasetConverter(tokenizer)

# Convert ShareGPT format
converter.convert_sharegpt_to_llada(
    "sharegpt_data.jsonl",
    "llada_sft_data.jsonl"
)

# Convert Alpaca format
converter.convert_alpaca_to_llada(
    "alpaca_data.json",
    "llada_sft_data.jsonl"
)
```

## Key Differences from Standard Transformers

1. **No Causal Masking**: The attention mechanism can attend to all positions
2. **Mask Token Training**: Uses a special mask token (126336) during training
3. **Diffusion Loss**: Loss is weighted by inverse masking probability
4. **Specialized Sampling**: Multiple iterative sampling strategies

## Performance Notes

### For Instruct Models:
- Use **semi-autoregressive padding** with **low-confidence remasking**
- Avoid **semi-autoregressive origin** method
- For long sequences (>512), use **random remasking** to avoid excessive EOS tokens

### For Base Models:
- **Low-confidence remasking** generally works best
- **Fixed-length** and **semi-autoregressive padding** perform similarly

## File Structure

```
LLaDA/
├── llada_model.py          # Core LLaDA model implementation
├── pretraining.py          # Pre-training code  
├── sft_training.py         # Supervised fine-tuning code
├── sampling.py             # Sampling methods implementation
├── train_llada.py          # Main training script (text generation)
├── utils.py                # Utilities and data processing
├── tts_training.py         # TTS training implementation
├── tts_config.py           # TTS configuration classes
├── tts_dataset.py          # TTS dataset handling and data collation
├── train_llada_tts.py      # TTS training script
├── sample_tts_config.yaml  # Sample TTS configuration
├── requirements.txt        # Dependencies
└── README.md              # This file
```
