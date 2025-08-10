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

### 1. Create Configuration Files

Generate sample configuration files:
```bash
python train_llada.py create-configs
```

This creates `pretraining_config.json` and `sft_config.json` with default settings.

### 2. Pre-training

```bash
python train_llada.py pretrain \
  --data_path ./pretraining_data \
  --output_dir ./llada_pretrained \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 100000
```

### 3. Supervised Fine-tuning

```bash
python train_llada.py sft \
  --model_name_or_path ./llada_pretrained \
  --data_path ./sft_data \
  --output_dir ./llada_sft \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --max_steps 10000
```

### 4. Inference

```bash
python train_llada.py inference \
  --model_name_or_path ./llada_sft \
  --sampling_method fixed_length \
  --remasking_strategy low_confidence \
  --interactive
```

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
    "max_steps": 100000,
    "warmup_steps": 2000,
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
    "max_steps": 10000,
    "warmup_steps": 500
  }
}
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
├── train_llada.py          # Main training script
├── utils.py                # Utilities and data processing
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Citation

If you use this implementation, please cite the original LLaDA paper:

```bibtex
@article{llada2024,
  title={LLaDA: Large Language and Diffusion Alignment},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Ensure you're using GPU and consider mixed precision training
3. **Poor Generation Quality**: Try different sampling methods or adjust temperature

### Performance Tips

- Use `fp16=True` for faster training
- Increase `gradient_accumulation_steps` if you can't fit larger batches
- Use multiple GPUs with `accelerate` for distributed training
- Monitor with Weights & Biases for better experiment tracking

For more help, please open an issue on GitHub. 