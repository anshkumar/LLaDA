# LLaDA Inference

High-quality Text-to-Speech inference using LLaDA + SNAC codec.

## 🚀 Quick Start

### 1. Command Line Interface

```bash
python llada_inference.py \
    --model-path ../checkpoints_llada_tts \
    --snac-path ./snac_24khz.onnx \
    --text "Hello, this is LLaDA TTS!" \
    --output speech.wav \
    --device cuda --dtype bfloat16
```

### 2. Python API

```python
from llada_inference import LLaDAInference, InferenceConfig

# Configure inference
config = InferenceConfig(
    llada_model_path="../checkpoints_llada_tts",
    snac_model_path="./snac_24khz.onnx",
    device="cuda",
    torch_dtype="bfloat16"
)

# Initialize and generate
engine = LLaDAInference(config)
audio = engine.generate_text_to_speech("Hello world!")
engine.save_audio(audio, "output.wav")
```

### 3. API Server

Start the server:
```bash
python llada_api_server.py \
    --model-path ../checkpoints_llada_tts \
    --snac-path ./snac_24khz.onnx \
    --host 0.0.0.0 --port 8000
```

Use the API:
```python
import requests

# Synchronous TTS
response = requests.post("http://localhost:8000/tts/sync", json={
    "text": "LLaDA text-to-speech synthesis!",
    "sampling_method": "fixed_length",
    "max_tokens": 1024
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## 📁 Files

- **`llada_inference.py`** - Core inference engine with SNAC integration
- **`llada_api_server.py`** - FastAPI server for REST API access
- **`example_inference.py`** - Comprehensive usage examples
- **`test_api_client.py`** - API testing and benchmarking client

## 🎯 Features

- ✅ **LLaDA Sampling Methods**: fixed_length, semi_autoregressive_*
- ✅ **SNAC Audio Codec**: Hierarchical audio token decoding
- ✅ **FlashAttention**: Memory-efficient non-causal attention
- ✅ **Streaming Generation**: Real-time audio synthesis
- ✅ **REST API**: Production-ready web service
- ✅ **Batch Processing**: Multiple concurrent requests

## 🔧 Configuration

### Sampling Methods
- **`fixed_length`** - Most consistent timing, good quality
- **`semi_autoregressive_padding`** - Often faster generation

### Remasking Strategies  
- **`low_confidence`** - Better quality, slower
- **`random`** - Faster generation, good quality

### Performance Settings
- **Device**: `cuda` (recommended) or `cpu`
- **Precision**: `bfloat16` (best), `float16`, or `float32`
- **Iterations**: 5-20 (more = better quality, slower)

## 📋 Requirements

Make sure you have the SNAC ONNX model file:
```bash
# Download or convert your SNAC model to ONNX format
# Place as snac_24khz.onnx in this directory
```

Install dependencies:
```bash
pip install onnxruntime-gpu fastapi uvicorn soundfile
```

## 🎵 Examples

See `example_inference.py` for detailed usage examples including:
- Streaming audio generation
- Batch processing
- Advanced configuration
- Performance optimization tips
- Troubleshooting guide

## 🧪 Testing

Test the API server:
```bash
python test_api_client.py --base-url http://localhost:8000
```

This will run comprehensive tests including:
- Health checks
- Synchronous TTS
- Asynchronous TTS  
- Batch processing
- Performance benchmarks 