#!/usr/bin/env python3
"""
LLaDA + SNAC TTS Inference Examples
Demonstrates different ways to use the LLaDA+SNAC TTS system
"""

import asyncio
import json
import time
from pathlib import Path

import requests
import numpy as np

def example_cli_inference():
    """Example using the command-line interface"""
    print("🎯 Example 1: CLI Inference")
    print("=" * 50)
    
    # Example command
    cmd = """
    python llada_inference.py \\
        --model-path ./checkpoints_llada_tts \\
        --snac-path ./snac_24khz.onnx \\
        --text "Hello, this is a test of LLaDA text-to-speech synthesis." \\
        --output hello_llada.wav \\
        --sampling-method fixed_length \\
        --remasking-strategy low_confidence \\
        --max-tokens 1024 \\
        --num-iterations 10 \\
        --device cuda \\
        --dtype bfloat16 \\
        --verbose
    """
    
    print("Command to run:")
    print(cmd)
    print("\nThis will:")
    print("- Load your trained LLaDA model")
    print("- Load the SNAC decoder")
    print("- Generate audio tokens using LLaDA's mask prediction")
    print("- Convert tokens to audio using SNAC")
    print("- Save the result as hello_llada.wav")

def example_python_api():
    """Example using the Python API directly"""
    print("\n🎯 Example 2: Python API")
    print("=" * 50)
    
    example_code = '''
from llada_inference import LLaDAInference, InferenceConfig

# Configure inference
config = InferenceConfig(
    llada_model_path="./checkpoints_llada_tts",
    snac_model_path="./snac_24khz.onnx",
    tokenizer_path="./checkpoints_llada_tts",
    max_new_tokens=1024,
    sampling_method="fixed_length",
    remasking_strategy="low_confidence",
    device="cuda",
    torch_dtype="bfloat16"
)

# Initialize inference engine
inference_engine = LLaDAInference(config)

# Generate speech
text = "LLaDA combines the power of masked language modeling with high-quality audio synthesis."
audio_samples = inference_engine.generate_text_to_speech(text)

# Save audio
inference_engine.save_audio(audio_samples, "python_api_output.wav")
print(f"Generated {len(audio_samples)} audio samples")
'''
    
    print("Python code:")
    print(example_code)

def example_streaming_inference():
    """Example of streaming audio generation"""
    print("\n🎯 Example 3: Streaming Inference")
    print("=" * 50)
    
    example_code = '''
# Enable streaming for real-time audio generation
audio_chunks = inference_engine.generate_text_to_speech(
    text="This is streaming audio generation with LLaDA and SNAC.",
    stream_audio=True
)

# Process chunks as they arrive
all_chunks = []
for i, chunk in enumerate(audio_chunks):
    print(f"Received chunk {i+1}: {len(chunk)} samples")
    all_chunks.append(chunk)
    
    # You could play this chunk immediately in a real application
    # audio_player.play(chunk)

# Combine all chunks
full_audio = np.concatenate(all_chunks)
inference_engine.save_audio(full_audio, "streaming_output.wav")
'''
    
    print("Streaming code:")
    print(example_code)

def example_api_server():
    """Example using the FastAPI server"""
    print("\n🎯 Example 4: API Server Usage")
    print("=" * 50)
    
    print("1. Start the server:")
    server_cmd = """
    python llada_api_server.py \\
        --model-path ./checkpoints_llada_tts \\
        --snac-path ./snac_24khz.onnx \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --device cuda \\
        --dtype bfloat16
    """
    print(server_cmd)
    
    print("\n2. Use the API:")
    
    # Async API example
    async_example = '''
# Asynchronous API (non-blocking)
import requests

# Submit TTS job
response = requests.post("http://localhost:8000/tts", json={
    "text": "This is an asynchronous text-to-speech request using LLaDA.",
    "sampling_method": "fixed_length",
    "remasking_strategy": "low_confidence",
    "max_tokens": 1024,
    "num_iterations": 10
})

job_data = response.json()
job_id = job_data["job_id"]
print(f"Created job: {job_id}")

# Check job status
status_response = requests.get(f"http://localhost:8000/tts/{job_id}")
status_data = status_response.json()
print(f"Job status: {status_data['status']}")

# Download audio when ready
if status_data["status"] == "completed":
    audio_response = requests.get(f"http://localhost:8000{status_data['audio_url']}")
    with open("api_output.wav", "wb") as f:
        f.write(audio_response.content)
'''
    
    print(async_example)
    
    # Sync API example
    print("\n3. Synchronous API (blocking):")
    sync_example = '''
# Synchronous API (blocking - returns audio immediately)
response = requests.post("http://localhost:8000/tts/sync", json={
    "text": "This is a synchronous request that returns audio immediately.",
    "sampling_method": "fixed_length",
    "max_tokens": 512
})

# Save the audio directly
with open("sync_output.wav", "wb") as f:
    f.write(response.content)

print(f"Generation time: {response.headers.get('X-Generation-Time')}s")
print(f"Audio duration: {response.headers.get('X-Audio-Duration')}s")
'''
    
    print(sync_example)

def example_advanced_configuration():
    """Example of advanced configuration options"""
    print("\n🎯 Example 5: Advanced Configuration")
    print("=" * 50)
    
    config_example = '''
# Advanced configuration for different use cases

# High-quality, slower generation
high_quality_config = InferenceConfig(
    llada_model_path="./checkpoints_llada_tts",
    snac_model_path="./snac_24khz.onnx",
    sampling_method="fixed_length",
    remasking_strategy="low_confidence",
    num_iterations=15,  # More iterations for quality
    remasking_ratio=0.9,  # Higher remasking ratio
    max_new_tokens=2048,  # Longer sequences
    torch_dtype="bfloat16",
    device="cuda"
)

# Fast generation, lower quality
fast_config = InferenceConfig(
    llada_model_path="./checkpoints_llada_tts", 
    snac_model_path="./snac_24khz.onnx",
    sampling_method="semi_autoregressive_padding",
    remasking_strategy="random",
    num_iterations=5,  # Fewer iterations for speed
    remasking_ratio=0.6,  # Lower remasking ratio
    max_new_tokens=512,
    torch_dtype="bfloat16",
    device="cuda"
)

# Memory-constrained setup
memory_efficient_config = InferenceConfig(
    llada_model_path="./checkpoints_llada_tts",
    snac_model_path="./snac_24khz.onnx",
    torch_dtype="float16",  # Use FP16 to save memory
    max_new_tokens=256,  # Shorter sequences
    audio_chunk_size=21,  # Smaller chunks
    device="cuda"
)
'''
    
    print(config_example)

def example_batch_processing():
    """Example of batch processing multiple texts"""
    print("\n🎯 Example 6: Batch Processing")
    print("=" * 50)
    
    batch_example = '''
import asyncio
from pathlib import Path

async def process_text_batch(texts, output_dir="batch_outputs"):
    """Process multiple texts in parallel"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize inference engine
    config = InferenceConfig(
        llada_model_path="./checkpoints_llada_tts",
        snac_model_path="./snac_24khz.onnx",
        device="cuda"
    )
    inference_engine = LLaDAInference(config)
    
    async def process_single_text(text, index):
        """Process a single text"""
        try:
            audio_samples = inference_engine.generate_text_to_speech(text)
            output_file = output_path / f"output_{index:03d}.wav"
            inference_engine.save_audio(audio_samples, str(output_file))
            return f"✅ Processed text {index}: {len(audio_samples)} samples"
        except Exception as e:
            return f"❌ Failed text {index}: {e}"
    
    # Process all texts
    tasks = [process_single_text(text, i) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(result)

# Example usage
texts = [
    "Hello, this is the first text to synthesize.",
    "This is the second text with different content.",
    "The third text demonstrates batch processing capabilities.",
    "Finally, this is the fourth and last text in our batch."
]

# Run batch processing
asyncio.run(process_text_batch(texts))
'''
    
    print(batch_example)

def example_performance_tips():
    """Performance optimization tips"""
    print("\n🎯 Example 7: Performance Tips")
    print("=" * 50)
    
    tips = '''
Performance Optimization Tips:

1. Model Loading:
   - Use bfloat16 for best speed/quality balance
   - Pre-load model to avoid startup delays
   - Use CUDA when available

2. Sampling Configuration:
   - fixed_length: Most consistent timing
   - semi_autoregressive_padding: Often faster
   - low_confidence remasking: Better quality
   - random remasking: Faster generation

3. SNAC Optimization:
   - Process audio in appropriate chunk sizes (28+ tokens)
   - Use GPU ONNX provider when available
   - Validate token ranges to avoid SNAC errors

4. Memory Management:
   - Use gradient checkpointing for training
   - Clear CUDA cache between large generations
   - Process long texts in smaller chunks

5. Distributed Inference:
   - Use the API server for multiple clients
   - Implement proper load balancing
   - Consider caching frequently used models
'''
    
    print(tips)

def example_troubleshooting():
    """Common issues and solutions"""
    print("\n🎯 Example 8: Troubleshooting")
    print("=" * 50)
    
    troubleshooting = '''
Common Issues and Solutions:

1. "No audio tokens generated":
   - Check if your model was trained with TTS data
   - Verify audio token range configuration
   - Try increasing max_new_tokens

2. "SNAC inference failed":
   - Ensure audio tokens are in range [0, 4096]
   - Check SNAC model path and format
   - Verify you have enough tokens (min 28)

3. "CUDA out of memory":
   - Reduce max_new_tokens
   - Use torch_dtype="float16"
   - Enable gradient checkpointing
   - Process shorter texts

4. "Model loading failed":
   - Check model path exists
   - Verify model format (PyTorch vs SafeTensors)
   - Ensure config.json is present
   - Try loading with trust_remote_code=True

5. "Poor audio quality":
   - Increase num_iterations (10-15)
   - Use low_confidence remasking
   - Try fixed_length sampling
   - Check if model is properly trained

6. "Slow generation":
   - Use semi_autoregressive_padding
   - Reduce num_iterations (5-8)
   - Use random remasking
   - Enable FlashAttention optimizations
'''
    
    print(troubleshooting)

def main():
    """Run all examples"""
    print("🎵 LLaDA + SNAC TTS Inference Examples")
    print("=" * 80)
    
    example_cli_inference()
    example_python_api()
    example_streaming_inference()
    example_api_server()
    example_advanced_configuration()
    example_batch_processing()
    example_performance_tips()
    example_troubleshooting()
    
    print("\n" + "=" * 80)
    print("🚀 Ready to use LLaDA + SNAC for high-quality TTS!")
    print("Check the individual scripts for working implementations.")

if __name__ == "__main__":
    main() 