#!/usr/bin/env python3
"""
LLaDA + SNAC Inference Script
Combines LLaDA text-to-speech model with SNAC audio codec for high-quality audio generation
"""

import torch
import numpy as np
import onnxruntime
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Generator, List, Dict, Any, Union
from dataclasses import dataclass
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from llada_model import LLaDAForMaskedLM
from sampling import create_sampler, SamplingConfig
from tts_config import TTSConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for LLaDA+SNAC inference"""
    # Model paths
    llada_model_path: str
    snac_model_path: str
    tokenizer_path: str
    
    # Generation parameters
    max_new_tokens: int = 1024
    sampling_method: str = "fixed_length"  # "fixed_length", "semi_autoregressive_padding"
    remasking_strategy: str = "low_confidence"  # "random", "low_confidence"
    num_iterations: int = 10
    remasking_ratio: float = 0.8
    
    # Audio parameters
    sample_rate: int = 24000
    min_audio_tokens: int = 28  # Minimum tokens needed for SNAC
    audio_chunk_size: int = 28  # Process audio in chunks of N tokens
    
    # Device settings
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Special tokens (TTS-specific)
    audio_token_start: int = 128256  # Start of audio token range
    audio_token_end: int = 156928    # End of audio token range (128256 + 7*4096)
    mask_token_id: int = 126336
    
    # Performance settings
    use_cache: bool = True
    batch_size: int = 1


class SNACAudioDecoder:
    """SNAC audio decoder using ONNX runtime"""
    
    def __init__(self, snac_path: str):
        self.snac_path = Path(snac_path)
        if not self.snac_path.exists():
            raise FileNotFoundError(f"SNAC model not found: {snac_path}")
        
        logger.info(f"Loading SNAC model from {snac_path}")
        self._snac_session = onnxruntime.InferenceSession(
            str(snac_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        
        # Get input names for the ONNX model
        self.input_names = [x.name for x in self._snac_session.get_inputs()]
        logger.info(f"SNAC input names: {self.input_names}")
    
    def _convert_to_audio(self, multiframe: List[int]) -> Optional[np.ndarray]:
        """Convert audio tokens to raw audio using SNAC decoder"""
        if len(multiframe) < 28:  # Ensure we have enough tokens
            logger.debug(f"Not enough tokens for audio conversion: {len(multiframe)} < 28")
            return None

        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames * 7]

        # Initialize empty numpy arrays
        codes_0 = np.array([], dtype=np.int32)
        codes_1 = np.array([], dtype=np.int32) 
        codes_2 = np.array([], dtype=np.int32)

        for j in range(num_frames):
            i = 7 * j
            # Append values to numpy arrays following SNAC's hierarchical structure
            codes_0 = np.append(codes_0, frame[i])
            codes_1 = np.append(codes_1, [frame[i + 1], frame[i + 4]])
            codes_2 = np.append(codes_2, [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]])

        # Reshape arrays to match the expected input format (add batch dimension)
        codes_0 = np.expand_dims(codes_0, axis=0)
        codes_1 = np.expand_dims(codes_1, axis=0)
        codes_2 = np.expand_dims(codes_2, axis=0)

        # Validate token ranges (SNAC codebook typically 0-4096)
        if (np.any(codes_0 < 0) or np.any(codes_0 > 4096) or
            np.any(codes_1 < 0) or np.any(codes_1 > 4096) or
            np.any(codes_2 < 0) or np.any(codes_2 > 4096)):
            logger.warning(f"Audio tokens out of range: codes_0 range [{codes_0.min()}, {codes_0.max()}], "
                         f"codes_1 range [{codes_1.min()}, {codes_1.max()}], "
                         f"codes_2 range [{codes_2.min()}, {codes_2.max()}]")
            return None

        try:
            # Create input dictionary for ONNX session
            input_dict = dict(zip(self.input_names, [codes_0, codes_1, codes_2]))
            
            # Run SNAC inference
            audio_hat = self._snac_session.run(None, input_dict)[0]
            
            # Process output - extract relevant audio segment
            audio_np = audio_hat[:, :, 2048:4096]  # Extract middle portion
            
            # Convert to int16 format for audio output
            audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
            
            return audio_int16.flatten()
            
        except Exception as e:
            logger.error(f"SNAC inference failed: {e}")
            return None


class LLaDAInference:
    """LLaDA inference engine with SNAC audio generation"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set up torch dtype
        if config.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif config.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        logger.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")
        
        # Load components
        self._load_tokenizer()
        self._load_model()
        self._setup_sampler()
        self._load_snac()
        
        logger.info("✅ LLaDA+SNAC inference engine ready!")
    
    def _load_tokenizer(self):
        """Load the tokenizer"""
        logger.info(f"Loading tokenizer from {self.config.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
    
    def _load_model(self):
        """Load the LLaDA model"""
        logger.info(f"Loading LLaDA model from {self.config.llada_model_path}")
        
        try:
            # Load model checkpoint
            self.model = LLaDAForMaskedLM.from_pretrained(
                self.config.llada_model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.config.device,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load with from_pretrained: {e}")
            # Fallback: load state dict
            from transformers import LlamaConfig
            
            # Try to load config
            config_path = Path(self.config.llada_model_path) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    model_config = json.load(f)
                llama_config = LlamaConfig(**model_config)
            else:
                raise FileNotFoundError(f"Model config not found: {config_path}")
            
            # Initialize model and load weights
            self.model = LLaDAForMaskedLM(llama_config)
            
            # Load state dict
            state_dict_path = Path(self.config.llada_model_path) / "pytorch_model.bin"
            if not state_dict_path.exists():
                state_dict_path = Path(self.config.llada_model_path) / "model.safetensors"
            
            if state_dict_path.exists():
                state_dict = torch.load(state_dict_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"Model weights not found in {self.config.llada_model_path}")
        
        self.model.to(self.device).to(self.torch_dtype)
        self.model.eval()
        
        logger.info(f"✅ Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_sampler(self):
        """Setup the LLaDA sampler"""
        sampling_config = SamplingConfig(
            sampling_method=self.config.sampling_method,
            remasking_strategy=self.config.remasking_strategy,
            max_length=self.config.max_new_tokens,
            num_iterations=self.config.num_iterations,
            remasking_ratio=self.config.remasking_ratio,
            mask_token_id=self.config.mask_token_id,
            bos_token_id=self.tokenizer.bos_token_id or 1,
            eos_token_id=self.tokenizer.eos_token_id or 2,
            pad_token_id=self.tokenizer.pad_token_id or 0,
        )
        
        self.sampler = create_sampler(self.model, **sampling_config.__dict__)
        logger.info(f"✅ Sampler configured: {self.config.sampling_method} + {self.config.remasking_strategy}")
    
    def _load_snac(self):
        """Load SNAC audio decoder"""
        self.snac_decoder = SNACAudioDecoder(self.config.snac_model_path)
        logger.info("✅ SNAC decoder loaded")
    
    def _token_to_audio_id(self, token: int) -> Optional[int]:
        """Convert model token ID to SNAC audio token ID"""
        if self.config.audio_token_start <= token <= self.config.audio_token_end:
            # Map to 0-4096 range for SNAC
            return token - self.config.audio_token_start
        return None
    
    def _extract_audio_tokens(self, token_ids: torch.Tensor) -> List[int]:
        """Extract and convert audio tokens from generated sequence"""
        audio_tokens = []
        
        for token in token_ids.flatten():
            token_id = token.item()
            audio_id = self._token_to_audio_id(token_id)
            if audio_id is not None:
                audio_tokens.append(audio_id)
        
        return audio_tokens
    
    def generate_text_to_speech(
        self, 
        text: str, 
        stream_audio: bool = False
    ) -> Union[np.ndarray, Generator[np.ndarray, None, None]]:
        """
        Generate speech from text using LLaDA + SNAC
        
        Args:
            text: Input text to convert to speech
            stream_audio: If True, yield audio chunks as they're generated
            
        Returns:
            Audio array or generator of audio chunks
        """
        logger.info(f"Generating TTS for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        
        start_time = time.time()
        
        # Generate tokens using LLaDA
        with torch.no_grad():
            generated_ids = self.sampler.sample(
                prompt_ids=input_ids,
                max_new_tokens=self.config.max_new_tokens
            )
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Token generation completed in {generation_time:.2f}s")
        
        # Extract audio tokens
        audio_tokens = self._extract_audio_tokens(generated_ids)
        logger.info(f"🎵 Extracted {len(audio_tokens)} audio tokens")
        
        if len(audio_tokens) < self.config.min_audio_tokens:
            logger.warning(f"Not enough audio tokens generated: {len(audio_tokens)} < {self.config.min_audio_tokens}")
            return np.array([], dtype=np.int16)
        
        if stream_audio:
            return self._stream_audio_generation(audio_tokens)
        else:
            return self._batch_audio_generation(audio_tokens)
    
    def _stream_audio_generation(self, audio_tokens: List[int]) -> Generator[np.ndarray, None, None]:
        """Stream audio generation in chunks"""
        logger.info("🎵 Starting streaming audio generation...")
        
        chunk_size = self.config.audio_chunk_size
        
        for i in range(0, len(audio_tokens), chunk_size):
            chunk = audio_tokens[i:i + chunk_size]
            
            if len(chunk) >= self.config.min_audio_tokens:
                audio_chunk = self.snac_decoder._convert_to_audio(chunk)
                if audio_chunk is not None:
                    yield audio_chunk
                    logger.debug(f"🎵 Generated audio chunk {i//chunk_size + 1}, samples: {len(audio_chunk)}")
    
    def _batch_audio_generation(self, audio_tokens: List[int]) -> np.ndarray:
        """Generate all audio at once"""
        logger.info("🎵 Starting batch audio generation...")
        
        start_time = time.time()
        audio_samples = self.snac_decoder._convert_to_audio(audio_tokens)
        audio_time = time.time() - start_time
        
        if audio_samples is not None:
            logger.info(f"✅ Audio generation completed in {audio_time:.2f}s, samples: {len(audio_samples)}")
            return audio_samples
        else:
            logger.error("❌ Audio generation failed")
            return np.array([], dtype=np.int16)
    
    def save_audio(self, audio_samples: np.ndarray, output_path: str):
        """Save audio to file"""
        try:
            import soundfile as sf
            sf.write(output_path, audio_samples, self.config.sample_rate)
            logger.info(f"✅ Audio saved to {output_path}")
        except ImportError:
            logger.error("soundfile not installed. Install with: pip install soundfile")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")


def create_inference_config_from_args(args) -> InferenceConfig:
    """Create inference config from command line arguments"""
    return InferenceConfig(
        llada_model_path=args.model_path,
        snac_model_path=args.snac_path,
        tokenizer_path=args.tokenizer_path or args.model_path,
        max_new_tokens=args.max_tokens,
        sampling_method=args.sampling_method,
        remasking_strategy=args.remasking_strategy,
        num_iterations=args.num_iterations,
        device=args.device,
        torch_dtype=args.dtype,
        use_cache=not args.no_cache,
        sample_rate=args.sample_rate,
    )


def main():
    parser = argparse.ArgumentParser(description="LLaDA + SNAC TTS Inference")
    
    # Model paths
    parser.add_argument("--model-path", required=True, help="Path to LLaDA model")
    parser.add_argument("--snac-path", required=True, help="Path to SNAC ONNX model")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer (default: same as model)")
    
    # Generation parameters
    parser.add_argument("--text", required=True, help="Text to convert to speech")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--sampling-method", default="fixed_length", 
                       choices=["fixed_length", "semi_autoregressive_padding"],
                       help="Sampling method")
    parser.add_argument("--remasking-strategy", default="low_confidence",
                       choices=["random", "low_confidence"], help="Remasking strategy")
    parser.add_argument("--num-iterations", type=int, default=10, help="Number of sampling iterations")
    
    # Audio parameters  
    parser.add_argument("--sample-rate", type=int, default=24000, help="Audio sample rate")
    parser.add_argument("--stream", action="store_true", help="Enable streaming audio generation")
    
    # Device settings
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"],
                       help="Model dtype")
    parser.add_argument("--no-cache", action="store_true", help="Disable KV caching")
    
    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create inference config
    config = create_inference_config_from_args(args)
    
    # Initialize inference engine
    inference_engine = LLaDAInference(config)
    
    # Generate speech
    if args.stream:
        logger.info("🎵 Streaming mode enabled")
        audio_chunks = []
        for chunk in inference_engine.generate_text_to_speech(args.text, stream_audio=True):
            audio_chunks.append(chunk)
        
        if audio_chunks:
            # Concatenate all chunks
            full_audio = np.concatenate(audio_chunks)
            inference_engine.save_audio(full_audio, args.output)
        else:
            logger.error("No audio chunks generated")
    else:
        logger.info("🎵 Batch mode")
        audio_samples = inference_engine.generate_text_to_speech(args.text, stream_audio=False)
        if len(audio_samples) > 0:
            inference_engine.save_audio(audio_samples, args.output)
        else:
            logger.error("No audio generated")


if __name__ == "__main__":
    main() 