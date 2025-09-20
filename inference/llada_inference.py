#!/usr/bin/env python3
"""
LLaDA + SNAC Inference Script
Combines LLaDA text-to-speech model with SNAC audio codec for high-quality audio generation
"""

import torch
import numpy as np
import logging
import argparse
import json
import time
import os
from pathlib import Path
from typing import Optional, Generator, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer
from snac import SNAC

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from llada_model import LLaDAForMaskedLM
from sampling import create_sampler, SamplingConfig
from tts_config import TTSConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for LLaDA+SNAC inference"""
    # Model paths
    llada_model_path: str
    tokenizer_path: str
    
    # Generation parameters
    max_new_tokens: int = 1024
    sampling_method: str = "fixed_length"  # "fixed_length", "semi_autoregressive_padding"
    remasking_strategy: str = "low_confidence"  # "random", "low_confidence"
    num_iterations: int = 10
    remasking_ratio: float = 0.8
    
    # Audio parameters
    sample_rate: int = 24000
    min_audio_tokens: int = 7   # Minimum tokens needed for SNAC (1 frame)
    audio_chunk_size: int = 28  # Process audio in chunks of N tokens (4 frames)
    
    # Device settings
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Special tokens (TTS-specific)
    audio_token_start: int = 128256  # Start of custom token range (includes 10 special tokens + audio tokens)
    audio_token_end: int = 156928    # End of audio token range (128256 + 10 + 7*4096)
    mask_token_id: int = 126336      # Must match MASK_TOKEN_ID from tts_forward_process.py training
    # Note: Custom token layout: positions 0-9 are special tokens, positions 10+ are audio tokens
    
    # Performance settings
    use_cache: bool = True
    batch_size: int = 1


class SNACAudioDecoder:
    """SNAC audio decoder using PyTorch SNAC model for high-quality audio generation"""
    
    def __init__(self):
        # Initialize PyTorch SNAC model
        logger.info("Loading PyTorch SNAC model...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        
        # Set device
        self.snac_device = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.snac_device)
        
        logger.info(f"SNAC model loaded on {self.snac_device}")
    
    def _convert_to_audio(self, multiframe: List[int]) -> Optional[np.ndarray]:
        """Convert audio tokens to raw audio using PyTorch SNAC decoder with proper hierarchical structure"""
        if len(multiframe) < 7:  # Need at least 1 frame (7 tokens)
            logger.debug(f"Not enough tokens for audio conversion: {len(multiframe)} < 7")
            return None

        # Process in groups of 7 tokens per frame (SNAC structure)
        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames * 7]  # Use complete frames only

        # Initialize hierarchical code tensors (following original working code)
        codes_0 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=self.snac_device, dtype=torch.int32)

        for j in range(num_frames):
            i = 7 * j
            
            # Build codes_0 (1 token per frame)
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=self.snac_device, dtype=torch.int32)
            else:
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=self.snac_device, dtype=torch.int32)])

            # Build codes_1 (2 tokens per frame: positions i+1, i+4)
            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor([frame[i+1]], device=self.snac_device, dtype=torch.int32)
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.snac_device, dtype=torch.int32)])
            else:
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=self.snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.snac_device, dtype=torch.int32)])
            
            # Build codes_2 (4 tokens per frame: positions i+2, i+3, i+5, i+6)
            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i+2]], device=self.snac_device, dtype=torch.int32)
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.snac_device, dtype=torch.int32)])
            else:
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.snac_device, dtype=torch.int32)])

        # Prepare codes list with batch dimension
        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        
        # Validate token ranges (SNAC codebook expects 0-4096)
        if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
            torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
            torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
            logger.warning(f"Audio tokens out of range: codes_0 range [{codes[0].min()}, {codes[0].max()}], "
                         f"codes_1 range [{codes[1].min()}, {codes[1].max()}], "
                         f"codes_2 range [{codes[2].min()}, {codes[2].max()}]")
            return None

        try:
            # Generate audio using PyTorch SNAC model
            with torch.inference_mode():
                audio_hat = self.model.decode(codes)
            
            # Extract audio slice (following original working code)
            audio_slice = audio_hat[:, :, 2048:4096]
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            
            # Convert to int16 format for audio output
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
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
        
        logger.info(f"✅ Using mask token ID: {self.config.mask_token_id} (added directly as token IDs)")
        
        # Debug: Check for custom tokens in vocabulary
        vocab = self.tokenizer.get_vocab()
        custom_tokens = [k for k in vocab.keys() if k.startswith("<custom_token_")]
        logger.info(f"Found {len(custom_tokens)} custom tokens in vocabulary")
        if custom_tokens:
            logger.info(f"Sample custom tokens: {custom_tokens[:5]} ...")
            logger.info(f"Custom token ID range: {min(vocab[t] for t in custom_tokens)} - {max(vocab[t] for t in custom_tokens)}")
        else:
            logger.warning("❌ NO custom tokens found in tokenizer vocabulary!")
    
    def _load_model(self):
        """Load the LLaDA model - exactly like training does it"""
        logger.info(f"Loading LLaDA model from {self.config.llada_model_path}")
        
        from transformers import LlamaConfig, AutoModelForCausalLM
        
        # Load the training config that was saved with the checkpoint
        config_path = Path(self.config.llada_model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path) as f:
            training_config = json.load(f)
        
        # Get the original base model name
        base_model_name = training_config.get('model_name_or_path', 'canopylabs/3b-hi-pretrain-research_release')
        logger.info(f"Loading base model architecture from: {base_model_name}")
        
        # Load base model to get architecture (exactly like training code does)
        hf_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        # Get and convert config (exactly like training code does)
        model_config = hf_model.config
        llama_config = LlamaConfig(
            vocab_size=len(self.tokenizer),  # Use current tokenizer size
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
        
        logger.info(f"Model architecture: hidden_size={llama_config.hidden_size}, num_layers={llama_config.num_hidden_layers}")
        
        # Create LLaDA model with correct architecture
        self.model = LLaDAForMaskedLM(llama_config)
        
        # Clean up base model
        del hf_model
        
        # Load the saved TTS model weights
        state_dict_path = Path(self.config.llada_model_path) / "pytorch_model.bin"
        if not state_dict_path.exists():
            raise FileNotFoundError(f"Model weights not found: {state_dict_path}")
        
        logger.info(f"Loading TTS weights from {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(self.device).to(self.torch_dtype)
        self.model.eval()
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"✅ Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"✅ Vocab size: {self.model.config.vocab_size}")
    
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
        self.snac_decoder = SNACAudioDecoder()
        logger.info("✅ SNAC decoder loaded")
    
    def _extract_audio_tokens(self, token_ids: torch.Tensor) -> List[int]:
        """Extract and convert audio tokens from generated sequence (for non-iterative generation)"""
        audio_tokens = []
        audio_token_position = 0
        
        for token in token_ids.flatten():
            token_id = token.item()
            
            # Check if it's an audio token
            if self.config.audio_token_start <= token_id <= self.config.audio_token_end:
                # Convert using formula from working code: account for 10 special tokens before audio tokens
                snac_token_id = (token_id - self.config.audio_token_start) - 10 - ((audio_token_position % 7) * 4096)
                
                # Ensure token is in valid SNAC range (0-4096)
                if 0 <= snac_token_id <= 4096:
                    audio_tokens.append(snac_token_id)
                else:
                    logger.debug(f"Token {token_id} converted to {snac_token_id} is out of SNAC range, skipping")
                
                # Increment position for every audio token
                audio_token_position += 1
        
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

    def _create_tts_prompt_with_masks(self, text: str, num_mask_tokens: int = 28) -> str:
        """
        Create TTS prompt with specified number of mask tokens
        
        Args:
            text: Input text (should contain conversation format)
            num_mask_tokens: Number of mask tokens to add after START_OF_SPEECH
        
        Returns:
            Formatted prompt with mask tokens
        """
        # Convert token IDs to actual token names in vocabulary
        # Token ID 128256 + X -> <custom_token_X>
        def id_to_token_name(token_id: int) -> str:
            return f"<custom_token_{token_id - 128256}>"
        
        start_of_human = id_to_token_name(128259)    # <custom_token_3>
        end_of_human = id_to_token_name(128260)      # <custom_token_4>
        start_of_ai = id_to_token_name(128261)       # <custom_token_5>
        start_of_speech = id_to_token_name(128257)   # <custom_token_1>
        end_of_speech = id_to_token_name(128258)     # <custom_token_2>
        end_of_ai = id_to_token_name(128262)         # <custom_token_6>
        
        # Note: We no longer need the mask token string since we add mask tokens as direct IDs
        
        # Create the base prompt without mask tokens (we'll add them as token IDs later)
        if start_of_speech in text:
            # Return text without additional mask tokens (they'll be added later)
            return text
        else:
            # Create complete conversation format without mask tokens
            return f"{start_of_human}{text}{end_of_human}{start_of_ai}{start_of_speech}"
    
    def _extract_generated_audio_tokens(self, original_ids: torch.Tensor, generated_ids: torch.Tensor) -> Tuple[List[int], List[int]]:
        """Extract newly generated tokens and return both SNAC tokens and original model tokens"""
        # Get the new tokens (everything after the original prompt)
        new_tokens = generated_ids[:, original_ids.size(1):]
        
        audio_tokens = []  # SNAC tokens (0-4095) 
        model_tokens = []  # Original model tokens (for next iteration)
        
        # Track position in audio token sequence (not just successful conversions)
        audio_token_position = 0
        
        for token in new_tokens.flatten():
            token_id = token.item()
            
            # Check for END_OF_SPEECH or END_OF_AI tokens
            if token_id in [128258, 128262]:  # END_OF_SPEECH, END_OF_AI
                logger.info(f"Found end token: {token_id}, stopping generation")
                break
                
            # Check if it's an audio token
            if self.config.audio_token_start <= token_id <= self.config.audio_token_end:
                # Convert using formula from working code: account for 10 special tokens before audio tokens
                # token_id - audio_token_start gives position in custom token range
                # Subtract 10 to get position in audio token range, then apply hierarchical offset
                snac_token_id = (token_id - self.config.audio_token_start) - 10 - ((audio_token_position % 7) * 4096)
                
                logger.debug(f"Audio token {audio_token_position}: {token_id} → {snac_token_id} (pos % 7 = {audio_token_position % 7})")
                
                # Ensure token is in valid SNAC range (0-4096)
                if 0 <= snac_token_id <= 4096:
                    audio_tokens.append(snac_token_id)
                    model_tokens.append(token_id)  # Keep original for next iteration
                    logger.debug(f"✅ Accepted SNAC token: {snac_token_id}")
                else:
                    logger.debug(f"❌ Token {token_id} converted to {snac_token_id} is out of SNAC range, skipping")
                
                # Increment position for every audio token (whether accepted or not)
                audio_token_position += 1
        
        logger.debug(f"Final result: {len(audio_tokens)} valid SNAC tokens from {audio_token_position} audio tokens")
        return audio_tokens, model_tokens
    
    def generate_text_to_speech_iterative(
        self,
        text: str,
        max_chunks: int = 50,
        chunk_size: int = 28,
        stream_audio: bool = False
    ) -> Union[np.ndarray, Generator[np.ndarray, None, None]]:
        """
        Generate speech using iterative chunk-based approach
        
        Args:
            text: Input text for TTS
            max_chunks: Maximum number of 28-token chunks to generate
            chunk_size: Number of mask tokens per chunk (28 for SNAC frames)
            stream_audio: If True, yield audio chunks as they're generated
            
        Returns:
            Audio array or generator of audio chunks
        """
        if stream_audio:
            return self._generate_iterative_streaming(text, max_chunks, chunk_size)
        else:
            return self._generate_iterative_batch(text, max_chunks, chunk_size)
    
    def _generate_iterative_batch(self, text: str, max_chunks: int, chunk_size: int) -> np.ndarray:
        """Non-streaming iterative generation that returns complete audio"""
        logger.info(f"Starting iterative TTS generation: {max_chunks} chunks of {chunk_size} tokens")
        logger.info(f"Input text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        all_audio_tokens = []
        current_sequence = text
        
        for chunk_idx in range(max_chunks):
            logger.info(f"Generating chunk {chunk_idx + 1}/{max_chunks}")
            
            # Create base prompt without mask tokens
            base_prompt = self._create_tts_prompt_with_masks(current_sequence, 0)  # 0 mask tokens for now
            logger.info(f"Chunk {chunk_idx + 1} base prompt: {base_prompt[:200]}...")
            
            # Tokenize the base prompt
            inputs = self.tokenizer(base_prompt, return_tensors="pt", padding=True, truncation=True)
            base_input_ids = inputs.input_ids.to(self.device)
            
            # Add mask tokens directly as token IDs
            mask_token_ids = torch.full((base_input_ids.shape[0], chunk_size), 
                                       self.config.mask_token_id, 
                                       dtype=base_input_ids.dtype, 
                                       device=self.device)
            
            # Concatenate base prompt + mask tokens
            input_ids = torch.cat([base_input_ids, mask_token_ids], dim=1)
            logger.info(f"Input IDs shape: {input_ids.shape}, prompt length: {input_ids.size(1)}")
            
            # Debug: Show token structure
            first_tokens = input_ids[0][:15].cpu().numpy().tolist()
            logger.info(f"First 15 token IDs: {first_tokens}")
            
            # Count mask tokens
            mask_count = (input_ids == self.config.mask_token_id).sum().item()
            logger.info(f"✅ Found {mask_count} mask tokens (ID: {self.config.mask_token_id})")
            
            # Check if we see the expected special tokens
            expected_tokens = [128259, 128260, 128261, 128257]
            found_expected = [tid for tid in input_ids[0].cpu().numpy() if tid in expected_tokens]
            logger.info(f"Found conversation tokens: {list(set(found_expected))}")
            
            # Generate tokens for this chunk
            with torch.no_grad():
                generated_ids = self.sampler.sample(
                    prompt_ids=input_ids,
                    max_new_tokens=chunk_size  # Only generate exactly chunk_size tokens
                )
            
            logger.info(f"Generated IDs shape: {generated_ids.shape}")
            
            # Debug: Check what tokens were actually generated
            new_tokens = generated_ids[:, input_ids.size(1):]
            new_tokens_list = new_tokens.flatten().cpu().numpy().tolist()
            logger.info(f"Generated tokens: {new_tokens_list[:10]}...")  # First 10 tokens
            
            # Check if any are in audio range
            audio_range_tokens = [t for t in new_tokens_list if self.config.audio_token_start <= t <= self.config.audio_token_end]
            logger.info(f"Audio range tokens found: {len(audio_range_tokens)} out of {len(new_tokens_list)}")
            if audio_range_tokens:
                logger.info(f"First 5 audio tokens: {audio_range_tokens[:5]}")
            
            # Extract only the new audio tokens from this generation
            chunk_audio_tokens, chunk_model_tokens = self._extract_generated_audio_tokens(input_ids, generated_ids)
            
            logger.info(f"Extracted {len(chunk_audio_tokens)} SNAC tokens, {len(chunk_model_tokens)} model tokens")
            if chunk_audio_tokens:
                logger.info(f"First 5 SNAC tokens: {chunk_audio_tokens[:5]}")
                logger.info(f"First 5 model tokens: {chunk_model_tokens[:5]}")
            
            if not chunk_audio_tokens:
                logger.info(f"No audio tokens generated in chunk {chunk_idx + 1}, stopping")
                break
            
            logger.info(f"Chunk {chunk_idx + 1}: Generated {len(chunk_audio_tokens)} audio tokens")
            all_audio_tokens.extend(chunk_audio_tokens)
            
            # Update current sequence with generated tokens (for next iteration)
            # Use the original model tokens directly
            generated_model_token_strings = []
            for model_token in chunk_model_tokens:
                generated_model_token_strings.append(f"<custom_token_{model_token}>")
            
            # Append to current sequence for next iteration
            current_sequence += "".join(generated_model_token_strings)
        
        logger.info(f"Iterative generation complete. Total audio tokens: {len(all_audio_tokens)}")
        
        if not all_audio_tokens:
            logger.warning("No audio tokens generated")
            return np.array([], dtype=np.int16)
        
        # Generate complete audio as numpy array
        return self._batch_audio_generation(all_audio_tokens)
    
    def _generate_iterative_streaming(self, text: str, max_chunks: int, chunk_size: int) -> Generator[np.ndarray, None, None]:
        """Streaming iterative generation that yields audio chunks"""
        logger.info(f"Starting iterative TTS generation: {max_chunks} chunks of {chunk_size} tokens")
        logger.info(f"Input text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        all_audio_tokens = []
        current_sequence = text
        
        for chunk_idx in range(max_chunks):
            logger.info(f"Generating chunk {chunk_idx + 1}/{max_chunks}")
            
            # Create prompt with mask tokens for this chunk
            prompt_with_masks = self._create_tts_prompt_with_masks(current_sequence, chunk_size)
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt_with_masks, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            
            # Generate tokens for this chunk
            with torch.no_grad():
                generated_ids = self.sampler.sample(
                    prompt_ids=input_ids,
                    max_new_tokens=chunk_size  # Only generate exactly chunk_size tokens
                )
            # Extract only the new audio tokens from this generation
            chunk_audio_tokens, chunk_model_tokens = self._extract_generated_audio_tokens(input_ids, generated_ids)
            
            if not chunk_audio_tokens:
                logger.info(f"No audio tokens generated in chunk {chunk_idx + 1}, stopping")
                break
            
            logger.info(f"Chunk {chunk_idx + 1}: Generated {len(chunk_audio_tokens)} audio tokens")
            all_audio_tokens.extend(chunk_audio_tokens)
            
            # Update current sequence with generated tokens (for next iteration)
            # Use the original model tokens directly
            generated_model_token_strings = []
            for model_token in chunk_model_tokens:
                generated_model_token_strings.append(f"<custom_token_{model_token}>")
            
            # Append to current sequence for next iteration
            current_sequence += "".join(generated_model_token_strings)
            
            # Check if we have enough tokens for SNAC (minimum 7)
            if len(all_audio_tokens) >= 7:
                # Generate audio for accumulated tokens
                if len(all_audio_tokens) >= chunk_size:
                    chunk_for_audio = all_audio_tokens[-chunk_size:]
                    audio_chunk = self.snac_decoder._convert_to_audio(chunk_for_audio)
                    if audio_chunk is not None:
                        yield audio_chunk
        
        # Generate final audio chunk if there are remaining tokens
        if len(all_audio_tokens) % chunk_size != 0:
            remaining_tokens = all_audio_tokens[-(len(all_audio_tokens) % chunk_size):]
            if len(remaining_tokens) >= self.config.min_audio_tokens:
                audio_chunk = self.snac_decoder._convert_to_audio(remaining_tokens)
                if audio_chunk is not None:
                    yield audio_chunk


def create_inference_config_from_args(args) -> InferenceConfig:
    """Create inference config from command line arguments"""
    return InferenceConfig(
        llada_model_path=args.model_path,
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

    parser.add_argument("--tokenizer-path", help="Path to tokenizer (default: same as model)")
    
    # Generation parameters
    parser.add_argument("--text", required=True, help="Text to convert to speech")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--iterative", action="store_true", help="Use iterative chunk-based generation (recommended)")
    parser.add_argument("--max-chunks", type=int, default=50, help="Maximum number of chunks for iterative generation")
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
    
    # Debug logging can be enabled with --verbose flag
    
    # Create inference config
    config = create_inference_config_from_args(args)
    
    # Initialize inference engine
    inference_engine = LLaDAInference(config)
    
    # Generate speech
    if args.iterative:
        logger.info("🎵 Using iterative chunk-based generation")
        if args.stream:
            logger.info("🎵 Streaming mode enabled")
            audio_chunks = []
            for chunk in inference_engine.generate_text_to_speech_iterative(
                args.text,
                max_chunks=args.max_chunks,
                chunk_size=28, # SNAC chunk size
                stream_audio=True
            ):
                audio_chunks.append(chunk)
            
            if audio_chunks:
                # Concatenate all chunks
                full_audio = np.concatenate(audio_chunks)
                inference_engine.save_audio(full_audio, args.output)
            else:
                logger.error("No audio chunks generated")
        else:
            logger.info("🎵 Batch iterative mode")
            audio_samples = inference_engine.generate_text_to_speech_iterative(
                args.text,
                max_chunks=args.max_chunks,
                chunk_size=28,
                stream_audio=False
            )
            if len(audio_samples) > 0:
                inference_engine.save_audio(audio_samples, args.output)
            else:
                logger.error("No audio generated")
    else:
        # Original generation method
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