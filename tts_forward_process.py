"""
TTS-Aware Forward Process for LLaDA
Only masks audio tokens, preserves text prompts
"""

import torch


def tts_forward_process(input_ids: torch.Tensor, eps: float = 1e-3, tokenizer_vocab_size: int = None, 
                       training_progress: float = None, use_linear_schedule: bool = True,
                       use_curriculum_learning: bool = False, curriculum_target_progress: float = 0.8) -> tuple:
    """
    TTS-aware forward diffusion process for LLaDA
    Only masks audio tokens between START_OF_SPEECH and END_OF_SPEECH
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
        tokenizer_vocab_size: UNUSED - kept for compatibility (using hardcoded values)
        eps: Minimum masking probability
        training_progress: Progress through training (0.0 to 1.0). If None, uses random masking
        use_linear_schedule: If True, use linear masking schedule. If False, use original random masking
        use_curriculum_learning: If True, use curriculum learning timestep schedule (CLTS)
        curriculum_target_progress: When to fully transition to curriculum learning (0.0-1.0)
    
    Returns:
        noisy_batch: Input with masked audio tokens only
        masked_indices: Boolean tensor indicating masked positions (only audio)
        p_mask: Masking probabilities for each position
    """
    b, l = input_ids.shape
    device = input_ids.device
    
    # Hardcoded TTS token boundaries from SNAC model constants
    TOKENISER_LENGTH = 128256
    START_OF_SPEECH = TOKENISER_LENGTH + 1  # 128257
    END_OF_SPEECH = TOKENISER_LENGTH + 2    # 128258
    MASK_TOKEN_ID = 126336
    
    if use_curriculum_learning and training_progress is not None:
        # Curriculum Learning Timestep Schedule (CLTS) from diffusion optimization paper
        # Gradually shift from uniform sampling to focusing on harder timesteps (lower masking rates)
        
        # Calculate curriculum factor: 0 -> 1 over curriculum_target_progress
        gamma = min(training_progress / curriculum_target_progress, 1.0)
        
        if gamma < 1.0:
            # Mixed distribution: uniform + gaussian focusing on harder timesteps
            # Sample base timesteps uniformly
            t_uniform = torch.rand(b, device=device)
            
            # Sample from gaussian focused on harder timesteps (lower t values = less masking)
            # Mean at 0.3 (30% masking), std covers full range
            t_gaussian = torch.normal(0.3, 0.5, (b,), device=device).clamp(0.0, 1.0)
            
            # Mix uniform and gaussian based on training progress
            t = (1 - gamma) * t_uniform + gamma * t_gaussian
        else:
            # Pure curriculum: focus on harder timesteps
            t = torch.normal(0.3, 0.5, (b,), device=device).clamp(0.0, 1.0)
        
        # Calculate masking probability from sampled timesteps
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)  # Shape: (b, l)
        
    elif use_linear_schedule and training_progress is not None:
        # Linear masking schedule: increase from eps to 1.0 over training
        # training_progress: 0.0 -> eps (1% masking)
        # training_progress: 1.0 -> 1.0 (100% masking)
        target_mask_rate = eps + (1.0 - eps) * training_progress
        
        # Use the same masking rate for all sequences in the batch
        p_mask = torch.full((b, l), target_mask_rate, device=device)
    else:
        # Original random masking approach
        # Sample random timesteps for each sequence in the batch
        t = torch.rand(b, device=device)
        
        # Calculate masking probability: p_mask = (1 - eps) * t + eps
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)  # Shape: (b, l)
    
    # Create audio region mask (only allow masking between START_OF_SPEECH and END_OF_SPEECH)
    audio_region_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    for batch_idx in range(b):
        sequence = input_ids[batch_idx]
        
        # Find START_OF_SPEECH and END_OF_SPEECH positions
        start_speech_positions = (sequence == START_OF_SPEECH).nonzero(as_tuple=True)[0]
        end_speech_positions = (sequence == END_OF_SPEECH).nonzero(as_tuple=True)[0]
        
        if len(start_speech_positions) > 0 and len(end_speech_positions) > 0:
            start_pos = start_speech_positions[0].item()
            end_pos = end_speech_positions[0].item()
            
            # Mark audio region (excluding the boundary tokens themselves)
            if start_pos + 1 < end_pos:
                audio_region_mask[batch_idx, start_pos + 1:end_pos] = True
    
    # Determine which tokens to mask (only in audio regions)
    random_mask = torch.rand((b, l), device=device) < p_mask
    masked_indices = audio_region_mask & random_mask  # Only mask audio tokens
    
    # Apply masking only to selected audio tokens
    noisy_batch = torch.where(masked_indices, MASK_TOKEN_ID, input_ids)
    
    return noisy_batch, masked_indices, p_mask


def analyze_tts_masking(input_ids: torch.Tensor, tokenizer=None, masked_indices: torch.Tensor = None) -> dict:
    """
    Analyze the TTS masking results for debugging
    
    Args:
        input_ids: Original input tokens
        tokenizer: UNUSED - kept for compatibility (using hardcoded values)
        masked_indices: Boolean mask of what was masked
        
    Returns:
        Analysis dictionary
    """
    batch_size, seq_len = input_ids.shape
    
    # Hardcoded token boundaries from SNAC model constants
    TOKENISER_LENGTH = 128256
    START_OF_SPEECH = TOKENISER_LENGTH + 1  # 128257
    END_OF_SPEECH = TOKENISER_LENGTH + 2    # 128258
    
    total_tokens = input_ids.numel()
    total_masked = masked_indices.sum().item()
    
    audio_tokens = 0
    text_tokens = 0
    masked_audio = 0
    masked_text = 0
    
    for batch_idx in range(batch_size):
        sequence = input_ids[batch_idx]
        mask = masked_indices[batch_idx]
        
        # Find audio region boundaries
        start_positions = (sequence == START_OF_SPEECH).nonzero(as_tuple=True)[0]
        end_positions = (sequence == END_OF_SPEECH).nonzero(as_tuple=True)[0]
        
        if len(start_positions) > 0 and len(end_positions) > 0:
            start_pos = start_positions[0].item()
            end_pos = end_positions[0].item()
            
            # Count tokens in different regions
            for pos in range(seq_len):
                if start_pos < pos < end_pos:  # Audio region
                    audio_tokens += 1
                    if mask[pos]:
                        masked_audio += 1
                else:  # Text region
                    text_tokens += 1
                    if mask[pos]:
                        masked_text += 1
    
    return {
        "total_tokens": total_tokens,
        "total_masked": total_masked,
        "mask_percentage": total_masked / total_tokens * 100,
        "audio_tokens": audio_tokens,
        "text_tokens": text_tokens,
        "masked_audio": masked_audio,
        "masked_text": masked_text,
        "audio_mask_rate": masked_audio / audio_tokens * 100 if audio_tokens > 0 else 0,
        "text_mask_rate": masked_text / text_tokens * 100 if text_tokens > 0 else 0,
        "correctly_targeted": masked_text == 0  # Should be True (no text masked)
    } 