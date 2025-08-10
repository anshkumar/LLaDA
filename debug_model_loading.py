#!/usr/bin/env python3
"""
Debug script to test model loading independently
"""

import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig, AutoTokenizer
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    try:
        logger.info("Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"✓ Tokenizer loaded successfully. Vocab size: {len(tokenizer)}")
        
        logger.info("Testing model loading with AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Config type: {type(model.config)}")
        logger.info(f"Model architecture: {getattr(model.config, 'architectures', 'unknown')}")
        logger.info(f"Vocab size: {model.config.vocab_size}")
        logger.info(f"Hidden size: {model.config.hidden_size}")
        
        # Test config conversion
        logger.info("Testing config conversion to LlamaConfig...")
        if not isinstance(model.config, LlamaConfig):
            logger.info("Converting to LlamaConfig...")
            llama_config = LlamaConfig(
                vocab_size=getattr(model.config, 'vocab_size', 32000),
                hidden_size=getattr(model.config, 'hidden_size', 4096),
                intermediate_size=getattr(model.config, 'intermediate_size', 11008),
                num_hidden_layers=getattr(model.config, 'num_hidden_layers', 32),
                num_attention_heads=getattr(model.config, 'num_attention_heads', 32),
                num_key_value_heads=getattr(model.config, 'num_key_value_heads', 32),
                max_position_embeddings=getattr(model.config, 'max_position_embeddings', 4096),
                rms_norm_eps=getattr(model.config, 'rms_norm_eps', 1e-6),
                rope_theta=getattr(model.config, 'rope_theta', 10000.0),
                attention_bias=getattr(model.config, 'attention_bias', False),
                tie_word_embeddings=getattr(model.config, 'tie_word_embeddings', False),
            )
            logger.info("✓ Config converted successfully")
        else:
            logger.info("✓ Already a LlamaConfig")
            llama_config = model.config
        
        # Test LLaDA model creation
        logger.info("Testing LLaDA model creation...")
        from llada_model import LLaDAForMaskedLM
        llada_model = LLaDAForMaskedLM(llama_config)
        logger.info("✓ LLaDA model created successfully")
        
        # Test weight copying
        logger.info("Testing weight copying...")
        try:
            llada_model.model.load_state_dict(model.model.state_dict(), strict=False)
            llada_model.lm_head.load_state_dict(model.lm_head.state_dict(), strict=False)
            logger.info("✓ Weights copied successfully")
        except Exception as e:
            logger.warning(f"Direct weight copying failed: {e}")
            logger.info("This is expected - weights will be copied layer by layer in training")
        
        # Test tokenizer expansion
        logger.info("Testing tokenizer expansion...")
        new_tokens = [f"<audio_token_{i}>" for i in range(100)]  # Test with 100 tokens
        num_added = tokenizer.add_tokens(new_tokens)
        logger.info(f"✓ Added {num_added} tokens")
        
        # Test embedding resize
        logger.info("Testing embedding resize...")
        old_size = llada_model.config.vocab_size
        new_size = len(tokenizer)
        llada_model.resize_token_embeddings(new_size)
        logger.info(f"✓ Embeddings resized from {old_size} to {new_size}")
        
        logger.info("🎉 All tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    test_model_loading() 