import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaRMSNorm
from typing import Optional, Tuple, Union
import math
from flash_attn import flash_attn_func

class LLaDAAttention(nn.Module):
    """Modified LLaMA attention without causal masking for LLaDA"""
    
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        
        # Initialize rotary embeddings
        self._init_rope()
    
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # Handle different rope_scaling formats
            if isinstance(self.config.rope_scaling, dict):
                scaling_type = self.config.rope_scaling.get("type", "linear")
                scaling_factor = self.config.rope_scaling.get("factor", 1.0)
            else:
                # Fallback for non-dict rope_scaling
                scaling_type = "linear"
                scaling_factor = float(self.config.rope_scaling) if self.config.rope_scaling else 1.0
            
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # Fallback to basic RoPE for unknown types
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        # FlashAttention expects (batch, seqlen, nheads, headdim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        # Apply RoPE - need to transpose for apply_rotary_pos_emb function
        query_states_rope = query_states.transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        key_states_rope = key_states.transpose(1, 2)      # (bsz, num_kv_heads, q_len, head_dim)
        
        kv_seq_len = key_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]  # past_key_value is in (batch, heads, seqlen, headdim)
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states_rope, key_states_rope = apply_rotary_pos_emb(query_states_rope, key_states_rope, cos, sin, position_ids)
        
        # Convert back to FlashAttention format
        query_states = query_states_rope.transpose(1, 2)  # (bsz, q_len, num_heads, head_dim)
        key_states = key_states_rope.transpose(1, 2)      # (bsz, q_len, num_kv_heads, head_dim)
        
        # Ensure tensors are in the correct dtype for FlashAttention
        flash_dtype = torch.bfloat16 if query_states.device.type == 'cuda' else torch.float16
        if query_states.dtype not in (torch.float16, torch.bfloat16):
            query_states = query_states.to(flash_dtype)
            key_states = key_states.to(flash_dtype)
            value_states = value_states.to(flash_dtype)
        
        # 🚀 FlashAttention-Only Implementation
        if self.training and not use_cache:
            # Training: Standard FlashAttention            
            attn_output = flash_attn_func(
                q=query_states,                    # (batch, seqlen, 24, 128)
                k=key_states,                      # (batch, seqlen, 8, 128) 
                v=value_states,                    # (batch, seqlen, 8, 128)
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=False,                      # 🎯 Non-causal for LLaDA!
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False
            )
            
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            # FlashAttention doesn't return attention weights
            return attn_output, None
        
        elif not use_cache:
            # Inference without cache: Standard FlashAttention            
            attn_output = flash_attn_func(
                q=query_states,
                k=key_states,
                v=value_states,
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=False,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=True
            )
            
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            # FlashAttention doesn't return attention weights
            return attn_output, None
        
        else:
            # Inference with cache: Use simple concatenation + FlashAttention            
            if past_key_value is not None:
                # past_key_value is in (batch, heads, seqlen, headdim) format
                past_k = past_key_value[0].transpose(1, 2)  # Convert to (batch, seqlen, heads, headdim)
                past_v = past_key_value[1].transpose(1, 2)
                
                # Concatenate past and current
                key_states = torch.cat([past_k, key_states], dim=1)
                value_states = torch.cat([past_v, value_states], dim=1)
            
            # Run FlashAttention on the full sequence
            attn_output = flash_attn_func(
                q=query_states,
                k=key_states,
                v=value_states,
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=False,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=True
            )
            
            # Update cache for next iteration
            new_past_key_value = (
                key_states.transpose(1, 2),    # Convert back to (batch, heads, seqlen, headdim)
                value_states.transpose(1, 2)
            )
            
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            # FlashAttention doesn't return attention weights
            return attn_output, None, new_past_key_value


class LLaDADecoderLayer(LlamaDecoderLayer):
    """Modified LLaMA decoder layer using LLaDA attention"""
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LLaDAAttention(config=config, layer_idx=layer_idx)


class LLaDAModel(LlamaModel):
    """LLaDA model with non-causal attention"""
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LLaDADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # Catch any legacy parameters like attention_mask, output_attentions
    ):
        # Call parent forward but remove unsupported parameters
        return super().forward(
            input_ids=input_ids,
            attention_mask=None,  # Not supported in FlashAttention-only mode
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,  # Not supported in FlashAttention-only mode
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class LLaDAForMaskedLM(nn.Module):
    """LLaDA model for masked language modeling"""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LLaDAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Mask token ID (as specified in the paper)
        self.mask_token_id = 126336
        
        # Initialize weights
        self.post_init()
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # Catch any legacy parameters like attention_mask, output_attentions
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # FlashAttention-only implementation - simplified parameters
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=None,  # Don't use past key values for training
            inputs_embeds=inputs_embeds,
            use_cache=False,  # Always False for training
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        return type('ModelOutput', (), {
            'logits': logits,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'attentions': None,  # FlashAttention doesn't return attention weights
        })()
    
    def post_init(self):
        """Initialize weights"""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """
        Resize input and output embeddings to new vocabulary size
        
        Args:
            new_num_tokens: New vocabulary size
            
        Returns:
            New embedding layer
        """
        old_embeddings = self.get_input_embeddings()
        old_num_tokens = old_embeddings.weight.size(0)
        
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        
        # Resize input embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.weight.size(1))
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        # Initialize new token embeddings
        if new_num_tokens > old_num_tokens:
            with torch.no_grad():
                # Initialize new embeddings with small random values
                new_embeddings.weight.data[old_num_tokens:, :].normal_(
                    mean=0.0, std=getattr(self.config, 'initializer_range', 0.02)
                )
        
        # Update model
        self.model.embed_tokens = new_embeddings
        
        # Resize output embeddings (lm_head)
        old_lm_head = self.get_output_embeddings()
        new_lm_head = nn.Linear(old_lm_head.in_features, new_num_tokens, bias=old_lm_head.bias is not None)
        new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        
        # Copy old weights
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        if old_lm_head.bias is not None:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
        
        # Initialize new output weights
        if new_num_tokens > old_num_tokens:
            with torch.no_grad():
                new_lm_head.weight.data[old_num_tokens:, :].normal_(
                    mean=0.0, std=getattr(self.config, 'initializer_range', 0.02)
                )
                if new_lm_head.bias is not None:
                    new_lm_head.bias.data[old_num_tokens:].zero_()
        
        # Update model
        self.lm_head = new_lm_head
        
        # Update config
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens
        
        return new_embeddings
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        else:
            # Fallback: set gradient_checkpointing flag
            self.model.gradient_checkpointing = True
            if hasattr(self.model, 'layers'):
                for layer in self.model.layers:
                    layer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        else:
            # Fallback: unset gradient_checkpointing flag
            self.model.gradient_checkpointing = False
            if hasattr(self.model, 'layers'):
                for layer in self.model.layers:
                    layer.gradient_checkpointing = False
    
    def get_input_embeddings(self):
        """Get input embeddings"""
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        """Set input embeddings"""
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        """Get output embeddings"""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings"""
        self.lm_head = new_embeddings


# Helper functions for RoPE (still needed for positional embeddings)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embedding"""
    # The cos and sin tensors in the original implementation were computed as:
    # cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    # sin = sin[position_ids].unsqueeze(1)
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Import necessary rotary embedding classes (simplified versions)
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)
    
    @torch.no_grad()
    def forward(self, x, seq_len=None):
        # Difference from the original RoPE: rescaling the position indices
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)
    
    @torch.no_grad()
    def forward(self, x, seq_len=None):
        # Difference from the original RoPE: rescaling the base
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
        else:
            inv_freq = self.inv_freq
        
        t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin 