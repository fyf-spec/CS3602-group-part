import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, apply_rotary_pos_emb
import copy

def scaled_dot_product_gqa(
    query,
    key,
    value,
    dropout=0.0,
    scale=None,
    mask=None,
    is_causal=False,
    head_mask=None
):
    """
    Scaled dot product attention with support for grouped queries.
    Adapted for GPTNeoX (Pythia) usage.
    """
    b, hq, n, d = query.shape
    b, hk, s, d = key.shape
    
    if scale is None:
        scale = d ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    
    # Reshape query to split heads into groups
    # query: (b, hq, n, d) -> (b, g, hk, n, d)
    # Assuming hq is ordered as (h1_g1, h1_g2, ..., h2_g1, ...)?
    # Standard GQA: hq = hk * g.  Query heads matching Key head k form a block.
    # So we view as (b, hk, g, n, d).
    # Then we verify if we need to swap hk and g dimensions for the math below.
    # Original logic using einops: rearrange(query, "b (h g) n d -> b g h n d")
    # This implies input is (h, g) flattened. Output is (g, h).
    # So we do: view(b, hk, g, n, d) -> permute(0, 2, 1, 3, 4)
    query = query.view(b, hk, num_head_groups, n, d).permute(0, 2, 1, 3, 4)
    
    # Reshape key to match dimensions
    # key: (b, hk, s, d) -> (b, 1, hk, s, d)
    key = key.unsqueeze(1)
    
    # Compute similarity --> (b, g, h, n, s)
    # query: (b, g, h, n, d)
    # key:   (b, 1, h, s, d)
    # Einsum broadcast over 'g' (dim 1) and '1' (dim 1)
    similarity = torch.einsum("b g h n d, b z h s d -> b g h n s", query, key)
    
    if is_causal:
        # Causal mask
        causal_mask = torch.ones((n, s), device=query.device, dtype=torch.bool).tril()
        similarity.masked_fill_(~causal_mask, torch.finfo(similarity.dtype).min)

    if mask is not None:
        # mask is usually (b, 1, 1, s) or similar from transformers
        if mask.ndim == 4:
             mask = mask.unsqueeze(1) # (b, 1, 1, 1, s)
        similarity = similarity + mask

    attention = F.softmax(similarity, dim=-1)
    
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)
        
    if head_mask is not None:
        attention = attention * head_mask
    
    # Value: (b, hk, s, d) -> (b, 1, hk, s, d)
    value = value.unsqueeze(1)
    
    # Out: (b, g, h, n, d)
    out = torch.einsum("b g h n s, b z h s d -> b g h n d", attention, value)
    
    # Recombine heads
    # original: rearrange(out, "b g h n d -> b (h g) n d")
    # We want to reverse the definition of hq.
    # previous: hq = (h g) flattened.
    # So we want (b, h, g, n, d) -> flatten to (b, h*g, n, d)
    # Current out is (b, g, h, n, d).
    # So permute(0, 2, 1, 3, 4) -> (b, h, g, n, d)
    # Then reshape.
    out = out.permute(0, 2, 1, 3, 4).reshape(b, hq, n, d)
    
    return out, attention

class GPTNeoXGQA(GPTNeoXAttention):
    """
    GPTNeoXAttention adapted for Grouped Query Attention.
    """
    def __init__(self, config, kv_heads=None):
        super().__init__(config)
        self.num_key_value_heads = kv_heads if kv_heads is not None else config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Override query_key_value to split Q, K, V
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        del self.query_key_value
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        b, n, _ = query.shape
        query = query.view(b, n, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(b, n, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(b, n, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            offset = past_key.shape[-2]
        else:
            offset = 0

        cos, sin = self.rotary_emb(value, seq_len=n + offset)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
            
        if use_cache:
            present = (key, value)
        else:
            present = None
            
        attn_output, attn_weights = scaled_dot_product_gqa(
            query,
            key,
            value,
            dropout=self.attention_dropout.p if self.training else 0.0,
            mask=attention_mask,
            is_causal=False,
            head_mask=head_mask
        )
        
        attn_output = self.dense(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs

def convert_gptneox_to_gqa(model, kv_heads=4):
    """
    Convert a GPTNeoX model to use Grouped Query Attention in-place.
    """
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    
    if num_heads % kv_heads != 0:
        raise ValueError(f"num_heads {num_heads} must be divisible by kv_heads {kv_heads}")
        
    group_size = num_heads // kv_heads
    
    for layer in model.gpt_neox.layers:
        original_attn = layer.attention
        
        gqa_attn = GPTNeoXGQA(config, kv_heads=kv_heads)
        
        qkv_weight = original_attn.query_key_value.weight.data
        qkv_weight = qkv_weight.view(num_heads, 3, head_dim, config.hidden_size)
        
        q_weight = qkv_weight[:, 0, :, :].reshape(num_heads * head_dim, config.hidden_size)
        k_weight = qkv_weight[:, 1, :, :].reshape(num_heads * head_dim, config.hidden_size)
        v_weight = qkv_weight[:, 2, :, :].reshape(num_heads * head_dim, config.hidden_size)
        
        gqa_attn.q_proj.weight.data = q_weight
        
        k_weight = k_weight.view(kv_heads, group_size, head_dim, config.hidden_size)
        v_weight = v_weight.view(kv_heads, group_size, head_dim, config.hidden_size)
        
        k_weight_avg = k_weight.mean(dim=1).reshape(kv_heads * head_dim, config.hidden_size)
        v_weight_avg = v_weight.mean(dim=1).reshape(kv_heads * head_dim, config.hidden_size)
        
        gqa_attn.k_proj.weight.data = k_weight_avg
        gqa_attn.v_proj.weight.data = v_weight_avg
        
        gqa_attn.dense.weight.data = original_attn.dense.weight.data
        if original_attn.dense.bias is not None:
             gqa_attn.dense.bias.data = original_attn.dense.bias.data
             
        layer.attention = gqa_attn.to(original_attn.dense.weight.device)
        
    return model
