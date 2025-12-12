from dataclasses import dataclass
import torch
from torch import nn
# from accelerated_inference.kvpress.base_press import BasePress
from kvpress.base_press import BasePress

@dataclass
class StreamLLMPress(BasePress):
    """
    StreamingLLM: Keep initial tokens (sinks) and recent window.
    """
    compression_ratio: float = 0.0
    num_sinks: int = 4

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        seq_len = keys.shape[2]
        n_kept = int(seq_len * (1 - self.compression_ratio))
        
        if n_kept >= seq_len:
            return keys, values
            
        if n_kept <= self.num_sinks:
             # Edge case: kept is smaller than sinks, just keep recent 
             return keys[:, :, -n_kept:], values[:, :, -n_kept:]
        
        # Keep sinks
        sinks_k = keys[:, :, :self.num_sinks]
        sinks_v = values[:, :, :self.num_sinks]
        
        # Keep recent
        window_size = n_kept - self.num_sinks
        recent_k = keys[:, :, -window_size:]
        recent_v = values[:, :, -window_size:]
        
        return torch.cat([sinks_k, recent_k], dim=2), torch.cat([sinks_v, recent_v], dim=2)

@dataclass
class SnapKVPress(BasePress):
    """
    SnapKV: Select important KV pairs based on attention scores from a 'window' of observation.
    Simplified implementation for benchmarking.
    """
    compression_ratio: float = 0.0
    window_size: int = 32 # Observation window size
    kernel_size: int = 5 
    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values
            
        seq_len = keys.shape[2]
        n_kept = int(seq_len * (1 - self.compression_ratio))
        
        # If we don't have attention scores (e.g. prefill), we can't prune effectively using SnapKV logic usually
        # But here 'attentions' might be passed?
        # In base_press, `output[1]` is passed as `attentions`.
        # If None, we cannot compress based on attention.
        
        if attentions is None:
            # Fallback to recent window (StreamingLLM style) if no attention scores
            return keys[:, :, -n_kept:], values[:, :, -n_kept:]
            
        # attentions shape: (bsz, num_heads, q_len, k_len)
        # We perform pruning based on the last few tokens' attention to the past
        
        # Take average attention over the observation window (last `window_size` queries)
        # We need to be careful with shapes.
        # If q_len is small (generation), we use it.
        
        # Sum attention over query dimension
        # attention_score: (bsz, num_heads, k_len)
        attention_score = attentions.sum(dim=-2) 
        
        # Select top-k
        indices = attention_score.topk(n_kept, dim=-1).indices
        indices = indices.sort(dim=-1).values # Sort to keep temporal order if needed/preferred
        
        # Gather
        # indices: (bsz, num_heads, n_kept)
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        
        keys = keys.gather(2, expanded_indices).contiguous()
        values = values.gather(2, expanded_indices).contiguous()
        
        return keys, values
