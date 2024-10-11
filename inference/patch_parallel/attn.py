from typing import Any, Dict, Optional, Tuple, Union

import torch

# from diffusers.models.attention import Attention
from diffusers.models.attention import Attention

from diffusers.utils import USE_PEFT_BACKEND
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from .base_module import BaseModule
from .utils import DistriConfig
from .logger import init_logger

logger = init_logger(__name__)

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.
    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class DistriAttentionPP(BaseModule):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriAttentionPP, self).__init__(module, distri_config)

        to_k = module.to_k
        to_v = module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape

        in_size, out_size = to_k.in_features, to_k.out_features
        to_kv = nn.Linear(
            in_size,
            out_size * 2,
            bias=to_k.bias is not None,
            device=to_k.weight.device,
            dtype=to_k.weight.dtype,
        )
        to_kv.weight.data[:out_size].copy_(to_k.weight.data)
        to_kv.weight.data[out_size:].copy_(to_v.weight.data)

        if to_k.bias is not None:
            assert to_v.bias is not None
            to_kv.bias.data[:out_size].copy_(to_k.bias.data)
            to_kv.bias.data[out_size:].copy_(to_v.bias.data)

        self.to_kv = to_kv


class DistriSelfAttentionPP(DistriAttentionPP):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriSelfAttentionPP, self).__init__(module, distri_config)

    def _forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, image_rotary_emb: Optional[torch.Tensor] = None):
        attn = self.module
        distri_config = self.distri_config
        assert isinstance(attn, Attention)

        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        kv = self.to_kv(hidden_states)        

        if distri_config.n_device_per_batch == 1:
            full_kv = kv
        else:
            if self.buffer_list is None:  # buffer not created
                print("buffer not created...")
                full_kv = torch.cat([kv for _ in range(distri_config.n_device_per_batch)], dim=1)
            elif distri_config.mode == "full_sync" or self.counter <= distri_config.warmup_steps:
                print("full_sync...")
                print("kv.shape:", kv.shape)
                print("self.counter:", self.counter)
                print("distri_config.warmup_steps", distri_config.warmup_steps)
                dist.all_gather(self.buffer_list, kv, group=distri_config.batch_group, async_op=False)
                full_kv = torch.cat(self.buffer_list, dim=1)
                print("full_kv.shape:", full_kv.shape)
            else:
                print("async...")
                new_buffer_list = [buffer for buffer in self.buffer_list]
                new_buffer_list[distri_config.split_idx()] = kv
                full_kv = torch.cat(new_buffer_list, dim=1)
                if distri_config.mode != "no_sync":
                    print("self.idx:", self.idx)
                    self.comm_manager.enqueue(self.idx, kv)

        key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        print("hidden_states.shape:", hidden_states.shape)
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        distri_config = self.distri_config

        if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
            print("waiting for comm...", self.idx)
            if self.comm_manager.handles[self.idx] is not None:
                self.comm_manager.handles[self.idx].wait()
                self.comm_manager.handles[self.idx] = None

        hidden_states_cat = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        b, l, c = hidden_states_cat.shape
        if distri_config.n_device_per_batch > 1 and self.buffer_list is None:
            if self.comm_manager.buffer_list is None:
                self.idx = self.comm_manager.register_tensor(
                    shape=(b, l, self.to_kv.out_features), torch_dtype=hidden_states.dtype, layer_type="attn"
                )
            else:
                self.buffer_list = self.comm_manager.get_buffer_list(self.idx)

        output = self._forward(hidden_states, encoder_hidden_states, image_rotary_emb)

        self.counter += 1
        return output
