"""VibeVoice-Realtime-0.5B MLX port.

Architecture:
  - Qwen2.5-0.5B split into 4 lower (text) + 20 upper (TTS) layers
  - 4-layer diffusion head (SwiGLU + AdaLN, no attention)
  - σ-VAE acoustic decoder (causal Conv1d with streaming cache)
  - DPM-Solver multistep scheduler (v-prediction, cosine beta)

Streaming pipeline:
  Text → 5-token windows → base LM → TTS LM → diffusion head (20 steps, CFG)
  → acoustic decoder → audio chunks at 24kHz (7.5Hz latent frame rate)
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Diffusion scheduler (DPM-Solver Multistep, cosine beta, v-prediction)
# ---------------------------------------------------------------------------

def cosine_betas(num_steps: int, max_beta: float = 0.999) -> mx.array:
    """Cosine beta schedule from 'Improved DDPM'."""
    steps = np.arange(num_steps + 1, dtype=np.float64) / num_steps
    alpha_bar = np.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0, max_beta)
    return mx.array(betas, dtype=mx.float32)


class DPMSolverScheduler:
    """Minimal DPM-Solver++ 2M for v-prediction."""

    def __init__(self, num_train_steps: int = 1000, num_inference_steps: int = 20):
        betas = np.array(cosine_betas(num_train_steps).tolist(), dtype=np.float64)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas)
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.timesteps = None
        self._prev_sample = None

    def set_timesteps(self, num_steps: int):
        self.num_inference_steps = num_steps
        step_ratio = self.num_train_steps / num_steps
        timesteps = np.round(np.arange(num_steps, 0, -1) * step_ratio - 1).astype(np.int64)
        self.timesteps = mx.array(timesteps, dtype=mx.int32)
        self._prev_sample = None

    def _alpha_sigma(self, t: int):
        acp = float(self.alphas_cumprod[t])
        alpha = math.sqrt(acp)
        sigma = math.sqrt(1 - acp)
        return alpha, sigma

    def step(self, model_output: mx.array, timestep: mx.array, sample: mx.array) -> mx.array:
        """Single DPM-Solver step with v-prediction → x0 conversion."""
        t = int(timestep.item())
        alpha_t, sigma_t = self._alpha_sigma(t)

        # v-prediction to x0
        x0_pred = alpha_t * sample - sigma_t * model_output

        # Simple first-order update (DDIM-like)
        tidx = None
        ts = self.timesteps.tolist()
        for i, tv in enumerate(ts):
            if tv == t:
                tidx = i
                break

        if tidx is not None and tidx + 1 < len(ts):
            t_next = ts[tidx + 1]
        else:
            t_next = 0

        alpha_next, sigma_next = self._alpha_sigma(t_next)
        prev_sample = alpha_next * x0_pred + sigma_next * ((sample - alpha_t * x0_pred) / sigma_t)
        return prev_sample


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,)) if affine else None

    def __call__(self, x: mx.array) -> mx.array:
        norm = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        if self.weight is not None:
            norm = norm * self.weight
        return norm


# ---------------------------------------------------------------------------
# Diffusion head
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.linear1 = nn.Linear(freq_dim, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, t: mx.array) -> mx.array:
        half = self.freq_dim // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return self.linear2(nn.silu(self.linear1(emb)))


class DiffusionFFN(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class HeadLayer(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.ffn = DiffusionFFN(dim, ffn_dim)
        self.norm = RMSNorm(dim, eps=eps)
        self.adaLN_modulation = nn.Linear(cond_dim, 3 * dim, bias=False)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = nn.silu(c)
        mod = self.adaLN_modulation(mod)
        shift, scale, gate = mx.split(mod, 3, axis=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.ffn(h)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, cond_size: int, eps: float = 1e-5):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=eps, affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Linear(cond_size, 2 * hidden_size, bias=False)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = nn.silu(c)
        mod = self.adaLN_modulation(mod)
        shift, scale = mx.split(mod, 2, axis=-1)
        x = self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)


class DiffusionHead(nn.Module):
    def __init__(self, hidden_size: int = 896, latent_size: int = 64,
                 head_layers: int = 4, ffn_ratio: float = 3.0, eps: float = 1e-5):
        super().__init__()
        self.noisy_images_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        ffn_dim = int(hidden_size * ffn_ratio)
        self.layers = [HeadLayer(hidden_size, ffn_dim, hidden_size, eps) for _ in range(head_layers)]
        self.final_layer = FinalLayer(hidden_size, latent_size, hidden_size, eps)

    def __call__(self, noisy: mx.array, timesteps: mx.array, condition: mx.array) -> mx.array:
        x = self.noisy_images_proj(noisy)
        t = self.t_embedder(timesteps)
        c = self.cond_proj(condition) + t
        for layer in self.layers:
            x = layer(x, c)
        return self.final_layer(x, c)


# ---------------------------------------------------------------------------
# Acoustic decoder (σ-VAE, causal Conv1d with streaming cache)
# ---------------------------------------------------------------------------

class StreamingCache:
    """Cache for causal Conv1d streaming, keyed by layer id."""
    def __init__(self):
        self.cache = {}

    def get(self, layer_id: str) -> Optional[mx.array]:
        return self.cache.get(layer_id)

    def set(self, layer_id: str, state: mx.array):
        self.cache[layer_id] = state


class CausalConv1d(nn.Module):
    """Causal 1D convolution with streaming cache support."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        # MLX conv1d: weight shape is (out_ch, kernel_size, in_ch // groups)
        self.weight = mx.zeros((out_ch, kernel_size, in_ch // groups))
        self.bias = mx.zeros((out_ch,)) if bias else None
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"conv1d_{id(self)}"
        return self._layer_id

    def __call__(self, x: mx.array, cache: Optional[StreamingCache] = None) -> mx.array:
        """x: (B, C, T) — channels-first like PyTorch Conv1d."""
        if cache is not None:
            return self._forward_streaming(x, cache)
        return self._forward_padded(x)

    def _forward_padded(self, x: mx.array) -> mx.array:
        # Causal left-pad
        pad_total = self.context_size
        if pad_total > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (pad_total, 0)])
        # MLX conv1d expects (B, T, C)
        x = mx.transpose(x, (0, 2, 1))
        out = mx.conv1d(x, self.weight, stride=self.stride, dilation=self.dilation, groups=self.groups)
        out = mx.transpose(out, (0, 2, 1))
        if self.bias is not None:
            out = out + self.bias[:, None]
        return out

    def _forward_streaming(self, x: mx.array, cache: StreamingCache) -> mx.array:
        cached = cache.get(self.layer_id)
        if cached is None and self.context_size > 0:
            cached = mx.zeros((x.shape[0], self.in_ch, self.context_size))

        if cached is not None and cached.shape[2] > 0:
            full = mx.concatenate([cached, x], axis=2)
        else:
            full = x

        # Conv without extra padding — context from cache
        xt = mx.transpose(full, (0, 2, 1))
        out = mx.conv1d(xt, self.weight, stride=self.stride, dilation=self.dilation, groups=self.groups)
        out = mx.transpose(out, (0, 2, 1))
        if self.bias is not None:
            out = out + self.bias[:, None]

        # Update cache
        if self.context_size > 0:
            total_len = full.shape[2]
            if total_len >= self.context_size:
                new_cache = full[:, :, total_len - self.context_size:]
            else:
                new_cache = full
            cache.set(self.layer_id, new_cache)

        return out


class CausalConvTranspose1d(nn.Module):
    """Causal transposed 1D convolution with streaming cache."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, bias: bool = True):
        super().__init__()
        # MLX conv_transpose1d weight: (in_ch, kernel_size, out_ch)
        # but we store as (out_ch, kernel_size, in_ch) and transpose at call time
        self.weight = mx.zeros((out_ch, kernel_size, in_ch))
        self.bias = mx.zeros((out_ch,)) if bias else None
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.padding_total = kernel_size - stride
        self.context_size = kernel_size - 1
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"convtr1d_{id(self)}"
        return self._layer_id

    def _apply_convtr(self, x: mx.array) -> mx.array:
        # MLX conv_transpose expects (B, T, C) and weight (out_ch, k, in_ch)
        # Actually: mlx.core.conv_transpose1d(input, weight, stride, padding, dilation, groups)
        # input: (B, T, C_in), weight: (C_out, K, C_in)
        xt = mx.transpose(x, (0, 2, 1))  # (B, T, C_in)
        # Weight needs to be (C_out, K, C_in) — but our stored shape matches
        # MLX conv_transpose1d weight: (C_out, K_w, C_in)
        out = mx.conv_transpose1d(xt, self.weight, stride=self.stride)
        out = mx.transpose(out, (0, 2, 1))  # (B, C_out, T_out)
        if self.bias is not None:
            out = out + self.bias[:, None]
        return out

    def __call__(self, x: mx.array, cache: Optional[StreamingCache] = None) -> mx.array:
        if cache is not None:
            return self._forward_streaming(x, cache)
        return self._forward_padded(x)

    def _forward_padded(self, x: mx.array) -> mx.array:
        y = self._apply_convtr(x)
        # Causal: remove right padding
        pad_right = math.ceil(self.padding_total)
        pad_left = self.padding_total - pad_right
        if pad_left + pad_right > 0:
            end = y.shape[2] - pad_right if pad_right > 0 else y.shape[2]
            y = y[:, :, pad_left:end]
        return y

    def _forward_streaming(self, x: mx.array, cache: StreamingCache) -> mx.array:
        cached_input = cache.get(self.layer_id)
        if cached_input is None:
            cached_input = mx.zeros((x.shape[0], self.in_ch, 0))

        full_input = mx.concatenate([cached_input, x], axis=2) if cached_input.shape[2] > 0 else x
        full_output = self._apply_convtr(full_input)

        # Remove padding
        pad_right = math.ceil(self.padding_total)
        pad_left = self.padding_total - pad_right
        if pad_left + pad_right > 0:
            end = full_output.shape[2] - pad_right if pad_right > 0 else full_output.shape[2]
            full_output = full_output[:, :, pad_left:end]

        # Return only new output
        if cached_input.shape[2] == 0:
            output = full_output
        else:
            expected_new = x.shape[2] * self.stride
            if full_output.shape[2] >= expected_new:
                output = full_output[:, :, -expected_new:]
            else:
                output = full_output

        # Update cache
        if full_input.shape[2] > self.context_size:
            new_cache = full_input[:, :, -self.context_size:]
        else:
            new_cache = full_input
        cache.set(self.layer_id, new_cache)

        return output


class ConvRMSNorm(nn.Module):
    """RMSNorm for conv features (channels-first → channels-last → norm → back)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) → (B, T, C)
        xt = mx.transpose(x, (0, 2, 1))
        norm = xt * mx.rsqrt(mx.mean(xt * xt, axis=-1, keepdims=True) + self.eps)
        norm = norm * self.weight
        return mx.transpose(norm, (0, 2, 1))


class ConvFFN(nn.Module):
    """FFN operating on (B, C, T) via transpose to (B, T, C)."""
    def __init__(self, dim: int, ffn_dim: int, bias: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(dim, ffn_dim, bias=bias)
        self.linear2 = nn.Linear(ffn_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) → (B, T, C)
        xt = mx.transpose(x, (0, 2, 1))
        h = nn.gelu(self.linear1(xt))
        h = self.linear2(h)
        return mx.transpose(h, (0, 2, 1))


class Block1D(nn.Module):
    """Residual block: depthwise conv mixer + FFN."""
    def __init__(self, dim: int, kernel_size: int = 7, groups: int = None,
                 ffn_expansion: int = 4, bias: bool = True, eps: float = 1e-5,
                 layer_scale: float = 1e-6):
        super().__init__()
        groups = groups or dim  # depthwise by default
        self.norm = ConvRMSNorm(dim, eps=eps)
        self.mixer = CausalConv1d(dim, dim, kernel_size, groups=groups, bias=bias)
        self.ffn_norm = ConvRMSNorm(dim, eps=eps)
        self.ffn = ConvFFN(dim, ffn_expansion * dim, bias=bias)
        self.gamma = mx.full((dim,), layer_scale) if layer_scale > 0 else None
        self.ffn_gamma = mx.full((dim,), layer_scale) if layer_scale > 0 else None

    def __call__(self, x: mx.array, cache: Optional[StreamingCache] = None) -> mx.array:
        # Mixer
        residual = x
        h = self.norm(x)
        h = self.mixer(h, cache=cache)
        if self.gamma is not None:
            h = h * self.gamma[:, None]
        x = residual + h

        # FFN
        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        if self.ffn_gamma is not None:
            h = h * self.ffn_gamma[:, None]
        x = residual + h
        return x


class AcousticDecoder(nn.Module):
    """σ-VAE decoder: latent → audio waveform.

    Weight layout matches HF: upsample_layers[0] = stem Conv1d,
    upsample_layers[1..N] = ConvTranspose1d upsamples.
    """
    def __init__(self, vae_dim: int = 64, n_filters: int = 32,
                 ratios: list = None, depths: list = None,
                 kernel_size: int = 7, bias: bool = True, eps: float = 1e-5,
                 layer_scale: float = 1e-6):
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 5, 4, 2, 2]
        if depths is None:
            depths = [8, 3, 3, 3, 3, 3, 3]  # reversed encoder depths

        self.ratios = ratios
        self.depths = depths
        n_stages = len(depths)

        # upsample_layers[0] = stem (Conv1d), [1..6] = ConvTranspose1d
        stem_out = n_filters * 2 ** (n_stages - 1)
        self.upsample_layers = [CausalConv1d(vae_dim, stem_out, kernel_size, bias=bias)]
        for i in range(len(ratios)):
            in_ch = n_filters * 2 ** (n_stages - 1 - i)
            out_ch = n_filters * 2 ** (n_stages - 1 - i - 1)
            self.upsample_layers.append(
                CausalConvTranspose1d(in_ch, out_ch, kernel_size=ratios[i] * 2, stride=ratios[i], bias=bias)
            )

        self.stages = []
        for i in range(n_stages):
            ch = n_filters * 2 ** (n_stages - 1 - i)
            stage = [Block1D(ch, kernel_size=kernel_size, bias=bias, eps=eps,
                             layer_scale=layer_scale) for _ in range(depths[i])]
            self.stages.append(stage)

        # Head
        final_ch = n_filters
        self.head = CausalConv1d(final_ch, 1, kernel_size=kernel_size, bias=bias)

    def __call__(self, x: mx.array, cache: Optional[StreamingCache] = None) -> mx.array:
        """x: (B, vae_dim, T) or (B, T, vae_dim) → (B, 1, T_audio)"""
        if x.shape[1] != 64 and x.shape[2] == 64:
            x = mx.transpose(x, (0, 2, 1))

        # upsample_layers[0] is the stem
        x = self.upsample_layers[0](x, cache=cache)
        for block in self.stages[0]:
            x = block(x, cache=cache)

        # upsample_layers[1..N] + stages[1..N]
        for i in range(len(self.ratios)):
            x = self.upsample_layers[i + 1](x, cache=cache)
            for block in self.stages[i + 1]:
                x = block(x, cache=cache)

        x = self.head(x, cache=cache)
        return x


# ---------------------------------------------------------------------------
# Qwen2 LM (minimal, for the 4+20 layer split)
# ---------------------------------------------------------------------------

class Qwen2Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, rope_theta: float = 1e6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rope = nn.RoPE(self.head_dim, base=rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 cache=None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Layer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 intermediate_size: int, rms_eps: float = 1e-6, rope_theta: float = 1e6):
        super().__init__()
        self.self_attn = Qwen2Attention(hidden_size, num_heads, num_kv_heads, rope_theta)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache=None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen2Model(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int,
                 num_kv_heads: int, intermediate_size: int, vocab_size: int,
                 rms_eps: float = 1e-6, rope_theta: float = 1e6,
                 has_norm: bool = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen2Layer(hidden_size, num_heads, num_kv_heads, intermediate_size, rms_eps, rope_theta)
            for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=rms_eps) if has_norm else None

    def __call__(self, input_ids: Optional[mx.array] = None,
                 inputs_embeds: Optional[mx.array] = None,
                 mask: Optional[mx.array] = None,
                 cache=None) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, cache=cache[i])

        if self.norm is not None:
            h = self.norm(h)
        return h


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, k: mx.array, v: mx.array):
        if self.keys is None:
            self.keys = k
            self.values = v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        self.offset = self.keys.shape[2]
        return self.keys, self.values


# ---------------------------------------------------------------------------
# Full VibeVoice model
# ---------------------------------------------------------------------------

class VibeVoiceMLX(nn.Module):
    """Complete VibeVoice-Realtime-0.5B in MLX."""

    def __init__(self, config: dict):
        super().__init__()
        dec = config["decoder_config"]
        diff = config["diffusion_head_config"]
        ac = config["acoustic_tokenizer_config"]
        tts_layers = config.get("tts_backbone_num_hidden_layers", 20)
        base_layers = dec["num_hidden_layers"] - tts_layers

        # Base LM (lower 4 layers)
        self.language_model = Qwen2Model(
            hidden_size=dec["hidden_size"],
            num_layers=base_layers,
            num_heads=dec["num_attention_heads"],
            num_kv_heads=dec["num_key_value_heads"],
            intermediate_size=dec["intermediate_size"],
            vocab_size=dec["vocab_size"],
            rms_eps=dec["rms_norm_eps"],
            rope_theta=dec["rope_theta"],
        )

        # TTS LM (upper 20 layers, has final norm)
        self.tts_language_model = Qwen2Model(
            hidden_size=dec["hidden_size"],
            num_layers=tts_layers,
            num_heads=dec["num_attention_heads"],
            num_kv_heads=dec["num_key_value_heads"],
            intermediate_size=dec["intermediate_size"],
            vocab_size=dec["vocab_size"],
            rms_eps=dec["rms_norm_eps"],
            rope_theta=dec["rope_theta"],
            has_norm=True,
        )

        # Type embeddings (text=1, speech=0)
        self.tts_input_types = nn.Embedding(2, dec["hidden_size"])

        # Speech components
        self.acoustic_connector_fc1 = nn.Linear(ac["vae_dim"], dec["hidden_size"])
        self.acoustic_connector_norm = nn.RMSNorm(dec["hidden_size"], eps=1e-6)
        self.acoustic_connector_fc2 = nn.Linear(dec["hidden_size"], dec["hidden_size"])

        self.prediction_head = DiffusionHead(
            hidden_size=diff["hidden_size"],
            latent_size=diff["latent_size"],
            head_layers=diff["head_layers"],
            ffn_ratio=diff["head_ffn_ratio"],
            eps=diff["rms_norm_eps"],
        )

        # Acoustic decoder
        encoder_depths_str = ac.get("encoder_depths", "3-3-3-3-3-3-8")
        if isinstance(encoder_depths_str, str):
            encoder_depths = [int(d) for d in encoder_depths_str.split('-')]
        else:
            encoder_depths = encoder_depths_str
        decoder_depths = list(reversed(encoder_depths))

        self.acoustic_decoder = AcousticDecoder(
            vae_dim=ac["vae_dim"],
            n_filters=ac["decoder_n_filters"],
            ratios=ac["decoder_ratios"],
            depths=decoder_depths,
            bias=ac.get("conv_bias", True),
            eps=ac.get("layernorm_eps", 1e-5),
            layer_scale=ac.get("layer_scale_init_value", 1e-6),
        )

        # Scaling factors
        self.speech_scaling_factor = mx.array(1.0)
        self.speech_bias_factor = mx.array(0.0)

        # EOS classifier
        self.tts_eos_fc1 = nn.Linear(dec["hidden_size"], dec["hidden_size"])
        self.tts_eos_fc2 = nn.Linear(dec["hidden_size"], 1)

        # Scheduler
        self.scheduler = DPMSolverScheduler(
            num_train_steps=diff["ddpm_num_steps"],
            num_inference_steps=diff["ddpm_num_inference_steps"],
        )

        self.config = config

    def acoustic_connector(self, x: mx.array) -> mx.array:
        h = self.acoustic_connector_fc1(x)
        h = self.acoustic_connector_norm(h)
        return self.acoustic_connector_fc2(h)

    def eos_classifier(self, x: mx.array) -> mx.array:
        return self.tts_eos_fc2(nn.relu(self.tts_eos_fc1(x)))

    def sample_speech_tokens(self, condition: mx.array, neg_condition: mx.array,
                             cfg_scale: float = 3.0, num_steps: int = 20) -> mx.array:
        """Diffusion sampling with classifier-free guidance."""
        self.scheduler.set_timesteps(num_steps)
        cond = mx.concatenate([condition, neg_condition], axis=0)
        vae_dim = self.config["acoustic_vae_dim"]
        speech = mx.random.normal((cond.shape[0], vae_dim))

        for t in self.scheduler.timesteps:
            half = speech[:speech.shape[0] // 2]
            combined = mx.concatenate([half, half], axis=0)
            t_batch = mx.broadcast_to(t, (combined.shape[0],))
            eps = self.prediction_head(combined, t_batch, condition=cond)
            cond_eps, uncond_eps = mx.split(eps, 2, axis=0)
            guided = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = mx.concatenate([guided, guided], axis=0)
            speech = self.scheduler.step(eps, t, speech)
            mx.eval(speech)

        return speech[:speech.shape[0] // 2]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _map_weights(raw: dict) -> dict:
    """Map HF weight keys to our MLX model structure."""
    mapped = {}

    for key, val in raw.items():
        new_key = key

        # Strip 'model.' prefix for internal components
        if new_key.startswith("model."):
            new_key = new_key[6:]

        # Acoustic connector
        if new_key.startswith("acoustic_connector."):
            new_key = new_key.replace("acoustic_connector.fc1", "acoustic_connector_fc1")
            new_key = new_key.replace("acoustic_connector.norm", "acoustic_connector_norm")
            new_key = new_key.replace("acoustic_connector.fc2", "acoustic_connector_fc2")

        # EOS classifier (these don't have model. prefix)
        if key.startswith("tts_eos_classifier."):
            new_key = key.replace("tts_eos_classifier.fc1", "tts_eos_fc1")
            new_key = new_key.replace("tts_eos_classifier.fc2", "tts_eos_fc2")

        # Acoustic tokenizer → acoustic_decoder
        if new_key.startswith("acoustic_tokenizer.decoder."):
            new_key = new_key.replace("acoustic_tokenizer.decoder.", "acoustic_decoder.")
            # upsample_layers.N.0.conv.conv → upsample_layers.N (stem, Conv1d)
            # upsample_layers.N.0.convtr.convtr → upsample_layers.N (ConvTranspose1d)
            new_key = new_key.replace(".0.convtr.convtr.", ".")
            new_key = new_key.replace(".0.conv.conv.", ".")
            # stages.N.M.mixer.conv.conv.conv → stages.N.M.mixer
            new_key = new_key.replace(".mixer.conv.conv.conv.", ".mixer.")
            # head.conv.conv → head
            if "head.conv.conv." in new_key:
                new_key = new_key.replace("head.conv.conv.", "head.")

        # Conv weight transposition for acoustic decoder
        if "acoustic_decoder" in new_key and new_key.endswith(".weight") and val.ndim == 3:
            # Detect Conv1d vs ConvTranspose1d by checking if this is an upsample layer > 0
            is_convtranspose = ("upsample_layers." in new_key and
                                not new_key.startswith("acoustic_decoder.upsample_layers.0."))

            if is_convtranspose:
                # PyTorch ConvTranspose1d: (in_ch, out_ch, kernel_size)
                # MLX conv_transpose1d weight: (out_ch, kernel_size, in_ch)
                val = mx.transpose(val, (1, 2, 0))
            else:
                # PyTorch Conv1d: (out_ch, in_ch_per_group, kernel_size)
                # MLX conv1d weight: (out_ch, kernel_size, in_ch_per_group)
                val = mx.transpose(val, (0, 2, 1))

        # Timestep embedder: sequential indices → named
        new_key = new_key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        new_key = new_key.replace("t_embedder.mlp.2.", "t_embedder.linear2.")

        # adaLN_modulation sequential: .1. → just the linear
        new_key = new_key.replace("adaLN_modulation.1.", "adaLN_modulation.")

        # Diffusion head FFN
        new_key = new_key.replace("final_layer.norm_final.", "final_layer.norm_final.")

        mapped[new_key] = val

    return mapped


def load_vibevoice(model_id: str = "microsoft/VibeVoice-Realtime-0.5B"):
    """Load VibeVoice-Realtime-0.5B weights into MLX model."""
    from huggingface_hub import hf_hub_download
    import json

    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    weights_path = hf_hub_download(model_id, "model.safetensors")
    raw_weights = mx.load(weights_path)

    model = VibeVoiceMLX(config)
    mapped = _map_weights(raw_weights)

    # Load weights
    model.load_weights(list(mapped.items()))
    mx.eval(model.parameters())

    return model, config


# ---------------------------------------------------------------------------
# Voice prompt conversion + generation
# ---------------------------------------------------------------------------

def convert_voice_prompt(pt_path: str) -> dict:
    """Convert a PyTorch .pt voice prompt to MLX arrays.

    Returns dict with 'lm', 'tts_lm', 'neg_lm', 'neg_tts_lm' each containing
    'last_hidden_state' as mx.array and 'kv_cache' as list of (K, V) mx.array pairs.
    """
    import torch as _torch

    data = _torch.load(pt_path, map_location="cpu", weights_only=False)
    result = {}

    for key in ("lm", "tts_lm", "neg_lm", "neg_tts_lm"):
        entry = data[key]
        hs = mx.array(entry["last_hidden_state"].float().numpy())

        kv_cache = []
        if "past_key_values" in entry and entry["past_key_values"] is not None:
            pkv = entry["past_key_values"]
            # DynamicCache or list of (K, V) tuples
            if hasattr(pkv, "key_cache"):
                for k, v in zip(pkv.key_cache, pkv.value_cache):
                    kv_cache.append((
                        mx.array(k.float().numpy()),
                        mx.array(v.float().numpy()),
                    ))
            else:
                for layer_kv in pkv:
                    k, v = layer_kv[0], layer_kv[1]
                    kv_cache.append((
                        mx.array(k.float().numpy()),
                        mx.array(v.float().numpy()),
                    ))

        result[key] = {"last_hidden_state": hs, "kv_cache": kv_cache}

    return result


TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6


def generate(
    model: VibeVoiceMLX,
    text: str,
    voice_prompt: dict,
    cfg_scale: float = 1.5,
    num_diffusion_steps: int = 5,
    max_speech_tokens: int = 2000,
    callback=None,
) -> mx.array:
    """Generate speech from text using VibeVoice MLX.

    Args:
        model: Loaded VibeVoiceMLX model.
        text: Input text to speak.
        voice_prompt: Converted voice prompt from convert_voice_prompt().
        cfg_scale: Classifier-free guidance scale.
        num_diffusion_steps: Number of diffusion denoising steps.
        max_speech_tokens: Maximum number of speech latent tokens to generate.
        callback: Optional fn(audio_chunk: mx.array) called per speech token.

    Returns:
        mx.array: Generated audio waveform (1D, 24kHz).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    text_tokens = tokenizer.encode(text.strip() + "\n", add_special_tokens=False)
    tts_text_ids = mx.array([text_tokens])

    # Restore KV caches from voice prompt
    def _restore_cache(kv_list):
        caches = []
        for k, v in kv_list:
            c = KVCache()
            c.keys = k
            c.values = v
            c.offset = k.shape[2]
            caches.append(c)
        return caches

    lm_cache = _restore_cache(voice_prompt["lm"]["kv_cache"])
    tts_lm_cache = _restore_cache(voice_prompt["tts_lm"]["kv_cache"])
    neg_lm_cache = _restore_cache(voice_prompt["neg_lm"]["kv_cache"])
    neg_tts_lm_cache = _restore_cache(voice_prompt["neg_tts_lm"]["kv_cache"])

    # Last hidden states from prefill
    lm_hidden = voice_prompt["lm"]["last_hidden_state"]
    tts_lm_hidden = voice_prompt["tts_lm"]["last_hidden_state"]
    neg_tts_lm_hidden = voice_prompt["neg_tts_lm"]["last_hidden_state"]

    audio_chunks = []
    acoustic_cache = StreamingCache()
    text_window_idx = 0
    total_speech_tokens = 0

    while total_speech_tokens < max_speech_tokens:
        # Get next text window
        start = text_window_idx * TTS_TEXT_WINDOW_SIZE
        end = start + TTS_TEXT_WINDOW_SIZE
        cur_text = tts_text_ids[:, start:end]
        text_window_idx += 1

        if cur_text.shape[1] > 0:
            # Forward through base LM
            cur_embeds = model.language_model.embed_tokens(cur_text)
            lm_hidden = model.language_model(inputs_embeds=cur_embeds, cache=lm_cache)
            mx.eval(lm_hidden)

            # Forward through TTS LM with LM hidden states spliced in
            tts_embeds = model.tts_language_model.embed_tokens(cur_text)
            splice_start = tts_embeds.shape[1] - lm_hidden.shape[1]
            tts_embeds = mx.concatenate([
                tts_embeds[:, :splice_start],
                lm_hidden
            ], axis=1) if splice_start > 0 else lm_hidden
            # Add type embedding (text=1)
            type_embed = model.tts_input_types(mx.ones(tts_embeds.shape[:2], dtype=mx.int32))
            tts_embeds = tts_embeds + type_embed

            tts_lm_hidden = model.tts_language_model(inputs_embeds=tts_embeds, cache=tts_lm_cache)
            mx.eval(tts_lm_hidden)

        # Generate speech tokens
        finished = False
        for speech_idx in range(TTS_SPEECH_WINDOW_SIZE):
            pos_cond = tts_lm_hidden[:, -1:, :].reshape(1, -1)
            neg_cond = neg_tts_lm_hidden[:, -1:, :].reshape(1, -1)

            speech_latent = model.sample_speech_tokens(
                pos_cond, neg_cond, cfg_scale=cfg_scale, num_steps=num_diffusion_steps
            )

            # Decode to audio
            scaled = speech_latent.reshape(1, 1, -1) / model.speech_scaling_factor - model.speech_bias_factor
            # acoustic decoder expects (B, C, T) where C=vae_dim
            scaled_for_decode = mx.transpose(scaled, (0, 2, 1))  # (1, 64, 1)
            audio_chunk = model.acoustic_decoder(scaled_for_decode, cache=acoustic_cache)
            mx.eval(audio_chunk)

            audio_chunks.append(audio_chunk.reshape(-1))
            total_speech_tokens += 1

            if callback:
                callback(audio_chunk.reshape(-1))

            # Feed speech embedding back into TTS LM
            acoustic_embed = model.acoustic_connector(speech_latent.reshape(1, 1, -1))
            type_embed_speech = model.tts_input_types(mx.zeros((1, 1), dtype=mx.int32))
            tts_input = acoustic_embed + type_embed_speech

            tts_lm_hidden = model.tts_language_model(inputs_embeds=tts_input, cache=tts_lm_cache)

            # Also feed through negative TTS LM
            neg_tts_lm_hidden = model.tts_language_model(inputs_embeds=tts_input, cache=neg_tts_lm_cache)
            mx.eval(tts_lm_hidden, neg_tts_lm_hidden)

            # Check EOS
            eos_logit = model.eos_classifier(tts_lm_hidden[:, -1, :])
            if mx.sigmoid(eos_logit).item() > 0.5:
                finished = True
                break

            if total_speech_tokens >= max_speech_tokens:
                break

        if finished:
            break

        # Check if we've consumed all text
        if text_window_idx * TTS_TEXT_WINDOW_SIZE >= tts_text_ids.shape[1] and cur_text.shape[1] == 0:
            # No more text, continue generating speech until EOS
            pass

    if audio_chunks:
        return mx.concatenate(audio_chunks)
    return mx.array([])
