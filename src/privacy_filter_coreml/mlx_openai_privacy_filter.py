from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, scaled_dot_product_attention
from mlx_lm.models.gpt_oss import SwiGLU, mlx_topk
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "openai_privacy_filter"
    vocab_size: int = 200064
    hidden_size: int = 640
    intermediate_size: int = 640
    num_hidden_layers: int = 8
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    head_dim: int = 64
    sliding_window: int = 128
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    pad_token_id: int = 199999
    id2label: dict[str, str] | None = None
    label2id: dict[str, int] | None = None
    rope_parameters: dict[str, Any] | None = None
    max_position_embeddings: int = 131072


def _linear(linear: nn.Linear, x: mx.array) -> mx.array:
    return linear(x)


class AttentionBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.sliding_window = config.sliding_window
        self.sinks = mx.zeros((config.num_attention_heads,))

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rope = PrivacyRotaryEmbedding(config)
        self.sm_scale = 1 / math.sqrt(config.head_dim)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        batch, length, _ = x.shape
        query = _linear(self.q_proj, x).reshape(batch, length, -1, self.head_dim).swapaxes(1, 2)
        key = _linear(self.k_proj, x).reshape(batch, length, -1, self.head_dim).swapaxes(1, 2)
        value = _linear(self.v_proj, x).reshape(batch, length, -1, self.head_dim).swapaxes(1, 2)

        query = self.rope(query)
        key = self.rope(key)
        if mask is not None:
            mask = mask.astype(query.dtype)
        out = scaled_dot_product_attention(query, key, value, None, self.sm_scale, mask, sinks=self.sinks)
        out = out.swapaxes(1, 2).reshape(batch, length, -1)
        return self.o_proj(out)


class PrivacyRotaryEmbedding(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        rope = config.rope_parameters or {"rope_type": "default", "rope_theta": 150000.0}
        self.head_dim = config.head_dim
        self.attention_scaling = 1.0
        if rope.get("rope_type", "default") == "yarn":
            inv_freq, attention_scaling = _compute_yarn_parameters(config, rope)
            self.inv_freq = inv_freq
            self.attention_scaling = attention_scaling
        else:
            base = rope.get("rope_theta", 150000.0)
            self.inv_freq = 1.0 / (
                base ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
            )

    def __call__(self, x: mx.array) -> mx.array:
        length = x.shape[-2]
        positions = mx.arange(length, dtype=mx.float32)
        freqs = mx.expand_dims(positions, 1) * mx.expand_dims(self.inv_freq, 0)
        cos = (mx.cos(freqs) * self.attention_scaling).astype(x.dtype)
        sin = (mx.sin(freqs) * self.attention_scaling).astype(x.dtype)
        cos = mx.expand_dims(mx.expand_dims(cos, 0), 0)
        sin = mx.expand_dims(mx.expand_dims(sin, 0), 0)

        first = x[..., ::2]
        second = x[..., 1::2]
        rotated_first = first * cos - second * sin
        rotated_second = second * cos + first * sin
        return mx.stack([rotated_first, rotated_second], axis=-1).reshape(*x.shape)


def _compute_yarn_parameters(config: ModelArgs, rope: dict[str, Any]) -> tuple[mx.array, float]:
    base = rope["rope_theta"]
    dim = config.head_dim
    factor = rope["factor"]
    original_max_position_embeddings = rope["original_max_position_embeddings"]
    beta_fast = rope.get("beta_fast") or 32
    beta_slow = rope.get("beta_slow") or 1

    attention_factor = rope.get("attention_factor")
    if attention_factor is None:
        mscale = rope.get("mscale")
        mscale_all_dim = rope.get("mscale_all_dim")
        if mscale and mscale_all_dim:
            attention_factor = _get_yarn_mscale(factor, mscale) / _get_yarn_mscale(
                factor,
                mscale_all_dim,
            )
        else:
            attention_factor = _get_yarn_mscale(factor)

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if rope.get("truncate", True):
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0)
    high = min(high, dim - 1)

    pos_freqs = base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    ramp = (mx.arange(dim // 2, dtype=mx.float32) - low) / (high - low if high != low else 0.001)
    ramp = mx.clip(ramp, 0, 1)
    extrapolation_factor = 1 - ramp
    inv_freq = inv_freq_interpolation * (1 - extrapolation_factor) + (
        inv_freq_extrapolation * extrapolation_factor
    )
    return inv_freq, float(attention_factor)


def _get_yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MLPBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
            num_experts=config.num_local_experts,
            activation=SwiGLU(),
            bias=True,
        )
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        router_logits = self.router(x.astype(mx.float32))
        expert_logits, indices = mlx_topk(router_logits, k=self.num_experts_per_tok, axis=-1)
        expert_weights = mx.softmax(expert_logits, axis=-1, precise=True) / self.num_experts_per_tok
        out = self.experts(x, indices)
        out = out * mx.expand_dims(expert_weights, axis=-1)
        return out.sum(axis=-2) * self.num_experts_per_tok


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = AttentionBlock(config)
        self.mlp = MLPBlock(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, mask)

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x


class OpenAIPrivacyFilterModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [EncoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.sliding_window = args.sliding_window

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        x = self.embed_tokens(input_ids)
        mask = make_bidirectional_sliding_window_mask(input_ids, attention_mask, self.sliding_window)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.config = args
        label_count = len(args.id2label or {}) or 33
        self.model = OpenAIPrivacyFilterModel(args)
        self.score = nn.Linear(args.hidden_size, label_count, bias=True)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None):
        hidden = self.model(input_ids, attention_mask=attention_mask)
        return SimpleNamespace(logits=self.score(hidden))

    def sanitize(self, weights):
        for index, layer in enumerate(self.model.layers):
            weights[f"model.layers.{index}.self_attn.rope.inv_freq"] = layer.self_attn.rope.inv_freq
        return weights


def make_bidirectional_sliding_window_mask(
    input_ids: mx.array,
    attention_mask: mx.array | None,
    sliding_window: int,
) -> mx.array | None:
    _, length = input_ids.shape
    if length <= 1 and attention_mask is None:
        return None

    positions = mx.arange(length)
    distance = mx.abs(mx.expand_dims(positions, 0) - mx.expand_dims(positions, 1))
    allowed = distance <= sliding_window
    if attention_mask is not None:
        key_allowed = attention_mask.astype(mx.bool_)
        allowed = mx.expand_dims(allowed, (0, 1)) & mx.expand_dims(key_allowed, (1, 2))
    else:
        allowed = mx.expand_dims(mx.expand_dims(allowed, 0), 0)

    return mx.where(allowed, mx.array(0.0, dtype=mx.float32), mx.array(-mx.inf, dtype=mx.float32))
