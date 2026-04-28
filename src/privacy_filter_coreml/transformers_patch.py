from __future__ import annotations


def patch_openai_privacy_filter_dense_experts() -> None:
    """Patch OpenAI Privacy Filter MoE experts into an export-friendly dense form.

    The upstream eager implementation loops over the experts that were hit by
    the router for the current input. That preserves the intended sparse compute
    path, but the loop length is data-dependent and blocks `torch.export`.

    This patch computes every expert for every token and then masks/sums the
    router top-k experts. It is mathematically equivalent for conversion
    feasibility tests, but it is not the desired production performance path.
    """

    try:
        from transformers.models.openai_privacy_filter import modeling_openai_privacy_filter
    except ImportError as exc:  # pragma: no cover - dependency-gated utility
        raise RuntimeError("transformers with OpenAI Privacy Filter support is required") from exc

    def dense_forward(self, hidden_states, router_indices=None, routing_weights=None):
        original_dtype = hidden_states.dtype
        x = hidden_states.float()
        token_count = x.shape[0]
        expert_weights = x.new_zeros((token_count, self.num_experts), dtype=x.dtype)
        expert_weights.scatter_add_(1, router_indices, routing_weights.float())

        gate_up = (
            _einsum("nh,ehi->nei", x, self.gate_up_proj.float())
            + self.gate_up_proj_bias.float().unsqueeze(0)
        )
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * _sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        out = (
            _einsum("nei,eih->neh", gated_output.float(), self.down_proj.float())
            + self.down_proj_bias.float().unsqueeze(0)
        )
        return (out * expert_weights.unsqueeze(-1)).sum(dim=1).to(original_dtype)

    # Import torch lazily so base package import stays lightweight.
    import torch

    _einsum = torch.einsum
    _sigmoid = torch.sigmoid
    modeling_openai_privacy_filter.OpenAIPrivacyFilterExperts.forward = dense_forward
