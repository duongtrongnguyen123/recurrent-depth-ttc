"""LoopedTransformer with switchable architecture variants.

Llama-style block: RMSNorm, SwiGLU MLP, RoPE attention, no biases.

Variants (selected via PretrainConfig):
  - "vanilla":          n_blocks=N distinct blocks, n_loops=1.
  - "looped":           1 shared block, n_loops=N.
  - "looped_aux":       same forward as looped; aux loss handled in train loop.
  - "pcc":              n_prelude unique + 1 core looped n_loops + n_coda unique.
  - "gated":            looped with gated state injection x' = (1-g)*x + g*block(x).
  - "per_loop_bias":    looped with per-loop learnable bias added to block input.
  - "stochastic_depth": looped; train loop randomly truncates n_loops to [r_min,r_max].
  - "curriculum":       looped+aux; train loop ramps n_loops over training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PretrainConfig:
    arch: str = "vanilla"          # see module docstring
    vocab_size: int = 50257
    d_model: int = 1024
    n_heads: int = 8
    ff_mult: int = 4               # FFN hidden = ff_mult * d_model (SwiGLU uses 2/3 of this internally)
    n_blocks: int = 8              # distinct blocks. For looped variants, set to 1.
    n_loops: int = 1               # times the (entire) stack is looped.
    n_prelude: int = 0             # for 'pcc': unique pre-recurrence layers
    n_coda: int = 0                # for 'pcc': unique post-recurrence layers
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    tie_embeddings: bool = True
    init_std: float = 0.02
    # --- Latent noise injection (looped_aux_robust, Branch 1 Result R recipe) ---
    p_noise: float = 0.0           # probability of injecting noise per forward pass
    noise_alpha: float = 0.0       # noise stddev = alpha * ||h|| / sqrt(d)
    # --- Skip residual from initial embedding (Branch 1 Result AJ recipe) ---
    skip_alpha_init: float = 0.0   # >0 enables learnable skip; init value of alpha
    use_skip: bool = False         # set True to enable α·h_0 skip at every loop
    # --- Cross-loop attention (novel arch): at loop r, attend to current K/V
    # plus K/V cached from all previous loops 0..r-1. Tests gradient flow
    # across depth + cross-loop token-token effects.
    use_cross_loop_attn: bool = False
    # --- Hierarchical recurrent (HR): nested outer x inner loops with goal register ---
    use_hr: bool = False
    n_hr_inner: int = 4
    n_hr_outer: int = 4
    # --- Step-aware per-loop bias: gives model an explicit "you are on loop r"
    # signal at every loop. Compatible with any arch (PCC, looped, etc.) — unlike
    # arch=="per_loop_bias" which gates on the arch string. Cheap (n_loops × d_model
    # parameters, ~5K total at d=1024 n_loops=4).
    use_loop_bias: bool = False
    # Per-loop QKV bias (archlab FN/FO): n_loops * 3 * d_model param tensor
    # added to qkv-linear output, gives each loop a distinct Q/K/V direction.
    use_loop_qkv_bias: bool = False
    # --- Vanilla blockwise iter-target: each "loop" applies ONE distinct block,
    # exposing per-block intermediate logits for iter-target lookahead supervision.
    # Set with n_blocks=N, n_loops=N, use_iter_target=True. Each block i predicts
    # input shifted by (i+1) positions. This gives vanilla N-layer transformer
    # with intermediate output heads + multi-token-prediction (MTP-style) loss.
    vanilla_blockwise_iter: bool = False
    # --- OpenMythos / Parcae LTI injection: h_{t+1} = A·h + B·e + Block(h+e)
    # diagonal A∈(0,1) per-channel, diagonal B per-channel. block input is h+e.
    # Untested at production scale (small-scale = no benefit per BT/BU/BV/BW/CI).
    use_lti: bool = False
    # --- Two-stream parallel (AS recipe): vanilla + looped streams, per-token gate
    # zero-init gate so model starts as 50/50 mix; stream B uses the looped core_blocks
    use_two_stream: bool = False
    n_stream_a_blocks: int = 8     # number of distinct blocks in stream A (vanilla branch)
    # Mixture-of-Loops (MoL): K shared looped cores, per-token-per-loop routing
    use_mol: bool = False
    n_mol_cores: int = 4           # K cores in the mixture


# ---------- Llama-style components ----------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(x.dtype) * self.weight


def _precompute_rope(seq_len: int, head_dim: int, base: float, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [T, head_dim/2]
    cos = freqs.cos()[None, None, :, :].to(dtype)  # [1,1,T,hd/2]
    sin = freqs.sin()[None, None, :, :].to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past_kvs: list | None = None,
                loop_qkv_bias: torch.Tensor | None = None,
                ) -> "tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | torch.Tensor":
        """If past_kvs is None: standard single-pass attention, returns out tensor.
        If past_kvs is a list (possibly empty): cross-loop attention. Q is from
        current x; K/V are concatenation of all past_kvs entries plus current K/V
        along the sequence dim. Returns (out, (k, v)) so caller can append to cache.

        loop_qkv_bias: optional [3*d_model] tensor. When provided, added to the
        qkv linear output BEFORE reshape — gives this loop's attention a
        distinct Q/K/V offset (ports archlab Result FN/FO from synthetic chain
        to real-text recurrence). Per-loop bias on QKV breaks the Q-collapse
        identified by repr_flow analysis (cross-loop Q cosine ~ 1.00).
        """
        B, T, C = x.shape
        qkv = self.qkv(x)
        if loop_qkv_bias is not None:
            qkv = qkv + loop_qkv_bias  # broadcast over [B, T]
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)        # [B,T,H,D]
        q = q.transpose(1, 2)              # [B,H,T,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos[:, :, :T, :], sin[:, :, :T, :])
        k = _apply_rope(k, cos[:, :, :T, :], sin[:, :, :T, :])
        if past_kvs is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, T, C)
            return self.proj(out)
        # Cross-loop: concat past loops' K/V along seq dim before current.
        if past_kvs:
            past_K = torch.cat([pk for pk, _ in past_kvs], dim=2)   # [B,H,T*r,D]
            past_V = torch.cat([pv for _, pv in past_kvs], dim=2)
            all_k = torch.cat([past_K, k], dim=2)                    # [B,H,T*(r+1),D]
            all_v = torch.cat([past_V, v], dim=2)
            n_past = len(past_kvs)
            # Mask: q[t] can attend to (loop r', pos t') iff t' <= t (causal in pos).
            # Across loops: full visibility (any past loop fully visible up to
            # causal pos). Mask shape [T, T*(n_past+1)].
            base = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            full = torch.cat([base] * (n_past + 1), dim=1)
            attn_mask = torch.where(
                full, torch.zeros((), device=x.device, dtype=q.dtype),
                torch.full((), float("-inf"), device=x.device, dtype=q.dtype))
            out = F.scaled_dot_product_attention(q, all_k, all_v,
                                                    attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        # Standard Llama SwiGLU sizing: hidden ≈ 2/3 of (ff_mult * d_model), rounded to 64.
        hidden = int(2 * cfg.ff_mult * cfg.d_model / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ln2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past_kvs: list | None = None,
                loop_qkv_bias: torch.Tensor | None = None):
        """If past_kvs is None: standard. Otherwise cross-loop attention is used
        and (out, (k, v)) is returned so caller can update the cache.
        loop_qkv_bias: optional [3*d_model] tensor passed to attn for per-loop
        QKV specialization."""
        if past_kvs is None:
            x = x + self.attn(self.ln1(x), cos, sin, loop_qkv_bias=loop_qkv_bias)
            x = x + self.mlp(self.ln2(x))
            return x
        attn_out, kv = self.attn(self.ln1(x), cos, sin, past_kvs=past_kvs,
                                  loop_qkv_bias=loop_qkv_bias)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, kv


# ---------- The model ----------

class LoopedTransformer(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_prelude)])
        self.core_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_blocks)])
        self.coda_blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_coda)])
        self.ln_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Variant-specific extras. Always assign both so __getattr__ never raises.
        self.gate_logits = None
        self.loop_bias = None
        if cfg.arch == "gated":
            # one scalar gate per loop step. init to logit(0.99) so we start near pure block(x).
            self.gate_logits = nn.Parameter(torch.full((cfg.n_loops,), 4.0))
        elif cfg.arch == "per_loop_bias":
            self.loop_bias = nn.Parameter(torch.zeros(cfg.n_loops, cfg.d_model))
        # Step-aware loop bias is ARCH-AGNOSTIC — works on top of pcc/pcc_iter/
        # looped/looped_aux/etc. Each loop r gets its own learned d-dim bias added
        # before the core pass. Tells the model "this is loop r of n".
        if cfg.use_loop_bias and self.loop_bias is None:
            self.loop_bias = nn.Parameter(torch.zeros(cfg.n_loops, cfg.d_model))

        # Per-loop QKV bias (archlab Result FN/FO ported to real-text). Adds a
        # learned [n_loops, 3*d_model] offset to the qkv-projection output,
        # giving each loop a distinct Q/K/V direction. Breaks the Q-collapse
        # documented by repr_flow analysis (cross-loop Q cosine ~ 1.00).
        # Cost: n_loops * 3 * d_model parameters (~24K at d=1024 n_loops=8).
        self.loop_qkv_bias = None
        if cfg.use_loop_qkv_bias:
            self.loop_qkv_bias = nn.Parameter(
                torch.zeros(cfg.n_loops, 3 * cfg.d_model))

        # Skip residual from h_0 (Branch 1 AJ recipe) — single learnable scalar.
        self.skip_alpha = (nn.Parameter(torch.tensor(cfg.skip_alpha_init))
                            if cfg.use_skip else None)

        # Hierarchical recurrent (HR): outer block + goal register
        if cfg.use_hr:
            self.outer_block = Block(cfg)
            self.goal_init = nn.Parameter(torch.randn(1, 1, cfg.d_model) * cfg.init_std)
        else:
            self.outer_block = None
            self.goal_init = None

        # Mixture-of-Loops (MoL): K shared looped cores + per-token-per-loop router
        if cfg.use_mol:
            self.mol_cores = nn.ModuleList(Block(cfg) for _ in range(cfg.n_mol_cores))
            self.mol_router = nn.Linear(cfg.d_model, cfg.n_mol_cores, bias=True)
            nn.init.zeros_(self.mol_router.weight)
            nn.init.zeros_(self.mol_router.bias)   # uniform init → all cores equally likely
        else:
            self.mol_cores = None
            self.mol_router = None

        # Two-stream parallel (AS recipe): vanilla branch + looped branch + gate
        if cfg.use_two_stream:
            self.stream_a_blocks = nn.ModuleList(
                Block(cfg) for _ in range(cfg.n_stream_a_blocks))
            # zero-init gate produces sigmoid(0)=0.5 → 50/50 mix at start
            self.stream_gate = nn.Linear(cfg.d_model, 1, bias=True)
            nn.init.zeros_(self.stream_gate.weight)
            nn.init.zeros_(self.stream_gate.bias)
        else:
            self.stream_a_blocks = None
            self.stream_gate = None

        # OpenMythos / Parcae LTI injection (h_{t+1} = A·h + B·e + Block(h+e))
        if cfg.use_lti:
            self.lti_log_A = nn.Parameter(torch.zeros(cfg.d_model))
            self.lti_log_dt = nn.Parameter(torch.zeros(1))
            self.lti_B = nn.Parameter(torch.ones(cfg.d_model) * 0.1)
            self.lti_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        else:
            self.lti_log_A = None
            self.lti_log_dt = None
            self.lti_B = None
            self.lti_norm = None

        self.apply(self._init_weights)

        # RoPE table cached lazily
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def _rope(self, T: int, device, dtype):
        cos, sin = self._cos, self._sin
        if cos is None or sin is None or cos.size(2) < T or cos.device != device:
            cos, sin = _precompute_rope(
                self.cfg.max_seq_len, self.cfg.d_model // self.cfg.n_heads,
                self.cfg.rope_base, device, dtype)
            self._cos, self._sin = cos, sin
        return cos[:, :, :T, :], sin[:, :, :T, :]

    def _core_pass(self, h: torch.Tensor, cos, sin, n_loops: int,
                   collect: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the core stack n_loops times. Returns (final h, list of per-loop hidden states).

        If self.cfg.use_cross_loop_attn, threads a KV cache across loops so
        each loop's attention can read all previous loops' K/V at the same
        sequence positions.
        If self.cfg.use_lti, applies OpenMythos/Parcae update rule:
          h_{t+1} = A·h + B·e + Block(LayerNorm(h+e))
        with diagonal A∈(0,1) per-channel and diagonal B per-channel.
        """
        outs: list[torch.Tensor] = []
        cross = self.cfg.use_cross_loop_attn
        lti = self.cfg.use_lti
        mol = self.cfg.use_mol
        past_kvs: list[tuple] = [] if cross else None
        e = h if lti else None  # save post-prelude state for re-injection
        for i in range(n_loops):
            h_in = h
            if self.loop_bias is not None and i < self.loop_bias.shape[0]:
                h_in = h_in + self.loop_bias[i]
            if lti:
                h_in = self.lti_norm(h_in + e)
            new_h = h_in
            if mol:
                # Per-token-per-loop routing: each token picks top-1 core
                router_logits = self.mol_router(h_in)   # [B, T, K]
                router_argmax = router_logits.argmax(dim=-1)  # [B, T]
                # Sequential per-core mask (avoids dynamic dispatch issues with grad)
                k_out = torch.zeros_like(h_in)
                for k_idx, blk in enumerate(self.mol_cores):
                    mask = (router_argmax == k_idx).unsqueeze(-1).float()
                    if mask.sum() > 0:
                        b_out = blk(h_in, cos, sin)   # apply this core to ALL tokens
                        k_out = k_out + mask * b_out
                # Soft router gradient: re-weight by softmax (Switch Transformer style)
                gate_p = router_logits.softmax(dim=-1).gather(-1, router_argmax.unsqueeze(-1))  # [B, T, 1]
                new_h = gate_p * k_out + (1 - gate_p).detach() * k_out  # gradient flows through router
            else:
                # Per-loop QKV bias: pick this loop's slice of [n_loops, 3*d]
                qb = (self.loop_qkv_bias[i] if self.loop_qkv_bias is not None
                      and i < self.loop_qkv_bias.shape[0] else None)
                for blk in self.core_blocks:
                    if cross:
                        new_h, kv = blk(new_h, cos, sin, past_kvs=past_kvs,
                                          loop_qkv_bias=qb)
                        past_kvs.append(kv)
                    else:
                        new_h = blk(new_h, cos, sin, loop_qkv_bias=qb)
            if lti:
                # h_{t+1} = A·h + B·e + transformer_out
                A = torch.exp(-torch.exp((self.lti_log_dt + self.lti_log_A).clamp(-20, 20)))
                h = A * h + self.lti_B * e + new_h
            elif self.gate_logits is not None and i < self.gate_logits.shape[0]:
                g = torch.sigmoid(self.gate_logits[i])
                h = (1 - g) * h + g * new_h
            else:
                h = new_h
            if collect:
                outs.append(h)
        return h, outs


    def _core_pass_hr(self, h, cos, sin, n_outer, n_inner, collect=False):
        """Hierarchical recurrent core: n_outer x n_inner loops with a per-position
        goal register. Causal version — summary at position t depends only on
        h[:, :t+1, :], so future tokens cannot leak into past predictions.

        Inner steps add the per-position goal to the input before each inner block
        pass. Outer step computes a causal cumulative-mean summary at every
        position, runs the outer block per-position on (summary, goal), and
        takes the updated per-position goal."""
        B, T, D = h.shape
        # Per-position goal (init: every position gets the same init vector)
        goal = self.goal_init.expand(B, T, D).contiguous()  # [B, T, D]
        cos_outer, sin_outer = self._rope(2, h.device, h.dtype)
        outs = []
        # Per-position normalizer for cumulative mean
        idx = torch.arange(1, T + 1, device=h.device).view(1, T, 1).to(h.dtype)
        for _ in range(n_outer):
            for _ in range(n_inner):
                h_in = h + goal  # per-position add
                for blk in self.core_blocks:
                    h_in = blk(h_in, cos, sin)
                h = h_in
                if collect:
                    outs.append(h)
            # Causal cumulative summary: summary[:, t, :] = mean(h[:, :t+1, :])
            summary = h.cumsum(dim=1) / idx  # [B, T, D]
            # Per-position outer-block call: stack (summary[t], goal[t]) -> [B*T, 2, D]
            outer_in = torch.stack([summary, goal], dim=2).view(B * T, 2, D)
            outer_out = self.outer_block(outer_in, cos_outer, sin_outer)
            goal = outer_out[:, 1, :].view(B, T, D)  # per-position updated goal
        return h, outs

    def forward(self, idx: torch.Tensor, n_loops: Optional[int] = None,
                collect_loops: bool = False) -> dict:
        """Standard forward. Returns dict {logits, loop_hiddens (optional)}."""
        T = idx.size(1)
        h = self.tok_emb(idx)
        cos, sin = self._rope(T, h.device, h.dtype)

        for blk in self.prelude_blocks:
            h = blk(h, cos, sin)

        n = n_loops if n_loops is not None else self.cfg.n_loops

        # Two-stream parallel: vanilla A + looped B with per-token gate
        if self.cfg.use_two_stream:
            h_input = h.clone()
            # Stream A: distinct vanilla blocks
            h_a = h_input
            for blk in self.stream_a_blocks:
                h_a = blk(h_a, cos, sin)
            # Stream B: shared looped core
            h_b, loop_hiddens = self._core_pass(h_input, cos, sin, n_loops=n, collect=collect_loops)
            # Per-token gate (zero-init produces 50/50 at start)
            g = torch.sigmoid(self.stream_gate(h_input))   # [B, T, 1]
            h = (1 - g) * h_a + g * h_b
        elif self.cfg.use_hr:
            h, loop_hiddens = self._core_pass_hr(
                h, cos, sin,
                n_outer=self.cfg.n_hr_outer, n_inner=self.cfg.n_hr_inner,
                collect=collect_loops)
        else:
            h, loop_hiddens = self._core_pass(h, cos, sin, n_loops=n, collect=collect_loops)

        for blk in self.coda_blocks:
            h = blk(h, cos, sin)

        logits = self.lm_head(self.ln_f(h))
        out = {"logits": logits}
        if collect_loops:
            out["loop_hiddens"] = loop_hiddens
        if self.cfg.use_two_stream:
            out["gate_mean"] = g.mean().item()  # diagnostic
        return out

    def forward_with_aux(self, idx: torch.Tensor, n_loops: int,
                         aux_min_loops: int = 1) -> dict:
        """Aux-loss forward: returns logits at every loop in [aux_min_loops, n_loops].
        Used for arch='looped_aux' and 'curriculum'.

        Optional Branch 1 recipes:
        - p_noise > 0: latent noise injection at random loop (Result R)
        - use_skip: additive α·h_0 residual at every loop's output (Result AJ)
        """
        T = idx.size(1)
        h = self.tok_emb(idx)
        cos, sin = self._rope(T, h.device, h.dtype)
        for blk in self.prelude_blocks:
            h = blk(h, cos, sin)

        # Anchor for skip residual (snapshot after prelude).
        h0 = h if self.skip_alpha is not None else None

        # Decide noise injection schedule for this forward pass.
        inject_at = -1
        if self.training and self.cfg.p_noise > 0.0 and torch.rand(1).item() < self.cfg.p_noise:
            inject_at = torch.randint(1, n_loops + 1, (1,)).item()

        per_loop_logits: list[torch.Tensor] = []
        cross = self.cfg.use_cross_loop_attn
        past_kvs: list = [] if cross else None
        for i in range(n_loops):
            h_in = h
            if self.loop_bias is not None and i < self.loop_bias.shape[0]:
                h_in = h_in + self.loop_bias[i]
            new_h = h_in
            if self.cfg.vanilla_blockwise_iter:
                # Vanilla mode: apply ONLY block[i] (1 distinct block per loop)
                # so per_loop_logits[i] = output after block i.
                if i < len(self.core_blocks):
                    new_h = self.core_blocks[i](new_h, cos, sin)
            else:
                for blk in self.core_blocks:
                    if cross:
                        new_h, kv = blk(new_h, cos, sin, past_kvs=past_kvs)
                        past_kvs.append(kv)
                    else:
                        new_h = blk(new_h, cos, sin)
            if self.gate_logits is not None and i < self.gate_logits.shape[0]:
                g = torch.sigmoid(self.gate_logits[i])
                h = (1 - g) * h + g * new_h
            else:
                h = new_h
            # Skip residual from h_0 (Branch 1 AJ recipe).
            if self.skip_alpha is not None and h0 is not None:
                h = h + self.skip_alpha * h0
            # Inject noise *after* the core pass at this loop.
            if (i + 1) == inject_at:
                noise_std = (self.cfg.noise_alpha
                              * h.norm(dim=-1, keepdim=True)
                              / (h.shape[-1] ** 0.5))
                h = h + torch.randn_like(h) * noise_std

            if (i + 1) >= aux_min_loops:
                h_out = h
                for blk in self.coda_blocks:
                    h_out = blk(h_out, cos, sin)
                per_loop_logits.append(self.lm_head(self.ln_f(h_out)))

        return {"logits": per_loop_logits[-1], "per_loop_logits": per_loop_logits}

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.tok_emb.weight.numel()
        return n


def build_model(cfg: PretrainConfig) -> LoopedTransformer:
    """Apply arch-specific config defaults, then build."""
    if cfg.arch == "vanilla":
        if not cfg.vanilla_blockwise_iter:
            cfg.n_loops = 1
        # else: keep n_loops as set (should equal n_blocks for blockwise iter)
    elif cfg.arch in ("looped", "looped_aux", "looped_aux_robust",
                      "looped_aux_skip", "gated", "per_loop_bias",
                      "stochastic_depth", "curriculum"):
        cfg.n_blocks = 1
    elif cfg.arch in ("pcc", "pcc_aux_robust", "pcc_iter_robust", "pcc_xloop"):
        cfg.n_blocks = 1     # one core block looped
    elif cfg.arch == "hr":
        cfg.n_blocks = 1
        cfg.use_hr = True
    elif cfg.arch == "lti":
        cfg.n_blocks = 1
        cfg.use_lti = True
    elif cfg.arch == "two_stream":
        cfg.n_blocks = 1
        cfg.use_two_stream = True
    elif cfg.arch == "mol":
        cfg.n_blocks = 1   # core_blocks unused; mol_cores holds K cores
        cfg.use_mol = True
    else:
        raise ValueError(f"unknown arch: {cfg.arch}")
    return LoopedTransformer(cfg)
