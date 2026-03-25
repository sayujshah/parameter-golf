# HELIX Architecture Design Spec

> **Hypernetwork-assisted Efficient Looped Information eXchange**
> Parameter Golf Challenge | 16MB artifact | 8×H100 | ≤10 min
> Target: Beat current SOTA BPB of **1.1194** (2026-03-23)

---

## 1. Goal

Implement HELIX — a novel transformer architecture combining three cutting-edge 2025 research innovations never previously combined or applied to this competition:

1. **D-TPA** — Differential Tensor Product Attention: outer-product factored K/V (TPA, ICML 2025) with noise-canceling differential maps (DIFF, ICLR 2025 Oral)
2. **MoR** — Mixture of Recursions: 5 shared blocks × 3 max iterations, where each token decides per-round whether to exit early or continue (arXiv:2507.10524, July 2025)
3. **Peri-LN** — Sandwich normalization (pre-LN + post-LN at every sublayer, ICML 2025, adopted by Gemma 2 / OLMo 2)

These are combined with the proven SOTA support stack: SmearGate, BigramHash, int6+zstd-22, partial RoPE, XSA on deep blocks, EMA(0.997), SWA, Muon optimizer.

**Projected BPB:** 1.097–1.107 (vs SOTA 1.1194). Margin of ≥0.012 if gains hold at this scale.

---

## 2. Architecture

### 2.1 High-Level Structure

```
input_ids [B, T]
    ↓ tok_emb(1024, 640) + bigram(2048, 128→640)
    ↓ RMSNorm → SmearGate
    ↓ x0 = x  (anchor for resid_mix)
    ↓
[Iteration r=0]
  Block_0(x, x0, r=0) → Block_1 → Block_2 → Block_3 → Block_4
  [collect U-Net skip from Block_0, Block_1 outputs]
  MoR gate_0(x) → optional early exit for tokens
    ↓
[Iteration r=1]
  Block_0(x, x0, r=1) → ... → Block_4
  MoR gate_1(x) → optional early exit for tokens
    ↓
[Iteration r=2]
  [inject U-Net skips (reversed) into Block_0, Block_1]
  Block_0(x, x0, r=2) → ... → Block_4
    ↓
RMSNorm(x) → lm_head (tied to tok_emb.T) → softcap → logits
```

### 2.2 Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| `NUM_UNIQUE_BLOCKS` (K) | 5 | 5×3=15 virtual layers > 11 SOTA layers |
| `NUM_ITERATIONS` (R) | 3 | Matches Universal Transformer "≥L/3 unique blocks" theorem |
| `MODEL_DIM` (d) | 640 | D-TPA saves ~72% attn params → reinvest in width |
| `NUM_HEADS` | 8 | d_head = 640/8 = 80 |
| `NUM_KV_HEADS` | 4 | GQA 8Q/4KV ratio maintained |
| `DTPA_RANK` (r) | 4 | Rank-4 outer-product factorization |
| `FFN_HIDDEN` | 1792 | SwiGLU 2.8× expansion (hidden = 1792) |
| `ROPE_DIMS` | 16 | Partial RoPE: 16/80 head dims |
| `XSA_LAST_N` | 2 | XSA on blocks 3,4 at iteration r=2 only |
| `MOR_LB_WEIGHT` | 0.01 | Load-balancing aux loss weight (decays to 0) |
| `BIGRAM_VOCAB_SIZE` | 2048 | BigramHash buckets |
| `BIGRAM_DIM` | 128 | BigramHash projection dim |

### 2.3 Parameter Budget

| Component | Params |
|---|---|
| D-TPA per block | ~502K |
| SwiGLU FFN per block (hidden=1792) | ~3,441K |
| Peri-LN (4 RMSNorm per block) + per-depth scales + MoR router | ~12K |
| **Per-block subtotal** | **~3,955K** |
| 5 unique blocks | ~19,775K |
| tok_emb (1024×640) | 655K |
| BigramHash (2048×128 + 128×640) | 344K |
| SmearGate gate (640) + skip_weights (2×640) + misc | ~2K |
| **Grand total** | **~20,776K ≈ 20.78M** |
| Artifact @ 0.726 B/param (int6+zstd-22) | **~15.08MB ✓** |

Headroom: **0.92MB** below 16MB limit.

---

## 3. D-TPA: Differential Tensor Product Attention

### 3.1 Motivation

Two independent ICLR/ICML 2025 papers each improve attention in orthogonal ways:
- **TPA** reduces K/V parameter count 14× via outer-product factorization, while maintaining or improving quality. Native RoPE integration (advantage over MLA).
- **DIFF** cancels attention noise by subtracting two attention maps, reducing 35% of training tokens needed to reach equivalent quality.

D-TPA combines them: TPA provides efficient factored K/V representations; DIFF provides cleaner attention patterns. These are orthogonal improvements to different aspects of the attention computation.

### 3.2 Computation

```python
# Notation: d=640, n_heads=8, n_kv=4, r=4, d_head=80
# DIFF splits each head into two d_head/2=40-dim sub-heads

# --- Factored projections ---
# W_cQ: [d, 2*n_heads*r] = [640, 64]   (2 for DIFF pairs × 8 heads × rank 4)
# A_Q:  [2*n_heads, r, d_head//2]       (static basis, trained)
c_Q = x @ W_cQ                          # [B, T, 64]
c_Q = c_Q.view(B, T, 2*n_heads, r)     # [B, T, 16, 4]
Q = einsum('bthd,hrd->bthr', c_Q, A_Q) # [B, T, 2*n_heads, d_head//2]
Q1, Q2 = Q.chunk(2, dim=2)             # each [B, T, n_heads, d_head//2]

# Same factored projection for K and V (n_kv_heads=4, two DIFF pairs)
K1, K2 = ...  # [B, T, n_kv, d_head//2] each
V1, V2 = ...  # [B, T, n_kv, d_head//2] each

# --- Apply partial RoPE to basis vectors A_Q, A_K (first 16 of 40 dims) ---
# Applied once at init-time to A_Q, A_K; recomputed each forward for positional encoding

# --- Differential attention ---
# lambda: [n_heads], initialized via: 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1))
A1 = softmax(Q1 @ K1.T / sqrt(d_head//2), mask=causal)  # [B, n_heads, T, T]
A2 = softmax(Q2 @ K2.T / sqrt(d_head//2), mask=causal)  # [B, n_heads, T, T]
A  = A1 - lam.view(1,n_heads,1,1) * A2                   # differential map

# V is averaged (not split) for the output
V = (V1 + V2) / 2                                         # or just V1
out = (A @ V).reshape(B, T, n_heads * d_head//2)          # [B, T, 320]
# Note: n_heads * d_head//2 = 8 * 40 = 320 ≠ d=640
# So W_O: [320, 640] — output projection back to model dim
out = out @ W_O                                            # [B, T, 640]
```

### 3.3 Parameter Count Detail

| Tensor | Shape | Params |
|---|---|---|
| W_cQ | 640 × (2×8×4) = 640×64 | 40,960 |
| A_Q (basis) | 2×8×4×40 | 2,560 |
| W_cK | 640 × (2×4×4) = 640×32 | 20,480 |
| A_K (basis) | 2×4×4×40 | 1,280 |
| W_cV | 640 × (2×4×4) = 640×32 | 20,480 |
| A_V (basis) | 2×4×4×40 | 1,280 |
| W_O | 320 × 640 | 204,800 |
| λ (per head) | 8 | 8 |
| q_gain (per head) | 8 | 8 |
| **Total D-TPA** | | **~291,856 ≈ 292K** |

> **Note:** Standard GQA at d=512 costs 1,769K. D-TPA at d=640 costs 292K. The freed ~1.5M params per block go into SwiGLU FFN width and U-Net expressiveness.

### 3.4 QK-Norm

Apply RMSNorm to Q and K after factored reconstruction (before attention) — standard QK-norm for numerical stability with differential maps. The subtraction `A1 - λ*A2` can produce small values that amplify gradient noise without QK-norm.

### 3.5 XSA Integration

For the last 2 blocks (k=3, k=4) at iteration r=2 only: drop the V projection entirely (XSA = "Exclusive Self-Attention", removes self-value pathway). This forces context-dependent representations in the final processing pass.

```python
if self.use_xsa and r == self.num_iterations - 1:
    # Use A (differential attention map) directly without V
    out = A.sum(dim=-1, keepdim=True) * x  # attention-weighted self
```

---

## 4. MoR: Mixture of Recursions

### 4.1 Structure

```python
def forward(self, input_ids, target_ids):
    x = self._embed(input_ids)  # embed + bigram + smear
    x0 = x.clone()
    first_iter_hidden = []

    for r in range(R_MAX):  # R_MAX = 3
        for k in range(K):  # K = 5
            # U-Net: inject encoder skips into last iteration
            if r == R_MAX - 1 and k < self.num_skip:
                skip_idx = self.num_skip - 1 - k
                x = x + self.skip_weights[skip_idx] * first_iter_hidden[skip_idx]

            x = self.blocks[k](x, x0, r)

            # U-Net: collect encoder skips from first iteration
            if r == 0 and k < self.num_skip:
                first_iter_hidden.append(x.detach())  # detach to avoid gradient loops

        # MoR gate: should tokens continue to next iteration?
        if r < R_MAX - 1:
            gate_logit = x @ self.mor_gate[r]  # [B, T, 1]
            # Training: soft — store logit for aux loss, don't hard-route
            # Inference: hard — freeze exited tokens
            self._mor_gate_logits[r] = gate_logit

    return self.final_norm(x)
```

### 4.2 MoR Router Parameters

```python
self.mor_gate = nn.ParameterList([
    nn.Parameter(torch.zeros(d, 1))   # one gate vector per inter-iteration boundary
    for _ in range(R_MAX - 1)         # = 2 gates
])
# Total: 2 × 640 × 1 = 1,280 params
```

The gate vectors are initialized to zero (sigmoid(0) = 0.5, equal probability of exit/continue). They should be routed to **scalar AdamW** (not Muon) since they are 2D weight tensors but semantically scalar controllers. Naming convention: `mor_gate` — add to CONTROL_TENSOR_NAME_PATTERNS check.

### 4.3 Load-Balancing Auxiliary Loss

```python
def compute_mor_aux_loss(self, lb_weight=0.01):
    # gate_logits: list of [B, T, 1] tensors
    p1 = torch.sigmoid(self._mor_gate_logits[0]).mean()   # exit rate at r=1
    p2 = torch.sigmoid(self._mor_gate_logits[1]).mean()   # exit rate at r=2 (of remaining)
    # Target: each exit point handles ~1/3 of tokens
    target = 1.0 / 3.0
    aux = lb_weight * ((p1 - target)**2 + (p2 * (1 - p1) - target)**2)
    return aux
```

The `lb_weight` starts at 0.01 and is linearly decayed to 0 during the last 1000 warmdown steps. This ensures the gates are trained to be meaningful but don't constrain the model at the end of training.

### 4.4 Inference (Eval)

At evaluation (`forward_logits`), all 3 iterations are always executed for all tokens. MoR gates are ignored. This is the safest policy for maximizing BPB. The gates serve primarily as a training signal — they force the model to learn iteration-depth specialization during training, even if hard routing is not used at eval.

---

## 5. Per-Depth Adaptation (Peri-LN + Scales)

### 5.1 Peri-LN (Sandwich Normalization)

Each sublayer (attention and MLP) gets pre-norm AND post-norm:

```python
# Attention sublayer
h = self.pre_norm_attn(x)     # Pre-LN (standard)
h = self.dtpa(h)
h = self.post_norm_attn(h)    # Post-LN  ← Peri-LN addition
x = x + self.iter_attn_scale[r] * h

# MLP sublayer
h = self.pre_norm_mlp(x)      # Pre-LN (standard)
h = self.swiglu(h)
h = self.post_norm_mlp(h)     # Post-LN  ← Peri-LN addition
x = x + self.iter_mlp_scale[r] * h
```

4 RMSNorm instances per block (vs 2 in standard Pre-LN). Parameter cost: 4 × 640 = 2,560 per block = 12,800 total across 5 blocks. Negligible.

### 5.2 Per-Iteration Scalars

```python
# Per block, per iteration — routes to float32 scalar AdamW (name-matched)
self.iter_attn_scale = nn.ParameterList([
    nn.Parameter(torch.ones(d))  for _ in range(R)
])   # name contains "attn_scale" → CONTROL_TENSOR_NAME_PATTERNS match

self.iter_mlp_scale = nn.ParameterList([
    nn.Parameter(torch.ones(d))  for _ in range(R)
])   # name contains "mlp_scale"

self.iter_resid_mix = nn.ParameterList([
    nn.Parameter(torch.stack([torch.ones(d), torch.zeros(d)]))  for _ in range(R)
])   # name contains "resid_mix"
# Per-block cost: 3 × (640 + 640 + 1280) = 7,680 params
```

### 5.3 SwiGLU FFN

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        self.gate = CastedLinear(dim, hidden)    # 640×1792 = 1,146,880
        self.fc   = CastedLinear(dim, hidden)    # 640×1792 = 1,146,880
        self.proj = CastedLinear(hidden, dim)    # 1792×640 = 1,146,880
        self.proj._zero_init = True
        # Total: 3,440,640

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.fc(x))
```

SwiGLU is isoparametric to relu²(3× hidden) at hidden=1792 vs d×3=1920, but provides:
- Smooth gradient everywhere (no dead zones)
- Multiplicative gating (XOR-like interactions)
- Validated in LLaMA, PaLM, and competition predecessor GRAFT-WX proposal

---

## 6. Training Configuration

### 6.1 Optimizer Setup

```python
# Matrix params (2D, no control pattern name) → Muon
muon_params = [p for n,p in base_model.blocks.named_parameters()
               if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)]
# + dtpa basis matrices A_Q, A_K, A_V are 3D → Adam (can't Muon 3D tensors)

# Scalar/control params → float32 AdamW
adam_params = [
    {'params': embedding_params, 'lr': TIED_EMBED_LR},
    {'params': scalar_params,    'lr': SCALAR_LR},
    {'params': dtpa_basis_params,'lr': SCALAR_LR},   # A_Q, A_K, A_V basis
    {'params': mor_gate_params,  'lr': SCALAR_LR},   # MoR gate vectors
]
```

**Critical:** The TPA basis tensors `A_Q`, `A_K`, `A_V` are 3D (`[2*n_heads, r, d_head//2]`). Muon requires 2D matrices. These must be routed to Adam. Add `"dtpa_basis"` (or name them `A_q_basis`, `A_k_basis` etc.) to bypass Muon routing.

### 6.2 Learning Rates

| Parameter Group | LR | Rationale |
|---|---|---|
| Matrix params (Muon) | 0.023 | `0.04 / √3` — 3× gradient accumulation from shared weights |
| Scalar/control (Adam) | 0.04 | Standard |
| Embeddings (Adam) | 0.05 | Standard |
| TPA basis A_Q/K/V (Adam) | 0.04 | 3D tensors, semantic role similar to scalars |

### 6.3 Full Environment Variables

```bash
RUN_ID=helix_v1 \
MODEL_DIM=640 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
DTPA_RANK=4 \
NUM_UNIQUE_BLOCKS=5 \
NUM_ITERATIONS=3 \
ROPE_DIMS=16 \
XSA_LAST_N=2 \
FFN_HIDDEN=1792 \
MOR_LB_WEIGHT=0.01 \
MOR_LB_DECAY_START=0     \
TRAIN_SEQ_LEN=2048 \
WARMDOWN_ITERS=3500 \
GRAD_CLIP_NORM=0.3 \
MATRIX_LR=0.023 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
MUON_WD=0.04 \
ADAM_WD=0.01 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
SWA_ENABLED=1 \
SWA_EVERY=200 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
LOGIT_SOFTCAP=30.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 6.4 Initialization

```python
def _init_weights(self):
    virtual_depth = K * R  # = 5 * 3 = 15
    for module in self.modules():
        if isinstance(module, CastedLinear):
            if getattr(module, '_zero_init', False):
                nn.init.zeros_(module.weight)
            else:
                nn.init.orthogonal_(module.weight)
                module.weight.data.mul_(1.0 / math.sqrt(2 * virtual_depth))
        elif isinstance(module, nn.Parameter) and module.ndim == 3:
            # TPA basis vectors: orthogonal init per head
            nn.init.orthogonal_(module.data.reshape(-1, module.shape[-1]))
    # Lambda (DIFF): initialized per layer
    for layer_idx, block in enumerate(self.blocks):
        lam_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        block.dtpa.lam.data.fill_(lam_init)
```

---

## 7. Interface Contract

HELIX exposes the standard SOTA training-loop interface:

```python
base_model.blocks          # nn.ModuleList of 5 HELIXBlock instances
base_model.smear           # SmearGate (has .gate attribute)
base_model.bigram          # BigramHashEmbedding
base_model.tok_emb         # nn.Embedding (tied with lm_head)
base_model.skip_weights    # nn.Parameter [2, 640]
base_model.mtp_heads       # nn.ModuleList([]) — unused
base_model.mtp_num_heads   # 0
base_model.forward_logits(input_ids)  # → [B, T, V] logits (all 3 iters, no MoR exit)
```

The `compute_mor_aux_loss()` method is called inside `forward()` and added to the cross-entropy loss before `.backward()`.

---

## 8. File Structure

| File | Role |
|---|---|
| `train_gpt.py` | Main training script — HELIX model class replaces `GPT`; all other infrastructure (optimizer, loader, artifact) unchanged |
| `train_gpt_mlx.py` | MLX mirror for local smoke tests (simplified D-TPA, no MoR) |

HELIX replaces only the `Block` and `GPT` classes (and adds `DTPA`, `SwiGLU`). Everything from line ~730 onward (training loop, DDP, quantization, artifact) is untouched.

**Line budget estimate:**
- `DTPA` class: ~80 lines
- `SwiGLU` class: ~20 lines
- `HELIXBlock` class: ~80 lines
- `HELIX_GPT` class: ~130 lines
- Hyperparameter additions: ~15 lines
- Optimizer routing changes (TPA basis, MoR gate): ~20 lines
- MoR aux loss in training loop: ~15 lines
- **Total additions: ~360 lines**
- Baseline script: ~1500 lines (but replaces ~250 lines of Block/GPT)
- **Net: within 1500-line limit** ✓

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| D-TPA not proven at <25M scale | Medium | TPA and DIFF each proven separately; combination is modular |
| Recursive convergence slower than independent layers | Medium | Orthogonal init + muP scaling + per-depth scales |
| MoR gate collapse (all tokens exit early) | Low-Medium | Load-balancing aux loss prevents this |
| 3D TPA basis tensors complicate optimizer routing | Low | Explicit routing in optimizer setup |
| Parameter count exceeds 16MB | Low | 0.92MB headroom; careful accounting above |
| Training time exceeds 10 minutes | Low | MoR doesn't add FLOPs during training (soft gates) |

---

## 10. References

1. **TPA**: "Tensor Product Attention Is All You Need" — arXiv:2501.06425 (Jan 2025, ICML 2025)
2. **Differential Attention**: "Differential Transformer" — arXiv:2410.05258 (Oct 2024, ICLR 2025 Oral)
3. **Peri-LN**: "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture" — arXiv:2502.02732 (Feb 2025, ICML 2025)
4. **MoR**: "Mixture of Recursions" — arXiv:2507.10524 (July 2025)
5. **SwiGLU**: Shazeer 2020 / LLaMA — validated in multiple large-scale LMs
6. **XSA**: "Exclusive Self-Attention" — arXiv:2603.09078 (2026)
7. **Looped Transformers**: "Looped Transformers as Programmable Computers" — Giannou et al. 2024
8. **Universal Transformers**: Dehghani et al. 2018
9. **SmearGate / BigramHash**: parameter-golf community contributions (PR #414 stack)
