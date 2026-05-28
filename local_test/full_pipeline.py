"""
Tiny LLaMA/GPT-style Transformer compiled by TVM.

Flow:
  PyTorch nn.Module
    -> torch.export
    -> TVM Relax IRModule
    -> Relax graph optimization / FuseTIR
    -> DLight GPU scheduling
    -> CUDA codegen
    -> Relax VM execution
    -> correctness check vs PyTorch

This is a fixed-shape prefill-only demo:
  - no KV cache
  - no autoregressive sampling
  - fixed batch and sequence length
  - tiny dimensions to keep compile time reasonable
"""

import math
import os
import sys
import re
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

try:
    from tvm import dlight as dl
except ImportError:
    # Some TVM branches expose DLight under tvm.s_tir.
    from tvm.s_tir import dlight as dl


# -----------------------------
# 0. Config
# -----------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 1024
    batch_size: int = 1
    seq_len: int = 16
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


CFG = ModelConfig()


# -----------------------------
# 1. Helpers
# -----------------------------

def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", s)


def dump_mod(title: str, mod, path: str):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)
    script = strip_ansi(mod.script())
    print(script)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(script)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def get_param_list(params):
    # relax.frontend.detach_params usually returns Dict[str, List[Tensor]].
    if isinstance(params, dict):
        if "main" in params:
            return params["main"]
        if "forward" in params:
            return params["forward"]
        # fall back to the first function's params
        return next(iter(params.values()))
    return params


# -----------------------------
# 2. LLaMA-style PyTorch model
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, T, D]
        variance = (x * x).mean(dim=-1, keepdim=True)
        # variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def precompute_rope(seq_len: int, head_dim: int, theta: float):
    """
    Return:
      cos: [1, 1, T, head_dim]
      sin: [1, 1, T, head_dim]
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, head_dim / 2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return cos, sin


def rotate_half(x):
    # x: [B, H, T, Dh]
    dh = x.shape[-1]
    x1 = x[..., : dh // 2]
    x2 = x[..., dh // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

        cos, sin = precompute_rope(cfg.seq_len, self.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        mask = torch.triu(
            torch.full((cfg.seq_len, cfg.seq_len), -1.0e4, dtype=torch.float32),
            diagonal=1,
        )
        self.register_buffer(
            "causal_mask",
            mask[None, None, :, :],
            persistent=False,
        )

    def forward(self, x):
        # x: [B, T, D]
        bsz, seq_len, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # [B, T, D] -> [B, H, T, Dh]
        q = q.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + self.causal_mask
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # [B, H, T, Dh]
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.dim)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)

    def forward(self, x):
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate)
        up = self.w3(x)
        return self.w2(gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: [B, T], int64
        x = self.tok_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------
# 3. Export PyTorch -> Relax
# -----------------------------

def export_to_relax(torch_model, input_ids):
    with torch.no_grad():
        exported = torch.export.export(torch_model, (input_ids,))

    # 1. frontend conveter
    mod = from_exported_program(
        exported,
        keep_params_as_input=True,
        unwrap_unit_return_tuple=True,
    )

    mod, params = relax.frontend.detach_params(mod)
    return mod, params


# -----------------------------
# 4. Optimize with TVM
# -----------------------------

def optimize_for_cuda(mod, target):
    with target:
        pipeline = tvm.ir.transform.Sequential(
            [
                # Official "zero" pipeline includes legalization and fusion passes.
                relax.get_pipeline("zero"),

                # Default GPU schedule rules. This is where matmul/reduction/etc.
                # TIR functions are scheduled for CUDA.
                dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                ),
            ]
        )
        return pipeline(mod)


# -----------------------------
# 5. Optional: inspect generated CUDA
# -----------------------------

def print_imported_device_sources(executable, max_chars=6000):
    print()
    print("=" * 90)
    print("GENERATED DEVICE MODULES")
    print("=" * 90)

    candidates = []
    if hasattr(executable, "mod"):
        candidates.append(executable.mod)
    candidates.append(executable)

    seen = set()
    found = False

    for rt_mod in candidates:
        if rt_mod is None or id(rt_mod) in seen:
            continue
        seen.add(id(rt_mod))

        if not hasattr(rt_mod, "imports"):
            continue

        for i, imp in enumerate(rt_mod.imports):
            found = True
            print(f"\n--- imported module #{i}, kind = {getattr(imp, 'kind', '<unknown>')} ---")
            try:
                src = imp.inspect_source()
            except Exception:
                try:
                    src = imp.get_source()
                except Exception as e:
                    print(f"Cannot inspect source: {e}")
                    continue

            print(strip_ansi(src[:max_chars]))
            if len(src) > max_chars:
                print(f"\n... truncated, full source has {len(src)} chars ...")

    if not found:
        print("No imported device module found. This may depend on the TVM executable layout.")


# -----------------------------
# 6. Main
# -----------------------------

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    torch_model = TinyTransformer(CFG).eval()

    input_ids = torch.randint(
        low=0,
        high=CFG.vocab_size,
        size=(CFG.batch_size, CFG.seq_len),
        dtype=torch.int64,
    )

    with torch.no_grad():
        torch_out = torch_model(input_ids).detach().cpu().numpy()

    print("PyTorch output shape:", torch_out.shape)

    # 1) PyTorch -> Relax
    # python/tvm/relax/frontend/torch/exported_program_translator.py
    # python/tvm/relax/frontend/torch/base_fx_graph_translator.py
    # python/tvm/relax/frontend/common.py
    mod, params = export_to_relax(torch_model, input_ids)
    # sys.exit(0)
    dump_mod(
        "STAGE 1: Imported Relax IRModule from PyTorch",
        mod,
        "logs/01_imported_relax.py",
    )
    # sys.exit(0)

    # 2) Legalize once so you can inspect Relax + TensorIR before fusion/schedule
    print("[Zazzle] after export to relax begin LegalizeOps", file=sys.stderr, flush=True)
    mod_legalized = relax.transform.LegalizeOps()(mod)
    dump_mod(
        "STAGE 2: After LegalizeOps: Relax call_tir + TensorIR PrimFuncs",
        mod_legalized,
        "logs/02_legalized.py",
    )
    sys.exit(0)

    # 3) Optimize for CUDA
    dev = tvm.cuda(0)
    assert dev.exist, "CUDA device is not available"

    target = tvm.target.Target.from_device(dev)
    mod_opt = optimize_for_cuda(mod, target)
    dump_mod(
        "STAGE 3: After Relax pipeline + DLight GPU schedule",
        mod_opt,
        "logs/03_optimized_scheduled.py",
    )

    # 4) Compile
    executable = tvm.compile(mod_opt, target=target)

    # 5) Inspect CUDA source if available
    print_imported_device_sources(executable)

    # 6) Run with Relax VM
    vm = relax.VirtualMachine(executable, dev)

    tvm_input = tvm.runtime.tensor(input_ids.cpu().numpy(), device=dev)

    param_list = get_param_list(params)
    tvm_params = [
        tvm.runtime.tensor(to_numpy(p), device=dev)
        for p in param_list
    ]

    tvm_out = vm["main"](tvm_input, *tvm_params)
    if isinstance(tvm_out, (tuple, list)):
        tvm_out = tvm_out[0]

    tvm_out_np = tvm_out.numpy()

    np.testing.assert_allclose(tvm_out_np, torch_out, rtol=1e-3, atol=1e-3)

    print()
    print("=" * 90)
    print("CORRECTNESS: PASS")
    print("=" * 90)
    print("TVM output shape:", tvm_out_np.shape)

    # 7) Benchmark
    for _ in range(5):
        _ = vm["main"](tvm_input, *tvm_params)
    dev.sync()

    repeat = 20
    t0 = time.time()
    for _ in range(repeat):
        _ = vm["main"](tvm_input, *tvm_params)
    dev.sync()
    t1 = time.time()

    print(f"Average latency: {(t1 - t0) * 1e3 / repeat:.3f} ms")


if __name__ == "__main__":
    main()

# ninja -C build

# PYTHONUNBUFFERED=1 TVM_TRACE_BLOCK_BUILDER=1 .venv/bin/python -u local_test/full_pipeline.py > run.log 2>&1

"""
from_exported_program()      <-  YOU ARE HERE
        ↓
detach_params()              <- step A
        ↓
LegalizeOps                  <- step B (inside relax.get_pipeline("zero"))
        ↓
relax.get_pipeline("zero")   <- step C: fusion, normalization, etc.
        ↓
DLight ApplyDefaultSchedule  <- step D: GPU scheduling
        ↓
tvm.compile                  <- step E: codegen â CUDA + host
        ↓
relax.VirtualMachine         <- step F: runtime execution
"""