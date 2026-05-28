"""
Dynamic-shape variant of `full_pipeline.py`.

Same model, same params, same execution flow — the only difference is the
torch.export call: `batch_size` is marked dynamic via `torch.export.Dim`,
so the imported Relax IRModule has a `tirx::Var` for that dimension.

Useful for observing how `tirx::Var` flows through:
  - the torch -> Relax frontend translator
    (python/tvm/relax/frontend/torch/exported_program_translator.py)
  - BlockBuilder::BeginScope -> AddDefinitionToScope
    (src/relax/ir/block_builder.cc)

Run:
  PYTHONUNBUFFERED=1 \
  TVM_TRACE_BLOCK_BUILDER=1 \
  .venv/bin/python -u local_test/dynamic_full_pipeline.py > run_dynamic.log 2>&1

Note on the choice of which axis is dynamic:
  Only `batch_size` is dynamic. `seq_len` stays static because the model's
  rope_cos / rope_sin / causal_mask buffers are precomputed with
  `cfg.seq_len` at construction time (see TinyTransformer in full_pipeline.py)
  and the forward path uses them as-is. Making seq_len dynamic would require
  changing the model itself.
"""

import os
import sys
import time

import numpy as np
import torch
from torch.export import Dim

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

try:
    from tvm import dlight as dl  # noqa: F401  (used by optimize_for_cuda)
except ImportError:
    from tvm.s_tir import dlight as dl  # noqa: F401

# Reuse the model + helpers from full_pipeline.py so the only diff is the
# export step. Add this file's directory to sys.path so the import works
# when invoked from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_pipeline import (  # noqa: E402
    CFG,
    TinyTransformer,
    dump_mod,
    to_numpy,
    get_param_list,
    optimize_for_cuda,
    print_imported_device_sources,
)


# -----------------------------
# 3'. Export PyTorch -> Relax with a dynamic batch dim
# -----------------------------

def export_to_relax_dynamic(torch_model, input_ids, dynamic_shapes):
    with torch.no_grad():
        exported = torch.export.export(
            torch_model,
            (input_ids,),
            dynamic_shapes=dynamic_shapes,
        )

    mod = from_exported_program(
        exported,
        keep_params_as_input=True,
        unwrap_unit_return_tuple=True,
    )

    mod, params = relax.frontend.detach_params(mod)
    return mod, params


# -----------------------------
# Main (mirrors full_pipeline.main step-for-step)
# -----------------------------

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    torch_model = TinyTransformer(CFG).eval()

    # Sample input shape is identical to full_pipeline's; only the export
    # annotation differs.
    input_ids = torch.randint(
        low=0,
        high=CFG.vocab_size,
        size=(CFG.batch_size, CFG.seq_len),
        dtype=torch.int64,
    )

    # PyTorch specializes batch=1 (treats it as a degenerate dimension), so
    # the *export-time* sample must have batch >= 2 even though we still
    # validate the compiled binary at batch=1 below.
    export_batch = max(CFG.batch_size, 2)
    input_ids_export = torch.randint(
        low=0,
        high=CFG.vocab_size,
        size=(export_batch, CFG.seq_len),
        dtype=torch.int64,
    )

    # Mark batch dim as dynamic. The exported program will have a symbolic
    # dim for axis 0 of `input_ids`, which the Relax frontend turns into a
    # `tirx::Var` named "batch" (or similar) inside TensorStructInfo.
    batch_dim = Dim("batch", min=1, max=64)
    dynamic_shapes = {"input_ids": {0: batch_dim}}

    with torch.no_grad():
        torch_out = torch_model(input_ids).detach().cpu().numpy()

    print("PyTorch output shape:", torch_out.shape)

    # 1) PyTorch -> Relax (with dynamic batch)
    # Trace will show:
    #   - exported_program_translator.py debug prints (line 2021 etc.)
    #   - BlockBuilder BeginScope/AddDefinitionToScope with var_map.size>=1
    mod, params = export_to_relax_dynamic(torch_model, input_ids_export, dynamic_shapes)

    # Same early-exit hatch as full_pipeline.py — comment this out to
    # continue through legalize / DLight / compile / VM.
    sys.exit(0)

    dump_mod(
        "STAGE 1: Imported Relax IRModule (dynamic batch)",
        mod,
        "logs/01_imported_relax_dynamic.py",
    )

    # 2) Legalize once for inspection.
    mod_legalized = relax.transform.LegalizeOps()(mod)
    dump_mod(
        "STAGE 2: After LegalizeOps (dynamic batch)",
        mod_legalized,
        "logs/02_legalized_dynamic.py",
    )

    # 3) Optimize for CUDA.
    dev = tvm.cuda(0)
    assert dev.exist, "CUDA device is not available"

    target = tvm.target.Target.from_device(dev)
    mod_opt = optimize_for_cuda(mod, target)
    dump_mod(
        "STAGE 3: After Relax pipeline + DLight GPU schedule (dynamic batch)",
        mod_opt,
        "logs/03_optimized_scheduled_dynamic.py",
    )

    # 4) Compile.
    executable = tvm.compile(mod_opt, target=target)

    # 5) Inspect CUDA source.
    print_imported_device_sources(executable)

    # 6) Run with the Relax VM. Verify dynamism by running at *two* batch
    #    sizes against the same compiled binary.
    vm = relax.VirtualMachine(executable, dev)

    param_list = get_param_list(params)
    tvm_params = [
        tvm.runtime.tensor(to_numpy(p), device=dev) for p in param_list
    ]

    print()
    print("=" * 90)
    print("CORRECTNESS at multiple batch sizes")
    print("=" * 90)
    for batch in (1, 4):
        ids = torch.randint(
            low=0,
            high=CFG.vocab_size,
            size=(batch, CFG.seq_len),
            dtype=torch.int64,
        )
        with torch.no_grad():
            ref = torch_model(ids).detach().cpu().numpy()

        tvm_input = tvm.runtime.tensor(ids.cpu().numpy(), device=dev)
        tvm_out = vm["main"](tvm_input, *tvm_params)
        if isinstance(tvm_out, (tuple, list)):
            tvm_out = tvm_out[0]
        tvm_out_np = tvm_out.numpy()

        np.testing.assert_allclose(tvm_out_np, ref, rtol=1e-3, atol=1e-3)
        print(f"  batch={batch}: TVM out shape={tvm_out_np.shape} -> MATCH")

    # 7) Benchmark at the reference shape.
    tvm_input = tvm.runtime.tensor(input_ids.cpu().numpy(), device=dev)
    for _ in range(5):
        _ = vm["main"](tvm_input, *tvm_params)
    dev.sync()

    repeat = 20
    t0 = time.time()
    for _ in range(repeat):
        _ = vm["main"](tvm_input, *tvm_params)
    dev.sync()
    t1 = time.time()

    print(f"Average latency at batch={CFG.batch_size}, "
          f"seq_len={CFG.seq_len}: {(t1 - t0) * 1e3 / repeat:.3f} ms")


if __name__ == "__main__":
    main()

# PYTHONUNBUFFERED=1 \
# TVM_TRACE_BLOCK_BUILDER=1 \
# .venv/bin/python -u local_test/dynamic_full_pipeline.py > run_dynamic.log 2>&1
