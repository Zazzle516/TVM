"""Pipeline test for using all three custom converter levels together.

This follows the same high-level flow as ``local_test/full_pipeline.py``:

  PyTorch nn.Module
    -> torch.export
    -> TVM Relax IRModule with custom converter plugin insertion
    -> Relax optimization
    -> compile
    -> Relax VM execution
    -> correctness check vs PyTorch

Run from the TVM repo root with the local build environment, for example:

    PYTHONPATH=/home/zazzle/TVM/python \
    LD_LIBRARY_PATH=/home/zazzle/TVM/build/lib:/home/zazzle/TVM/build/llvm-root/usr/lib/llvm-18/lib:/home/zazzle/TVM/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/home/zazzle/TVM/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/lib/wsl/lib \
    CUDA_PATH=/home/zazzle/TVM/.venv/lib/python3.12/site-packages/nvidia/cu13 \
    TVM_CUDA_COMPILE_MODE=nvrtc \
    .venv/bin/python local_test/test_custom_converter_pipeline.py
"""

import os
import sys

import numpy as np
import torch

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

sys.path.insert(0, os.path.dirname(__file__))

from custom_converter import (  # noqa: E402
    AddMulSoftmaxOp,
    CUSTOM_CONVERT_MAP_ALL,
    register_python_softmax_dps_for_testing,
)


def get_param_list(params):
    if isinstance(params, dict):
        if "main" in params:
            return params["main"]
        if "forward" in params:
            return params["forward"]
        if params:
            return next(iter(params.values()))
        return []
    return params


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def export_to_relax(torch_model, inputs):
    with torch.no_grad():
        exported = torch.export.export(torch_model, inputs)

    register_python_softmax_dps_for_testing()

    mod = from_exported_program(
        exported,
        keep_params_as_input=True,
        unwrap_unit_return_tuple=True,
        custom_convert_map=CUSTOM_CONVERT_MAP_ALL,
    )
    mod, params = relax.frontend.detach_params(mod)
    return mod, params


def assert_custom_converter_insertions(mod):
    text = mod.script(show_meta=False)

    assert "R.add(" in text, "Relax add converter was not visible in imported IR"
    assert "zazzle_mul_1d_tir" in text, "TIR mul converter was not visible in imported IR"
    assert "R.call_tir(" in text, "TIR mul converter did not emit call_tir"
    assert 'R.call_dps_packed("zazzle.cudnn.softmax_dps"' in text, (
        "Runtime softmax converter did not emit call_dps_packed"
    )


def select_target_and_device():
    cuda_dev = tvm.cuda(0)
    if cuda_dev.exist:
        return tvm.target.Target.from_device(cuda_dev), cuda_dev

    return tvm.target.Target("llvm"), tvm.cpu(0)


def optimize_for_target(mod, target):
    # The full zero pipeline includes FuseTIR.  This smoke test intentionally
    # keeps the hand-written call_tir boundary intact while still running a
    # small Relax optimization stage before tvm.compile's default VM pipeline.
    passes = [
        relax.transform.LegalizeOps(),
        relax.transform.FoldConstant(),
        relax.transform.ToNonDataflow(),
    ]

    with target:
        return tvm.ir.transform.Sequential(passes)(mod)


def run_vm(mod, params, target, dev, inputs):
    executable = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(executable, dev)

    tvm_inputs = [tvm.runtime.tensor(to_numpy(x), device=dev) for x in inputs]
    tvm_params = [
        tvm.runtime.tensor(to_numpy(param), device=dev)
        for param in get_param_list(params)
    ]

    out = vm["main"](*tvm_inputs, *tvm_params)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.numpy()


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    torch_model = AddMulSoftmaxOp().eval()
    inputs = (
        torch.randn(5, dtype=torch.float32),
        torch.randn(5, dtype=torch.float32),
        torch.randn(5, dtype=torch.float32),
    )

    with torch.no_grad():
        torch_out = torch_model(*inputs).detach().cpu().numpy()

    mod, params = export_to_relax(torch_model, inputs)
    assert_custom_converter_insertions(mod)

    target, dev = select_target_and_device()
    mod_opt = optimize_for_target(mod, target)
    tvm_out = run_vm(mod_opt, params, target, dev, inputs)

    np.testing.assert_allclose(tvm_out, torch_out, rtol=1e-5, atol=1e-5)

    print("CUSTOM CONVERTER PIPELINE: PASS")
    print("target:", target)
    print("output shape:", tvm_out.shape)


if __name__ == "__main__":
    main()
