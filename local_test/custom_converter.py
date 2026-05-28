"""Examples of PyTorch ExportedProgram custom converters.

Each converter maps one exported ATen op name to Relax IR.  Pick exactly one
map for a given ATen key when calling ``from_exported_program``.
"""

import numpy as np
import torch
import torch.nn as nn

import tvm
from tvm import relax
from tvm.script import tirx as T
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter


class AddOp(nn.Module):
    def forward(self, x, y):
        return x + y


class MulOp(nn.Module):
    def forward(self, x, y):
        return x * y


class SoftmaxOp(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class AddMulSoftmaxOp(nn.Module):
    def forward(self, x, y, z):
        return torch.softmax((x + y) * z, dim=-1)


def _same_tensor_sinfo(expr: relax.Expr) -> relax.TensorStructInfo:
    sinfo = expr.struct_info
    if not isinstance(sinfo, relax.TensorStructInfo):
        raise TypeError(f"Expected TensorStructInfo, got {type(sinfo)}")
    return sinfo


# ---------------------------------------------------------------------------
# 1. Pure Relax converter
# ---------------------------------------------------------------------------


def custom_add_relax_converter(
    node: torch.fx.Node,
    importer: ExportedProgramImporter,
) -> relax.Var:
    """Lower ``aten.add.Tensor`` using normal Relax ops."""

    x, y = importer.retrieve_args(node)
    return importer.block_builder.emit(relax.op.add(x, y))


CUSTOM_CONVERT_MAP_RELAX = {
    "add.Tensor": custom_add_relax_converter,
}


# ---------------------------------------------------------------------------
# 2. TIR converter
# ---------------------------------------------------------------------------


@T.prim_func(s_tir=True)
def zazzle_mul_1d_tir(var_x: T.handle, var_y: T.handle, var_out: T.handle, n: T.int64):
    x = T.match_buffer(var_x, (n,), "float32")
    y = T.match_buffer(var_y, (n,), "float32")
    out = T.match_buffer(var_out, (n,), "float32")

    for i in range(n):
        out[i] = x[i] * y[i]


def custom_mul_tir_converter(
    node: torch.fx.Node,
    importer: ExportedProgramImporter,
) -> relax.Var:
    """Lower ``aten.mul.Tensor`` to a custom TIR PrimFunc via ``R.call_tir``.

    This example intentionally supports only 1-D float32 tensors.  A production
    converter should either support all expected ranks/dtypes or reject them
    explicitly, as this one does.
    """

    x, y = importer.retrieve_args(node)
    x_sinfo = _same_tensor_sinfo(x)
    y_sinfo = _same_tensor_sinfo(y)

    if x_sinfo.dtype != "float32" or y_sinfo.dtype != "float32":
        raise NotImplementedError("zazzle_mul_1d_tir example only supports float32")

    x_shape = importer.shape_of(x)
    y_shape = importer.shape_of(y)
    if len(x_shape) != 1 or len(y_shape) != 1:
        raise NotImplementedError("zazzle_mul_1d_tir example only supports 1-D tensors")

    n = x_shape[0]
    out_sinfo = relax.TensorStructInfo([n], "float32")

    # Cache the GlobalVar on the importer instance so multiple mul nodes reuse
    # the same PrimFunc inside this one conversion.
    cache_key = "_zazzle_mul_1d_tir_gvar"
    gvar = getattr(importer, cache_key, None)
    if gvar is None:
        gvar = importer.block_builder.add_func(zazzle_mul_1d_tir, "zazzle_mul_1d_tir")
        setattr(importer, cache_key, gvar)

    return importer.block_builder.emit(
        relax.call_tir(
            gvar,
            (x, y),
            out_sinfo=out_sinfo,
            tir_vars=[n],
        )
    )


CUSTOM_CONVERT_MAP_TIR = {
    "mul.Tensor": custom_mul_tir_converter,
}


# ---------------------------------------------------------------------------
# 3. Runtime PackedFunc converter
# ---------------------------------------------------------------------------


def register_python_softmax_dps_for_testing() -> None:
    """Register a CPU Python DPS PackedFunc for local testing.

    Real cuDNN integration should be a C++/CUDA PackedFunc with this signature:

        zazzle.cudnn.softmax_dps(x, axis, out)

    That wrapper can then call cuDNN internally.  Do not directly use
    ``tvm.contrib.cudnn.softmax.forward`` with ``call_dps_packed``; cuDNN's TVM
    PackedFunc currently expects ``(x, out, axis)``, while ``call_dps_packed``
    appends allocated outputs after the user-provided args, i.e. ``(x, axis, out)``.
    """

    @tvm.register_global_func("zazzle.cudnn.softmax_dps", override=True)
    def _softmax_dps(x, axis, out):
        axis = int(axis)
        data = x.numpy()
        shifted = data - np.max(data, axis=axis, keepdims=True)
        result = np.exp(shifted)
        result = result / np.sum(result, axis=axis, keepdims=True)
        out.copyfrom(result.astype(data.dtype, copy=False))


def custom_softmax_runtime_converter(
    node: torch.fx.Node,
    importer: ExportedProgramImporter,
) -> relax.Var:
    """Lower softmax to a runtime PackedFunc via ``R.call_dps_packed``."""

    x = importer.env[node.args[0]]
    x_sinfo = _same_tensor_sinfo(x)

    axis = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
    if not isinstance(axis, int):
        raise NotImplementedError("runtime softmax example requires a static integer axis")

    if axis < 0:
        axis += len(importer.shape_of(x))

    return importer.block_builder.emit(
        relax.call_dps_packed(
            "zazzle.cudnn.softmax_dps",
            (x, axis),
            out_sinfo=x_sinfo,
        )
    )


CUSTOM_CONVERT_MAP_RUNTIME = {
    # The observed key depends on whether PyTorch decomposition has run.
    "softmax.int": custom_softmax_runtime_converter,
    "_softmax.default": custom_softmax_runtime_converter,
}


CUSTOM_CONVERT_MAP_ALL = {
    **CUSTOM_CONVERT_MAP_RELAX,
    **CUSTOM_CONVERT_MAP_TIR,
    **CUSTOM_CONVERT_MAP_RUNTIME,
}


# Default export for simple imports from full_pipeline.py.
CUSTOM_CONVERT_MAP = CUSTOM_CONVERT_MAP_ALL


def _print_call_function_names(exported: torch.export.ExportedProgram) -> None:
    for node in exported.graph.nodes:
        if node.op == "call_function":
            print(node.target.__name__, node.target)


if __name__ == "__main__":
    model = AddMulSoftmaxOp()
    args = (
        torch.randn(5, dtype=torch.float32),
        torch.randn(5, dtype=torch.float32),
        torch.randn(5, dtype=torch.float32),
    )
    exported = torch.export.export(model, args)

    print("ATen names after decomposition:")
    _print_call_function_names(exported.run_decompositions())

    register_python_softmax_dps_for_testing()

    mod = from_exported_program(
        exported,
        custom_convert_map=CUSTOM_CONVERT_MAP_ALL,
    )
    mod.show()


# 这三个转换层级可以同时发生  但是前提不能是同一个算子
# eg. RELAX_CONVERTER(add) 和 TIR_CONVERTER(add) 不可以
# 但是 RELAX_CONVERTER(add) 和 TIR_CONVERTER(mul) 可以
