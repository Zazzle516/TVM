"""
Minimal pipeline to observe how `tirx::Var` (a symbolic shape variable) is
processed by `BlockBuilder::AddDefinitionToScope` in
`src/relax/ir/block_builder.cc`.

Run with the trace flag enabled so the C++ side prints the var_map contents:

  PYTHONUNBUFFERED=1 \
  TVM_TRACE_BLOCK_BUILDER=1 \
  .venv/bin/python -u local_test/dynamic_shape_trace.py > run_dyn.log 2>&1

Expected: with tirx.Var dimensions in the parameter struct_info, you should see
`var_map.size` > 0 and lines like
    [Zazzle] AddDefinitionToScope var_map[0] key=n value=n
for each new symbolic dim, instead of the 0-size you saw with fully-static
shapes in `full_pipeline.py`.
"""

from tvm import relax, tirx


def build_with_dynamic_shape():
    # 1. Create symbolic shape variables. These are the things that, when
    #    they appear inside a TensorStructInfo's shape, get harvested by
    #    StructInfoVarCollector and registered into the current ScopeFrame.
    n = tirx.Var("n", "int64")  # batch (or seq_len) — dynamic
    m = tirx.Var("m", "int64")  # hidden dim — dynamic
    k = tirx.Var("k", "int64")  # extra dim, only appears via add result

    # 2. Mix of dynamic dims, expressions, and constants in parameter shapes.
    #    - x: both dims are bare tirx.Var  -> both should be collected
    #    - w: one bare tirx.Var, one constant -> only the var is collected
    #    - b: a single bare tirx.Var
    #    - c: shape is (m + 1, 16) -> the `m + 1` is NOT a bare Var, so it is
    #         NOT collected (matches the comment in StructInfoVarCollector).
    #    - d: fully static, like full_pipeline.py params -> var_map.size == 0
    x = relax.Var("x", relax.TensorStructInfo([n, m], "float32"))
    w = relax.Var("w", relax.TensorStructInfo([m, 64], "float32"))
    b = relax.Var("b", relax.TensorStructInfo([k], "float32"))
    c = relax.Var("c", relax.TensorStructInfo([m + 1, 16], "float32"))
    d = relax.Var("d", relax.TensorStructInfo([1, 16, 128], "float32"))

    bb = relax.BlockBuilder()

    with bb.function("dynamic_main", [x, w, b, c, d]):
        with bb.dataflow():
            # A trivial body so we still go through Emit/Normalize, which
            # also routes through scope state.
            t0 = bb.emit(relax.op.matmul(x, w))      # [n, 64]
            t1 = bb.emit(relax.op.add(t0, t0))       # [n, 64]
            out = bb.emit_output(t1)
        bb.emit_func_output(out)

    mod = bb.finalize()
    return mod


def main():
    print("[trace-test] building module with dynamic-shape params...")
    mod = build_with_dynamic_shape()
    print("[trace-test] module built. printing IR:")
    mod.show()


if __name__ == "__main__":
    main()
