# Adding the `Zazzle` Operator to TVM (Relax layer)

`zazzle(A, B, padding)` computes **`A @ B + padding`**, where:

- `A`, `B` — input tensors (matmul operands)
- `padding` — a **compile-time `float` constant**, broadcast from a scalar to the
  matmul output shape
- returns one tensor

It is registered as a **primitive Relax op** (`relax.zazzle` appears as a single node in
the IR), and is **lowered at legalization time by composing existing ops**:
`matmul` → `broadcast_to` → `add`.

---

## Files to add / modify

All `zazzle` code lives in the existing `linear_algebra` files (it is matmul-based), so
**no `CMakeLists.txt` change is needed** — `src/relax/op/**` is auto-globbed.

| # | File | Action | What to do |
|---|------|--------|-----------|
| 1 | `include/tvm/relax/attrs/linear_algebra.h` | modify | Add a `ZazzleAttrs` struct (fields `double padding`, `DataType out_dtype`) next to `MatmulAttrs`, with `RegisterReflection()` + `TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ZazzleAttrs", ...)`. |
| 2 | `src/relax/op/tensor/linear_algebra.h` | modify | Declare `Expr zazzle(Expr x1, Expr x2, double padding, ffi::Optional<DataType> out_dtype);` |
| 3 | `src/relax/op/tensor/linear_algebra.cc` | modify | (a) Register `ZazzleAttrs::RegisterReflection()`. (b) Define `zazzle()` building `ZazzleAttrs` and returning `Call(Op::Get("relax.zazzle"), {x1, x2}, Attrs(attrs), {})`. (c) FFI: `GlobalDef().def("relax.op.zazzle", zazzle)`. (d) `InferStructInfoZazzle` (output = matmul result of `A`,`B`; reuse matmul shape logic). (e) `TVM_REGISTER_OP("relax.zazzle")` with 2 tensor args, `FInferStructInfo`, `FPurity`. |
| 4 | `python/tvm/relax/op/linear_algebra.py` | modify | Add `def zazzle(x1, x2, padding=0.0, out_dtype=None)` → `_ffi_api.zazzle(x1, x2, float(padding), out_dtype)`. |
| 5 | `python/tvm/relax/op/__init__.py` | modify | Export `zazzle` in the `from .linear_algebra import ...` line. |
| 6 | `python/tvm/relax/transform/legalize_ops/linear_algebra.py` | modify | `@register_legalize("relax.zazzle")` that emits `matmul(A,B)`, makes a scalar `const(padding)`, `broadcast_to` the matmul output shape, then `add`. The sub-ops legalize recursively. |
| 7 | `tests/python/relax/test_op_linear_algebra.py` | add test | `test_zazzle_infer_struct_info` — shape/dtype inference matches matmul of `(A, B)`. |
| 8 | `tests/python/relax/test_transform_legalize_ops_linear_algebra.py` | add test | Run `LegalizeOps()`, assert decomposition into matmul + broadcast_to + add. |
| 9 | `tests/python/relax/test_vm_build.py` (optional) | add test | Build + run on VM, compare against numpy `A @ B + padding`. |

### Reference snippets

`ZazzleAttrs` (file 1):
```cpp
struct ZazzleAttrs : public AttrsNodeReflAdapter<ZazzleAttrs> {
  double padding;
  DataType out_dtype;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ZazzleAttrs>()
        .def_ro("padding", &ZazzleAttrs::padding, "Scalar padding added after matmul")
        .def_ro("out_dtype", &ZazzleAttrs::out_dtype, "The data type of the output tensor");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ZazzleAttrs", ZazzleAttrs, BaseAttrsNode);
};
```

Python API (file 4):
```python
def zazzle(x1: Expr, x2: Expr, padding: float = 0.0,
           out_dtype: str | DataType | None = None) -> Expr:
    return _ffi_api.zazzle(x1, x2, float(padding), out_dtype)  # type: ignore
```

Legalization (file 6):
```python
@register_legalize("relax.zazzle")
def _zazzle(bb: BlockBuilder, call: Call) -> Expr:
    from ...op import matmul, add, broadcast_to
    from ... import const
    attrs = call.attrs
    out_sinfo = call.struct_info                       # matmul-result sinfo
    prod = bb.emit(matmul(call.args[0], call.args[1], out_dtype=attrs.out_dtype))
    pad = const(attrs.padding, out_sinfo.dtype)        # 0-D scalar constant
    pad = broadcast_to(pad, out_sinfo.shape)           # scalar -> output shape
    return add(prod, pad)
```

---

## Reference patterns to copy from

- **matmul** — Python `python/tvm/relax/op/linear_algebra.py:28`; C++/registration
  `src/relax/op/tensor/linear_algebra.cc:44-55, 57-160, 167-174`; header
  `src/relax/op/tensor/linear_algebra.h:44`; attrs
  `include/tvm/relax/attrs/linear_algebra.h:32-42`.
- **broadcast_to** — `python/tvm/relax/op/manipulate.py:31`,
  `src/relax/op/tensor/manipulate.cc:63-72`.
- **add** — macro `src/relax/op/tensor/binary.h:40-62`; legalize
  `python/tvm/relax/transform/legalize_ops/binary.py:46`.
- **register_legalize** — `python/tvm/relax/transform/legalize_ops/common.py:110-121`.

---

## Verification

1. **Rebuild C++** (steps 1–3 change C++): cmake/ninja build of TVM.
2. **Tests:** `pytest tests/python/relax/test_op_linear_algebra.py -k zazzle` plus the
   legalize test.
3. **End-to-end sanity:**
   ```python
   import numpy as np, tvm
   from tvm import relax
   from tvm.relax.transform import LegalizeOps
   # Build an IR module with R.zazzle(A, B) (padding=2.0),
   # apply LegalizeOps(), relax.build, run on the VM,
   # assert_allclose(res, A @ B + 2.0)
   ```
4. Confirm `R.zazzle` is a single op before legalization and decomposes into
   matmul / broadcast_to / add afterward.
