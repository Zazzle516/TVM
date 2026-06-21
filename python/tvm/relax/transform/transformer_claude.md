# Function of `python/tvm/relax/transform`

This folder contains Python-facing Relax transformation passes and helper
modules. These passes operate on Relax IRModules during compilation. They do
not run as part of the compiled model; instead, they rewrite, normalize, lower,
or optimize the IR before later TIR lowering and target code generation.

The Python files in this folder are mostly wrappers or pass definitions exposed
through TVM's FFI. Some pass implementations live in C++ under `src/relax`,
while Python provides the user-facing API and, in some cases, compile-time
callback logic.

## `legalize_ops`

The `legalize_ops` subfolder provides default legalization rules for Relax
operators. Legalization means converting a high-level Relax operator into a
lower-level TIR-backed implementation.

For example:

```text
R.matmul(x, y)
    |
    | relax.transform.LegalizeOps
    v
R.call_tir(matmul, (x, y), ...)
    +
@T.prim_func
def matmul(...):
    loops, buffers, reductions
```

So files such as `legalize_ops/linear_algebra.py` sit between high-level Relax
ops and lower-level TIR implementations. They provide per-op conversion rules.

This is similar in spirit to an MLIR `RewritePattern`:

```text
MLIR RewritePattern
  match:    a high-level op
  rewrite: replace it with lower-level ops or dialect ops

TVM Relax legalization rule
  match:    relax.matmul
  rewrite: replace it with R.call_tir(...) plus a generated TIR PrimFunc
```

The legalization functions are written in Python because they are compile-time
rewrite callbacks, not runtime model code. They build TVM IR objects using
`BlockBuilder`, TE, and TOPI. For example, `bb.call_te(...)` converts a TE/TOPI
compute description into a TIR `PrimFunc` and inserts a corresponding
`R.call_tir` into the Relax program.

In short:

- `R.matmul` is a high-level Relax op.
- `LegalizeOps` finds the registered legalization rule for `relax.matmul`.
- The Python rule describes how to build the lower-level TE/TIR computation.
- The result is still TVM IR, not generated C++ source code.
- Later compiler stages lower TIR to LLVM, CUDA, or another target backend.

## Relax, TOPI, and TE

TOPI stands for **Tensor Operator Inventory**. It is TVM's library of common
tensor operators.

TE stands for **Tensor Expression**. TE is the lower-level tensor compute DSL
used to describe computations such as:

```python
C = te.compute((n,), lambda i: A[i] + B[i])
```

TOPI is built on top of TE. A TOPI function is usually a reusable TE compute
definition:

```python
topi.add(A, B)
topi.nn.matmul(A, B)
topi.reshape(A, shape)
```

So the relationship is:

```text
TE
  primitive tensor expression API

TOPI
  common tensor operator library implemented using TE
```

The rough layer order is:

```text
Relax IR
  high-level program graph: R.matmul, R.add, R.zazzle
        |
        | LegalizeOps
        v
TE / TOPI
  tensor compute descriptions: topi.add, topi.nn.matmul, te.compute
        |
        v
TIR PrimFunc
  explicit loops, buffers, reductions
        |
        v
Target codegen / runtime
```

Relax expressions and TOPI tensors are different layers.

- `call.args[0]` in a legalization callback is a Relax `Expr`.
- TOPI functions generally expect TE `Tensor` inputs.
- Therefore, do not call TOPI directly on Relax expressions.

Wrong:

```python
topi.add(call.args[0], call.args[1])
```

Correct:

```python
bb.call_te(topi.add, call.args[0], call.args[1])
```

`bb.call_te` is the bridge. It creates TE placeholders for the Relax inputs,
calls the TE/TOPI compute function, generates a TIR `PrimFunc`, and returns a
Relax `R.call_tir`.

In other words, `bb.call_te` performs the layer transition:

```text
Relax Expr inputs
  -> TE placeholders
  -> TOPI/TE compute
  -> generated TIR PrimFunc
  -> Relax R.call_tir(...)
```

Without `bb.call_te`, TOPI receives the wrong kind of object. A Relax `Expr`
describes a value in the high-level program. A TE `Tensor` describes an input or
output of a tensor compute expression. They are not interchangeable.

## TE/TOPI Is Not a Relax Optimization Layer

TE/TOPI is not the basis for most Relax-level optimizations. Relax passes
optimize Relax IR: operators, calls, functions, dataflow blocks, tuples,
symbolic shapes, and `R.call_tir` boundaries.

Examples of Relax-level optimizations include graph rewrites, fusion decisions,
dead-code elimination, function-level rewrites, memory planning, and
shape/dataflow simplification. These operate on Relax structure, not on TE
compute graphs.

Kernel-level optimization belongs to TIR. Once a TIR `PrimFunc` exists, later
TIR passes and scheduling/codegen machinery optimize loops, buffers, memory
accesses, vectorization, thread binding, and target-specific details.

Therefore, in `LegalizeOps`, TE/TOPI is best understood as a reusable library of
tensor compute recipes:

```text
Relax semantic op
  R.add(x, y)

legalization recipe
  use TOPI's TE definition of elementwise add

generated implementation
  TIR PrimFunc for add

Relax replacement
  R.call_tir(add, (x, y), ...)
```

This is similar to a mapping table, but it is executable Python/TE code rather
than a static dictionary. The TE/TOPI function can inspect shapes, attrs, dtype
choices, broadcasting cases, reductions, and multi-output structure, then build
the corresponding tensor compute. `bb.call_te` turns that compute recipe into a
TIR `PrimFunc` and returns the Relax call boundary.

So the role of TE/TOPI here is:

```text
not: optimize Relax using kernel internals
but: define/generate the TIR implementation that a Relax op should call
```

## BlockBuilder, `emit`, and `call_te`

`BlockBuilder` is a Relax IR construction helper. It is not a separate builder
for every TVM layer. In particular, TE/TOPI do not have their own
`BlockBuilder.emit` equivalent in this flow.

The same `BlockBuilder` concept appears in multiple compilation phases because
those phases are all constructing or rewriting Relax IR:

```text
Torch frontend
  uses BlockBuilder to build initial Relax bindings

LegalizeOps
  uses BlockBuilder to rewrite Relax ops into lower-level Relax expressions
  such as R.call_tir(...)
```

For example, during frontend import:

```python
lv0 = bb.emit(relax.op.matmul(x, y))
```

means "add this computation as a binding in the current Relax block":

```python
lv0 = R.matmul(x, y)
```

During legalization:

```python
lv0 = bb.emit(bb.call_te(topi.add, x, y))
```

means "generate a TIR implementation for `topi.add`, create a Relax
`R.call_tir(...)` expression, and add that call as a binding":

```python
lv0 = R.call_tir(add, (x, y), ...)
```

So `emit` is still a Relax operation in both places. It does not mean "enter the
TE layer". It means "append a Relax binding to the current block".

### Binding

A binding is one statement inside a Relax binding block. The common form is:

```text
VarBinding(var, value)
```

which corresponds to:

```python
lv0 = R.add(x, y)
```

`bb.emit(expr)` creates a fresh `Var` or `DataflowVar`, normalizes `expr`,
creates a `VarBinding`, appends it to the current block, and records the mapping
in the builder's binding table.

`bb.match_cast(...)` creates the other important binding form:

```text
MatchCast(var, value, struct_info)
```

which is used to assert or refine shape/type information.

### Normalization

In `bb.emit(expr)`, normalization means making the expression well-formed before
binding it. It fills in struct info, checks scope, canonicalizes expressions,
and may lift nested non-leaf expressions into separate bindings.

For example, a nested expression such as:

```python
bb.emit(relax.op.add(relax.op.multiply(x, y), z))
```

can be normalized conceptually as:

```python
lv0 = R.multiply(x, y)
lv1 = R.add(lv0, z)
```

Normalization is not the same as legalization. Normalization completes and
canonicalizes Relax IR. Legalization replaces high-level Relax ops with
lower-level implementations, often `R.call_tir`.

### `call_te`

`bb.call_te(...)` is the bridge from Relax operands to TE/TOPI and then to TIR.
It does not add a binding to the current Relax block by itself.

```python
expr = bb.call_te(topi.add, x, y)
```

does the following:

```text
1. Convert Relax arguments into TE placeholders.
2. Call the TE/TOPI function.
3. Generate a TIR PrimFunc.
4. Add that PrimFunc to the IRModule.
5. Return a Relax R.call_tir(...) expression.
```

To bind the returned call in the current Relax function, use:

```python
lv = bb.emit(bb.call_te(topi.add, x, y))
```

or the shorthand:

```python
lv = bb.emit_te(topi.add, x, y)
```

### IRModule vs Relax Binding

An `IRModule` is TVM's top-level container. It is not TIR-only. A Relax pipeline
commonly produces an `IRModule` that contains both Relax functions and TIR
PrimFuncs:

```text
IRModule
  @main = R.function(...)
  @add = T.prim_func(...)
  @matmul = T.prim_func(...)
```

Adding something to the `IRModule` means adding a global function, such as a
Relax `Function` or a TIR `PrimFunc`.

Adding a binding through `BlockBuilder.emit` means adding a statement inside the
current Relax function or dataflow block.

Therefore:

```python
bb.call_te(topi.add, x, y)
```

adds a generated TIR `PrimFunc` to the `IRModule` and returns a Relax
`R.call_tir(...)` expression, but it does not append a Relax binding.

```python
bb.emit(bb.call_te(topi.add, x, y))
```

also appends the returned `R.call_tir(...)` expression as a binding in the
current Relax block.

## Why Relax Calls TIR

After legalization, the outer program remains Relax, while individual tensor
kernels are implemented as TIR PrimFuncs. This is intentional:

```text
Relax
  program/dataflow/control layer
  tracks values, functions, tuples, symbolic shapes, and calls

TIR
  tensor kernel layer
  describes loops, buffers, memory access, reductions, and scheduling

TE/TOPI
  helper DSL/library for generating TIR kernels
```

Relax does not inline raw TIR loops directly into a Relax function. Instead, a
Relax function calls global TIR functions through `R.call_tir`.

Before legalization:

```python
z = R.matmul(x, y)
```

After legalization:

```python
z = R.call_tir(matmul, (x, y), ...)
```

and the module also contains:

```python
@T.prim_func
def matmul(...):
    ...
```

This split lets Relax remain useful for whole-program graph rewrites, dataflow
analysis, fusion, shape reasoning, memory planning, and function-level
transforms, while TIR remains useful for low-level loop and memory
optimization.

The TIR function is stored in the same `IRModule` because the Relax
`R.call_tir(...)` expression must reference a concrete global function. Relax
usually does not need the loop body for Relax-level optimization; it needs the
call boundary and interface information, such as the callee, inputs, output
shape/dtype, symbolic variables, and purity/effect behavior. The TIR body is
mainly for downstream TIR optimization and target code generation.

## `zazzle` Legalization Choice

`zazzle(A, B, padding)` can be legalized in two ways.

Option 1: decompose into existing Relax ops:

```text
R.zazzle(A, B, padding)
  -> R.matmul(A, B)
  -> R.broadcast_to(R.const(padding), output_shape)
  -> R.add(...)
```

This is usually the better first implementation because existing Relax ops
already define the correct Relax semantics. In particular, `R.matmul` handles
Relax's rank promotion, batched broadcasting, dtype behavior, and shape
inference.

Option 2: implement a single TE/TOPI compute:

```python
@register_legalize("relax.zazzle")
def _zazzle(bb, call):
    attrs = call.attrs

    def te_zazzle(a, b):
        prod = topi.nn.matmul(a, b, out_dtype=attrs.out_dtype)
        return topi.add(prod, tir.const(attrs.padding, prod.dtype))

    return bb.call_te(te_zazzle, call.args[0], call.args[1])
```

This is possible, but then the `zazzle` legalizer must correctly implement all
of the semantics that Relax `matmul` already handles. TOPI matmul operators are
lower-level and may have narrower assumptions, such as requiring rank at least
2 or using specific transpose/layout conventions.

## `call.args` vs `call.attrs`

Use `call.args` for runtime IR operands. These are Relax expressions flowing
through the program.

Use `call.attrs` for compile-time operator configuration.

For `zazzle`, the C++ constructor creates:

```cpp
return Call(op, {std::move(x1), std::move(x2)}, Attrs{attrs}, {});
```

So the call has exactly two runtime arguments:

```python
call.args[0]        # x1
call.args[1]        # x2
```

The padding value is stored in the attributes:

```cpp
attrs->padding = padding;
attrs->out_dtype = out_dtype.value_or(DataType::Void());
```

So the legalizer must read:

```python
call.attrs.padding
call.attrs.out_dtype
```

There is no `call.args[2]` unless the op is explicitly registered as a
three-input op and the third input is passed as a Relax expression.
