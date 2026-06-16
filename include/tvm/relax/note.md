# TVM Object

这里的文件基本都遵循着 `XXXNode` 和 `XXX` 这样的定义方式

就是因为 TVM 要在内部进行运行状态的管理  所以底层真正的实现是 `XXXNode` 但是暴露给外部的引用是 `XXX`

这个表格是这个文件夹下各个文件的功能

在前端转换阶段最重要的两个文件 `include/tvm/relax/struct_info.h` 和 `include/tvm/relax/expr.h`

分别对应了 TVM 中怎么定义变量的属性和表达式的结构

`StructInfoNode` 也是 `expr` 的一部分   可以把 `expr.h` 看成 root file

| File | Function |
|---|---|
| `analysis.h` | Relax analysis APIs: shape equality, static type derivation, struct-info checks, free/bound vars, recursion, well-formedness, layout suggestions. |
| `attrs/ccl.h` | Attribute structs for collective communication ops such as all-reduce, all-gather, scatter collectives. |
| `attrs/create.h` | Attribute structs for tensor creation ops such as init and triangular matrix ops. |
| `attrs/datatype.h` | Attribute structs for dtype-related ops such as `astype` and parameter wrapping. |
| `attrs/distributed.h` | Attribute structs for distributed Relax operators. |
| `attrs/image.h` | Attribute structs for image ops such as resize and grid sample. |
| `attrs/index.h` | Attribute structs for indexing ops such as take and strided slice. |
| `attrs/linear_algebra.h` | Attribute structs for linear algebra ops such as matmul and einsum. |
| `attrs/manipulate.h` | Attribute structs for shape/layout/data manipulation ops: concat, split, squeeze, tile, gather, scatter, one-hot, etc. |
| `attrs/nn.h` | Attribute structs for neural-network ops: convolutions, pooling, normalization, activations, attention, padding, loss, etc. |
| `attrs/op.h` | Attribute structs for core Relax call forms such as `call_tir`, inplace calls, and device annotations. |
| `attrs/qdq.h` | Attribute structs for quantize/dequantize style ops. |
| `attrs/sampling.h` | Attribute structs for sampling ops such as multinomial-from-uniform. |
| `attrs/search.h` | Attribute structs for search ops such as argmax/argmin and bucketize. |
| `attrs/sorting.h` | Attribute structs for sort, argsort, and top-k. |
| `attrs/statistical.h` | Attribute structs for statistical and scan-style reductions. |
| `attrs/vision.h` | Attribute structs for vision ops such as NMS, ROI align/pool, valid-count, multibox transform. |
| `backend.h` | Backend lowering pass declarations, e.g. runtime builtin lowering and VM shape lowering. |
| `backend/adreno/transform.h` | Adreno-specific Relax transform pass declarations. |
| `binding_rewrite.h` | Object/ref API for rewriting bindings inside dataflow blocks. |
| `block_builder.h` | `BlockBuilderNode` / `BlockBuilder`; imperative construction API for Relax functions, blocks, calls, bindings, and modules. |
| `dataflow_matcher.h` | APIs for matching and rewriting Relax dataflow graphs using dataflow patterns. |
| `dataflow_pattern.h` | Object/ref hierarchy for Relax dataflow-pattern objects and constraints. |
| `dataflow_pattern_functor.h` | Visitor/functor framework for traversing dataflow-pattern objects. |
| `distributed/axis_group_graph.h` | Internal graph utilities for reasoning about distributed axis groups and sharding propagation. |
| `distributed/global_info.h` | Distributed global-info objects, especially `DeviceMeshNode` / `DeviceMesh`. |
| `distributed/struct_info.h` | Distributed struct-info objects: placement specs, placements, and `DTensorStructInfo`. |
| `distributed/transform.h` | Distributed Relax pass declarations: sharding propagation, local-view lowering, redistribution legalization, DistIR lowering. |
| `exec_builder.h` | `ExecBuilderNode` / `ExecBuilder`; builder for VM executable instructions, constants, and function metadata. |
| `expr.h` | Core Relax IR object hierarchy: expressions, vars, constants, calls, tuples, bindings, blocks, functions, extern funcs, and base `StructInfo`. |
| `expr_functor.h` | Visitor and mutator framework for Relax expressions. |
| `nested_msg.h` | Template utility for nested diagnostic/message construction and FFI type traits. |
| `op_attr_types.h` | Operator attribute callback type aliases: infer struct info, normalize, validate, legalize, lower builtin, gradients, op pattern kind. |
| `script/builder/frame.h` | Object/ref frame hierarchy used by Relax TVMScript IR builder. |
| `script/builder/ir.h` | TVMScript Relax builder entry points for functions, args, blocks, emits, and match-cast bindings. |
| `struct_info.h` | Concrete `StructInfo` hierarchy: object, primitive, shape, tensor, tuple, function struct info, plus helpers to get/update expr struct info. |
| `struct_info_functor.h` | Visitor and mutator framework for `StructInfo` objects. |
| `tir_pattern.h` | TIR pattern matching support for Relax codegen, including `MatchResultNode` / `MatchResult`. |
| `transform.h` | Main Relax transform pass declarations and pattern-fusion helper objects. |
| `type.h` | Relax static type objects: shape, tensor, object, and packed-function types. |
| `utils.h` | Miscellaneous Relax utilities: binding substitution, symbolic var inference, bool/leaf checks, purity checks, function variable copying. |


```sh
ffi::Object
├─ IdNode
├─ StructInfoNode
├─ BindingNode
│  ├─ MatchCastNode
│  └─ VarBindingNode
├─ BindingBlockNode
│  └─ DataflowBlockNode
└─ ExprNode / RelaxExprNode
   ├─ CallNode
   ├─ TupleNode
   ├─ TupleGetItemNode
   ├─ LeafExprNode
   │  ├─ ShapeExprNode
   │  ├─ VarNode
   │  │  └─ DataflowVarNode
   │  ├─ ConstantNode
   │  ├─ PrimValueNode
   │  ├─ StringImmNode
   │  └─ DataTypeImmNode
   ├─ SeqExprNode
   └─ IfNode

BaseFuncNode
├─ FunctionNode
└─ ExternFuncNode
```