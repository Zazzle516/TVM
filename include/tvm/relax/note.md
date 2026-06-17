# TVM Object

这里的文件基本都遵循着 `XXXNode` 和 `XXX` 这样的定义方式

就是因为 TVM 要在内部进行运行状态的管理  所以底层真正的实现是 `XXXNode` 但是暴露给外部的引用是 `XXX`

这个表格是这个文件夹下各个文件的功能

在前端转换阶段最重要的两个文件 `include/tvm/relax/struct_info.h` 和 `include/tvm/relax/expr.h`

分别对应了 TVM 中怎么定义变量的属性和表达式的结构

`StructInfoNode` 也是 `expr` 的一部分   可以把 `expr.h` 看成 root file

| File                             | 功能                                                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `analysis.h`                     | Relax 分析 API：形状相等性判断、静态类型推导、StructInfo 检查、自由变量 / 绑定变量分析、递归分析、well-formedness 检查、layout 建议等。                 |
| `attrs/ccl.h`                    | 集合通信算子的属性结构体，例如 all-reduce、all-gather、scatter 类 collective 操作。                                              |
| `attrs/create.h`                 | 张量创建类算子的属性结构体，例如初始化算子和三角矩阵算子。                                                                               |
| `attrs/datatype.h`               | dtype 相关算子的属性结构体，例如 `astype` 和参数包装。                                                                         |
| `attrs/distributed.h`            | 分布式 Relax 算子的属性结构体。                                                                                         |
| `attrs/image.h`                  | 图像算子的属性结构体，例如 resize 和 grid sample。                                                                         |
| `attrs/index.h`                  | 索引类算子的属性结构体，例如 take 和 strided slice。                                                                        |
| `attrs/linear_algebra.h`         | 线性代数算子的属性结构体，例如 matmul 和 einsum。                                                                            |
| `attrs/manipulate.h`             | 形状、layout、数据变换类算子的属性结构体，例如 concat、split、squeeze、tile、gather、scatter、one-hot 等。                              |
| `attrs/nn.h`                     | 神经网络算子的属性结构体，例如卷积、池化、归一化、激活、attention、padding、loss 等。                                                       |
| `attrs/op.h`                     | 核心 Relax 调用形式的属性结构体，例如 `call_tir`、inplace call 和设备标注。                                                       |
| `attrs/qdq.h`                    | 量化 / 反量化风格算子的属性结构体。                                                                                         |
| `attrs/sampling.h`               | 采样类算子的属性结构体，例如 multinomial-from-uniform。                                                                    |
| `attrs/search.h`                 | 搜索类算子的属性结构体，例如 argmax / argmin 和 bucketize。                                                                 |
| `attrs/sorting.h`                | sort、argsort 和 top-k 的属性结构体。                                                                                |
| `attrs/statistical.h`            | 统计类和 scan 风格 reduction 算子的属性结构体。                                                                            |
| `attrs/vision.h`                 | 视觉算子的属性结构体，例如 NMS、ROI align / pool、valid-count、multibox transform。                                          |
| `backend.h`                      | 后端 lowering pass 的声明，例如 runtime builtin lowering 和 VM shape lowering。                                       |
| `backend/adreno/transform.h`     | Adreno 专用的 Relax transform pass 声明。                                                                         |
| `binding_rewrite.h`              | 用于重写 dataflow block 内部 binding 的对象 / 引用 API。                                                                |
| `block_builder.h`                | `BlockBuilderNode` / `BlockBuilder`；用于命令式构造 Relax function、block、call、binding 和 module 的 API。               |
| `dataflow_matcher.h`             | 使用 dataflow pattern 匹配和重写 Relax dataflow graph 的 API。                                                       |
| `dataflow_pattern.h`             | Relax dataflow-pattern 对象及其约束的对象 / 引用层级。                                                                    |
| `dataflow_pattern_functor.h`     | 用于遍历 dataflow-pattern 对象的 visitor / functor 框架。                                                             |
| `distributed/axis_group_graph.h` | 内部图工具，用于分析分布式 axis group 和 sharding propagation。                                                            |
| `distributed/global_info.h`      | 分布式 global-info 对象，尤其是 `DeviceMeshNode` / `DeviceMesh`。                                                     |
| `distributed/struct_info.h`      | 分布式 StructInfo 对象：placement 规格、placement，以及 `DTensorStructInfo`。                                            |
| `distributed/transform.h`        | 分布式 Relax pass 声明：sharding propagation、local-view lowering、redistribution legalization、DistIR lowering。     |
| `exec_builder.h`                 | `ExecBuilderNode` / `ExecBuilder`；用于构造 VM 可执行指令、常量和函数元数据的 builder。                                          |
| `expr.h`                         | Relax 核心 IR 对象层级：表达式、变量、常量、调用、tuple、binding、block、function、extern func，以及基础 `StructInfo`。                   |
| `expr_functor.h`                 | Relax 表达式的 visitor 和 mutator 框架。                                                                            |
| `nested_msg.h`                   | 用于构造嵌套诊断 / 消息以及 FFI type traits 的模板工具。                                                                      |
| `op_attr_types.h`                | 算子属性回调的类型别名：infer struct info、normalize、validate、legalize、lower builtin、gradients、op pattern kind。          |
| `script/builder/frame.h`         | Relax TVMScript IR builder 使用的对象 / 引用 frame 层级。                                                             |
| `script/builder/ir.h`            | TVMScript Relax builder 的入口点，用于构造 function、参数、block、emit 操作和 match-cast binding。                            |
| `struct_info.h`                  | 具体的 `StructInfo` 层级：object、primitive、shape、tensor、tuple、function struct info，以及获取 / 更新表达式 StructInfo 的辅助函数。 |
| `struct_info_functor.h`          | `StructInfo` 对象的 visitor 和 mutator 框架。                                                                      |
| `tir_pattern.h`                  | Relax codegen 使用的 TIR pattern matching 支持，包括 `MatchResultNode` / `MatchResult`。                             |
| `transform.h`                    | 主要的 Relax transform pass 声明，以及 pattern-fusion 辅助对象。                                                         |
| `type.h`                         | Relax 静态类型对象：shape、tensor、object 和 packed-function 类型。                                                      |
| `utils.h`                        | Relax 杂项工具：binding 替换、符号变量推导、bool / leaf 检查、纯函数检查、函数变量拷贝等。                                                  |



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

对应的大致流程

```sh
torch.export graph
  -> Relax Function
     -> params 表示为 relax::Var
     -> body 表示为 SeqExpr
        -> BindingBlock/DataflowBlock
           -> VarBinding(var, Call(...))
        -> final body Expr
```

For example

```py
torch.export:
  x = aten.add(a, b)
  y = aten.relu(x)
  return y
```

对应在 relax 转换后的结果

```sh
Function(params=[a, b],
  body=SeqExpr(
    blocks=[
      DataflowBlock([
        VarBinding(x, Call(relax.add, [a, b])),
        VarBinding(y, Call(relax.nn.relu, [x]))
      ])
    ],
    body=y
  )
)
```

把图节点 lowering 成 binding / dataflow block 内部的 Relax Call 表达式