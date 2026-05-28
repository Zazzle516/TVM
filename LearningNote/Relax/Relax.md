# Relax

最外层是 function  function 里面有若干个 block   block 里面有一条条 binding

- Global Context Management

- Scope Management

- Normalization

Scope is BlockBuilder State: tracks what symbolic shape variables are valid while building or normalizing a function body

global context is the module-level environment, scope is the builder’s lexical/shape-inference environment, and blocks are the actual IR containers receiving bindings

Normalization sits across all three: it reads global context, uses scope for well-formed struct info, and emits normalized bindings into the current block

```sh
Relax Function
└── SeqExpr
    ├── BindingBlock(DataflowBlock)
    │   ├── Binding
    │   ├── Binding
    │   └── Binding
    └── final output Expr
```

function 是一个图级别的函数

Binding Block: 储存 Binding 的普通容器

Q: Binding 在 Binding Block 容器内外能访问的环境有什么不同  有没有基于 Binding Block 的变量生命

A: Binding Block 中的变量在 Function 中全部可见

Q: function 和 

```sh
IRModule scope
└── Function scope
    ├── BindingBlock scope
    │   ├── 普通 Var
    │   └── 普通 Var
    │
    ├── DataflowBlock scope
    │   ├── DataflowVar / 内部临时 Var
    │   ├── DataflowVar / 内部临时 Var
    │   └── output Var
    │
    └── return expression
```

```sh
GlobalVar:
    整个 IRModule 可见

Function parameter:
    整个 function body 可见

BindingBlock 中定义的普通 Var:
    当前 function 后续位置可见

DataflowBlock 中定义的内部变量:
    当前 dataflow block 内可见

DataflowBlock 中 R.output 的变量:
    当前 dataflow block 外的后续 function body 可见
```

BindingBlock:
    普通顺序绑定块。
    里面定义的 Var 是 function-local，
    通常可以被当前 function 后续 block / return 使用。

DataflowBlock:
    特殊绑定块。
    内部的 DataflowVar 默认只在该 DataflowBlock 内可见。
    只有通过 R.output(...) 导出的变量，才能在 block 外继续使用。
    DataflowBlock 是有作用域和优化边界意义的划分，而不是直接的内存分配边界。
    它让编译器更容易判断哪些中间值不会逃逸，从而为后续内存规划和算子融合提供信息。

内存生命周期:
    不由 BindingBlock / DataflowBlock 直接决定。
    真正的运行时内存生命周期要看最后一次使用、lowering、memory planning、storage reuse 等后续分析。

Binding: Block 中一条最基本的计算语句

Dataflow Block: 中间变量默认是局部的，只有通过 R.output(...) 标记的变量才能逃出这个 block

比如一般的 binding 计算出的中间值

Q: function 是一整个模型计算图吗  一个 IRModule 中可以有多个 function 吗

整个 relax 中可以有多个 Module 吗

A: 可以的  一个 function 表示子图  可以嵌套和拼接

一个 IRModule 里面可以同时放多个 Relax function 和 TIR PrimFunc

高层图 IR 和低层 kernel IR 可以共存在同一个 module 里

```sh
IRModule
├── main: Relax Function          # 可能是整个模型入口
├── encode: Relax Function        # 可能是子图/辅助函数
├── decode: Relax Function        # 可能是另一个入口或子图
├── fused_matmul_add: PrimFunc    # TIR kernel
└── fused_softmax: PrimFunc       # TIR kernel
```

Q: emit() 函数是什么意思

A: 把一个 Relax 表达式变成当前 block 里的一条有名字的 binding，并返回这个变量，供后续计算引用

生成一条 binding 语句

| API                         | 作用                      | 生成什么                      |
| --------------------------- | ----------------------- | ------------------------- |
| `bb.emit(expr)`             | 生成普通中间变量                | `lv = expr`               |
| `bb.emit_output(expr)`      | 生成 dataflow block 的输出变量 | `gv = expr; R.output(gv)` |
| `bb.emit_func_output(expr)` | 结束整个 function           | `return expr`             |



```sh
Relax Function
└── SeqExpr
    ├── BindingBlock
    │   ├── Binding
    │   └── Binding
    ├── DataflowBlock
    │   ├── Binding
    │   ├── Binding
    │   ├── Binding
    │   └── R.output(...)
    └── final output expression
```

Q: 那比如算子融合这种优化的话，必须是发生在 Dataflow Block 内部的吗  不能是 Binding Block 吗

因为 Dataflow 的特性  融合优化通常发生在这里  并且很少跨 Block 进行优化

融合的单位是 bindingOp  所以 Binding Block 也可以

不过也可以去手动融合  只要依赖关系分析清楚  怎么融合都可以


```py
bb = relax.BlockBuilder()

with bb.function("main", params):
    with bb.dataflow():
        lv = bb.emit(R.astype(input_ids, dtype="int32"))
        lv1 = bb.emit(R.reshape(lv, R.shape([16])))
        ...
        lv179 = bb.emit(R.reshape(lv178, R.shape([1, 16, 1024])))

        gv = bb.emit_output(lv179)

    bb.emit_func_output(gv)
```

Q: 全局层级怎么表示

A: 只有被 GlobalVar 声明的才是真正的全局

## Ragion

Q: 转换层级是如何定义的

A: 本身是由 Importer 决定

```sh
torch.export ExportedProgram / GraphModule
    ↓
一个 Relax IRModule
    ↓
一个入口 Relax Function，通常叫 main
    ↓
main 里面一个或多个 DataflowBlock / BindingBlock
    ↓
每个 torch graph node 转成一条或多条 Relax binding
```


```sh
模型 forward / exported graph
    → Relax Function

图中的 node
    → Relax Binding

图中的 tensor value
    → Relax Var / DataflowVar

图中的最终 output
    → R.output(...) + return

图中的 parameter / buffer
    → Relax function 参数或 Constant

后续 lowered kernel
    → TIR PrimFunc
```

## 转换层级

Q: 因为 relax.function 表示一个计算图  计算图是图优化的一个重要标准

在从 torch 导出到 TVM 的时候  这个边界是怎么定义的

究竟是一个图整体被转换到 function  还是单个算子被转换到 function  还是子图

Currently, during TVM's frontend conversion, a Torch computation graph is typically converted into a function. If there are subgraphs, each subgraph is also converted into a corresponding function.

What I would like to understand is: during frontend conversion, what determines the boundary of a function? Is this boundary defined by Torch itself, or is it introduced by TVM during the conversion process?

```sh
torch.export 决定“有哪些 FX graph / subgraph”。

TVM frontend importer 决定“这些 FX graph / subgraph 要不要变成 Relax function”。

在你这段代码路径里：
    top-level ExportedProgram graph
        → 一个 Relax function: main

    torch.cond 这种显式 FX subgraph
        → 额外的 Relax function，例如 cond_true_branch / cond_false_branch

    普通 torch op
        → main 里的 binding，不是 function
```


main is the top-level Relax function boundary created by this Torch frontend

在 `local_test/create_multi_func.py` 示例中确实也可以创建多个 function 但是也是只能有一个 main  并且其他 function 也是从 `main` 强制修改名称来的

- Separate EncoderOnly and DecoderOnly exports.

- Renaming imported main functions to encode and decode.

- Merging them into one IRModule.

- Building a wrapper Relax main with BlockBuilder


在 Relax 层级  Shape 是 first-class-value  不只是 Tensor 的一个属性

每个 Relax 表达式都有且只有一个 `StructInfo`  也就是它的类型

所以，只要某个东西可以被写成 Relax 表达式，它就需要一个对应的 `StructInfo` 子类

```sh
StructInfo
├─ TensorStructInfo   # Tensor value 的类型
├─ ShapeStructInfo    # Shape value 的类型
├─ PrimStructInfo     # 标量 prim value 的类型  例如 R.Prim
├─ TupleStructInfo    # Tuple value 的类型
└─ FuncStructInfo     # callable 的类型
```

Shape 是一个运行时 Relax value，它可能携带符号化的 TIR 级别 pattern

因为它是一个 value，所以它需要一个 StructInfo，也就是 ShapeStructInfo

因为同一个 shape 也可能出现在 Tensor 的类型中，所以 TensorStructInfo 通过它的 Optional<Expr> shape 字段复用这套 shape 机制

这个字段的值被约束为一个 Relax 表达式，并且该表达式的 struct_info 是 ShapeStructInfo

# Params 是怎么传递的

```sh
# python/tvm/relax/block_builder.py
_enter_function_scope() => self.begin_scope(func_scope._params)

class BlockBuilder.function()
```