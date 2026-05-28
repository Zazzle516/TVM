# Notes about FX.Node and Graph

TVM 前端支持多种模型表示方式  这篇笔记是针对 torch 的模型表达方式

在用 torch 描述一个模型后  使用 torch 本身提供的 `torch.export.export()` 把模型代码导出为 `torch IR fx` 表示

TVM 的前端再去处理 `fx` 表达

那么为了理解 TVM 的转换  首先要对 `torch.fx` 的输出结构有基本了解

# Output

首先 `torch.export` 有这几个重要的部分

```sh
ExportedProgram
├── graph_module
│   ├── graph           # DAG
│   └── module state    # Tensor value but for `export mode` is none
├── graph_signature
├── state_dict          # Tensor value
├── range_constraints   # Dynamic shape constraint
├── constants
├── module_call_graph   # 原始 module 调用结构信息  Q: ???
├── example_inputs
└── verifier / dialect / metadata
```

目前 `full_pipeline.py` 对应的导出结果在 `run.log`

从打印出的结果上主要分为了三部分 `class GraphModule(torch.nn.Module)`, `Graph signature` 和 `Range constraints`


# GraphModule

FX is a flat list of Nodes forming a DAG

```sh
GraphModule
├── graph ── node  # fx.Graph (A doubly-linked list of Nodes)
├── module state
│   ├── _parameter
│   └── _buffer
```

The `fx.Graph` is an ordered sequence of fx.Node objects. But there still are branches, eg. `ExportedProgramImporter._cond()`

Every intermediate value gets its own Node.

It's essentially SSA: each Node produces one FX value(might be tuple valued node), named once, used by other Nodes.

**Q: 这其中 parameter 和 buffer 的区别是什么**

A: 
Parameter = 注册为模型可训练参数的 Tensor

Buffer = 固定的预计算值  模型状态的一部分  不会用来训练

```sh
p_tok_embeddings_weight: PARAMETER target='tok_embeddings.weight'
p_layers_0_attn_wq_weight: PARAMETER target='layers.0.attn.wq.weight'
p_norm_weight: PARAMETER target='norm.weight'
p_lm_head_weight: PARAMETER target='lm_head.weight'

# 不参与梯度计算  不被优化更新
b_layers_0_attn_rope_cos: BUFFER target='layers.0.attn.rope_cos'
b_layers_0_attn_rope_sin: BUFFER target='layers.0.attn.rope_sin'
b_layers_0_attn_causal_mask: BUFFER target='layers.0.attn.causal_mask'
```

### For each `fx.node`

Take `linear: "f32[1, 16, 128]" = torch.ops.aten.linear.default(mul_2, p_layers_0_attn_wq_weight)` for example

| Field  | Value                                                   | What it means                                    |
|--------|---------------------------------------------------------|--------------------------------------------------|
| name   | `"linear"`                                             | Unique SSA name                                  |
| op     | `"call_function"`                                      | One of 6 categories (see below)                  |
| target | `torch.ops.aten.linear.default`                         | The thing to call                                |
| args   | `(Node("mul_2"), Node("p_layers_0_attn_wq_weight"))`   | Inputs — references to other Nodes, not values   |
| kwargs | `{}`                                                    | Keyword inputs                                   |
| meta   | `{"val": FakeTensor(shape=[1,16,128], dtype=f32), ...}` | Shape/dtype info                                 |
|

Note that no args contain real Tensor values, they are all just reference

### For each `op` in `fx.node`

| op              | target is…                                             | Meaning                                              |
|-----------------|--------------------------------------------------------|------------------------------------------------------|
| `placeholder`   | a string name                                          | Function argument (`input_ids`, `p_*`, `b_*`)        |
| `get_attr`      | attribute path string                                  | Read `self.foo.bar` (legacy trace only, not export)  |
| `call_function` | a callable (`aten.linear.default`, `operator.add`)     | Free function call                                   |
| `call_method`   | method name string (`"reshape"`)                       | `args[0].method(*args[1:])`                          |
| `call_module`   | submodule path string                                  | Invoke a child `nn.Module`                           |
| `output`        | `"output"`                                             | Return node — `args[0]` is a tuple of return Nodes   |

### QA

**Q: Node 中的重要属性有什么**

A: op 是 Node 的操作类别，target 是这个类别下的具体目标；call_function 是 op 的一种取值，表示“这个 Node 调用一个普通函数 / aten op”，而真正决定它调用 linear、mul、embedding 还是 softmax 的，是 node.target

**Q: 从哪看到具体的 node 内容**

A: `.venv/lib/python3.10/site-packages/torch/fx/node.py`

**Q: 这个 `call_function`, `call_method`, `call_module` 这三者的关系是什么**

A: call_function 是 torch 本身提供的一些计算函数调用  aten 系列  名称是很特殊的 `torch.ops.aten.add.default`

而 call_method 记录的是字符串 调用自定义的对象的方法 method 名称和函数名保持一致

**Q: 在代码明明就是 `x.reshape()` 这种调用对象的方法  但是 fx 结果中却变成了 `call_function()`**

A: 首先它最开始确实是 call_method 记录的  在 FX 中用字符串 target 记录方法名，语义上等价于 `args[0].method(*args[1:]);` 

后续如果走 `interpreter/codegen`，就按这个方法名调用；如果走 `export/decomposition/lowering`，并且该方法被规范化到 aten，那么它会变成具体的 `torch.ops.aten.xxx callable`

在 lowering 的过程中  torch 会尽可能把 call_method 转换到 call_function

call_module: 调用子模块  target: 从 Model 最外层访问到自己的路径

**Q: 为什么 `call_module` 明明是调用的子模块  但是还是和 `call_method` 和 `call_function` 平级**

A: 从程序执行的角度  都是一条 fx.IR 语句产生一个 SSA 结果  在编译的角度确实是平级的

但是 call_module 相比而言确实更上层

**Q: 既然 `call_module` 相比更上层  那么 torch 在 lowering 的时候  为什么不把 `call_module` lower 到 `call_function`**

A: 是可以的  首先 fx 的目标并不是把模型完全转换到底层的结构  它希望保留模型的高层信息  否则传递到后续的编译处理时  这个信息就消失了

而且在 module 内部可能有 module state 内部环境

而目前的 run.log 中已经存在着 lowering 后的示例  `linear: "f32[1, 16, 128]" = torch.ops.aten.linear.default` 对应 `full_pipeline.py` 中的 `CausalSelfAttention` 中的第一个 `linear` 调用  它理论上也应该是 `call_module` 的

此时参数和 buffer 也被 lifting 成了 placeholder

Summary: `call_module` 确实大多数情况下可以进一步 lowering，但普通 FX 不会总是立刻这么做，因为它要保留 `nn.Module` 的结构、参数、buffer、类型和边界，方便做高层 graph rewrite；而 `torch.export` 这种面向编译器的流程会更积极地把 `call_module` 展开成 `aten call_function`

**Q: 这种导出方式和 TVM 的关系**

A: 如果把 TVM 作为编译器后端的话  那么这种转换层级很合适

而如果想用 TVM 进行高层图优化的话  就不适合  因为失去了高层的图信息  但是对于算子级别的融合就很方便

```sh
高层 FX:
    node.op     = "call_module"
    node.target = "layers.0.attn.wq"
    含义：调用这个子模块

lowering / export 后:
    node.op     = "call_function"
    node.target = torch.ops.aten.linear.default
    含义：调用具体 aten linear 算子
```

**Q: kwargs 是什么  和 args 是什么关系**

A: kwargs 只针对有值的参数

```py
torch.clamp(x, min=0.0, max=1.0)
x.to(dtype=torch.float32)

# after torch.export
node.args = (x,)
node.kwargs = {
    "min": 0.0,
    "max": 1.0,
}

node.args = (x, )
node.kwargs = {
    "dtype": torch.float32,
}
```

args 和 kwargs 只是记录这次 FX Node 调用时的参数形式  经过 torch.export 规范化后，很多原本的 kwargs 可能会被转成 args

**Q: 真正的参数在哪  `Graph signature` 吗  和 kwargs 有关系吗**

A: 没有  kwargs 只局限在一个 Node 内部  而 Graph Signature 是针对整个图的输入输出

# Graph Signature

Graph Signature 是针对整个模型计算图输入输出的说明    Signature 有主要三种类型 Parameter, Buffer, User Input

`.venv/lib/python3.12/site-packages/torch/export/graph_signature.py`

在用 torch 写模型代码的时候  一般情况下只有一个输入 User Input(Prompt)  但是模型中还有很多隐式的参数 `Embedding Weights`, `attention.weights`

```py
class Model(nn.Module):
    def __init__(self):
        # Parameter
        self.tok_embeddings = nn.Embedding(...)
        # Buffer
        self.rope_cos = ...
        self.rope_sin = ...

    # User Input
    def forward(self, input_ids):
        ...
```

在使用 `torch.export` 导出的时候  会访问到这些隐式参数把它们显式的记录到 `Graph Signature` 中

这些参数是通过 `Input Lifting` 才写入到 `Graph Sigature` 中  对应的 `op` 属性被更新为 `placeholder`

### QA

**Q: 怎么判断一个隐式参数是否被 input lifting 到显式参数**

A: 根据 `node.op` 和 `graph signature` 一起判断

**case1**: `node.op` == `placeholder`

此时说明它是 `graph` 的函数参数 Parameter, Buffer, User Input 具体这个参数是怎么来的  要结合 `Graph Signature`

```sh
placeholder + graph_signature.PARAMETER => input lifting.Parameter

placeholder + graph_signature.BUFFER => input lifting.Buffer

placeholder + graph_signature.USER_INPUT => Input
```

**case2**: `node.op` == `get_attr`

说明这是运行时读取的参数  该 Tensor 值存在 `GraphModule.module_state` 中

并不会被 Graph Signature 记录

**Q: 什么决定了一个参数是储存在 `GraphModule.module_state` 中还是转成 `GraphSignature` 中的静态映射**

A: 首先一个参数是既可以存在 `GraphModule.module_state` 中  也可以选择转成 `GraphSignature` 中的静态映射

区别在于你的转换方式  理论上任何值都可以进行 `input lifting`

如果只是导出到 GraphModule  那么 Tensor Value 就会被挂到 `GraphModule.module_state` 中

如果导出到 `fx.IR`  那么 torch 编译器更倾向于进行 `input lifting` 所以此时的结果是一个序列化的图

更适合 TVM 去处理

**Q: 既然已经有 `Graph Signature`  还需要存储 `args` 和 `kwargs` 吗**

A: 需要的  描述的是单个 node 和其他 node 的输入输出连接关系


# State Dict

存储参数名称到真实 Tensor 引用的字典

**Q: 其中的 `GraphModule.module_state` 和 `State_Dict` 是什么关系**

A:
```sh
graph_module.graph
    只描述计算
    例如：embedding = aten.embedding(p_tok_embeddings_weight, input_ids)

graph_signature
    描述 placeholder 的身份
    例如：p_tok_embeddings_weight 是 PARAMETER target='tok_embeddings.weight'

state_dict
    保存真实 Tensor
    例如：state_dict["tok_embeddings.weight"] = 原始 embedding 权重
```


# Connected to TVM

Every node has a `relax.Expr` in `self.env`, and the function body is a flat linked list of `block_builder.emit`

> Tip: except for `output node` and `branch node`
> the `output_node` don't have a `relax.Expr` in `self.env`, `branch.node` use `get_attr`

Will become a single Relax `DataflowBlock`

> Tip: Not for `branch.node`