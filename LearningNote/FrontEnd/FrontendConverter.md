# TVM Converter

After `LearningNote/TorchFXGraph.md` right now TVM step in and convert the GraphNode into TVM.relax

## Self Define (Plugin)

抛开 `add`, `mul` 等等这种 TVM 本身提供的基础算子转换

TVM 也支持用户在多个层级进行自定义 `TVM.relax`, `TIR Function`, `Runtime Call`

参考例子 `eg.local_test/custom_converter.py`

但需要同时提供该算子对应的 `Converter Func` (转换到 `TVM-IR` 的结果)

以 `dict` 的形式传入 `class ExportedProgramImporter.from_exported_program()`

```py
# dict[str, Callable]
custom_convert_map={
    "add.Tensor": custom_add_converter,
    "mul.Tensor": custom_mul_converter,
}
```

TVM 本身不负责处理用户自定义的计算逻辑  只负责执行用户定义的 `converter` 函数  

```sh
torch.export graph node
    name: "add.Tensor"
        |
        v
TVM lookup in convert_map (python/tvm/relax/frontend/torch/exported_program_translator.py)
        |
        v
call custom_add_converter(node, importer)
        |
        v
converter emits Relax IR
```

**Q: 怎么保证用户插入的算子的前后连接**

A: 首先在算子定义的 torch 端  本身就包含了算子的前后关系

在执行 `torch.export` 的时候  如果当前自定义算子的参数是其他算子  那么这个 argument 在转换后作为 Node 存在  而输出同理  该自定义算子也会被其他算子调用

```py
# python/tvm/relax/frontend/torch/exported_program_translator.py
def _translate_fx_graph():
    ...
    if func_name in custom_ops:
        self.env[node] = self.convert_map[func_name](node, self)
    ...
```

TVM Converter 会严格按照 torch 的执行流程来自动转换  以此保证算子的前后连接

```py
# python/tvm/relax/frontend/torch/exported_program_translator.py
nodes: list[fx.Node] = exported_program.graph.nodes
```

**Q: 和 tilelang 的关系**

A: 这里的 plugin 机制和 tilelang 无关  tilelang 是 kernel 级别的

**Q: 为什么在 TVM 层面还会调用 `run_ep_decomposition=True` 这种 torch 的设置**

A: `run_ep_decomposition` 参数的功能

Indicating whether to run Torch's decomposition on the exported program before translation. 

High-level operators will be decomposed into their constituent parts.

首先当前的层级是 TVM Converter 此时 torch 的状态仍然存在于 `exported_program`  所以仍然可以调用

```py
exported_program = exported_program.run_decompositions()
```

而执行流程如下  try to decompose supported high-level ATen ops into lower/core ATen ops

而如果 `custom-defined op` 是这些 high-level ATenOp 组成的  那么可能会被拆分而你自定义的 convert func 失效  所以需要 `set run_ep_decomposition=False`

或者为 decomposition 后出现的 op-key 注册 converter

```sh
PyTorch nn.Module
    |
    v
torch.export.export(...)
    |
    v
ExportedProgram(still PyTorch FX/ATen graph)
    |
    v
exported_program.run_decompositions()
PyTorch rewrite into simpler/core ATen ops
    |
    v
TVM walks graph nodes
    |
    v
TVM Relax IR
```

# Convert Pipeline

## Converter Struc

The driver loops over FX nodes, looks up the converter in `convert_map`, calls it, and stashes the resulting `relax.Var` in `self.env[node]`.

Later nodes that reference this node retrieve its translated value through `self.env`.

```sh
self.env         : dict[fx.Node, relax.Expr]   # FX node → translated Relax Expr
self.params      : dict[torch.Tensor, relax.Expr]  # PyTorch param tensor → Relax Var/const
self.block_builder : relax.BlockBuilder        # Emits Relax IR (DataflowBlock)
self.convert_map : dict[op_target, callable]   # op dispatcher, filled by subclass
```

## Input Converter

注意在导出的 `graph_module` 中所有参数经过 lifting 后都作为输入的 placeholder

在 `GraphSignature` 中的四种类型 (Parameter, Buffer, Constant, UserInput) 对于 `graph_module` 是不可见的

而 TVM 要根据 `graph_module` 提供的图结构进行转换  所以首先要根据 `GraphSignature` 提供的信息进行参数处理

```sh
1. create_input_vars
    Create a map to relax.Var
    USER_INPUT(runtime) vs parameters / buffers / constants (static)

2. convert_data_type: 
    Use the mapping from torchVar.type to relaxVar.type

3. merge_input_vars: 
   Merge the previously categorized parameters together and define the input order
   (USER_INPUT, [parameters / buffers / constants])

3. _translate_fx_graph
   Use input_vars to create bindings
```

**Q: 为什么在 `update_inputs_var` 中顺序要这么定义**

A: 为了和 `keep_params_as_input=True` 保持一致

如果 `keep_params_as_input=True`  那么 `named parameters` 会保留为 Relax function input

buffers/constants 仍然会被 BindParams 绑定进 IRModule

如果 `keep_params_as_input=False`  那么 parameters/buffers/constants 都会被绑定进 IRModule  main 最终只需要 user inputs


也要和后续在 vm ABI 保持一致

```sh
# keep_params_as_input=True
vm["main"](tvm_input, *tvm_params)
# input_ids, weight_0, weight_1, weight_2, ...

# keep_params_as_input=False
vm["main"](tvm_input)
```

**Q: 这个参数的意义是区别训练和推理吗   TVM 有支持训练吗**

A: No  训练还需要 gradient、optimizer、state update 等

只是 TVM 提供的一种选择  和是否是训练推理无关

# Rest

`python/tvm/relax/frontend/torch/exported_program_translator.py`

基本可以这样理解，但要稍微修正一句：

不是“把 lifted 参数绑定到**每个对应的 Node 上**”，而是：

**把每个 lifted 参数对应的 `placeholder fx.Node` 绑定到它对应的 `relax.Var` 上。**

也就是说，input lifting 做的是：

```text
原来模型内部的 parameter / buffer / constant
        ↓
被提升成整个 FX graph 的输入
        ↓
在 FX graph 里表现为 placeholder node
        ↓
在 Relax 函数签名里表现为 relax.Var
```

但是这里有两套表示：

```text
Relax 函数参数表：
  "p_linear_weight"  →  relax.Var

FX graph 内部引用：
  fx.Node(p_linear_weight placeholder)  → 被其他 fx.Node.args 引用
```

所以还需要这一句桥接：

```python
self.env[node] = inputs_vars[node.name]
```

它的意思是：

```text
这个 FX placeholder node
  对应 Relax 里的这个函数参数 Var
```

之后如果有一个计算节点：

```text
call_function linear(x, p_linear_weight, p_linear_bias)
```

它的参数不是直接写字符串 `"p_linear_weight"`，而是引用 FX graph 里的 placeholder node：

```text
node.args = [
  fx.Node(x),
  fx.Node(p_linear_weight),
  fx.Node(p_linear_bias),
]
```

converter 在翻译这个 `call_function` 时，会通过：

```python
self.env[node.args[i]]
```

找到对应的 Relax 表达式。

所以更准确的说法是：

**被 lifting 的参数已经成为整个模型计算图的输入；但在 FX graph 内部，它们仍然以 `placeholder node` 的形式被其他节点引用。因此翻译器需要把这些 placeholder node 映射到 Relax 函数参数 `relax.Var`，让后续节点翻译时能通过 `self.env` 查到它们。**

一句话总结：

**input lifting 解决“参数属于函数签名”的问题；`self.env` 映射解决“FX 节点引用如何找到对应 Relax 值”的问题。**


从“按名字索引的参数表”到“按 FX node 索引的 SSA 表”的一次性桥接。后续所有 converter 都会通过 self.env[node.args[i]] 或 retrieve_args 来读取这个 SSA 表

input lifting 把参数放进了函数签名；而 _translate_fx_graph 里处理 placeholder 的分支，则把这个参数放进了后续遍历所使用的 SSA 符号表