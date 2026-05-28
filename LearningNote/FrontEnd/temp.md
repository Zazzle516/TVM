The difference is only for `named_parameters()`.

So in your demo:

```text
Parameters / weights:
  tok_embeddings.weight
  lm_head.weight
  attn weights
  ffn weights
  norm weights

With keep_params_as_input=True:
  kept as function inputs, then extracted by detach_params

Buffers:
  rope_cos
  rope_sin
  causal_mask

Always:
  bound into the module as constants
```

The short reason: `keep_params_as_input=True` matches the later `detach_params` + `vm["main"](tvm_input, *tvm_params)` flow. It keeps the compiled model structure separate from the learned weight values.


# BaseFXGraphImporter

This file defines `BaseFXGraphImporter`, an abstract base class that translates a PyTorch FX graph into a TVM Relax IR program, one FX node at a time.

Concrete subclasses (e.g., `FXGraphImporter`, `ExportedProgramImporter`) inherit it and provide the actual `create_convert_map()` mapping from PyTorch ops → converter methods defined here.






## 2. Utilities (lines 49–196)

- _convert_torch_tensor_to_relax (104) — turns a PyTorch param tensor into a relax.const.
- shape_of (110) — returns shape from a Relax TensorStructInfo or PyTorch tensor.
- _promote_common_dtype (123) — applies PyTorch type-promotion rules for binary ops.
- retrieve_args / _retrieve_args (169) — recursively resolves a node's args (which may be FX nodes, lists, tuples, dicts, or constants) into Relax exprs by looking each up in self.env.
- _check_unsupported_func_type (188) — pre-flight check that every call_function node has a converter.

## 4. Common patterns

- Layout normalization — pool/conv converters expand a missing batch dim then squeeze it back (e.g., _avg_pool2d_impl 731–760).
- Bias fusion — conv ops emit the conv, reshape bias to (1, C, 1, …), then add (e.g., _conv2d_impl 993).

- Dtype promotion — _binary_op (476) inspects both TensorStructInfos, promotes via _promote_common_dtype, inserts astype casts as needed before calling
  the Relax op.
- In-place semantics — ops like _copy_ (2294), _inplace_fill (2394), _inplace_masked_fill (2439), _zeros_inplace (2703), and the index_put_ branch in _index_put (1925–1941) update self.env[node.args[0]] so later reads of the original FX node see the mutated tensor.
- Tuple-returning ops — _max_poolNd_with_indices (1401/1417/1433), _sort (2198), _topk (2264), _unbind (1554) emit a relax.Tuple; downstream _getitem then extracts elements via relax.TupleGetItem.
- Higher-order helpers — _unary_op, _binary_op, _tril_triu, _argmax_argmin are factories that return a closure, letting the subclass register the same generic converter against many op targets.
  
## 5. The abstract hook (lines 2713–2717)

```py
@abc.abstractmethod
def create_convert_map(self) -> dict[op_target, Callable[[fx.Node], relax.Var]]: ...
```

Subclasses must build the dispatch table that pairs PyTorch ops (e.g., `torch.nn.Conv2d`, `aten.add.Tensor`, `operator.getitem`) with the converters here.

That table is what turns this base class into a working translator.


## How it fits into full_pipeline.py

When you call from_fx / from_exported_program:

1. The driver builds an fx.GraphModule and walks its nodes in topological order.

2. placeholder nodes become relax.Var function inputs and get registered in self.env.

3. get_attr nodes (parameters/buffers) become relax.const (or function params) via _convert_torch_tensor_to_relax / self.params.

4. call_function / call_method / call_module nodes look up self.convert_map[target] and invoke the converter — that's where everything in this file runs. The returned relax.Var is stored back into `self.env[node]`.

5. output node packages the final self.env[...] values into a Relax function return.

# Questions

Q: 用字符串去映射数据类型吗

_convert_torch_tensor_to_relax() -> _convert_data_type()    Q: 为什么要包裹一层

Q: 为什么 _convert_data_type for binary ops     这里对应的 torch 数据类型提升是什么  利用 torch promote 映射回来

Q: 为什么会有这个功能

Q: fx 的 6 中 op 类型是怎么一一对应到 tvm 的




# Translation

  6. Why retrieve_args exists

  Look at this Node's args from your log:

  cat: ... = torch.ops.aten.cat.default([neg, slice_1], -1)
              └─ args = ([Node("neg"), Node("slice_1")], -1)

  args[0] is a Python list containing fx.Nodes, and args[1] is a plain Python int. Some Nodes have nested structures (lists, tuples,
  dicts) mixing fx.Nodes with constants. That's why _retrieve_args (line 172 in the base file) is a recursive walker:

  def _retrieve_args(self, node):
      if isinstance(node, fx.Node): return self.env[node]   # replace Node with Relax expr
      if isinstance(node, tuple):    return tuple(self._retrieve_args(x) for x in node)
      if isinstance(node, list):     return [self._retrieve_args(x) for x in node]
      if isinstance(node, dict):     return {k: self._retrieve_args(v) for k,v in node.items()}
      if node is None:               return None
      return node                            # plain int/float/str passes through

  It mirrors the exact shape of node.args/node.kwargs, replacing every fx.Node with the Relax expression it stands for. Without this,
  every converter would have to repeat the same recursion.

  7. The dispatch table mirrors the IR's typing

  self.convert_map is keyed by node.target:

  {
      torch.ops.aten.linear.default      : self._linear,
      torch.ops.aten.mul.Tensor          : self._binary_op(relax.op.multiply, operator.mul),
      torch.ops.aten.embedding.default   : self._embedding,
      torch.ops.aten.softmax.int         : self._softmax,
      "reshape"                          : self._reshape,    # for call_method
      torch.nn.Linear                    : self._linear_module,  # for call_module
      ...
  }

  Lookup is O(1). The reason converters take node (not unpacked args) is that they need access to node.kwargs, node.meta, and the
  original op overload name — which the dispatcher couldn't generically unpack.

  8. Putting it on top of your run.log

  For the embedding line:

  embedding: "f32[1, 16, 128]" = torch.ops.aten.embedding.default(p_tok_embeddings_weight, input_ids)
      torch.ops.aten.embedding.default   : self._embedding,
      torch.ops.aten.softmax.int         : self._softmax,
      "reshape"                          : self._reshape,    # for call_method
      torch.nn.Linear                    : self._linear_module,  # for call_module
      ...
  }

  Lookup is O(1). The reason converters take node (not unpacked args) is that they need access to node.kwargs, node.meta, and the
  original op overload name — which the dispatcher couldn't generically unpack.

  8. Putting it on top of your run.log

  For the embedding line:

  embedding: "f32[1, 16, 128]" = torch.ops.aten.embedding.default(p_tok_embeddings_weight, input_ids)

  Translation step-by-step:

  1. Walk the linked list, reach this Node. node.op == "call_function", node.target == aten.embedding.default.
  2. convert_map[aten.embedding.default] returns self._embedding.
  3. Inside _embedding (calls _embedding_impl, line 1191): it does self.env[node.args[0]] and self.env[node.args[1]] to get the Relax
  Vars previously bound for p_tok_embeddings_weight (a placeholder lifted to a Relax param) and input_ids (another placeholder).
  4. It emits relax.op.take(weight, indices_int32, axis=0), gets back a fresh relax.Var.
  5. self.env[node] = that_var.
  6. Loop continues. The next Node (mul) will find its inputs by looking up self.env[Node("embedding")] twice — and that's why
  aten.mul.Tensor(embedding, embedding) becomes relax.op.multiply(v_embedding, v_embedding).

  9. The transformation mechanism in one sentence

  ▎ FX gives you a flat, SSA-like DAG of Nodes whose args reference earlier Nodes; TVM's translator walks this DAG once, maintaining an
  ▎ fx.Node → relax.Expr map (self.env), dispatching each Node through convert_map to a per-op handler that emits Relax via
  ▎ block_builder.

  Everything else in base_fx_graph_translator.py — the dtype promotion, the bias reshape for conv, the layout dance for adaptive pools —
  is just the inevitable busywork that comes from PyTorch and Relax having slightly different op signatures. The skeleton is pure
  DAG-walking.



## 1. Why params/inputs are separated

In legacy `torch.fx.symbolic_trace`, parameters live as `get_attr` nodes inside the graph — the graph reaches into `self`.

In `torch.export`, the graph is functionalized: every parameter, buffer, and user input is lifted to be a placeholder argument of `forward(...)`.

The graph signature tells you the role of each placeholder:

```text
p_tok_embeddings_weight: PARAMETER target='tok_embeddings.weight'   ← prefix p_
b_layers_0_attn_rope_cos: BUFFER  target='layers.0.attn.rope_cos'   ← prefix b_
input_ids: USER_INPUT                                                ← real input
```

The translator uses this signature to decide what each placeholder becomes in Relax.

In the subclass that handles `ExportedProgram` (`exported_program_translator.py`, not this base file), you'll see logic that walks `exported_program.graph_signature` and:

- For each `PARAMETER` → store the underlying tensor in `self.params` and bind it to a Relax Var, or inline as `relax.const`.
- For each `BUFFER` → same treatment as parameters.
- For each `USER_INPUT` → becomes a real Relax function input Var.

After this lift step, `self.env[placeholder_node]` holds a `relax.Expr` for every argument, and the rest of the file we discussed simply consumes `self.env` per node.


Q: 代码有两个 IR Module `include/tvm/ir/module.h` 然后在 tvm.py 端有多个 IR Module

它们是什么关系