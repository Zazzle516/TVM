# PrimExpr

```sh
IRModule
  ├── Relax Function       # 高层图 / tensor-level 程序
  │     └── call_tir(...)
  │
  └── TIR PrimFunc         # 低层 tensor kernel / loop program
        ├── Stmt           # 语句：For, Block, IfThenElse, BufferStore...
        └── PrimExpr       # 表达式：i + j, n * 4, A[i], i < 128...
```