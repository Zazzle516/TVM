"""Create a Relax IRModule with multiple frontend-created functions.

This file demonstrates the important boundary rule:

1. Exporting a normal PyTorch module with Python methods such as ``encode`` and
   ``decode`` still gives the TVM Torch frontend one top-level Relax ``main``.
   The Python method calls are traced through and flattened.
2. To get Relax functions named ``encode`` and ``decode``, create explicit
   frontend boundaries by exporting those Torch submodules separately, rename
   each imported ``main``, merge the modules, and build a wrapper Relax
   ``main`` that calls them.

Run from the TVM repo root, for example:

    PYTHONPATH=/home/zazzle/TVM/python \
    LD_LIBRARY_PATH=/home/zazzle/TVM/build/lib:/home/zazzle/TVM/build/llvm-root/usr/lib/llvm-18/lib:/home/zazzle/TVM/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/lib/wsl/lib \
    .venv/bin/python local_test/create_multi_func.py
"""

import torch
import torch.nn as nn

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


BATCH_SIZE = 2
INPUT_DIM = 16
LATENT_DIM = 4


class AutoEncoder(nn.Module):
    """One Torch module with Python-level encode/decode methods."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(INPUT_DIM, LATENT_DIM)
        self.decoder = nn.Linear(LATENT_DIM, INPUT_DIM)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        return self.decode(self.encode(x))


class EncoderOnly(nn.Module):
    """A separate export boundary for the encoder."""

    def __init__(self, model: AutoEncoder):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, x):
        return torch.relu(self.encoder(x))


class DecoderOnly(nn.Module):
    """A separate export boundary for the decoder."""

    def __init__(self, model: AutoEncoder):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, z):
        return torch.sigmoid(self.decoder(z))


def export_to_relax(model: nn.Module, example_args: tuple[torch.Tensor, ...]) -> tvm.IRModule:
    """Export one Torch module and import it as a Relax IRModule.

    The Torch frontend names the imported top-level Relax function ``main``.
    ``keep_params_as_input=True`` keeps model weights as explicit Relax function
    parameters, which makes the composed wrapper show all call edges clearly.
    """

    model.eval()
    with torch.no_grad():
        exported = torch.export.export(model, example_args)

    return from_exported_program(
        exported,
        keep_params_as_input=True,
        unwrap_unit_return_tuple=True,
    )


def rename_imported_main(mod: tvm.IRModule, new_name: str) -> tvm.IRModule:
    """Rename the frontend-created ``main`` function to ``new_name``."""

    renamed = mod.replace_global_vars({"main": new_name})
    renamed[new_name] = renamed[new_name].with_attr("global_symbol", new_name)
    return renamed


def merge_modules(*mods: tvm.IRModule) -> tvm.IRModule:
    """Merge imported modules into one IRModule."""

    merged = tvm.IRModule()
    for mod in mods:
        merged.update(mod)
    return merged


def build_composed_main(mod: tvm.IRModule) -> tvm.IRModule:
    """Add ``main(x) = decode(encode(x))`` to a module with encode/decode."""

    encode_gv = mod.get_global_var("encode")
    decode_gv = mod.get_global_var("decode")
    encode_func = mod[encode_gv]
    decode_func = mod[decode_gv]

    def clone_param(param: relax.Var, name_hint: str) -> relax.Var:
        return relax.Var(name_hint, param.struct_info)

    x = clone_param(encode_func.params[0], "x")
    encode_params = [
        clone_param(param, f"encode_{param.name_hint}") for param in encode_func.params[1:]
    ]
    decode_params = [
        clone_param(param, f"decode_{param.name_hint}") for param in decode_func.params[1:]
    ]
    main_params = [x] + encode_params + decode_params

    bb = relax.BlockBuilder(mod)
    with bb.function("main", params=main_params):
        with bb.dataflow():
            z = bb.emit(relax.Call(encode_gv, [x, *encode_params]), name_hint="z")
            y = bb.emit(relax.Call(decode_gv, [z, *decode_params]), name_hint="y")
            out = bb.emit_output(y)
        bb.emit_func_output(out)

    return bb.get()


def function_names(mod: tvm.IRModule) -> list[str]:
    return sorted(gv.name_hint for gv in mod.get_global_vars())


def main():
    torch.manual_seed(0)

    model = AutoEncoder()
    x = torch.randn(BATCH_SIZE, INPUT_DIM, dtype=torch.float32)
    z = torch.randn(BATCH_SIZE, LATENT_DIM, dtype=torch.float32)

    whole_mod = export_to_relax(model, (x,))
    print("Single AutoEncoder export:")
    print("  Relax functions:", function_names(whole_mod))
    print("  Note: Python methods encode/decode were flattened into main.\n")

    encode_mod = rename_imported_main(export_to_relax(EncoderOnly(model), (x,)), "encode")
    decode_mod = rename_imported_main(export_to_relax(DecoderOnly(model), (z,)), "decode")

    composed_mod = merge_modules(encode_mod, decode_mod)
    composed_mod = build_composed_main(composed_mod)

    print("Separate EncoderOnly/DecoderOnly exports, then merged:")
    print("  Relax functions:", function_names(composed_mod))
    print()
    print(composed_mod.script(show_meta=False))


if __name__ == "__main__":
    main()
