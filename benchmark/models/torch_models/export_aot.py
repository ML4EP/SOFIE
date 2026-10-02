#!/usr/bin/env python3
"""
Compiles a PyTorch AOTInductor .pt2 package per dynamic-family shape point

Usage:
    python3 benchmark/models/torch_models/export_aot.py [family ...]

With no arguments, exports every point for every available family. Skips
any family whose torch_models/<family>.py
fails to import — e.g. mambav2 (see torch_models/mambav2.py).
"""

import sys
import traceback
from pathlib import Path

import torch
import torch._inductor

MODEL_DIR = Path(__file__).resolve().parent.parent
AOT_DIR = MODEL_DIR / "aot_models"

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Mirrors benchmark/CMakeLists.txt's sofie_benchmark_sweep_n_e/_mlpf/_transformer.
_N_E_SWEEP = [(n, n * 5) for n in (100, 300, 1000, 3000, 10000, 30000, 100000)]
_MLPF_SWEEP = [128, 256, 512, 1024, 2048, 4096]
_TRANSFORMER_D32_SWEEP = [10, 20, 30, 40, 50, 60]
_TRANSFORMER_D64_SWEEP = [10, 20, 30, 40, 50, 60, 64, 128, 256, 512, 1024, 2048]

# (family module name, [(model_name_suffix, shape_overrides), ...])
FAMILIES = {
    "gnn_h32_k2": [(f"_n{n}_e{e}", {"n_nodes": n, "n_edges": e}) for n, e in _N_E_SWEEP],
    "gnn_h64_k4": [(f"_n{n}_e{e}", {"n_nodes": n, "n_edges": e}) for n, e in _N_E_SWEEP],
    "gnn_large": [(f"_n{n}_e{e}", {"n_nodes": n, "n_edges": e}) for n, e in _N_E_SWEEP],
    "punet_h32_k2_heads4_layers2": [(f"_n{n}_e{e}", {"n_nodes": n, "n_edges": e}) for n, e in _N_E_SWEEP],
    "punet_h64_k4_heads4_layers2": [(f"_n{n}_e{e}", {"n_nodes": n, "n_edges": e}) for n, e in _N_E_SWEEP],
    "mlpf_fp32_fused": [(f"_b1_n{n}", {"num_batch": 1, "num_elements": n}) for n in _MLPF_SWEEP],
    "mlpf_fp32_unfused": [(f"_b1_n{n}", {"num_batch": 1, "num_elements": n}) for n in _MLPF_SWEEP],
    "transformer_d32_h2_L6_ff32": [(f"_n{s}_s{s}", {"n_nodes": s, "seq_length": s}) for s in _TRANSFORMER_D32_SWEEP],
    "transformer_d64_h4_L6_ff128": [(f"_n{s}_s{s}", {"n_nodes": s, "seq_length": s}) for s in _TRANSFORMER_D64_SWEEP],
}


def export_family(family_name, points):
    try:
        module = __import__(family_name)
    except Exception as exc:
        print(f"[export_aot] SKIPPING {family_name}: {exc}")
        return

    AOT_DIR.mkdir(exist_ok=True)

    varying_dim_names = set()
    seen_values = {}
    max_value_by_dim_name = {}
    for _, shape_overrides in points:
        for dim_name, value in shape_overrides.items():
            if dim_name in seen_values and seen_values[dim_name] != value:
                varying_dim_names.add(dim_name)
            seen_values[dim_name] = value
            max_value_by_dim_name[dim_name] = max(max_value_by_dim_name.get(dim_name, value), value)

    for suffix, shape_overrides in points:
        model_name = f"{family_name}{suffix}"
        out_path = AOT_DIR / f"{model_name}.pt2"
        if out_path.exists():
            print(f"[export_aot] {model_name}: already exists, skipping")
            continue

        model, example_inputs = module.build(shape_overrides)

        model = model.to("cuda")
        example_inputs = tuple(t.to("cuda") for t in example_inputs)

        dims_by_name = {}
        dynamic_shapes = []
        for input_tensor, (input_name, axis_map) in zip(example_inputs, module.DYNAMIC_DIMS.items()):
            spec = {}
            for axis, dim_name in axis_map.items():
                if dim_name not in varying_dim_names:
                    continue
                if dim_name not in dims_by_name:
                    dims_by_name[dim_name] = torch.export.Dim(
                        dim_name, min=1, max=max_value_by_dim_name[dim_name]
                    )
                spec[axis] = dims_by_name[dim_name]
            dynamic_shapes.append(spec)

        try:
            with torch.inference_mode():
                exported = torch.export.export(model, example_inputs, dynamic_shapes=tuple(dynamic_shapes))
                torch._inductor.aoti_compile_and_package(exported, package_path=str(out_path))
            print(f"[export_aot] {model_name}: wrote {out_path}")
        except Exception:  # noqa: BLE001 - one shape failing shouldn't stop the rest of the sweep
            print(f"[export_aot] {model_name}: FAILED\n{traceback.format_exc()}")


def main():
    requested = sys.argv[1:] or list(FAMILIES.keys())
    for family_name in requested:
        if family_name not in FAMILIES:
            print(f"[export_aot] Unknown family: {family_name} (known: {', '.join(FAMILIES)})")
            continue
        export_family(family_name, FAMILIES[family_name])


if __name__ == "__main__":
    main()
