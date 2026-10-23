"""
Loading script for the model families, whose
architecture source lives in the TICL-GNN-Trackster-Linking repo
(https://github.com/cms-patatrack/TICL-GNN-Trackster-Linking).

"""

import os
import sys
import types
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper

TICL_REPO = Path(os.environ.get("SOFIE_TICL_REPO", "~/TICL-GNN-Trackster-Linking")).expanduser()

_path_ready = False


def _ensure_ticl_on_path():
    global _path_ready
    if _path_ready:
        return
    if not TICL_REPO.exists():
        raise RuntimeError(
            f"TICL-GNN-Trackster-Linking not found at {TICL_REPO}. Configure "
            f"the benchmark with -DSOFIE_BENCHMARK_AOT=ON once to clone it "
            f"automatically, "
            f"or set SOFIE_TICL_REPO to point at an existing checkout."
        )
    sys.path.insert(0, str(TICL_REPO / "tracksterLinker"))

    dummy = types.ModuleType("tracksterLinker.datasets.GNNDataset")
    dummy.GNNDataset = object
    sys.modules["tracksterLinker.datasets.GNNDataset"] = dummy

    _path_ready = True


def load_onnx_initializers(onnx_path):
    """{name: torch.Tensor}, read from the .onnx file's own weights."""
    model = onnx.load(str(onnx_path), load_external_data=True)
    return {
        init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
        for init in model.graph.initializer
    }


def fix_anonymous_linear_weights(torch_model, initializers, onnx_path):
    """
    Some nn.Linear weights in the transformer/PUNet .onnx export lose their
    parameter name in the graph
    """
    import onnx

    state_dict_keys = set(torch_model.state_dict().keys())
    graph = onnx.load(str(onnx_path)).graph
    renamed = 0
    for node in graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2:
            continue
        weight_name = node.input[1]
        if weight_name not in initializers:
            continue

        segments = node.name.strip("/").split("/")[:-1]
        candidates = [".".join(segments) + ".weight"]
        if segments:
            candidates.append(segments[-1] + ".weight")
        candidate = next((c for c in candidates if c in state_dict_keys and c not in initializers), None)
        if candidate is not None:
            initializers[candidate] = initializers.pop(weight_name).T.contiguous()
            renamed += 1
    if renamed:
        print(f"  [ticl] {onnx_path}: recovered {renamed} anonymized Linear weight name(s) from the graph")
    return initializers


def fix_anonymous_linear_weights_by_bias_topology(torch_model, initializers, onnx_path):
    """
    Same problem as fix_anonymous_linear_weights but for a graph whose node
    names are also anonymous
    """
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    node_by_output = {}
    for node in graph.node:
        for out in node.output:
            node_by_output[out] = node
    nodes_by_input = {}
    for node in graph.node:
        for inp in node.input:
            nodes_by_input.setdefault(inp, []).append(node)

    state_dict_keys = set(torch_model.state_dict().keys())
    renamed = 0
    for bias_name in list(initializers.keys()):
        if not bias_name.endswith(".bias"):
            continue
        weight_name = bias_name[: -len(".bias")] + ".weight"
        if weight_name not in state_dict_keys or weight_name in initializers:
            continue

        add_nodes = [n for n in nodes_by_input.get(bias_name, []) if n.op_type == "Add"]
        if len(add_nodes) != 1:
            continue
        add_node = add_nodes[0]
        other_input = next((i for i in add_node.input if i != bias_name), None)
        matmul_node = node_by_output.get(other_input)
        if matmul_node is None or matmul_node.op_type != "MatMul":
            continue

        anon_weight_name = next((i for i in matmul_node.input if i in initializers), None)
        if anon_weight_name is None:
            continue

        initializers[weight_name] = initializers.pop(anon_weight_name).T.contiguous()
        renamed += 1

    if renamed:
        print(f"  [ticl] {onnx_path}: recovered {renamed} anonymized Linear weight name(s) via bias topology")
    return initializers


def fix_anonymous_attention_weights_by_position(torch_model, initializers, original_by_name, onnx_path, num_decoder_layers):
    """
    Recovers decoder_layers.{i}.{self_attn,cross_attn}.{W_q,W_k,W_v,W_o}.weight
    """
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    weighted_matmuls = []
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        weight_inputs = [i for i in node.input if i in original_by_name]
        if len(weight_inputs) == 1:
            weighted_matmuls.append((node, weight_inputs[0]))

    order = ("self_attn.W_q", "self_attn.W_k", "self_attn.W_v", "self_attn.W_o",
             "cross_attn.W_q", "cross_attn.W_k", "cross_attn.W_v", "cross_attn.W_o")

    renamed = 0
    for i in range(num_decoder_layers):
        fc1_tensor = initializers.get(f"decoder_layers.{i}.feed_forward.fc1.weight")
        if fc1_tensor is None:
            continue
        fc1_idx = next(
            (idx for idx, (_, wname) in enumerate(weighted_matmuls)
             if original_by_name[wname].shape == fc1_tensor.T.shape
             and torch.equal(original_by_name[wname], fc1_tensor.T)),
            None,
        )
        if fc1_idx is None or fc1_idx < len(order):
            continue

        for offset, suffix in enumerate(order):
            _, anon_name = weighted_matmuls[fc1_idx - len(order) + offset]
            if anon_name not in initializers:
                continue
            target = f"decoder_layers.{i}.{suffix}.weight"
            if target not in torch_model.state_dict() or target in initializers:
                continue
            initializers[target] = initializers.pop(anon_name).T.contiguous()
            renamed += 1

    if renamed:
        print(f"  [ticl] {onnx_path}: recovered {renamed} anonymized attention weight name(s) by position")
    return initializers


def load_weights_strict(torch_model, initializers, allow_missing_prefixes=(), allow_missing_exact=(), ignore_unexpected=False):
    """
    Loads `initializers` into `torch_model` and raises if anything doesn't
    match exactly.
    """
    missing, unexpected = torch_model.load_state_dict(initializers, strict=False)
    missing = [m for m in missing
               if not any(m.startswith(p) for p in allow_missing_prefixes)
               and m not in allow_missing_exact]

    if ignore_unexpected and not missing:
        unexpected = []
    if missing or unexpected:
        raise RuntimeError(
            f"Weight loading mismatch: missing={missing}, unexpected={unexpected}. "
            "The constructed module's architecture doesn't match this .onnx file — "
            "check the dimensions parsed from the model name/onnx shapes."
        )
    model_state_dict = torch_model.state_dict()
    for name, tensor in initializers.items():
        if name not in model_state_dict:
            continue  # an ignored `unexpected` key (ignore_unexpected=True) — never loaded, nothing to compare
        model_tensor = model_state_dict[name]
        if not torch.equal(model_tensor, tensor.to(dtype=model_tensor.dtype)):
            raise RuntimeError(f"Value mismatch after loading {name} — weights did not load correctly.")
    return torch_model


def import_gnn_track_linking_net():
    _ensure_ticl_on_path()
    from tracksterLinker.GNN.TrackLinkingNet import GNN_TrackLinkingNet

    return GNN_TrackLinkingNet


def _patch_encoder_layer_for_export():
    from tracksterLinker.transformer.Transformer import EncoderLayer

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=mask)
        x = x + self.dropout(attn_output)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        x = self.final_norm(x)
        return x

    EncoderLayer.forward = forward


def import_punet():
    _ensure_ticl_on_path()
    _patch_encoder_layer_for_export()
    from tracksterLinker.multiGNN.PUNet import PUNet

    return PUNet


def import_transformer():
    _ensure_ticl_on_path()
    from tracksterLinker.transformer.Transformer import Transformer

    return Transformer
