"""Bounded MXFP4 W13 same-row M1/M8 witness, derived from the pinned v2 order probe.

Synthetic operation evidence only: cannot identify the first divergent model layer.
W2, dense FP8, reductions across devices and full-model state are outside scope.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import time

BASE_PROBE_SHA256 = '23deadbe773b27106afd1dcb4c6d73ac6219c6ae9373fdd8e4ea316defe3baf1'


def block_size(m, local=48, global_experts=384, topk=8):
    estimated_m = math.ceil(m * local / global_experts)
    for size in (8, 16, 32, 48, 64):
        if estimated_m * topk / local / size < 0.9:
            return size
    return size


def reverse_legal_regions(ids, experts, count, block, valid_count):
    """Reverse only valid entries inside each expert region; padding never moves."""
    result = list(ids)
    start = 0
    while start < count:
        expert = experts[start // block]
        end = start + block
        while end < count and experts[end // block] == expert:
            end += block
        offsets = [i for i in range(start, end) if ids[i] < valid_count]
        for offset, value in zip(offsets, reversed([ids[i] for i in offsets])):
            result[offset] = value
        start = end
    return result


def validate_layout(ids, experts, count, block, routes, mapping):
    seen = []
    if count % block or count > len(ids):
        raise ValueError("invalid active extent")
    for start in range(0, count, block):
        expert = experts[start // block]
        pad_seen = False
        for route in ids[start:start + block]:
            if route >= len(routes):
                pad_seen = True
                continue
            if route < 0 or pad_seen:
                raise ValueError("invalid route or padding before a valid block entry")
            if mapping[routes[route]] != expert or expert < 0:
                raise ValueError("route assigned to wrong expert")
            seen.append(route)
    expected = [i for i, expert in enumerate(routes) if mapping[expert] >= 0]
    if sorted(seen) != expected:
        raise ValueError("missing or duplicate routed rows")


def run(args):
    import torch
    import vllm._custom_ops as ops
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as marlin
    import vllm.model_executor.layers.fused_moe.moe_align_block_size as align_module
    from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_workspace
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import rand_marlin_weight_mxfp4_like
    from vllm.scalar_type import scalar_types

    began = time.monotonic()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    def emit(record):
        record["elapsed_seconds"] = round(time.monotonic() - began, 3)
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        print(line, flush=True)
        with (output / "results.jsonl").open("a") as stream: stream.write(line + "\n")
    def budget():
        if time.monotonic() - began > args.max_seconds:
            raise RuntimeError("diagnostic time budget exhausted")
    guards = [(marlin, "e048731e130ecc3100be0ab4a168777a7697131b41538f9d9b6a73c4f66e14ff"),
              (align_module, "2addb33632b611b4c0df21f2b874dc1124d2fc867785f19f05d4ce33d4b872c6")]
    for module, expected in guards:
        actual = hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("unexpected pinned source: " + module.__name__)
    spec = importlib.util.spec_from_file_location("candidate", args.candidate)
    candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
    canonical = candidate._canonicalize_marlin_moe_token_order
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(args.gpu_memory_limit_gib * 2**30 / total_memory, 1.0))
    torch.manual_seed(20260922)
    torch.cuda.manual_seed_all(20260922)
    E, G, K, N, TOPK = 48, 384, 6144, 4096, 8
    emit({"event": "start", "torch": torch.__version__, "cuda": torch.version.cuda,
          "capability": torch.cuda.get_device_capability(), "selected_operation": "MXFP4 W13 moe_wna16_marlin_gemm",
          "shape": {"local_experts": E, "global_experts": G, "K": K, "N": N, "topk": TOPK},
          "activation_dtype": "bfloat16", "scale_group_size": 32, "use_atomic_add": False,
          "use_fp32_reduce": True, "synthetic_weights": True, "candidate_sha256": hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
          "gpu_allocator_limit_gib": args.gpu_memory_limit_gib})
    # Build sequentially, keeping one dequantized expert for an independent GEMM witness.
    packed, scales, reference_weight = [], [], None
    for expert in range(E):
        budget()
        placeholder = torch.empty((N, K), dtype=torch.bfloat16, device="cuda")
        reference, q, s = rand_marlin_weight_mxfp4_like(placeholder, 32, input_dtype=torch.bfloat16)
        packed.append(q); scales.append(s)
        if expert == 0: reference_weight = reference
        else: del reference
        del placeholder
    weights = torch.stack(packed).contiguous(); weight_scales = torch.stack(scales).contiguous()
    del packed, scales, q, s
    torch.cuda.empty_cache()
    master = torch.randn((1498, K), dtype=torch.bfloat16, device="cuda") / 5
    routes = torch.argsort(torch.rand((1498, G), device="cuda"), dim=1)[:, :TOPK].to(torch.int32).contiguous()
    routes[0] = torch.arange(TOPK, device="cuda", dtype=torch.int32)  # Ensure q1 exercises local experts.
    mapping = torch.full((G,), -1, dtype=torch.int32, device="cuda")
    mapping[:E] = torch.arange(E, dtype=torch.int32, device="cuda")
    workspace = get_marlin_workspace(master.device)
    weight_ref = reference_weight
    reference_weight = None

    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def compare(a, b):
        delta = (a.float() - b.float()).abs()
        return {"byte_equal": bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8))),
                "different_values": int((a != b).sum()), "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
                "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a.float()).clamp_min(1e-12))}

    row0_witnesses = {}
    shared_weights_identity = (weights.data_ptr(), weight_scales.data_ptr())
    for m in (1, 8):
        budget()
        a = master[:m].contiguous(); ids = routes[:m].contiguous()
        bsz = block_size(m)
        top_weights = torch.ones((m, TOPK), device="cuda", dtype=torch.float32)
        route_cpu = ids.flatten().cpu().tolist(); map_cpu = mapping.cpu().tolist()
        valid_rows = (mapping[ids.long()].flatten() >= 0).nonzero().flatten()
        valid_rows_cpu = valid_rows.cpu().tolist()
        assert valid_rows_cpu[:TOPK] == list(range(TOPK)), "row zero must use eight active local experts"
        assert (weights.data_ptr(), weight_scales.data_ptr()) == shared_weights_identity
        row0_inputs = {"activation_sha256": digest(a[0]),
                       "routes_sha256": digest(ids[0]),
                       "topk_weights_sha256": digest(top_weights[0]),
                       "expert_ids": ids[0].cpu().tolist(),
                       "weight_and_scale_storage_shared_across_m": True}
        layouts, outputs = {}, {}
        for repeat in range(3):
            layout = align_module.moe_align_block_size(ids, bsz, G, mapping, ignore_invalid_experts=True)
            sids, eids, count = layout
            count_cpu = int(count.item())
            validate_layout(sids.cpu().tolist(), eids.cpu().tolist(), count_cpu, bsz, route_cpu, map_cpu)
            layouts["native_" + str(repeat)] = layout
            canon = canonical(sids, eids, count, bsz, ids.numel())
            validate_layout(canon.cpu().tolist(), eids.cpu().tolist(), count_cpu, bsz, route_cpu, map_cpu)
            layouts["canonical_" + str(repeat)] = (canon, eids, count)
        sids, eids, count = layouts["native_0"]
        reversed_ids = torch.tensor(reverse_legal_regions(sids.cpu().tolist(), eids.cpu().tolist(), int(count.item()), bsz, ids.numel()), device="cuda", dtype=sids.dtype)
        validate_layout(reversed_ids.cpu().tolist(), eids.cpu().tolist(), int(count.item()), bsz, route_cpu, map_cpu)
        layouts["reversed_legal"] = (reversed_ids, eids, count)
        layouts["reversed_canonical"] = (canonical(reversed_ids, eids, count, bsz, ids.numel()), eids, count)
        for label, (sorted_ids, expert_ids, padded_count) in layouts.items():
            budget()
            result = torch.zeros((m * TOPK, N), device="cuda", dtype=torch.bfloat16)
            result = ops.moe_wna16_marlin_gemm(
                a, result, weights, None, weight_scales, None, None, None, workspace,
                sorted_ids, expert_ids, padded_count, top_weights, moe_block_size=bsz,
                top_k=TOPK, mul_topk_weights=False, b_q_type=scalar_types.float4_e2m1f,
                size_m=m, size_n=N, size_k=K, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
            observed = result[valid_rows].detach().cpu()
            if not bool(torch.isfinite(observed).all()): raise RuntimeError("nonfinite W13 output")
            outputs[label] = observed
            emit({"event": "operation", "m": m, "block_size_m": bsz, "variant": label,
                  "valid_routed_rows": len(valid_rows_cpu), "num_tokens_post_padded": int(padded_count.item()),
                  "active_sorted_ids_sha256": digest(sorted_ids[:int(padded_count.item())]),
                  "output_sha256": digest(observed),
                  "versus_native_0": compare(outputs["native_0"], observed)})
            del result
        row0_witnesses[m] = {"inputs": row0_inputs,
                              "outputs": [outputs["canonical_" + str(i)][:TOPK].clone() for i in range(3)]}
        torch.save({"activation_row0": a[0].detach().cpu(),
                    "routes_row0": ids[0].detach().cpu(),
                    "canonical_outputs_row0": row0_witnesses[m]["outputs"]},
                   output / ("row0-m" + str(m) + ".pt"))
        emit({"event": "same_row_repeat", "operation": "W13", "m": m,
              "inputs": row0_inputs,
              "repeat_comparisons": [compare(row0_witnesses[m]["outputs"][0], item)
                                     for item in row0_witnesses[m]["outputs"][1:]]})
        oracle_rows = [i for i, route in enumerate(valid_rows_cpu) if route_cpu[route] == 0][:16]
        activation_rows = torch.tensor([valid_rows_cpu[i] // TOPK for i in oracle_rows], device="cuda")
        oracle = (a[activation_rows] @ weight_ref).cpu()
        actual = outputs["canonical_0"][oracle_rows]
        relerr = float((actual.float() - oracle.float()).abs().mean() / oracle.float().abs().mean().clamp_min(1e-12))
        emit({"event": "case_summary", "m": m,
              "native_repeat_equal": all(torch.equal(outputs["native_0"], outputs["native_" + str(i)]) for i in (1, 2)),
              "canonical_repeat_equal": all(torch.equal(outputs["canonical_0"], outputs["canonical_" + str(i)]) for i in (1, 2)),
              "canonical_legal_permutation_equal": torch.equal(outputs["canonical_0"], outputs["reversed_canonical"]),
              "native_legal_permutation_change": compare(outputs["native_0"], outputs["reversed_legal"]),
              "reference": {"expert": 0, "sample_rows": len(oracle_rows), "relative_mean_error_vs_dequantized_bf16": relerr}})
        if relerr > 0.05: raise RuntimeError("W13 numerical witness exceeds reference error bound")
        if m in (8, 1498):
            sids, eids, count = layouts["native_0"]
            try:
                warm_stream = torch.cuda.Stream(); warm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm_stream):
                    canonical(sids, eids, count, bsz, ids.numel())
                torch.cuda.current_stream().wait_stream(warm_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = canonical(sids, eids, count, bsz, ids.numel())
                graph.replay(); first = graph_output.clone()
                graph.replay(); torch.cuda.synchronize()
                repeats = 20
                eager_start, eager_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                eager_start.record()
                for _ in range(repeats): canonical(sids, eids, count, bsz, ids.numel())
                eager_end.record(); eager_end.synchronize()
                graph_start, graph_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                graph_start.record()
                for _ in range(repeats): graph.replay()
                graph_end.record(); graph_end.synchronize()
                emit({"event": "canonical_cuda_graph", "m": m, "success": True,
                      "repeat_equal": torch.equal(first, graph_output),
                      "matches_eager": torch.equal(graph_output, layouts["canonical_0"][0]),
                      "timed_repeats": repeats,
                      "eager_stream_elapsed_ms_per_call": eager_start.elapsed_time(eager_end) / repeats,
                      "graph_stream_elapsed_ms_per_replay": graph_start.elapsed_time(graph_end) / repeats,
                      "latency_scope": "helper only, CUDA stream elapsed time; no model throughput claim"})
            except Exception as error:
                emit({"event": "canonical_cuda_graph", "m": m, "success": False, "error_type": type(error).__name__, "error": str(error)[:500]})
        del outputs, layouts, observed, oracle, actual
    assert row0_witnesses[1]["inputs"] == row0_witnesses[8]["inputs"], "M1/M8 row-zero inputs differ"
    first, second = row0_witnesses[1]["outputs"][0], row0_witnesses[8]["outputs"][0]
    emit({"event": "cross_shape_same_row", "operation": "W13", "m_pair": [1, 8],
          "base_probe_sha256": BASE_PROBE_SHA256,
          "inputs": row0_witnesses[1]["inputs"],
          "row0_comparison": compare(first, second),
          "per_expert_comparison": [{"global_expert": expert, **compare(first[expert], second[expert])}
                                    for expert in range(TOPK)],
          "scope": "Same synthetic W13 activation row, weights, scales and active expert routes. Shape differences are diagnostic; whole-model attribution is unproven."})
    emit({"event": "complete", "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
          "scope": "synthetic selected-kernel correctness witness; not full-model quality or throughput proof"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=2.0)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({"m": [1, 8], "block_m": [block_size(m) for m in (1, 8)],
                          "K": 6144, "N": 4096, "local_experts": 48, "global_experts": 384,
                          "topk": 8, "weights": "synthetic MXFP4 group32", "activations": "BF16"}, indent=2))
    else:
        run(args)

