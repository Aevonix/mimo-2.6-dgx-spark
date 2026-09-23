#!/usr/bin/env python3
"""Two synthetic candidate GEMMs. No model, TP, inference, or runtime patch."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import statistics
from mimo_stable_dense import StableDenseWitness

PINNED_VLLM = "0961bbae2894d574be790d219651824eb199318e"
PINNED_MODEL = "MiMo V2.6 Pro RL geometry; synthetic tensors only, no checkpoint loaded"
CASES = (
    {"name": "attention_o_proj_pre_tp", "k": 2048, "n": 6144,
     "operation": "torch.nn.functional.linear(x, weight, bias=None)",
     "output_dtype": "bfloat16", "seed": 20260922},
    {"name": "router", "k": 6144, "n": 384,
     "operation": "torch.mm(x, weight.T, out_dtype=torch.float32)",
     "output_dtype": "float32", "seed": 20260923},
)


def digest(tensor):
    # numpy ships in the target image; no package installation is needed.
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(actual, reference):
    a, b = actual.to(torch.float64), reference.to(torch.float64)
    delta = a - b
    absolute = delta.abs()
    denom = float(torch.linalg.vector_norm(b))
    same_dtype = actual.dtype == reference.dtype
    max_index = int(absolute.reshape(-1).argmax())
    return {
        "byte_equal": digest(actual) == digest(reference) if same_dtype else None,
        "actual_dtype": str(actual.dtype), "reference_dtype": str(reference.dtype),
        "actual_sha256": digest(actual), "reference_sha256": digest(reference),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "unequal_elements": int(torch.count_nonzero(delta)),
        "elements": actual.numel(), "max_abs": float(absolute.max()),
        "rel_l2": float(torch.linalg.vector_norm(delta)) / max(denom, 1e-300),
        "max_abs_flat_index": max_index,
        "actual_at_max_abs": float(a.reshape(-1)[max_index]),
        "reference_at_max_abs": float(b.reshape(-1)[max_index]),
    }


def snapshot(output):
    torch.cuda.synchronize()
    return output.detach().cpu().clone()


def repeat_summary(outputs):
    pairs = [compare(output[0], outputs[0][0]) for output in outputs[1:]]
    full_hashes = [digest(output) for output in outputs]
    return {"first_row_all_byte_equal": all(p["byte_equal"] for p in pairs),
            "full_output_all_byte_equal": len(set(full_hashes)) == 1,
            "full_output_sha256": full_hashes,
            "first_row_against_repeat0": pairs}


def run_case(case, repeats):
    generator = torch.Generator(device="cpu").manual_seed(case["seed"])
    x_cpu = torch.randn((8, case["k"]), generator=generator).to(torch.bfloat16)
    weight_cpu = (torch.randn((case["n"], case["k"]), generator=generator) * 0.02).to(torch.bfloat16)
    # CPU FP32 uses the SAME rounded BF16 operands, one reference row for both M.
    # No TF32, BF16 reduction, or cuBLAS policy is changed for the tested kernels.
    reference = torch.nn.functional.linear(x_cpu[:1].float(), weight_cpu.float())[0]
    reference64 = torch.nn.functional.linear(x_cpu[:1].double(), weight_cpu.double())[0]
    # Preserve the original x8 and weights byte-for-byte. Extra rows use
    # an independent generator so the prior weight stream cannot move.
    extra_generator = torch.Generator(device="cpu").manual_seed(case["seed"] + 1000)
    extra = torch.randn((1490, case["k"]), generator=extra_generator).to(torch.bfloat16)
    master_cpu = torch.cat((x_cpu, extra), dim=0)
    master, weight = master_cpu.cuda(), weight_cpu.cuda()
    inputs = {m: master[:m] for m in (1, 8, 1498)}
    assert inputs[1].data_ptr() == inputs[8].data_ptr()
    assert torch.equal(inputs[1].cpu(), inputs[8][:1].cpu())

    candidate = StableDenseWitness(case["name"])

    def original_operation(m):
        if case["name"] == "router":
            return torch.mm(inputs[m], weight.T, out_dtype=torch.float32)
        return torch.nn.functional.linear(inputs[m], weight, bias=None)

    def operation(m):
        return candidate(inputs[m], weight)

    for _ in range(3):
        for m in (1, 8, 1498):
            operation(m)
    torch.cuda.synchronize()
    eager = {m: [] for m in inputs}
    for repeat in range(repeats):
        for m in (tuple(inputs) if repeat % 2 == 0 else tuple(reversed(inputs))):
            eager[m].append(snapshot(operation(m)))

    original_rows = {m: snapshot(original_operation(m))[0] for m in inputs}

    # Each graph contains exactly one GEMM; inputs and weight have stable storage.
    graphs, graph_outputs = {}, {}
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            for m in (1, 8, 1498):
                operation(m)
    capture_stream.synchronize()
    for m in (1, 8, 1498):
        graphs[m] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[m], stream=capture_stream):
            graph_outputs[m] = operation(m)
    captured = {m: [] for m in inputs}
    for repeat in range(repeats):
        for m in (tuple(inputs) if repeat % 2 == 0 else tuple(reversed(inputs))):
            graphs[m].replay()
            captured[m].append(snapshot(graph_outputs[m]))

    def event_median_ms(fn, m):
        for _ in range(5):
            fn(m)
        torch.cuda.synchronize()
        timings = []
        for _ in range(11):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            measured = fn(m)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        return {"median_ms": statistics.median(timings), "samples_ms": timings,
                "effective_tflops": 2 * m * case["n"] * case["k"] / (statistics.median(timings) * 1e9),
                "warmup_calls": 5, "timed_calls": 11,
                "measurement": "warm CUDA events, eager standalone operation, descriptive only"}

    timings = {m: {"original": event_median_ms(original_operation, m),
                   "candidate": event_median_ms(operation, m)} for m in (1, 8, 1498)}
    result = {
        **case, "scope": "synthetic_single_gpu_no_tp_reduction",
        "operation": "fixed-config pinned vLLM matmul_kernel_persistent",
        "candidate": candidate.describe(), "timings": timings,
        "activation_generation": "CPU seeded FP32 N(0,1), rounded to BF16",
        "weight_generation": "CPU seeded FP32 N(0,0.02^2), rounded to BF16",
        "weight_shape": list(weight.shape), "weight_stride": list(weight.stride()),
        "weight_sha256": digest(weight_cpu), "activation_m8_sha256": digest(x_cpu),
        "first_input_row_sha256": digest(x_cpu[0]),
        "row_counts": [1, 8, 1498],
        "master_input_sha256": digest(master_cpu),
        "same_first_row_storage": len({x.data_ptr() for x in inputs.values()}) == 1,
        "input_strides": {m: list(x.stride()) for m, x in inputs.items()},
        "reference": "CPU F.linear of BF16 operands converted to FP32; one row; not exact arithmetic",
        "reference64": "CPU F.linear of identical BF16 operands converted to FP64; row0 only",
        "cpu_fp32_vs_fp64_reference": compare(reference, reference64),
        "original_against_fp64_reference": {m: compare(original_rows[m], reference64) for m in inputs},
        "candidate_against_fp64_reference": {m: compare(eager[m][0][0], reference64) for m in inputs},
        "original_against_fp64_rounded_output": {m: compare(original_rows[m], reference64.to(original_rows[m].dtype)) for m in inputs},
        "candidate_against_fp64_rounded_output": {m: compare(eager[m][0][0], reference64.to(eager[m][0].dtype)) for m in inputs},
        "eager_cross_m_first_row": compare(eager[1][0][0], eager[8][0][0]),
        "graph_cross_m_first_row": compare(captured[1][0][0], captured[8][0][0]),
        "eager_m1_vs_m1498_first_row": compare(eager[1][0][0], eager[1498][0][0]),
        "graph_m1_vs_m1498_first_row": compare(captured[1][0][0], captured[1498][0][0]),
        "eager_m8_vs_m1498_first_row": compare(eager[8][0][0], eager[1498][0][0]),
        "graph_m8_vs_m1498_first_row": compare(captured[8][0][0], captured[1498][0][0]),
        "eager_repeats": {m: repeat_summary(eager[m]) for m in (1, 8, 1498)},
        "graph_repeats": {m: repeat_summary(captured[m]) for m in (1, 8, 1498)},
        "eager_vs_graph_first_row": {m: compare(eager[m][0][0], captured[m][0][0]) for m in (1, 8, 1498)},
        "against_fp32_reference": {
            mode: {m: compare(rows[m][0][0], reference) for m in (1, 8, 1498)}
            for mode, rows in (("eager", eager), ("graph", captured))
        },
    }
    if case["name"] == "attention_o_proj_pre_tp":
        result["against_bf16_rounded_fp32_reference"] = {
            mode: {m: compare(rows[m][0][0], reference.to(torch.bfloat16)) for m in (1, 8, 1498)}
            for mode, rows in (("eager", eager), ("graph", captured))
        }
    if case["name"] == "router":
        def route_info(logits):
            # Same FP32 sigmoid scoring for every source. Actual checkpoint
            # correction bias is not loaded; this witness uses zero bias.
            scores = torch.sigmoid(logits.to(torch.float32))
            values, ids = torch.topk(scores, 9, sorted=True)
            top8 = ids[:8].tolist()
            return {"ordered_top8": top8, "membership_top8": sorted(top8),
                    "score8": float(values[7]), "score9": float(values[8]),
                    "gap8_vs9": float(values[7] - values[8])}
        reference_route = route_info(reference64.float())
        result["synthetic_router_top8"] = {
            "scoring": "FP32 sigmoid; zero correction bias; no actual model routing claim",
            "fp64_rounded_fp32_reference": reference_route,
            "cpu_fp32_reference": route_info(reference),
            "original": {m: route_info(original_rows[m]) for m in inputs},
            "candidate": {m: route_info(eager[m][0][0]) for m in inputs},
        }
        for key in ("original", "candidate"):
            for info in result["synthetic_router_top8"][key].values():
                info["same_membership_as_fp64_reference"] = info["membership_top8"] == reference_route["membership_top8"]
                info["same_order_as_fp64_reference"] = info["ordered_top8"] == reference_route["ordered_top8"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--describe", action="store_true", help="Print fixed shapes without importing torch or using CUDA")
    parser.add_argument("--repeats", type=int, choices=range(2, 6), default=3)
    parser.add_argument("--max-seconds", type=int, choices=range(1, 181), default=120)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({"pinned_vllm": PINNED_VLLM, "pinned_model": PINNED_MODEL, "cases": CASES, "row_counts": [1, 8, 1498]}, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --describe is selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    with result_path.open("x") as result_file:
        result_file.write("{}\n")
    started = time.monotonic()

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Probe exceeded {args.max_seconds}s wall budget")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_seconds)
    global torch
    import torch
    torch.set_num_threads(1)
    assert str(torch.__version__).startswith("2.13."), torch.__version__
    assert torch.version.cuda == "13.0", torch.version.cuda
    assert torch.cuda.is_available(), "CUDA is required; do not use CPU as the tested backend"
    assert torch.cuda.device_count() == 1, "Expose only one idle diagnostic GPU"
    assert torch.cuda.get_device_capability(0) == (12, 1), "Expected GB10 SM121"
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = None
    report = {
        "schema": "dense-stable-prefill-v2", "status": "running",
        "pinned_vllm": PINNED_VLLM, "pinned_model": PINNED_MODEL,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256((Path(__file__).parent / "mimo_stable_dense.py").read_bytes()).hexdigest(),
        "original_probe_sha256": "9faf40ab6a487a855ce875b8b31583d859521773ac0e8a7485a323d2b60d3f12",
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "vllm_distribution_version": vllm_version,
        "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "environment": {key: os.environ.get(key) for key in (
            "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "VLLM_BATCH_INVARIANT")},
        "repeats_per_shape_per_mode": args.repeats,
        "limitations": ["synthetic inputs and weights", "no model or checkpoint load",
                        "no TP collectives", "no inference", "no claim of first divergent model layer",
                        "isolated candidate witness, no runtime or model patch",
                        "CUDA graphs contain standalone GEMMs, not compiled full-model graphs"],
        "cases": [],
    }
    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        report["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    try:
        with torch.inference_mode():
            for case in CASES:
                report["cases"].append(run_case(case, args.repeats))
                save()
                last = report["cases"][-1]
                print(json.dumps({"case": case["name"],
                                  "eager_cross_m": last["eager_cross_m_first_row"],
                                  "graph_cross_m": last["graph_cross_m_first_row"],
                                  "m1_vs_m1498": last["eager_m1_vs_m1498_first_row"]}), flush=True)
        report["status"] = "complete"
        save()
    except BaseException as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        signal.alarm(0)
    print(json.dumps({"status": report["status"], "result_path": str(result_path),
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
