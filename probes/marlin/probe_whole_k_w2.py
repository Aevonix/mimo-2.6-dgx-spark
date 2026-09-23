"""Matched stock/control/whole-K operation witness, never a serving benchmark."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time

BASE_SHA = "c4112762cd0b5d1cf100cfdfcbc8bb13a8877fd2e33c1b09b90f729ef4770f0a"
LABELS = ("native_0", "canonical_0", "native_1", "canonical_1", "native_2", "canonical_2", "reversed_legal", "reversed_canonical")


def main(args):
    import torch
    import vllm._custom_ops as ops
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256(Path(args.base_probe).read_bytes()).hexdigest() == BASE_SHA
    torch.ops.load_library(args.whole_k_library)
    private = torch.ops._mimo_marlin_whole_k_v1
    stock = ops.moe_wna16_marlin_gemm
    spec = importlib.util.spec_from_file_location("original_order_probe", args.base_probe)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    counters, canonical_rows, canonical_configs, input_hashes = {}, {}, {}, {}
    began = time.monotonic()

    def emit(record):
        record["elapsed_seconds"] = time.monotonic() - began
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        with (output / "whole-k-w2-results.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def compare(a, b):
        a, b = a.detach().cpu(), b.detach().cpu()
        delta = (a.float() - b.float()).abs()
        return {"byte_equal": torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)),
                "different_values": int((a != b).sum()), "max_abs": float(delta.max()),
                "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a.float()).clamp_min(1e-12))}

    def invoke(call_args, kw, destination, whole_k):
        positional = list(call_args)
        positional[1] = destination
        positional += [kw["moe_block_size"], kw["top_k"], kw["mul_topk_weights"], kw["b_q_type"].id,
                       kw["size_m"], kw["size_n"], kw["size_k"], kw["use_atomic_add"], kw["use_fp32_reduce"],
                       kw["is_zp_float"], kw.get("thread_k", -1), kw.get("thread_n", -1), kw.get("blocks_per_sm", -1), whole_k]
        return private.moe_wna16_marlin_gemm(*positional)

    def witness(*call_args, **kw):
        assert kw["top_k"] == 1 and kw["mul_topk_weights"] is True
        assert kw["size_m"] % 8 == 0
        m = kw["size_m"] // 8
        number = counters.get(m, 0)
        assert m in (1, 8, 1498) and number < len(LABELS)
        label = LABELS[number]
        counters[m] = number + 1
        original = stock(*call_args, **kw)
        control = invoke(call_args, kw, torch.zeros_like(original), False)
        control_config = private.last_config()
        fixed = invoke(call_args, kw, torch.zeros_like(original), True)
        fixed_config = private.last_config()
        assert control_config == fixed_config, "candidate and flag-off control selected different configurations"
        active = call_args[9][:int(call_args[11].item())]
        valid = active[(active >= 0) & (active < kw["size_m"] * kw["top_k"])].long().unique(sorted=True)
        assert torch.equal(valid[:8], torch.arange(8, device=valid.device))
        assert bool(torch.isfinite(fixed[valid]).all())
        stock_control = compare(original[valid], control[valid])
        emit({"event": "whole_k_operation", "m": m, "variant": label,
              "config_fields": ["thread_k", "thread_n", "threads", "blocks_per_sm", "sms", "moe_block_size", "m", "n", "k"],
              "control_config": control_config, "fixed_config": fixed_config,
              "stock_vs_flag_off_control": stock_control,
              "stock_vs_whole_k": compare(original[valid], fixed[valid]),
              "whole_k_output_sha256": digest(fixed[valid])})
        if not stock_control["byte_equal"]:
            raise RuntimeError("rebuilt flag-off operation differs from stock; do not attribute differences to whole-K")
        if label.startswith("canonical_"):
            canonical_rows.setdefault(m, []).append(fixed[:8].detach().cpu().clone())
        if label == "canonical_0":
            canonical_configs[m] = fixed_config
            input_hashes[m] = digest(call_args[0][:8])
            torch.save({"activation_row0": call_args[0][:8].detach().cpu(),
                        "whole_k_row0": fixed[:8].detach().cpu(),
                        "stock_row0": original[:8].detach().cpu()}, output / f"whole-k-row0-m{m}.pt")
            timing = {}
            for name, operation in (("stock", lambda: stock(*call_args, **kw)),
                                    ("whole_k", lambda: invoke(call_args, kw, fixed, True))):
                for _ in range(3): operation()
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20): operation()
                stop.record(); stop.synchronize()
                timing[name] = start.elapsed_time(stop) / 20
            emit({"event": "warm_operation_timing", "m": m, "stream_ms_per_call": timing,
                  "repeats": 20, "scope": "custom-op CUDA stream time including wrapper allocation effects; no serving-speed claim"})
            if m in (8, 1498):
                eager_sha = digest(fixed[valid])
                warm_stream = torch.cuda.Stream(); warm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm_stream): invoke(call_args, kw, fixed, True)
                torch.cuda.current_stream().wait_stream(warm_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): graph_output = invoke(call_args, kw, fixed, True)
                graph.replay(); first = graph_output[valid].clone()
                graph.replay(); torch.cuda.synchronize()
                equality = compare(first, graph_output[valid])
                emit({"event": "whole_k_cuda_graph", "m": m, "repeat": equality,
                      "matches_eager_sha256": digest(graph_output[valid]) == eager_sha})
                assert equality["byte_equal"]
                assert digest(graph_output[valid]) == eager_sha
        return fixed  # The base probe's independent dequantized oracle now checks the candidate.

    ops.moe_wna16_marlin_gemm = witness
    try:
        base.run(args)
    finally:
        ops.moe_wna16_marlin_gemm = stock
    for m, rows in canonical_rows.items():
        emit({"event": "whole_k_same_shape_repeat", "m": m,
              "comparisons": [compare(rows[0], row) for row in rows[1:]]})
    for m in (8, 1498):
        assert input_hashes[m] == input_hashes[1]
        emit({"event": "whole_k_cross_shape_row0", "m_pair": [1, m],
              "activation_row0_sha256": input_hashes[m],
              "comparison": compare(canonical_rows[1][0], canonical_rows[m][0]),
              "per_expert": [{"expert": i, **compare(canonical_rows[1][0][i], canonical_rows[m][0][i])} for i in range(8)],
              "configs": [canonical_configs[1], canonical_configs[m]]})
    emit({"event": "whole_k_complete", "scope": "synthetic weighted W2 kernel witness; full-model causation remains unproven"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whole-k-library", required=True)
    parser.add_argument("--base-probe", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=2.0)
    main(parser.parse_args())
