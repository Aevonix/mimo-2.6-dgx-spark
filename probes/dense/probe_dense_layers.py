#!/usr/bin/env python3
"""Real vLLM layer/loader + eager/graph/Inductor smoke; no model or TP8 run."""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

HELPER_SHA = "c96e4580f3ad159a60bc7f0d3045a7805dee14024ef238c83c29f2d73bb2b58e"
INTEGRATION_SHA = "3fefdc7b57691d0b405ce181991e3ad708550c41c53b6e682068c70738805fac"


def import_exact(name, path, expected_sha):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha:
        raise RuntimeError(f"Source hash mismatch: {path}: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, choices=range(30, 241), default=240)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    with output_path.open("x") as f:
        f.write("{}\n")
    started = time.monotonic()
    report = {"schema": "dense-layer-smoke-v1", "status": "running", "stage": "imports",
              "helper_sha256": HELPER_SHA, "integration_sha256": INTEGRATION_SHA,
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "real layers and real world-size1 Gloo context; no mocks, no model, no TP8 collective",
              "cases": []}

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        output_path.write_text(json.dumps(report, indent=2) + "\n")

    def deadline(signum, frame):
        raise TimeoutError(f"Smoke exceeded {args.max_seconds}s")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    # Local loopback only. The rendezvous store is an output-directory file.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment, ensure_model_parallel_initialized,
        destroy_model_parallel, destroy_distributed_environment,
    )
    from vllm.model_executor.layers.linear import RowParallelLinear

    torch.set_num_threads(1)
    assert str(torch.__version__).startswith("2.13.") and torch.version.cuda == "13.0"
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (12, 1)
    here = Path(__file__).resolve().parent
    import_exact("vllm.model_executor.models._mimo_stable_dense", here / "mimo_stable_dense.py", HELPER_SHA)
    integration = import_exact("vllm.model_executor.models._mimo_stable_row_layers",
                               here / "_mimo_stable_row_layers.py", INTEGRATION_SHA)
    report.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))

    def snapshot(value):
        torch.cuda.synchronize()
        return value.detach().cpu().clone()

    def digest(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def same(actual, expected):
        return {"byte_equal": digest(actual) == digest(expected),
                "dtype": str(actual.dtype), "shape": list(actual.shape),
                "sha256": digest(actual), "expected_sha256": digest(expected),
                "max_abs": float((actual.double() - expected.double()).abs().max())}

    class TensorOutput(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            output, bias = self.layer(x)
            return output

    try:
        config = VllmConfig()
        with set_current_vllm_config(config):
            report["stage"] = "world1_gloo_setup"
            save()
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                distributed_init_method="file://" + str((args.output_dir / "gloo-store").resolve()),
                backend="gloo", timeout=timedelta(seconds=30))
            ensure_model_parallel_initialized(1, 1, backend="gloo")
            report["group"] = {"world_size": torch.distributed.get_world_size(),
                               "backend": torch.distributed.get_backend(),
                               "socket_interface": "lo", "rendezvous": "file"}
            with torch.inference_mode():
                for name, k, n, seed in (("projection", 2048, 6144, 20260922),
                                          ("router", 6144, 384, 20260923)):
                    report["stage"] = name + "_construct"
                    save()
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    x8 = torch.randn((8, k), generator=generator).to(torch.bfloat16)
                    weights = (torch.randn((n, k), generator=generator) * 0.02).to(torch.bfloat16).cuda()
                    extra_gen = torch.Generator(device="cpu").manual_seed(seed + 1000)
                    extra = torch.randn((1490, k), generator=extra_gen).to(torch.bfloat16)
                    master = torch.cat((x8, extra), dim=0).cuda()
                    inputs = {m: master[:m] for m in (1, 8, 1498)}
                    with torch.device("cuda"):
                        if name == "router":
                            layer = integration.StableMimoGateLinear(k, n, bias=False,
                                params_dtype=torch.bfloat16, out_dtype=torch.float32,
                                prefix="model.layers.1.mlp.gate")
                        else:
                            # Actual constructor with local TP8 matrix width,
                            # TP disabled: no claim about distributed sharding.
                            layer = RowParallelLinear(k, n, bias=False, input_is_parallel=True,
                                params_dtype=torch.bfloat16, reduce_results=True,
                                quant_config=None, prefix="model.layers.0.self_attn.o_proj",
                                disable_tp=True)
                            old_weight = layer.weight
                            old_loader = layer.weight.weight_loader
                            integration.install_stable_mimo_output(layer)
                            assert layer.weight is old_weight
                            assert layer.weight.weight_loader == old_loader
                    if name == "router":
                        layer.weight_loader(layer.weight, weights)
                    else:
                        layer.weight_loader_v2(layer.weight, weights)
                    assert torch.equal(layer.weight, weights)
                    module = TensorOutput(layer).eval()
                    case = {"name": name, "layer_class": type(layer).__name__,
                            "method_class": type(layer.quant_method).__name__,
                            "weight_shape": list(layer.weight.shape),
                            "weight_loader_used": "replicated" if name == "router" else "row_parallel_v2",
                            "tp_size": layer.tp_size, "bias_is_none": layer.bias is None,
                            "reduce_results": getattr(layer, "reduce_results", None),
                            "input_strides": {m: list(x.stride()) for m, x in inputs.items()},
                            "eager": {}, "graph": {}, "compiled": {}, "compiled_graph": {}}
                    report["cases"].append(case)
                    eager = {}
                    report["stage"] = name + "_eager_graph"
                    save()
                    for m, x in inputs.items():
                        output, bias = layer(x)
                        assert bias is None
                        eager[m] = snapshot(output)
                        assert eager[m].dtype == (torch.float32 if name == "router" else torch.bfloat16)
                        assert tuple(eager[m].shape) == (m, n)
                        case["eager"][m] = same(snapshot(module(x)), eager[m])
                        for _ in range(3):
                            module(x)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            graph_output = module(x)
                        graph.replay()
                        case["graph"][m] = same(snapshot(graph_output), eager[m])
                    report["stage"] = name + "_inductor_fullgraph_dynamic"
                    save()
                    compiled = torch.compile(module, backend="inductor", fullgraph=True, dynamic=True)
                    for m in (8, 1, 1498):
                        x = inputs[m]
                        case["compiled"][m] = same(snapshot(compiled(x)), eager[m])
                        case["compiled"][m]["repeat_byte_equal"] = same(snapshot(compiled(x)), eager[m])["byte_equal"]
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            compiled_graph_output = compiled(x)
                        graph.replay()
                        case["compiled_graph"][m] = same(snapshot(compiled_graph_output), eager[m])
                        save()
                    case["row0_cross_m_byte_equal"] = len({digest(value[0]) for value in eager.values()}) == 1
                    assert case["row0_cross_m_byte_equal"]
                    for mode in ("eager", "graph", "compiled", "compiled_graph"):
                        assert all(check["byte_equal"] for check in case[mode].values()), (name, mode)
                    assert all(check["repeat_byte_equal"] for check in case["compiled"].values())
                    case["status"] = "passed"
                    save()
                    print(json.dumps({"case": name, "status": "passed", "m": [1, 8, 1498]}), flush=True)
        report["status"] = "complete"
        report["stage"] = "finished"
        save()
    except BaseException as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        signal.alarm(0)
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
