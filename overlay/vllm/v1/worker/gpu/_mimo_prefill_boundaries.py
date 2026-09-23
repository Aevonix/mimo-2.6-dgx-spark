"""Private one-prefill witness inside existing opaque operations, never model math."""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import re

PROMPT_SHA = "6236b707c3fac44d95d759971808bfe43184fc0955ab2212ea5d547995ab79fd"
PROMPT_LEN = 4083
MAX_BYTES = 32 * 1024 * 1024
TARGET_LAYERS = 70
COLLECTIVE_STAGES = {f"tp_all_reduce.{i}.{s}" for i in range(5) for s in ("input_last", "output_last")}
SOURCE_GUARDS = {
    "runner": "d2440c63ecfc4ecaaa4fe1eaa08b463c722b1bf59ba321cf7b9d39128abb17a7",
    "attention": "40c9ead67d8e82d71bdfd4de0c0abb6774ff9e1939ad89ff262af33c0ee36317",
    "backend": "5639817e1fa82e180bd20ab0b0efd444ecc8cbc8837121f9e81576ef50ac5a4d",
    "moe": "4bceadd548107fe9e20afd3dd30ee971e744355561a56ac17e293860cf1fff6b",
    "parallel": "2bcd7ed5a26d94da57f8c9a0ed5c08818a0bc6515ba4be9e8c596b1c84baedd5",
}
LOG = logging.getLogger(__name__)


def token_sha(tokens):
    return hashlib.sha256(json.dumps(list(map(int, tokens)), separators=(",", ":")).encode()).hexdigest()


def expected_stages(cfg):
    full = cfg.get("full_kv_layers", [0, 1])
    if not isinstance(full, (list, tuple)) or len(full) != 2 or len(set(full)) != 2 or any(type(x) is not int or not 0 <= x < TARGET_LAYERS for x in full):
        raise ValueError("full_kv_layers must name exactly two distinct target layers")
    result = {f"attention.{i}.{s}" for i in range(TARGET_LAYERS) for s in ("q_last", "output_last")}
    result |= {f"attention.{i}.{s}" for i in full for s in ("k_all", "v_all")}
    result |= {"moe.1.input_last", "moe.1.router_last", "moe.1.output_last"}
    return result | (COLLECTIVE_STAGES if cfg.get("capture_collectives", True) else set())


def choose_layers(context):
    selected = {}
    for name, layer in context.items():
        if any(part in name.lower() for part in ("draft", "mtp", "nextn")):
            continue
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if not match:
            continue
        idx = int(match[1])
        if idx >= TARGET_LAYERS:
            continue
        key = None
        if hasattr(layer, "impl") and callable(getattr(layer.impl, "forward", None)):
            key = ("attention", idx)
            if getattr(layer, "use_direct_call", False):
                raise ValueError("attention is not using its opaque operation")
        elif idx == 1 and callable(getattr(layer, "_forward_impl", None)):
            key = ("moe", idx)
        if key is not None:
            if key in selected:
                raise ValueError(f"ambiguous target boundary {key}")
            selected[key] = (name, layer)
    if set(selected) != {*(('attention', idx) for idx in range(TARGET_LAYERS)), ("moe", 1)}:
        raise ValueError(f"missing target boundaries: {sorted(selected)}")
    return selected


class Capture:
    def __init__(self, cfg, rank):
        self.cfg, self.rank = cfg, rank
        self.records, self.errors, self.saved = [], [], []
        self.bytes = 0
        self.bindings = []
        self.expected = expected_stages(cfg)

    def error(self, where, exc):
        self.errors.append({"where": where, "type": type(exc).__name__, "message": str(exc)[:500]})

    def snapshot(self, stage, tensor, *, full=False):
        """Clone before buffers may be reused; read back only after model forward."""
        try:
            import torch
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[0] != PROMPT_LEN:
                raise ValueError(f"unexpected witness shape {getattr(tensor, 'shape', None)}")
            if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
                raise ValueError("refusing capture inside a CUDA graph capture")
            selected = tensor if full else tensor[-1:]
            count = selected.numel() * selected.element_size()
            if self.bytes + count > MAX_BYTES:
                raise ValueError("bounded witness byte budget exceeded")
            meta = {"stage": stage, "shape": list(tensor.shape), "stride": list(tensor.stride()),
                    "storage_offset": tensor.storage_offset(), "data_ptr_mod128": tensor.data_ptr() % 128,
                    "dtype": str(tensor.dtype), "device": str(tensor.device), "full_tensor": full,
                    "selected_shape": list(selected.shape), "selected_stride": list(selected.stride()),
                    "selected_storage_offset": selected.storage_offset(),
                    "selected_data_ptr_mod128": selected.data_ptr() % 128}
            self.saved.append((meta, selected.detach().clone(memory_format=torch.contiguous_format)))
            self.bytes += count
        except Exception as exc:
            self.error(stage, exc)

    def context(self, name, metadata=None):
        try:
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            record = {"event": "boundary", "layer": name,
                      "cudagraph_runtime_mode": str(context.cudagraph_runtime_mode)}
            if metadata is not None:
                record.update({"metadata_type": type(metadata).__name__,
                               "max_query_len": getattr(metadata, "max_query_len", None),
                               "num_actual_tokens": getattr(metadata, "num_actual_tokens", None)})
            self.records.append(record)
        except Exception as exc:
            self.error("context:" + name, exc)

    def wrap(self, selected):
        for (kind, idx), (name, layer) in selected.items():
            if kind == "attention":
                obj, attr = layer.impl, "forward"
                old = getattr(obj, attr)

                def factory(old, idx, name):
                    @functools.wraps(old)
                    def wrapped(layer, query, key, value, kv_cache, attn_metadata, output,
                                output_scale=None, output_block_scale=None):
                        self.context(name, attn_metadata)
                        self.snapshot(f"attention.{idx}.q_last", query)
                        if idx in self.cfg.get("full_kv_layers", [0, 1]):
                            self.snapshot(f"attention.{idx}.k_all", key, full=True)
                            self.snapshot(f"attention.{idx}.v_all", value, full=True)
                        # Never catch or suppress an exception from the model operation.
                        result = old(layer, query, key, value, kv_cache, attn_metadata, output,
                                     output_scale=output_scale, output_block_scale=output_block_scale)
                        self.snapshot(f"attention.{idx}.output_last", output)
                        return result
                    return wrapped

                wrapped = factory(old, idx, name)
            else:
                obj, attr = layer, "_forward_impl"
                old = getattr(obj, attr)

                def factory(old, name):
                    @functools.wraps(old)
                    def wrapped(hidden_states, router_logits, shared_experts_input, input_ids=None):
                        self.context(name)
                        self.snapshot("moe.1.input_last", hidden_states)
                        self.snapshot("moe.1.router_last", router_logits)
                        result = old(hidden_states, router_logits, shared_experts_input, input_ids)
                        self.snapshot("moe.1.output_last", result)
                        return result
                    return wrapped

                wrapped = factory(old, name)
            self.bindings.append((obj, attr, old, attr in getattr(obj, "__dict__", {})))
            setattr(obj, attr, wrapped)

    def restore(self):
        for obj, attr, old, had_instance_attr in reversed(self.bindings):
            try:
                if had_instance_attr:
                    setattr(obj, attr, old)
                else:
                    delattr(obj, attr)
            except Exception as exc:
                self.error("restore:" + attr, exc)
        self.bindings.clear()

    def wrap_collectives(self, group):
        old = group._all_reduce_out_place
        count = 0

        @functools.wraps(old)
        def wrapped(tensor):
            nonlocal count
            selected = count < 5 and tuple(tensor.shape) == (PROMPT_LEN, 6144)
            idx = count
            if selected:
                count += 1
                self.records.append({"event": "collective", "index": idx,
                                     "group": group.unique_name, "world_size": group.world_size})
                self.snapshot(f"tp_all_reduce.{idx}.input_last", tensor)
            result = old(tensor)
            if selected:
                self.snapshot(f"tp_all_reduce.{idx}.output_last", result)
            return result

        self.bindings.append((group, "_all_reduce_out_place", old,
                              "_all_reduce_out_place" in getattr(group, "__dict__", {})))
        group._all_reduce_out_place = wrapped

    def finish(self):
        import torch
        directory = Path(self.cfg["output_directory"])
        directory.mkdir(parents=True, exist_ok=True)
        snapshots = []
        for meta, tensor in self.saved:
            try:
                cpu = tensor.cpu().contiguous()
                raw = cpu.view(torch.uint8).numpy().tobytes()
                values = cpu.float().reshape(-1)
                number = lambda x: x if math.isfinite(x) else str(x)
                record = dict(meta, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                              first8=[number(x) for x in values[:8].tolist()], mean=number(float(values.mean())),
                              mean_square=number(float(values.square().mean())), max_abs=number(float(values.abs().max())))
                if self.cfg.get("save_tensors", False):
                    if meta["stage"] not in self.expected:
                        raise ValueError("unknown stage name for raw witness")
                    run_key = hashlib.sha256(self.cfg["run_id"].encode()).hexdigest()[:16]
                    filename = f"rank{self.rank}.{run_key}.{meta['stage']}.bin"
                    with (directory / filename).open("xb") as stream:
                        stream.write(raw)
                    record["raw_file"] = filename
                snapshots.append(record)
            except Exception as exc:
                self.error("readback:" + meta["stage"], exc)
        stages = [record["stage"] for record in snapshots]
        expected = self.expected
        complete = set(stages) == expected and len(stages) == len(expected) and not self.errors
        payload = {"event": "prefill_boundaries", "version": 1, "run_id": self.cfg["run_id"],
                   "rank": self.rank, "prompt_token_sha256": PROMPT_SHA, "prompt_len": PROMPT_LEN,
                   "compiled_caller_preserved": True, "diagnostic_changes_timing": True,
                   "complete": complete, "bytes_cloned": self.bytes,
                   "boundaries": self.records, "snapshots": snapshots, "errors": self.errors,
                   "missing_stages": sorted(expected - set(stages))}
        with (directory / f"boundaries-rank{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        self.saved.clear()
        return payload


def validate_sources(runner_class):
    from vllm.model_executor.layers.attention import attention
    from vllm.model_executor.layers.fused_moe.runner import moe_runner
    from vllm.v1.attention.backends import triton_attn
    from vllm.distributed import parallel_state
    subjects = {"runner": runner_class, "attention": attention, "moe": moe_runner,
                "backend": triton_attn, "parallel": parallel_state}
    for name, subject in subjects.items():
        actual = hashlib.sha256(Path(inspect.getfile(subject)).read_bytes()).hexdigest()
        if actual != SOURCE_GUARDS[name]:
            raise ValueError(f"unsupported {name} source {actual}")


def install(runner_class):
    """Called after the existing sampling helper, under its unchanged runner guard."""
    if getattr(runner_class, "_mimo_opaque_prefill_v1", False):
        return
    validate_sources(runner_class)
    old_execute = runner_class.execute_model
    seen = set()

    @functools.wraps(old_execute)
    def execute(self, scheduler_output, *args, **kwargs):
        capture = None
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else -1
            # Only inspect config for actual newly admitted requests, never warmup or decode.
            new = getattr(scheduler_output, "scheduled_new_reqs", ())
            if rank >= 0 and len(new) == 1:
                cfg = json.loads(Path(os.environ["MIMO_SPEC_TRACE_CONFIG"]).read_text()).get("opaque_prefill", {})
                run_id = cfg.get("run_id")
                ranks = cfg.get("ranks", [0])
                if not isinstance(ranks, list) or not 1 <= len(ranks) <= 8 or any(type(x) is not int or not 0 <= x < 8 for x in ranks):
                    raise ValueError("diagnostic rank list must contain one to eight ranks in [0,7]")
                if cfg.get("enabled") and rank in ranks and isinstance(run_id, str) and run_id and run_id not in seen:
                    tokens = new[0].prefill_token_ids
                    if tokens is not None and len(tokens) == PROMPT_LEN and token_sha(tokens) == PROMPT_SHA:
                        seen.add(run_id)
                        capture = Capture(cfg, rank)
                        if (scheduler_output.total_num_scheduled_tokens != PROMPT_LEN or
                                scheduler_output.num_scheduled_tokens != {new[0].req_id: PROMPT_LEN} or
                                new[0].prompt_len != PROMPT_LEN or self.batch_sharder is not None):
                            raise ValueError("requires exactly one complete fresh prefill, without sharding")
                        capture.wrap(choose_layers(self.vllm_config.compilation_config.static_forward_context))
                        if cfg.get("capture_collectives", True):
                            from vllm.distributed.parallel_state import get_tp_group
                            capture.wrap_collectives(get_tp_group())
        except Exception as exc:
            if capture is not None:
                capture.error("admission", exc)
                capture.restore()
            else:
                LOG.warning("MiMo opaque-prefill diagnostic could not arm: %s", exc)
        try:
            return old_execute(self, scheduler_output, *args, **kwargs)
        except BaseException as exc:
            if capture is not None:
                capture.error("model_execution", exc)
            raise
        finally:
            if capture is not None:
                capture.restore()
                try:
                    capture.finish()
                except Exception:
                    LOG.exception("MiMo opaque-prefill evidence write failed; capture is incomplete")

    runner_class.execute_model = execute
    runner_class._mimo_opaque_prefill_v1 = True
