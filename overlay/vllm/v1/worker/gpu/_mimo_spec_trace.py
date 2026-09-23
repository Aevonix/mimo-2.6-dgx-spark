"""Pinned V2, synthetic-only sampling diagnostic. Not a performance measurement.

Installed only when MIMO_SPEC_TRACE_CONFIG is set. No model/sampler outputs are
modified. GPU-to-host copies deliberately synchronize the traced rank.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from pathlib import Path
import threading


def token_sha(tokens):
    return hashlib.sha256(json.dumps([int(x) for x in tokens], separators=(",", ":")).encode()).hexdigest()


def reconstruct_prefix(history, positions, token_ids, position):
    """Use actual query tokens over request history, never unverified draft state."""
    if len(positions) != len(token_ids):
        raise ValueError("query position/token length mismatch")
    patch = dict(zip(map(int, positions), map(int, token_ids)))
    if len(patch) != len(positions):
        raise ValueError("duplicate query positions")
    result = []
    for p in range(position + 1):
        if p in patch:
            result.append(patch[p])
        elif p < len(history):
            result.append(int(history[p]))
        else:
            raise ValueError("prefix has unknown token position")
    return result


def first_difference(reference, candidate):
    """Classify only an exactly shared prefix; no cross-prefix score comparison."""
    if reference["prefix_sha256"] != candidate["prefix_sha256"]:
        raise ValueError("different prefixes")
    decisions = {}
    ambiguous = []
    for kind in ("raw", "processed"):
        pair = []
        for label, record in (("reference", reference), ("candidate", candidate)):
            summary = record[kind]
            if "argmax" in summary:
                pair.append(summary["argmax"])
            elif isinstance(summary.get("margin"), (int, float)) and summary["margin"] > 0:
                pair.append(summary["ids"][0])
            else:
                pair.append(None)
                ambiguous.append(label + "." + kind)
        decisions[kind] = pair
    # Legacy topk tie order cannot establish the actual greedy decision. In
    # particular, topk(ids)[0] is not a stable lowest-index argmax under ties.
    if ambiguous:
        return {"kind": "ambiguous_legacy_tie", "is_divergence": False,
                "ambiguous_fields": ambiguous}
    for kind in ("raw", "processed"):
        a, b = decisions[kind]
        if a != b:
            return {"kind": "target_logits" if kind == "raw" else "logits_processing", "is_divergence": True, "reference_token": a, "candidate_token": b}
    a, b = reference.get("sampled_token"), candidate.get("sampled_token")
    if a is not None and b is not None and a != b:
        return {"kind": "sampling_or_acceptance", "is_divergence": True, "reference_token": a, "candidate_token": b}
    return None


def _list(tensor):
    return tensor.detach().cpu().tolist()


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _summary(logits, eos):
    import torch
    values, ids = torch.topk(logits.float(), 2, dim=-1)
    argmax = _list(torch.argmax(logits, dim=-1))
    out = []
    ids, values = _list(ids), _list(values)
    eos_scores = _list(logits[:, eos].float())
    def number(x):
        return x if math.isfinite(x) else str(x)
    for i, (row_ids, row_values) in enumerate(zip(ids, values)):
        out.append({"ids": row_ids, "argmax": argmax[i], "values": [number(x) for x in row_values],
                    "margin": number(row_values[0] - row_values[1]),
                    "eos_scores": [number(x) for x in eos_scores[i]], "dtype": str(logits.dtype)})
    return out


def _fingerprints(hidden):
    import torch
    h = hidden.float()
    summaries = torch.stack((h.mean(-1), h.square().mean(-1), h.abs().max(-1).values), dim=-1)
    return [{"mean": a, "mean_square": b, "max_abs": c, "first8": x}
            for (a, b, c), x in zip(_list(summaries), _list(h[:, :8]))]


class Trace:
    def __init__(self, config):
        self.config = config
        self.directory = Path(config["output_directory"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.allowed = set(config["prompt_token_sha256"])
        if not 1 <= len(self.allowed) <= 4:
            raise ValueError("allow exactly one to four synthetic prompt hashes")
        self.max_positions = int(config.get("max_output_positions", 192))
        self.start_position = int(config.get("output_start_position", 0))
        self.max_requests = int(config.get("max_requests", 2))
        if not 1 <= self.max_positions <= 512 or not 0 <= self.start_position <= 4096 or not 1 <= self.max_requests <= 4:
            raise ValueError("diagnostic scope exceeds hard cap")
        self.requests = {}
        self.count = 0
        self.current = threading.local()
        self.stopped = False
        self.reference = {}
        if config.get("reference_trace"):
            for line in Path(config["reference_trace"]).read_text().splitlines():
                record = json.loads(line)
                if record.get("event") == "step":
                    for row in record["rows"]:
                        if row.get("emitted"):
                            self.reference.setdefault(row["prefix_sha256"], row)
        self.emit({"event": "start", "config": config, "diagnostic_only": True,
                   "synchronizes_sampling_rank": True, "pid": os.getpid()})

    def emit(self, record):
        with (self.directory / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n")

    def register(self, scheduler):
        for data in scheduler.scheduled_new_reqs:
            tokens = data.prefill_token_ids
            if tokens is None or len(tokens) != data.prompt_len:
                continue  # A resumed/preempted request is not a fresh witness.
            sha = token_sha(tokens)
            if sha not in self.allowed or self.count >= self.max_requests:
                continue
            self.count += 1
            self.requests[data.req_id] = {"prompt_sha": sha, "prompt_len": len(tokens),
                                         "ordinal": self.count, "steps": 0}
            self.emit({"event": "request", "ordinal": self.count, "prompt_token_sha256": sha,
                       "prompt_len": len(tokens), "sampling_params": {
                           k: getattr(data.sampling_params, k, None)
                           for k in ("temperature", "top_p", "top_k", "seed", "max_tokens", "presence_penalty", "frequency_penalty", "repetition_penalty")}})

    def begin(self, runner, hidden, batch):
        if self.stopped or batch.num_reqs != 1 or runner.batch_sharder is not None:
            return None
        witness = self.requests.get(batch.req_ids[0])
        if witness is None:
            return None
        idx = int(batch.idx_mapping_np[0])
        indices = _list(batch.logits_indices)
        all_positions = _list(batch.positions[:batch.num_tokens])
        all_ids = _list(batch.input_ids[:batch.num_tokens])
        positions = [int(all_positions[x]) for x in indices]
        if not positions or min(positions) + 1 < witness["prompt_len"]:
            return None
        first_output = min(positions) + 1 - witness["prompt_len"]
        last_output = max(positions) + 1 - witness["prompt_len"]
        if first_output >= self.start_position + self.max_positions or last_output < self.start_position:
            return None
        witness["steps"] += 1
        if witness["steps"] > self.max_positions + 16:
            return None
        # all_token_ids stores committed history; query positions override draft rows.
        history = _list(runner.req_states.all_token_ids.gpu[idx, :max(positions) + 1])
        fingerprints = _fingerprints(hidden[batch.logits_indices])
        rows = []
        for i, position in enumerate(positions):
            prefix = reconstruct_prefix(history, all_positions, all_ids, position)
            if token_sha(prefix[:witness["prompt_len"]]) != witness["prompt_sha"]:
                raise RuntimeError("trace prefix does not match allowlisted prompt")
            rows.append({"position": position, "output_position": position + 1 - witness["prompt_len"],
                         "prefix_sha256": token_sha(prefix), "prefix_tail": prefix[-16:],
                         "query_input_token": all_ids[indices[i]], "hidden": fingerprints[i]})
        block_tables = []
        for group, table in enumerate(runner.block_tables.block_tables):
            n = int(runner.block_tables.num_blocks.np[group, idx])
            block_tables.append({"group": group, "block_size": runner.block_tables.block_sizes[group],
                                 "kernel_block_size": runner.block_tables.kernel_block_sizes[group],
                                 "block_ids": _list(table.gpu[idx, :n])})
        context = {"event": "step", "ordinal": witness["ordinal"], "step": witness["steps"],
                   "prompt_token_sha256": witness["prompt_sha"], "req_state_index": idx,
                   "num_draft_tokens": int(batch.num_draft_tokens), "num_tokens": int(batch.num_tokens),
                   "num_tokens_after_padding": int(batch.num_tokens_after_padding),
                   "query_start_loc": _list(batch.query_start_loc),
                   "seq_lens": _list(batch.seq_lens), "positions": all_positions,
                   "query_token_ids": all_ids, "logits_indices": indices,
                   "cu_num_logits": _list(batch.cu_num_logits),
                   "computed_tokens_cpu": batch.num_computed_tokens_np.tolist(),
                   "computed_tokens_gpu": int(runner.req_states.num_computed_tokens.gpu[idx].item()),
                   "blocks": block_tables, "rows": rows}
        self.current.context = context
        return context

    def capture_params(self, raw, processed, positions):
        ctx = getattr(self.current, "context", None)
        if ctx is None:
            return
        eos = self.config.get("eos_token_ids", [151643, 151645, 151672])
        raw_s, processed_s = raw, _summary(processed, eos)
        by_pos = {row["position"]: row for row in ctx["rows"]}
        for position, a, b in zip(_list(positions), raw_s, processed_s):
            row = by_pos[int(position)]
            row["raw"], row["processed"] = a, b

    def finish(self, context, output):
        self.current.context = None
        sampler_output, num_sampled, num_rejected = output
        tokens = _list(sampler_output.sampled_token_ids)[0]
        n, rejected = int(_list(num_sampled)[0]), int(_list(num_rejected)[0])
        context.update(sampled_tokens=tokens[:n], num_sampled=n, num_rejected=rejected)
        for i, row in enumerate(context["rows"]):
            row["sampled_token"] = int(tokens[i]) if i < n else None
            row["draft_next_token"] = context["rows"][i + 1]["query_input_token"] if i + 1 < len(context["rows"]) else None
            row["emitted"] = i < n
            if "processed" not in row:
                raise RuntimeError("trace missed actual sampling-parameter application")
        self.emit(context)
        for row in context["rows"]:
            ref = self.reference.get(row["prefix_sha256"])
            difference = first_difference(ref, row) if ref and row["emitted"] else None
            if difference and not difference.get("is_divergence", True):
                self.emit({"event": "ambiguous_legacy_tie", "difference": difference,
                           "position": row["position"], "prefix_sha256": row["prefix_sha256"]})
            elif difference:
                marker = {"event": "first_divergence", "difference": difference,
                          "reference": ref, "candidate": row, "ordinal": context["ordinal"],
                          "step": context["step"], "request_cancellation_required": True}
                self.emit(marker)
                (self.directory / "first-divergence.json").write_text(json.dumps(marker, indent=2) + "\n")
                self.stopped = True
                break


def install(runner_class):
    """Called by a guarded footer in the exact pinned model_runner source."""
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    import inspect
    import torch.distributed as dist

    config = json.loads(Path(os.environ["MIMO_SPEC_TRACE_CONFIG"]).read_text())
    sampler_path = Path(inspect.getfile(Sampler))
    expected = "ddbcdab46010bf38a5fab9b9b6a2deed2ec5e883ebfe81e83f6d22f41e546771"
    if hashlib.sha256(sampler_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("unsupported Sampler source for MiMo trace")
    rejection_path = sampler_path.parent.parent / "spec_decode" / "rejection_sampler.py"
    if hashlib.sha256(rejection_path.read_bytes()).hexdigest() != "f815df94f0a6cb38da07b814b274ea903386a2b0678b6d7083afc43d5f62b502":
        raise RuntimeError("unsupported RejectionSampler source for MiMo trace")
    state = {"trace": None}

    def get_trace(refresh=False):
        if not dist.is_initialized() or dist.get_rank() != 0:
            return None
        current_config = json.loads(Path(os.environ["MIMO_SPEC_TRACE_CONFIG"]).read_text()) if refresh else config
        if state["trace"] is None or (refresh and state["trace"].config != current_config):
            state["trace"] = Trace(current_config)
        return state["trace"]

    old_add, old_sample, old_params = runner_class.add_requests, runner_class.sample, Sampler.apply_sampling_params

    @functools.wraps(old_add)
    def add(self, scheduler):
        result = old_add(self, scheduler)
        trace = get_trace(refresh=bool(scheduler.scheduled_new_reqs))
        if trace:
            trace.register(scheduler)
        return result

    @functools.wraps(old_sample)
    def sample(self, hidden_states, input_batch, grammar_output):
        trace = get_trace()
        context = trace.begin(self, hidden_states, input_batch) if trace else None
        try:
            result = old_sample(self, hidden_states, input_batch, grammar_output)
            if context is not None:
                trace.finish(context, result)
            return result
        finally:
            if trace:
                trace.current.context = None

    @functools.wraps(old_params)
    def params(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos, input_ids,
               expanded_local_pos, seq_lens_upper_bound_np, skip_top_k_top_p=False):
        trace = state["trace"]
        active = trace is not None and getattr(trace.current, "context", None) is not None
        raw = _summary(logits, config.get("eos_token_ids", [151643, 151645, 151672])) if active else None
        result = old_params(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos,
                            input_ids, expanded_local_pos, seq_lens_upper_bound_np, skip_top_k_top_p)
        if active:
            trace.capture_params(raw, result, pos)
        return result

    runner_class.add_requests, runner_class.sample = add, sample
    Sampler.apply_sampling_params = params


# Private diagnostic extension for pinned V2 MiMo sampling trace.
# Appended to the existing immutable, tie-aware sampling helper by its builder.
CONSUMED_KV_TRACE_VERSION = 2
CONSUMED_KV_OUTPUT_RANGE = (80, 121)
CONSUMED_KV_PAGES = (99, 100)


def _selected_pages(table, row):
    return [int(table[row, page].item()) if page < table.shape[1] else None
            for page in CONSUMED_KV_PAGES]


def _capture_consumed_kv(runner, state, batch, context):
    """Read after target forward, before sampler and draft proposal mutate state."""
    outputs = [row["output_position"] for row in context["rows"]]
    lo, hi = CONSUMED_KV_OUTPUT_RANGE
    if not outputs or max(outputs) < lo or min(outputs) >= hi:
        return None
    if state is None or state.input_batch is not batch:
        raise RuntimeError("consumed KV trace lacks the matching execute-model state")
    if batch.num_reqs != 1 or runner.batch_sharder is not None:
        raise RuntimeError("consumed KV trace supports only one unsharded request")
    metadata = state.attn_metadata
    layer_slots = state.slot_mappings_by_layer
    if not isinstance(metadata, dict) or not isinstance(layer_slots, dict):
        raise RuntimeError("consumed KV trace requires per-layer target metadata")
    idx = int(batch.idx_mapping_np[0])
    groups = []
    skipped_metadata = []
    captured_target_metadata = 0
    for group, spec in enumerate(runner.kv_cache_config.kv_cache_groups):
        source = runner.block_tables.block_tables[group].gpu
        gathered = runner.block_tables.input_block_tables[group]
        group_record = {
            "group": group,
            "block_size": runner.block_tables.block_sizes[group],
            "kernel_block_size": runner.block_tables.kernel_block_sizes[group],
            "num_blocks_cpu": int(runner.block_tables.num_blocks.np[group, idx]),
            "num_blocks_gpu": int(runner.block_tables.num_blocks.gpu[group, idx].item()),
            "source_pages": _selected_pages(source, idx),
            "gathered_pages": _selected_pages(gathered, 0),
            "source_ptr": int(source.data_ptr()),
            "gathered_ptr": int(gathered.data_ptr()),
            "runner_slots": _list(runner.block_tables.slot_mappings[group, :batch.num_tokens]),
            "consumed_metadata": [],
        }
        seen = {}
        for layer in spec.layer_names:
            if layer not in metadata:
                continue  # Draft-only groups are not target-forward consumers.
            meta = metadata[layer]
            # The target uses this inspected Triton DiffKV metadata contract.
            # DFlash also contributes FlashInferMetadata to this state; its
            # fields differ and must not be interpreted as target metadata.
            meta_type = type(meta)
            if (meta_type.__name__ != "TritonAttentionMetadata" or
                    meta_type.__module__ != "vllm.v1.attention.backends.triton_attn"):
                skipped_metadata.append({
                    "group": group, "layer": layer,
                    "metadata_class": meta_type.__name__,
                    "metadata_module": meta_type.__module__,
                    "reason": ("draft_flashinfer_metadata_not_target_diffkv"
                               if meta_type.__name__ == "FlashInferMetadata"
                               else "uninspected_metadata_contract"),
                })
                continue
            slots = layer_slots[layer]
            key = (id(meta), int(slots.data_ptr()))
            if key in seen:
                seen[key]["layers"].append(layer)
                continue
            item = {
                "layers": [layer],
                "metadata_class": type(meta).__name__,
                "block_table_ptr": int(meta.block_table.data_ptr()),
                "pages": _selected_pages(meta.block_table, 0),
                "seq_lens": _list(meta.seq_lens),
                "query_start_loc": _list(meta.query_start_loc),
                "num_actual_tokens": int(meta.num_actual_tokens),
                "max_query_len": int(meta.max_query_len),
                "slot_mapping_ptr": int(slots.data_ptr()),
                "slot_mapping": _list(slots[:batch.num_tokens]),
            }
            captured_target_metadata += 1
            seen[key] = item
            group_record["consumed_metadata"].append(item)
        groups.append(group_record)
    return {
        "version": CONSUMED_KV_TRACE_VERSION,
        "capture_status": "partial" if skipped_metadata else ("complete" if captured_target_metadata else "empty"),
        "capture_complete": bool(captured_target_metadata) and not skipped_metadata,
        "captured_target_metadata_count": captured_target_metadata,
        "skipped_metadata": skipped_metadata,
        "capture_point": "after_target_forward_before_sampling_and_draft",
        "page_indices": list(CONSUMED_KV_PAGES),
        "request_state_index": idx,
        "positions": _list(batch.positions[:batch.num_tokens]),
        "seq_lens": _list(batch.seq_lens),
        "query_start_loc": _list(batch.query_start_loc),
        "groups": groups,
    }


_consumed_original_begin = Trace.begin


def _consumed_begin(self, runner, hidden, batch):
    context = _consumed_original_begin(self, runner, hidden, batch)
    if context is not None:
        try:
            captured = _capture_consumed_kv(
                runner, getattr(runner, "_mimo_consumed_execute_state", None), batch, context
            )
        except Exception as exc:
            # Diagnostics must never turn an otherwise valid request into an
            # inference failure. A failed capture is evidence of nothing.
            captured = {
                "version": CONSUMED_KV_TRACE_VERSION,
                "capture_status": "error",
                "capture_complete": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
        if captured is not None:
            context["consumed_kv"] = captured
    return context


Trace.begin = _consumed_begin
_consumed_original_install = install


def install(runner_class):
    import inspect
    runner_file = Path(inspect.getfile(runner_class))
    expected_runner = "d2440c63ecfc4ecaaa4fe1eaa08b463c722b1bf59ba321cf7b9d39128abb17a7"
    if hashlib.sha256(runner_file.read_bytes()).hexdigest() != expected_runner:
        raise RuntimeError("consumed KV diagnostic requires the exact pinned runner overlay")
    _consumed_original_install(runner_class)
    old_sample_tokens = runner_class.sample_tokens

    @functools.wraps(old_sample_tokens)
    def sample_tokens(self, grammar_output):
        self._mimo_consumed_execute_state = self.execute_model_state
        try:
            return old_sample_tokens(self, grammar_output)
        finally:
            self._mimo_consumed_execute_state = None

    runner_class.sample_tokens = sample_tokens


# Private bounded opaque-prefill extension. Existing trace and runner guards remain intact.
_opaque_previous_install = install
def install(runner_class):
    _opaque_previous_install(runner_class)
    from ._mimo_prefill_boundaries import install as _install_prefill_boundaries
    _install_prefill_boundaries(runner_class)
