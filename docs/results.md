# Results

Measured 2026-09-23 on eight DGX Sparks with the official normal checkpoint and pinned runtime in `manifest.json`. Native and DFlash K7 used the **same numerical overlays**. The comparison measures enabling speculation, not the speedup of our patches over unmodified vLLM.

## Frozen native/speculative comparison

These measurements used async scheduling, as shipped in v0.1.0. They are unchanged by the later configuration update.

![Historical structured-extraction comparison; this is not a prose speedup measurement](../assets/normal-results.png)

| Long task | Output tokens | Native seconds | DFlash seconds | Speedup |
| --- | ---: | ---: | ---: | ---: |
| decode-0 | 1,829 | 102.631 | 26.593 | 3.859x |
| decode-1 | 1,829 | 102.702 | 26.820 | 3.829x |
| decode-2 | 1,829 | 103.059 | 26.862 | 3.837x |
| decode-3 | 1,829 | 102.624 | 26.779 | 3.832x |

Aggregate output tokens divided by aggregate end-to-end seconds: **17.80 versus 68.34 tokens/s**, a **3.84x** speedup. Prefill and request overhead are included. This is not decode-only TPS. Work on another cohort overlapped these requests, so this is not an isolated peak-throughput record.

All twelve visible answers matched byte for byte: eight short evidence tasks and four long CSV-to-JSON extraction tasks. Both modes scored **6/8 short and 4/4 long**. P05.partial-execution and P06.parallel-or-sequential failed in both. Short-task speedups ranged from 1.36x to 1.78x. A long-output speedup is not representative of all interaction latency.

Sampling: temperature 0, seed 20260922, maximum 4,096 output tokens, thinking disabled, fresh request cache salt. The exact synthetic UTF-8 fixtures and independent expected answers are in `fixtures/`. A single enclosing JSON fence is permitted; extra prose or repeated JSON is not. All required values and natural completion must match. The public runner additionally enforces exact JSON value types; it reproduces the recorded outcomes on every captured answer locally.

Machine-readable paired measurements and output hashes: [normal-native-vs-speculative.json](../results/normal-native-vs-speculative.json). Captured visible answers, including failures: [normal-answers.json](../results/normal-answers.json). No private agent conversations are used. This small frozen collection is a regression witness, not a comprehensive intelligence benchmark.

## Additional checks on the original profile

| Check | Result |
| --- | --- |
| Long task repeats | Three correct repeats; 26.57-26.64 seconds |
| Mixed workload | One long request plus three staggered short requests; 4/4 correct; observed running-request peak four |
| Mixed long-answer latency | 34.37 seconds |
| Long-context witness | 257,531 actual input tokens; 77 output; natural stop; exact three facts |
| Context time to first model output | 217.60 seconds |
| Context end-to-end time | 219.03 seconds |
| Context requested output reserve | 4,096 tokens |
| API contracts | 4/4: two schema cases, forced tool call, automatic tool call |

The context witness used thinking enabled; its 262144 file label is a sizing target, not its exact token count. Tool tests validate the returned call and arguments and never execute a tool. API requests and original check receipts are included under `fixtures/api/` and `results/api-checks.json`.

Maximum configured context is 1,048,576 and the scheduler has 16 slots. Neither maximum was qualified. No claims are made for other GPU architectures, TP sizes, quantizations, model variants, sampled generation, multimodal input, agent integration or long unattended operation.

## Subsequent runtime failure

After the recorded checks, a mixed batch containing structured-output decoding and a newly admitted short request caused a fatal CUDA error on the head rank. The error surfaced during hidden-state selection; asynchronous CUDA reporting does not identify that operation as the cause. No OOM kill or GPU Xid was observed in the collected host diagnostics.

The prior measured results remain unchanged. They did not establish reliability for this mixed workload. The current default disables async scheduling, informed by [this Flash recipe](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe/tree/13621bb3cc6fd30a94d53609320599d1f1134686) and [the upstream Pro report](https://github.com/vllm-project/vllm/issues/46669). Neither proves the cause of our crash.

## Synchronous scheduling checks

The numerical overlays and weights stayed unchanged. These finite checks passed with `--no-async-scheduling`:

| Check | Result |
| --- | --- |
| Two long extraction tasks | 1,829 tokens each; 26.78 and 26.92 seconds; **68.13 tokens/s** combined end-to-end |
| Long-task first output | 0.98 and 0.96 seconds |
| Resource planning task | Correct; 2.83 seconds |
| Structured output plus new short request | Both complete; observed peak two requests |
| Four-request mixed workload | 4/4 correct; observed peak four; long answer 33.10 seconds |
| API contracts | 4/4 schema and tool-call checks; no tools executed |

The schema check emitted the exact 512-item integer sequence and stopped naturally. Its short request began after 484 output tokens and intentionally stopped at its one-token limit. Client overlap does not prove the exact scheduler geometry from the incident. The original crash has not been reproduced, so this is a configuration mitigation with regression evidence, not a proven causal fix or an unattended reliability claim.

Captured synthetic answers and timings: [synchronous-validation.json](../results/synchronous-validation.json). These speculative-mode checks are not a new matched native/speculative comparison. The large-context witness above has not been repeated with this setting.

## Prose throughput

Three prompts, each run twice, on the current async-off DFlash K7 profile. Temperature 0, thinking disabled, maximum 2,048 output tokens; all six stopped naturally. One benchmark client used the live endpoint. Four requests overlapped background inference, and another cohort was loading weights. These are descriptive measurements, not an isolated performance comparison.

| Prose task | Run | Output tokens | End-to-end TPS | First output | Observed running requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explanation | 1 | 855 | 24.44 | 0.359 s | 1 |
| Fiction | 1 | 867 | 19.96 | 0.371 s | 1 |
| Analysis | 1 | 798 | 21.91 | 0.394 s | 2 |
| Explanation | 2 | 711 | 17.95 | 0.428 s | 2 |
| Fiction | 2 | 802 | 13.88 | 0.402 s | 2 |
| Analysis | 2 | 779 | 21.77 | 0.427 s | 2 |

All prompts, requests and answers are included in [prose-throughput.json](../results/prose-throughput.json). Responses differed between repeats despite greedy sampling. No result was discarded. Natural completion is not a factual-accuracy or writing-quality score; the science explanation includes assumptions about spoon geometry and heat capacity.

There is **no matched native prose baseline**. Do not apply the structured-output 3.84× speedup to these results or divide these rates by the native structured-extraction rate. Prompt lengths, output lengths, scheduling and background load differ.

```sh
python -s scripts/plot_workloads.py
```

## Reproduce the chart

The graph is computed from the paired measurement JSON, not manually entered bar heights:

```sh
python -m pip install matplotlib numpy
python scripts/plot_results.py results/normal-native-vs-speculative.json assets
```

Keep plotting dependencies separate from the serving image. Use the CLI in [setup](setup.md) for new endpoint measurements; it writes results to a new local file and leaves the published baseline unchanged.
