# Results

Measured 2026-09-23 on eight DGX Sparks with the official normal checkpoint and pinned runtime in `manifest.json`. Native and DFlash K7 used the **same numerical overlays**. The comparison measures enabling speculation, not the speedup of our patches over unmodified vLLM.

## Frozen native/speculative comparison

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

## Additional checks

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

The prior measured results remain unchanged. They did not establish reliability for this mixed workload. Investigation and a synthetic reproduction are pending; do not treat this recipe as qualified for unattended production.

## Reproduce the chart

The graph is computed from the paired measurement JSON, not manually entered bar heights:

```sh
python -m pip install matplotlib numpy
python scripts/plot_results.py results/normal-native-vs-speculative.json assets
```

Keep plotting dependencies separate from the serving image. Use the CLI in [setup](setup.md) for new endpoint measurements; it writes results to a new local file and leaves the published baseline unchanged.
