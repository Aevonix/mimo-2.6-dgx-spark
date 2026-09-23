# Setup

Use eight idle GB10/SM121 Sparks with enough free memory to load the model and KV cache. The recipe uses one rank per Spark, TP8, EP, BF16 KV, MXFP4 expert weights, Marlin and DFlash K7. Async scheduling is disabled. The vLLM image and source revision are pinned in `manifest.json`. This is a text-only recipe.

## Weights

Download the complete official snapshot, including `dflash/`, once to storage accessible by every rank, or copy it to local storage on all eight hosts. Use the same path on each host. The HF CLI is supplied by `huggingface_hub`:

```sh
hf download XiaomiMiMo/MiMo-V2.6-Pro-RL \
  --revision 73875d00b30a89ef8cc353a0b60b0e9f9561952d \
  --local-dir /srv/models/MiMo-V2.6-Pro-RL/73875d00b30a89ef8cc353a0b60b0e9f9561952d
```

Do not mix a tokenizer or chat template from another model revision. The serving configuration disables network downloads. Shared read-only storage works, but loading and restart then depend on that storage host.

## Network and image

Inspect your network with `ip -br address` and `rdma link`. Set eight rank addresses, the socket interface, RDMA HCA names and an IPv4 RoCE v2 GID index in `cluster.json`. The example addresses use the documentation-only `192.0.2.0/24` range and cannot start unchanged. The same interface/HCA names are assumed on each host; use a host-local config if device names differ.

The recipe sets NCCL IB/RoCE v2 and four channels; it does not force an NCCL algorithm or protocol. These settings are measured on our interconnect, not necessarily optimal on yours. Keep the API and distributed ports on a trusted cluster network. The example has no public authentication layer and is not an Internet gateway.

Build the image on a Spark using the README command, then copy it to the other seven hosts with your normal image distribution method. A local build only installs source-checked overlays over the pinned base and the included qualified SM121 kernel. No model weights enter the image. There is no published prebuilt container to pull in this release.

## Start and stop

Preview `scripts/launch.py --rank N` on each host before adding `--execute`. Start all eight ranks with the same settings. Rank 0 serves the HTTP API; ranks 1-7 are headless. Local preflight checks the rank's IP, active RoCE links, required model files and an idle GPU. It never terminates an existing container or resets the host. Container startup can take several minutes.

```sh
docker logs -f mimo-spark-r0
curl http://YOUR_RANK_ZERO_ADDRESS:8000/health
curl http://YOUR_RANK_ZERO_ADDRESS:8000/v1/models
```

Stop only this recipe's container on each host: `docker stop mimo-spark-rN`. Remove that stopped container with `docker rm mimo-spark-rN` before restarting the same rank. There is no automatic restart/service manager here.

The example preserves the deployed 1,048,576 maximum context and 16 sequence slots. Only 257,531 input tokens and four simultaneous requests were tested. These configuration values are not a capacity promise. `--native` disables DFlash and retains all numerical overlays for a matched comparison. Do not start native and speculative variants on the same occupied GPUs.

## Validate

Use a dedicated endpoint while benchmarking. The tests consume real inference and can delay other users. They contain synthetic data and never execute returned tool calls.

```sh
python3 scripts/benchmark.py --base-url http://YOUR_RANK_ZERO_ADDRESS:8000/v1 \
  --suite frozen --output local-results/speculative.jsonl
python3 scripts/benchmark.py --base-url http://YOUR_RANK_ZERO_ADDRESS:8000/v1 \
  --suite api --output local-results/api.jsonl
python3 scripts/benchmark.py --base-url http://YOUR_RANK_ZERO_ADDRESS:8000/v1 \
  --suite context --output local-results/context.jsonl
python3 scripts/check_mixed_requests.py --url http://YOUR_RANK_ZERO_ADDRESS:8000 \
  --output local-results/mixed
```

The frozen suite takes several minutes and deliberately retains the two known model mistakes. A nonzero exit reports any failure, including those known failures. Every JSONL row includes its outcome, usage, end-to-end latency and output hash. The streaming suite records first model output separately from first visible content; neither is inferred from total request duration. Set `MIMO_API_KEY` in the environment only when your endpoint requires it.

The mixed check overlaps a long schema-constrained response with a short prefill after 482 output tokens. It checks completion, exact contents and overlap, and stops on failure. It exercises the workload class of the recorded crash; it does not reproduce its exact scheduler geometry. No server restart or configuration change is performed.

Repeat the frozen suite under `--native` into a separate output file to compare complete content hashes case by case. An unchanged pass count can hide regressions; inspect each case. Keep model/revision, source manifest, sampling, fixtures and any changed network/runtime settings with your results.
