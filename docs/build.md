# Source and build

`manifest.json` pins the official checkpoint, original vLLM revision/image and every mounted overlay. `patches/` shows changes to existing upstream modules; `overlay/` contains the exact selected Python files. Installation checks every input hash and refuses a different upstream source. It does not patch an arbitrary vLLM version.

## Numerical changes

| Change | Scope |
| --- | --- |
| Canonical Marlin token order | bxchange's upstream PR #52532, credited in NOTICE |
| DiffKV attention | Per-query partition bounds, padded/empty-row guards, mixed short/long query dispatch |
| Whole-K Marlin | Complete K reductions per stripe in the selected BF16/MXFP4 W13/W2 geometries |
| Selected target projections/router | Pinned vLLM persistent matmul kernel; BF16 projection and FP32 router output |
| Normalization | Existing `vllm_c` providers selected through `--kernel-config` |

The MiMo layer integration preserves parameter objects, loaders and TP all-reduce. Draft layers are excluded from the dense changes. Global batch-invariant mode is disabled. The numerical recipe was measured as a bundle; the results do not isolate each component's accuracy contribution or establish the smallest required patch set.

The measured bundle contains three diagnostic files: the `model_runner.py` footer, `_mimo_spec_trace.py` and `_mimo_prefill_boundaries.py`. The public image retains their exact bytes. `config/trace-disabled.json` selects no real prompt hash and disables opaque prefill tracing, matching the tested no-capture state. Do not enable traces for ordinary user prompts. Removing these wrappers is a runtime change to qualify separately; it is not silently included in this release.

## Included binary and full source

The 1.7 MB `artifacts/mimo_marlin_whole_k_v1.so` is the selected tested operation, not a model weight. SHA256:

```text
43d7889b9011cf7f39945e161c74842870de2d9b9a3f05ddca35314fcb8fd517
```

The complete corresponding source is in `whole-k/src/`; original files and source hashes are retained. It builds 15 BF16/MXFP4 variants for SM121 using C++20, PyTorch 2.13.0+cu130, the stable ABI target `0x020B000000000000ULL`, CUDA compiler/headers and ninja. The build sets `MAX_JOBS=2` by default. It does not rebuild all of vLLM.

On an idle Spark, inside the exact base image with this repository mounted at `/work`:

```sh
python whole-k/check_scheduler.py
MAX_JOBS=2 python whole-k/build.py --build-dir build/whole-k --verbose
python probes/marlin/probe_whole_k_w13.py \
  --whole-k-library build/whole-k/mimo_marlin_whole_k_v1.so \
  --base-probe probes/marlin/base_w13.py \
  --candidate probes/marlin/canonical_order.py --output-dir local-results/w13
python probes/marlin/probe_whole_k_w2.py \
  --whole-k-library build/whole-k/mimo_marlin_whole_k_v1.so \
  --base-probe probes/marlin/base_w2.py \
  --candidate probes/marlin/canonical_order.py --output-dir local-results/w2
```

The operation checks compare stock, rebuilt flag-off control and candidate, including an independent numerical reference, row consistency, repeats and CUDA graphs. Other supplied probes cover mixed attention, real dense layer loading and parent/helper integration; see each script's `--help` and source guards.

The build is reproducible from source, but **byte-identical binaries across paths/compiler installations have not been demonstrated**. The runtime helper intentionally accepts only the qualified binary hash above. A rebuilt different hash is a new artifact: run the operation and parent-integration checks, record its hash, then qualify its native/speculative outputs before updating the manifest/helper. Do not remove the checksum check just to make a different build load.

The generic Docker packaging has source/hash and local fixture checks. The measured deployment used these exact overlays through read-only mounts. An end-to-end run of the newly packaged image on another cluster remains a replication step, not a result claimed by this repository.
