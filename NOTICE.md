# Attribution

The vLLM-derived source is copyright contributors to the vLLM project, under Apache-2.0. The pinned source is `0961bbae2894d574be790d219651824eb199318e`. The complete license is in `LICENSE`; source headers are retained.

Marlin source retains the copyright of Elias Frantar and notices of modifications by Neural Magic. The whole-K operator adapts the pinned vLLM/Marlin implementation; it is not a new original matrix multiplication implementation.

The canonical token-order helper comes from **bxchange**, [vLLM PR #52532](https://github.com/vllm-project/vllm/pull/52532), head `e8a07dcccf8e48dd4b9b42a355fc6d5b9db59073`, under Apache-2.0. It is attributed upstream work. That PR was draft/unmerged when this release was prepared.

The dense wrapper uses the unmodified `matmul_kernel_persistent` from pinned vLLM `model_executor/determinism/batch_invariant.py`. The local changes select launch geometry and preserve output dtype. Global batch-invariant mode is not enabled.

Aevonix's changes to the attention dispatch/partitioning, whole-K operator, selected MiMo layer integration, diagnostics, build tooling and tests are supplied under Apache-2.0. Modified upstream files are enumerated with original and resulting hashes in `manifest.json`, with diffs under `patches/`. The source is intended for the explicitly documented geometry and runtime.

The included binary `artifacts/mimo_marlin_whole_k_v1.so` was built from the corresponding `whole-k/src/` source and build script supplied here. It uses the same Apache-2.0 source license. Its checksum, target and build options are recorded in `manifest.json` and `docs/build.md`.

The model and tokenizer are not distributed by this repository. XiaomiMiMo's [pinned metadata](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Pro-RL/revision/73875d00b30a89ef8cc353a0b60b0e9f9561952d) declares MIT; obtain the snapshot and its terms directly from its publisher. Runtime/container dependencies keep their own licenses. Neither Xiaomi nor the vLLM project is represented as endorsing this release.
