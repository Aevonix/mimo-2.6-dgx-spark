# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated witness wrapper around an unmodified pinned vLLM Triton kernel.

No installation, monkeypatching, global invariant mode, or model integration.
The caller must explicitly select one of the two MiMo TP8 operation shapes.
"""
import hashlib
import os
from pathlib import Path

PINNED_SOURCE_SHA256 = "e12ff8666faf99ae70027f1065ef0a5e4fac8b021a6ee27c754be2885a859b71"
SHAPES = {"attention_o_proj_pre_tp": (6144, 2048), "router": (384, 6144)}
CONFIG = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
          "GROUP_SIZE_M": 8, "num_stages": 2, "num_warps": 4}


class StableDenseWitness:
    def __init__(self, kind):
        if kind not in SHAPES:
            raise ValueError(f"Unsupported operation: {kind}")
        if os.environ.get("VLLM_BATCH_INVARIANT", "0") not in ("", "0"):
            raise RuntimeError("This isolated witness requires global batch invariance disabled")
        import torch
        from vllm.model_executor.determinism import batch_invariant

        if batch_invariant._batch_invariant_MODE:
            raise RuntimeError("Global batch invariant mode is already active")
        source_hash = hashlib.sha256(Path(batch_invariant.__file__).read_bytes()).hexdigest()
        if source_hash != PINNED_SOURCE_SHA256:
            raise RuntimeError(f"Pinned batch_invariant.py source mismatch: {source_hash}")
        if torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError("This witness is scoped to GB10 SM121")
        self.torch = torch
        self.kind = kind
        self.n, self.k = SHAPES[kind]
        self.output_dtype = torch.float32 if kind == "router" else torch.bfloat16
        self.kernel = batch_invariant.matmul_kernel_persistent
        # Host metadata is resolved before warmup/capture, never during a launch.
        self.num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.source_hash = source_hash

    def describe(self):
        return {"operation": self.kind, "source_sha256": self.source_hash,
                "kernel": "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent",
                "weight_shape": [self.n, self.k], "output_dtype": str(self.output_dtype),
                "config": CONFIG, "large_m_policy": {"threshold": 64, "BLOCK_SIZE_M": 64},
                "num_sms": self.num_sms,
                "global_invariant_mode_enabled": False, "tp_reduction": False}

    def __call__(self, x, weight):
        # Only tensor metadata is inspected. No item(), CPU copy, sync, or
        # activation/weight conversion occurs in this operation.
        if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != self.k or tuple(weight.shape) != (self.n, self.k):
            raise ValueError("Expected the exact selected MiMo TP8 matrix shapes")
        if x.dtype != self.torch.bfloat16 or weight.dtype != self.torch.bfloat16:
            raise ValueError("Both operands must be BF16")
        if x.device != self.device or weight.device != self.device:
            raise ValueError("Both operands must be on the prepared CUDA device")
        if not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("Both operands must be contiguous")
        m = x.shape[0]
        if m < 1:
            raise ValueError("At least one activation row is required")
        output = self.torch.empty((m, self.n), device=x.device, dtype=self.output_dtype)
        block_m = 64 if m >= 64 else 16
        config = dict(CONFIG, BLOCK_SIZE_M=block_m)
        grid = (min(self.num_sms, ((m + block_m - 1) // block_m) * ((self.n + 63) // 64)),)
        self.kernel[grid](
            x, weight, output, None, m, self.n, self.k,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0),  # Kernel sees weight.T.
            output.stride(0), output.stride(1),
            NUM_SMS=self.num_sms,
            A_LARGE=x.numel() > 2**31,
            B_LARGE=weight.numel() > 2**31,
            C_LARGE=output.numel() > 2**31,
            HAS_BIAS=False, **config,
        )
        return output
