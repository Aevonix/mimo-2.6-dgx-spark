"""Private MiMo Marlin adapter; load and register metadata before compilation."""
from pathlib import Path
import hashlib
import os

import torch
from vllm._custom_ops import moe_wna16_marlin_gemm as _stock_gemm
from vllm.scalar_type import ScalarType, scalar_types

BINARY_SHA256 = "43d7889b9011cf7f39945e161c74842870de2d9b9a3f05ddca35314fcb8fd517"
PARENT_SHA256 = "efcbc1475e9811d6840ec6c6398c94ce738a32f3161ad71c6b4ab4fc38789560"
_library = Path(os.environ.get("MIMO_WHOLE_K_LIBRARY", str(Path(__file__).with_name("mimo_marlin_whole_k_v1.so"))))
if hashlib.sha256(_library.read_bytes()).hexdigest() != BINARY_SHA256:
    raise RuntimeError("Unexpected private whole-K Marlin binary")
torch.ops.load_library(str(_library))
_operation = torch.ops._mimo_marlin_whole_k_v1.moe_wna16_marlin_gemm


def verify_parent_source(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != PARENT_SHA256:
        raise RuntimeError("Unexpected canonical Marlin parent source")


@torch.library.register_fake("_mimo_marlin_whole_k_v1::moe_wna16_marlin_gemm")
def _whole_k_fake(
    a, c_or_none, b_q_weight, b_bias_or_none, b_scales, a_scales,
    global_scale, b_zeros_or_none, workspace, sorted_token_ids, expert_ids,
    num_tokens_past_padded, topk_weights, moe_block_size, top_k,
    mul_topk_weights, b_type_id, size_m, size_n, size_k, use_atomic_add,
    use_fp32_reduce, is_zp_float, thread_k, thread_n, blocks_per_sm,
    whole_k=False,
):
    # Same output metadata as the pinned original op; no device reads.
    return torch.empty((size_m * top_k, size_n), dtype=a.dtype, device=a.device)


def moe_wna16_marlin_gemm(
    input: torch.Tensor,
    output: torch.Tensor | None,
    b_qweight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    b_qzeros: torch.Tensor | None,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_past_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
    thread_k: int = -1,
    thread_n: int = -1,
    blocks_per_sm: int = -1,
):
    supported = (
        input.dtype == torch.bfloat16
        and b_q_type == scalar_types.float4_e2m1f
        and b_scales.dtype == torch.float8_e8m0fnu
        and b_scales.ndim == 3
        and b_scales.shape[1] * 32 == size_k
        and (size_k, size_n) in ((6144, 4096), (2048, 6144))
        and moe_block_size in (8, 16, 32, 48, 64)
    )
    if supported:
        if a_scales is not None or global_scale is not None or b_qzeros is not None:
            raise RuntimeError("Unexpected activation scales or zero points in qualified MXFP4 path")
        if use_atomic_add or not use_fp32_reduce or is_zp_float:
            raise RuntimeError("Unexpected arithmetic flags in qualified whole-K path")
        return _operation(
            input, output, b_qweight, b_bias, b_scales, a_scales,
            global_scale, b_qzeros, workspace, sorted_token_ids, expert_ids,
            num_tokens_past_padded, topk_weights, moe_block_size, top_k,
            mul_topk_weights, b_q_type.id, size_m, size_n, size_k,
            use_atomic_add, use_fp32_reduce, is_zp_float,
            thread_k, thread_n, blocks_per_sm, True,
        )
    return _stock_gemm(
        input, output, b_qweight, b_bias, b_scales, a_scales,
        global_scale, b_qzeros, workspace, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, b_q_type, size_m, size_n, size_k,
        use_atomic_add, use_fp32_reduce, is_zp_float,
        thread_k, thread_n, blocks_per_sm,
    )
