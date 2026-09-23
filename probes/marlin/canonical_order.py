# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from bxchange/vllm PR 52532, head e8a07dcccf8e48dd4b9b42a355fc6d5b9db59073.
# https://github.com/vllm-project/vllm/pull/52532
# Private diagnostic candidate, not enabled in any runtime by this file.
# Upstream helper is unchanged; see MARLIN-CANONICAL-NOTICE.md.
import torch


def _canonicalize_marlin_moe_token_order(
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_size_m: int,
    num_valid_tokens: int,
) -> torch.Tensor:
    """Canonicalize routed token IDs within contiguous expert regions.

    moe_align_block_size guarantees expert grouping but does not guarantee
    token ordering within an expert. Marlin's multi-threadblock split-K
    reduction can depend on physical block position, so equivalent routed
    layouts need a common ordering before entering the kernel.

    Alignment buffers may contain unspecified data after
    num_tokens_post_padded. Keep that inactive tail strictly after the active
    prefix so its contents cannot affect canonicalization.
    """
    block_experts = expert_ids.to(dtype=torch.int64)
    block_positions = torch.arange(
        expert_ids.numel(),
        device=expert_ids.device,
        dtype=torch.int64,
    )
    active_blocks = (
        block_positions * block_size_m < num_tokens_post_padded
    )

    region_starts = torch.cat(
        (
            torch.ones_like(block_experts[:1], dtype=torch.bool),
            (block_experts[1:] != block_experts[:-1])
            | (active_blocks[1:] != active_blocks[:-1]),
        ),
        dim=0,
    )
    block_regions = (
        torch.cumsum(region_starts.to(dtype=torch.int64), dim=0) - 1
    )

    slot_regions = torch.repeat_interleave(
        block_regions,
        block_size_m,
    )[: sorted_token_ids.numel()]

    slot_positions = torch.arange(
        sorted_token_ids.numel(),
        device=sorted_token_ids.device,
        dtype=torch.int64,
    )
    active_slots = slot_positions < num_tokens_post_padded

    token_ids = sorted_token_ids.to(dtype=torch.int64)
    key_stride = num_valid_tokens + 1

    # Valid token IDs are in [0, num_valid_tokens), with the padding sentinel
    # at num_valid_tokens. Values in the inactive tail are unspecified, so do
    # not let them contribute to the sort key.
    token_key = torch.where(
        active_slots,
        token_ids,
        torch.zeros_like(token_ids),
    )
    sort_key = slot_regions * key_stride + token_key

    order = torch.argsort(sort_key)
    return sorted_token_ids.index_select(0, order)

