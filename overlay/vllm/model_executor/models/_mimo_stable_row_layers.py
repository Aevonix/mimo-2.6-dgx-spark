# SPDX-License-Identifier: Apache-2.0
"""Private opt-in MiMo integration for the qualified stable-row operations.

This module is loaded only by the separately prepared, hash-guarded MiMo
overlay. It does not register a global linear backend or change TP reduction.
"""
import torch

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models._mimo_stable_dense import StableDenseWitness


class StableMimoGateLinear(GateLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is not None or self.out_dtype != torch.float32:
            raise ValueError("The MiMo router overlay requires no bias and FP32 output")
        if tuple(self.weight.shape) != (384, 6144):
            raise ValueError("The MiMo router overlay requires weight shape (384,6144)")
        if self.weight.dtype != torch.bfloat16:
            raise ValueError("The MiMo router overlay requires BF16 weights")
        self._stable_mimo_router = StableDenseWitness("router")

    def forward(self, x):
        return self._stable_mimo_router(x, self.weight), None


class StableMimoOutputMethod(UnquantizedLinearMethod):
    def __init__(self):
        super().__init__()
        self._stable_mimo_output = StableDenseWitness("attention_o_proj_pre_tp")

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("The MiMo output overlay requires an unbiased projection")
        return self._stable_mimo_output(x, layer.weight)


def install_stable_mimo_output(layer):
    # Retain the existing layer, parameter objects, weight-loading path and
    # forward method. RowParallelLinear continues to own the TP all-reduce.
    if type(layer.quant_method) is not UnquantizedLinearMethod:
        raise TypeError("The MiMo output overlay requires the unquantized method")
    if tuple(layer.weight.shape) != (6144, 2048):
        raise ValueError("The MiMo TP8 output overlay requires shape (6144,2048)")
    if layer.bias is not None or not layer.reduce_results:
        raise ValueError("The MiMo output layer must be unbiased and reduce over TP")
    if layer.weight.dtype != torch.bfloat16:
        raise ValueError("The MiMo output overlay requires BF16 weights")
    layer.quant_method = StableMimoOutputMethod()
