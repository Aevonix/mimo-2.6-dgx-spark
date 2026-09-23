// SPDX-License-Identifier: Apache-2.0
#include "kernel.h"
#include "marlin_template.h"
namespace MARLIN_NAMESPACE_NAME {
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(), 256, 4, 16, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(), 128, 4, 8, 4, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(), 128, 4, 4, 8, false, 4, 2, false>(MARLIN_KERNEL_PARAMS);
}
