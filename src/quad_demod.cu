/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cuda_runtime.h>

#include "cuComplexOperatorOverloads.cuh"
#include "gsdr/quad_demod.h"
#include "gsdr/util.h"

__global__ static void k_quadFmDemod(const cuComplex* input, float* output, float gain, uint32_t numOutputElements) {
  uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index >= numOutputElements) {
    return;
  }

  const cuComplex m = input[index + 1] * cuConjf(input[index]);
  const float outputValue = gain * atan2f(m.y, m.x);
  output[index] = outputValue;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  // printf("index [%u] outputValue [%f]\n", index, outputValue);
#endif
}

__global__ static void k_quadAmDemod(const cuComplex* input, float* output, uint32_t numOutputElements) {
  uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index >= numOutputElements) {
    return;
  }

  cuComplex val = input[index];
  float magnitude = hypotf(val.x, val.y);
  float outputValue = scalbnf(__saturatef(magnitude), 1) - 1.0f;
  output[index] = outputValue;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  // printf("index [%u] magnitude [%f] amplitude [%f]\n", index, magnitude, outputValue);
#endif
}

GSDR_C_LINKAGE cudaError_t gsdrQuadFmDemod(
    const cuComplex* input,
    float* output,
    float gain,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_quadFmDemod()")
  k_quadFmDemod<<<blocks, threads, 0, cudaStream>>>(input, output, gain, numElements);
  SIMPLE_CUDA_FNC_END("k_quadFmDemod()")
}

GSDR_C_LINKAGE cudaError_t
gsdrQuadAmDemod(const cuComplex* input, float* output, size_t numElements, int32_t cudaDevice, cudaStream_t cudaStream)
    GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_quadAmDemod()")
  k_quadAmDemod<<<blocks, threads, 0, cudaStream>>>(input, output, numElements);
  SIMPLE_CUDA_FNC_END("k_quadAmDemod()")
}
