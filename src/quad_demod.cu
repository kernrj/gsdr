/*
 * Copyright (C) 2023 Rick Kern <kernrj@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
 * Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Affero General Public License along with this program.  If not, see
 * <https://www.gnu.org/licenses/>.
 */

#include <cuda_runtime.h>

#include "cuComplexOperatorOverloads.cuh"
#include "gsdr/quad_demod.h"
#include "gsdr/util.h"

__global__ static void k_quadFrequencyDemod(
    const cuComplex* input,
    float* output,
    float gain,
    uint32_t numOutputElements) {
  uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

  if (index >= numOutputElements) {
    return;
  }

  const cuComplex m = input[index + 1] * cuConjf(input[index]);
  const float outputValue = gain * atan2f(m.y, m.x);  // / 3.14159f * 180.0f;
  output[index] = outputValue;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  // printf("index [%u] outputValue [%f]\n", index, outputValue);
#endif
}

GSDR_C_LINKAGE cudaError_t gsdrQuadFrequencyDemod(
    const cuComplex* input,
    float* output,
    float gain,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  SIMPLE_CUDA_FNC_START("k_quadFrequencyDemod()");
  k_quadFrequencyDemod<<<blocks, threads, 0, cudaStream>>>(input, output, gain, numElements);
  SIMPLE_CUDA_FNC_END("k_quadFrequencyDemod()");
}
