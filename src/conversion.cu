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

#include "cuComplexOperatorOverloads.cuh"
#include "gsdr/conversion.h"

__global__ void k_int8ToFloat(const int8_t* input, float* output, size_t numElements) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x > numElements) {
    return;
  }

  output[x] = max(-1.0f, static_cast<float>(input[x]) / 127.0f);
}

GSDR_C_LINKAGE cudaError_t gsdrInt8ToNormFloat(
    const int8_t* input,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {
  SIMPLE_CUDA_FNC_START("k_int8ToFloat()");
  k_int8ToFloat<<<blocks, threads, 0, cudaStream>>>(input, output, numElements);
  SIMPLE_CUDA_FNC_END("k_int8ToFloat()");
}
