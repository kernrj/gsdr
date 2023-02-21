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
#include "gsdr/arithmetic.h"

__global__ void k_Magnitude(const cuComplex* in, float* out, size_t numElements) {
  const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x > numElements) {
    return;
  }

  cuComplex input = in[x];
  out[x] = hypotf(input.x, input.y);
}

__global__ void k_Abs(const float* in, float* out, size_t numElements) {
  const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x > numElements) {
    return;
  }

  out[x] = fabsf(in[x]);
}

GSDR_C_LINKAGE cudaError_t
gsdrMagnitude(const cuComplex* in, float* out, size_t numElements, int32_t cudaDevice, cudaStream_t cudaStream)
    GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_Magnitude()");
  k_Magnitude<<<blocks, threads, 0, cudaStream>>>(in, out, numElements);
  SIMPLE_CUDA_FNC_END("k_Magnitude()");
}

GSDR_C_LINKAGE cudaError_t
gsdrAbs(const float* in, float* out, size_t numElements, int32_t cudaDevice, cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_Abs()");
  k_Abs<<<blocks, threads, 0, cudaStream>>>(in, out, numElements);
  SIMPLE_CUDA_FNC_END("k_Abs()");
}
