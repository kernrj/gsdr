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
