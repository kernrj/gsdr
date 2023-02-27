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
#include "gsdr/conversion.h"

__global__ void k_int8ToFloat(const int8_t* input, float* output, size_t numElements) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x > numElements) {
    return;
  }

  output[x] = max(-1.0f, static_cast<float>(input[x]) / 127.0f);
}

GSDR_C_LINKAGE cudaError_t
gsdrInt8ToNormFloat(const int8_t* input, float* output, size_t numElements, int32_t cudaDevice, cudaStream_t cudaStream)
    GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_int8ToFloat()");
  k_int8ToFloat<<<blocks, threads, 0, cudaStream>>>(input, output, numElements);
  SIMPLE_CUDA_FNC_END("k_int8ToFloat()");
}
