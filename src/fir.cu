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

#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include "cuComplexOperatorOverloads.cuh"
#include "gsdr/cuda_util.h"
#include "gsdr/fir.h"

using namespace std;

template <class IN_T, class OUT_T, class TAP_T>
__global__ void k_Fir(
    const IN_T* input,
    const TAP_T* tapsReversed,
    uint32_t numTaps,
    OUT_T* output,
    uint32_t numOutputs) {
  uint32_t outputIndex = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t inputIndex = outputIndex;

  if (outputIndex >= numOutputs) {
    return;
  }

  OUT_T& outputSample = output[outputIndex];
  const IN_T* inputSample = input + inputIndex;

  outputSample = zero<OUT_T>();
  for (uint32_t i = 0; i < numTaps; i++, inputSample++) {
    outputSample += *inputSample * tapsReversed[i];
  }
}

template <class IN_T, class OUT_T, class TAP_T>
__global__ void k_FirDecimate(
    const IN_T* __restrict__ input,
    const TAP_T* __restrict__ tapsReversed,
    uint32_t numTaps,
    uint32_t decimation,
    OUT_T* __restrict__ output,
    uint32_t numOutputs) {
  uint32_t outputIndex = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t inputIndex = decimation * outputIndex;

  if (outputIndex >= numOutputs) {
    return;
  }

  OUT_T& outputSample = output[outputIndex];
  const IN_T* inputSample = input + inputIndex;

  outputSample = zero<OUT_T>();
  for (uint32_t i = 0; i < numTaps; i++, inputSample++) {
    outputSample += *inputSample * tapsReversed[i];
  }
}

GSDR_C_LINKAGE cudaError_t gsdrFirFC(
    size_t decimation,
    const float* taps,
    size_t tapCount,
    const cuComplex* input,
    cuComplex* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numOutputs;
  SIMPLE_CUDA_FNC_START("FIR FC")

  if (decimation == 1) {
    CHECK_CUDA_RET("Before k_Fir() FC");
    k_Fir<<<blocks, threads, 0, cudaStream>>>(input, taps, tapCount, output, numOutputs);
    CHECK_CUDA_RET("After k_Fir() FC");
  } else {
    CHECK_CUDA_RET("Before k_FirDecimate() FC");
    k_FirDecimate<<<blocks, threads, 0, cudaStream>>>(input, taps, tapCount, decimation, output, numOutputs);
    CHECK_CUDA_RET("After k_FirDecimate() FC");
  }

  SIMPLE_CUDA_FNC_END("FIR FC")
}

GSDR_C_LINKAGE cudaError_t gsdrFirFF(
    size_t decimation,
    const float* taps,
    size_t tapCount,
    const float* input,
    float* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numOutputs;
  SIMPLE_CUDA_FNC_START("FIR FF")

  if (decimation == 1) {
    CHECK_CUDA_RET("Before k_Fir() FF");
    k_Fir<<<blocks, threads, 0, cudaStream>>>(input, taps, tapCount, output, numOutputs);
    CHECK_CUDA_RET("After k_Fir() FF");
  } else {
    CHECK_CUDA_RET("Before k_FirDecimate() FF");
    k_FirDecimate<<<blocks, threads, 0, cudaStream>>>(input, taps, tapCount, decimation, output, numOutputs);
    CHECK_CUDA_RET("After k_FirDecimate() FF");
  }

  SIMPLE_CUDA_FNC_END("FIR FF")
}

GSDR_C_LINKAGE cudaError_t gsdrFirCC(
    size_t decimation,
    const cuComplex* taps,
    size_t tapCount,
    const cuComplex* input,
    cuComplex* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numOutputs;
  SIMPLE_CUDA_FNC_START("FIR CC")

  if (decimation == 1) {
    CHECK_CUDA_RET("Before k_Fir() CC");
    k_Fir<<<blocks, threads, 0, cudaStream>>>(input, taps, tapCount, output, numOutputs);
    CHECK_CUDA_RET("After k_Fir() CC");
  } else {
    CHECK_CUDA_RET("Before k_FirDecimate() CC");
    k_FirDecimate<<<blocks, threads, 0, cudaStream>>>(input, taps, tapCount, decimation, output, numOutputs);
    CHECK_CUDA_RET("After k_FirDecimate() CC");
  }

  SIMPLE_CUDA_FNC_END("FIR CC")
}

GSDR_C_LINKAGE cudaError_t gsdrFirCF(
    size_t decimation,
    const cuComplex* taps,
    size_t tapCount,
    const float* input,
    cuComplex* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  const size_t numElements = numOutputs;
  SIMPLE_CUDA_FNC_START("FIR CF")

  if (decimation == 1) {
    CHECK_CUDA_RET("Before k_Fir() CF");
    k_Fir<<<blocks, threads, 0, cudaStream>>>(input, taps, tapCount, output, numOutputs);
    CHECK_CUDA_RET("After k_Fir() CF");
  } else {
    CHECK_CUDA_RET("Before k_FirDecimate() CF");
    k_FirDecimate<<<blocks, threads, 0, cudaStream>>>(input, taps, tapCount, decimation, output, numOutputs);
    CHECK_CUDA_RET("After k_FirDecimate() CF");
  }

  SIMPLE_CUDA_FNC_END("FIR CF")
}
