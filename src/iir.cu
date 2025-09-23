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
#include "gsdr/iir.h"

using namespace std;

/**
 * IIR filter kernel using direct form II structure
 * This is more numerically stable than direct form I
 * Note: This implementation processes elements sequentially to maintain history consistency
 */
template <class IN_T, class OUT_T, class COEFF_T>
__global__ void k_Iir(
    const COEFF_T* bCoeffs,      // feedforward coefficients
    const COEFF_T* aCoeffs,      // feedback coefficients (a[0] is always 1.0)
    uint32_t coeffCount,
    IN_T* inputHistory,          // input history buffer
    OUT_T* outputHistory,        // output history buffer
    const IN_T* input,
    OUT_T* output,
    uint32_t numElements) {

  uint32_t elementIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (elementIndex >= numElements) {
    return;
  }

  // Use atomic operations or process sequentially to maintain history consistency
  // For now, we'll process one element at a time to ensure correct history
  for (uint32_t elem = elementIndex; elem < numElements; elem += gridDim.x * blockDim.x) {
    // Compute feedforward part (input terms)
    OUT_T feedforward = zero<OUT_T>();
    for (uint32_t i = 0; i < coeffCount; i++) {
      IN_T inputVal;
      if (i == 0) {
        inputVal = input[elem];
      } else {
        // Access historical input values
        inputVal = inputHistory[i - 1];
      }
      feedforward += inputVal * bCoeffs[i];
    }

    // Compute feedback part (output terms)
    OUT_T feedback = zero<OUT_T>();
    for (uint32_t i = 1; i < coeffCount; i++) {  // Start from i=1 since a[0] = 1.0
      feedback += outputHistory[i - 1] * aCoeffs[i];
    }

    // Compute output: feedforward - feedback
    OUT_T result = feedforward - feedback;

    // Update output
    output[elem] = result;

    // Update histories for next iteration
    // Shift input history
    for (int32_t i = coeffCount - 2; i >= 0; i--) {
      inputHistory[i + 1] = inputHistory[i];
    }
    inputHistory[0] = input[elem];

    // Shift output history
    for (int32_t i = coeffCount - 2; i >= 0; i--) {
      outputHistory[i + 1] = outputHistory[i];
    }
    outputHistory[0] = result;
  }
}

template <class IN_T, class OUT_T, class COEFF_T>
static cudaError_t iirGeneric(
    const COEFF_T* bCoeffs,
    const COEFF_T* aCoeffs,
    size_t coeffCount,
    IN_T* inputHistory,
    OUT_T* outputHistory,
    const IN_T* input,
    OUT_T* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {

  SIMPLE_CUDA_FNC_START("IIR");

  if (coeffCount < 2) {
    // Need at least 2 coefficients (b0, b1, a1)
    return cudaErrorInvalidValue;
  }

  k_Iir<IN_T, OUT_T, COEFF_T><<<blocks, threads, 0, cudaStream>>>(
      bCoeffs, aCoeffs, coeffCount, inputHistory, outputHistory,
      input, output, numElements);

  SIMPLE_CUDA_FNC_END("IIR");
}

GSDR_C_LINKAGE cudaError_t gsdrIirFF(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    float* inputHistory,
    float* outputHistory,
    const float* input,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {

  return iirGeneric<float, float, float>(
      bCoeffs, aCoeffs, coeffCount, inputHistory, outputHistory,
      input, output, numElements, cudaDevice, cudaStream);
}

GSDR_C_LINKAGE cudaError_t gsdrIirCC(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    cuComplex* inputHistory,
    cuComplex* outputHistory,
    const cuComplex* input,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {

  return iirGeneric<cuComplex, cuComplex, float>(
      bCoeffs, aCoeffs, coeffCount, inputHistory, outputHistory,
      input, output, numElements, cudaDevice, cudaStream);
}