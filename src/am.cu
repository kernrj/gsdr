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

#include "adjustFrequency.cuh"
#include "cuComplexOperatorOverloads.cuh"
#include "gsdr/am.h"

__global__ void k_Am(
    const cuComplex* __restrict__ rfInput,
    const float* __restrict__ lowPassTaps,
    uint32_t numTaps,
    uint32_t decimation,
    uint32_t firstSampleIndex,
    float sampleFrequency,
    float frequencyShift,
    float* __restrict__ output,
    uint32_t numOutputs) {
  uint32_t outputIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (outputIndex >= numOutputs) {
    return;
  }

  float& outputSample = output[outputIndex];

  const uint32_t initialInputIndex = decimation * outputIndex;
  const uint32_t firstSampleIndexForThisThread = firstSampleIndex + initialInputIndex;
  cuComplex sample = k_AdjustFrequency(
      frequencyShift,
      firstSampleIndexForThisThread,
      sampleFrequency,
      rfInput + initialInputIndex,
      lowPassTaps,
      numTaps);

  outputSample = __saturatef(hypotf(sample.x, sample.y)) * 2.0f - 1.0f;
}

cudaError_t gsdrAmDemod(
    float rfSampleRate,
    float centerFrequency,
    float channelFrequency,
    uint32_t decimation,
    size_t firstSampleIndex,
    const float* lowPassTaps,
    size_t numLowPassTaps,
    const cuComplex* input,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  SIMPLE_CUDA_FNC_START("k_Am()")

  const auto firstSampleIndexUInt32 = (uint32_t)fmodf((float)firstSampleIndex, rfSampleRate);
  const float cosineFrequency = centerFrequency - channelFrequency;
  k_Am<<<blocks, threads, 0, cudaStream>>>(
      input,
      lowPassTaps,
      numLowPassTaps,
      decimation,
      firstSampleIndexUInt32,
      rfSampleRate,
      cosineFrequency,
      output,
      numElements);

  SIMPLE_CUDA_FNC_END("k_Am()")
}
