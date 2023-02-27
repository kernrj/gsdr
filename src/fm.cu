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
#include "gsdr/fm.h"

__global__ void k_Fm(
    const cuComplex* __restrict__ rfInput,
    const float* __restrict__ lowPassTaps,
    uint32_t numLowPassTaps,
    uint32_t decimation,
    uint32_t firstSampleIndex,  // % sampleRate is ok
    float rfSampleRate,
    float frequencyShift,
    float quadFmDemodGain,
    float* __restrict__ output,
    uint32_t numOutputs) {
  uint32_t outputIndex = blockDim.x * blockIdx.x + threadIdx.x;

  // Warps overlap by one FIR value. This function outputs 31 values per 32-thread warp, so -1 for every 32 threads.
  uint32_t duplicates = outputIndex >> 5;
  outputIndex -= duplicates;
  const size_t totalThreadCount = numOutputs + 1;  // numOutputs = # outputs from low-pass filter + 1
  const size_t laneId = threadIdx.x & 31;

  if (outputIndex >= totalThreadCount) {
    return;
  }

  float& outputSample = output[outputIndex];

  const uint32_t initialInputIndex = decimation * outputIndex;

  // firstSampleIndex is the index of the first sample from the buffer the kernel was launched with.
  const uint32_t firstSampleIndexForThisThread = firstSampleIndex + initialInputIndex;
  cuComplex sample = k_AdjustFrequency(
      frequencyShift,
      firstSampleIndexForThisThread,
      rfSampleRate,
      rfInput + initialInputIndex,
      lowPassTaps,
      numLowPassTaps);

  cuComplex nextSample;
  nextSample.x = __shfl_down_sync(0xffffffff, sample.x, 1);
  nextSample.y = __shfl_down_sync(0xffffffff, sample.y, 1);

  if (laneId == 31) {
    return;
  }

  const cuComplex c = nextSample * cuConjf(sample);

  outputSample = quadFmDemodGain * atan2f(c.y, c.x);
}

__global__ void k_Fm4x(
    const cuComplex* __restrict__ rfInput,
    const float* __restrict__ tapsReversed,
    uint32_t numTaps,
    uint32_t decimation,
    float phi,
    float sampleFrequency,
    float cosineFrequency0Times2Pi,
    float cosineFrequency1Times2Pi,
    float cosineFrequency2Times2Pi,
    float cosineFrequency3Times2Pi,
    float gain0,
    float gain1,
    float gain2,
    float gain3,
    float* __restrict__ output0,
    float* __restrict__ output1,
    float* __restrict__ output2,
    float* __restrict__ output3,
    uint32_t numOutputsPerArray) {
  uint32_t outputIndex = blockDim.x * blockIdx.x + threadIdx.x;

  // Warps overlap by one FIR value. This function outputs 31 values per 32-thread warp, so -1 for every 32 threads.
  uint32_t duplicates = outputIndex >> 5;
  outputIndex -= duplicates;

  const size_t laneId = threadIdx.x & 31;

  if (outputIndex >= numOutputsPerArray) {
    return;
  }

  float& outputSample0 = output0[outputIndex];
  float& outputSample1 = output1[outputIndex];
  float& outputSample2 = output2[outputIndex];
  float& outputSample3 = output3[outputIndex];

  const uint32_t initialInputIndex = decimation * outputIndex;

  const float frac = outputIndex / sampleFrequency;
  const float theta0 = fmaf(phi, frac, cosineFrequency0Times2Pi);
  const float theta1 = fmaf(phi, frac, cosineFrequency1Times2Pi);
  const float theta2 = fmaf(phi, frac, cosineFrequency2Times2Pi);
  const float theta3 = fmaf(phi, frac, cosineFrequency3Times2Pi);

  const cuComplex* inputSample = rfInput + initialInputIndex;

  cuComplex sample0 = zero<cuComplex>();
  cuComplex sample1 = zero<cuComplex>();
  cuComplex sample2 = zero<cuComplex>();
  cuComplex sample3 = zero<cuComplex>();

  for (uint32_t i = 0; i < numTaps; i++, inputSample++) {
    const float tapVal = tapsReversed[i];
    const cuComplex inVal = *inputSample;

    cuComplex cosVal0;
    cuComplex cosVal1;
    cuComplex cosVal2;
    cuComplex cosVal3;

    __sincosf(theta0, &cosVal0.y, &cosVal0.x);
    __sincosf(theta1, &cosVal1.y, &cosVal1.x);
    __sincosf(theta2, &cosVal2.y, &cosVal2.x);
    __sincosf(theta3, &cosVal3.y, &cosVal3.x);

    const cuComplex multipliedValue0 = inVal * cosVal0;
    const cuComplex multipliedValue1 = inVal * cosVal1;
    const cuComplex multipliedValue2 = inVal * cosVal2;
    const cuComplex multipliedValue3 = inVal * cosVal3;

    const cuComplex filteredValue0 = multipliedValue0 * tapVal;
    const cuComplex filteredValue1 = multipliedValue1 * tapVal;
    const cuComplex filteredValue2 = multipliedValue2 * tapVal;
    const cuComplex filteredValue3 = multipliedValue3 * tapVal;

    sample0 += filteredValue0;
    sample1 += filteredValue1;
    sample2 += filteredValue2;
    sample3 += filteredValue3;
  }

  cuComplex nextSample0;
  cuComplex nextSample1;
  cuComplex nextSample2;
  cuComplex nextSample3;
  nextSample0.x = __shfl_down_sync(0xffffffff, sample0.x, 1);
  nextSample0.y = __shfl_down_sync(0xffffffff, sample0.y, 1);
  nextSample1.x = __shfl_down_sync(0xffffffff, sample1.x, 1);
  nextSample1.y = __shfl_down_sync(0xffffffff, sample1.y, 1);
  nextSample2.x = __shfl_down_sync(0xffffffff, sample2.x, 1);
  nextSample2.y = __shfl_down_sync(0xffffffff, sample2.y, 1);
  nextSample3.x = __shfl_down_sync(0xffffffff, sample3.x, 1);
  nextSample3.y = __shfl_down_sync(0xffffffff, sample3.y, 1);

  if (laneId == 31) {
    return;
  }

  const cuComplex c0 = nextSample0 * cuConjf(sample0);
  const cuComplex c1 = nextSample1 * cuConjf(sample1);
  const cuComplex c2 = nextSample2 * cuConjf(sample2);
  const cuComplex c3 = nextSample3 * cuConjf(sample3);

  outputSample0 = gain0 * atan2f(c0.y, c0.x);
  outputSample1 = gain1 * atan2f(c1.y, c1.x);
  outputSample2 = gain2 * atan2f(c2.y, c2.x);
  outputSample3 = gain3 * atan2f(c3.y, c3.x);
}

cudaError_t gsdrFmDemod(
    float rfSampleRate,
    float centerFrequency,
    float channelFrequency,
    float channelWidth,
    uint32_t decimation,
    size_t firstSampleIndex,
    const float* lowPassTaps,
    size_t numLowPassTaps,
    const cuComplex* input,
    float* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {
  /*
   * Define numElements here because SIMPLE_CUDA_FNC_START requires it.
   * The kernel processes numElements low-pass values, and the FM quad demod outputs one less value.
   */
  const size_t numElements = numOutputs + 1;
  SIMPLE_CUDA_FNC_START("k_Fm()")

  const auto firstSampleIndexUInt32 = (uint32_t)fmodf((float)firstSampleIndex, rfSampleRate);
  const float quadFmDemodGain = rfSampleRate / (2.0f * M_PIf * channelWidth);
  const float frequencyShift = centerFrequency - channelFrequency;
  k_Fm<<<blocks, threads, 0, cudaStream>>>(
      input,
      lowPassTaps,
      numLowPassTaps,
      decimation,
      firstSampleIndexUInt32,
      rfSampleRate,
      frequencyShift,
      quadFmDemodGain,
      output,
      numOutputs);

  SIMPLE_CUDA_FNC_END("k_Fm()")
}
