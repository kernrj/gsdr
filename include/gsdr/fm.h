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

#ifndef GSDR_INCLUDE_GSDR_FM_H_
#define GSDR_INCLUDE_GSDR_FM_H_

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gsdr/gsdr_export.h>
#include <gsdr/util.h>
#include <stdint.h>

/**
 * Repeated calls to fmDemod requires an overlap of numLowPassTaps input values from the previous call.
 *
 * @param rfSampleRate The number of RF samples per second
 * @param tuningFrequency The tuning frequency the samples were captured at.
 * @param channelFrequency The center frequency of the channel to demodulate
 * @param frequencyDeviation The frequency deviation of the FM channel.
 * @param decimation Reduce the number of output samples to 1 / (# input samples)
 * @param firstSampleIndex The sample offset within [input].
 * @param lowPassTaps FIR taps
 * @param numLowPassTaps The number of elements in [lowPassTaps].
 * @param input Number of elements must be at least (numOutputs + 1) * decimation.
 * @param output Must contain 'numOutputs' outputs.
 * @param numOutputs The number of elements to write to [output].
 * @param cudaDevice The CUDA device index to run on.
 * @param cudaStream The CUDA stream to use for processing.
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrFmDemod(
    float rfSampleRate,
    float tuningFrequency,
    float channelFrequency,
    float frequencyDeviation,
    uint32_t decimation,
    size_t firstSampleIndex,
    const float* lowPassTaps,
    size_t numLowPassTaps,
    const cuComplex* input,
    float* output,
    size_t numOutputs,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

#endif  // GSDR_INCLUDE_GSDR_FM_H_
