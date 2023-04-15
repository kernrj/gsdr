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

#ifndef GSDR_INCLUDE_GSDR_AM_H_
#define GSDR_INCLUDE_GSDR_AM_H_

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gsdr/gsdr_export.h>
#include <gsdr/util.h>
#include <stdint.h>

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrAmDemod(
    float rfSampleRate,
    float tuningFrequency,
    float channelFrequency,
    uint32_t decimation,
    size_t firstSampleIndex,
    const float* lowPassTaps,
    size_t numLowPassTaps,
    const cuComplex* input,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

#endif  // GSDR_INCLUDE_GSDR_AM_H_
