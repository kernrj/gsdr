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

#ifndef GPUSDR_QUAD_DEMOD_H
#define GPUSDR_QUAD_DEMOD_H

#include <cuComplex.h>
#include <gsdr/gsdr_export.h>
#include <gsdr/util.h>
#include <stddef.h>
#include <stdint.h>

/**
 * The number of elements in input must be at least numOutputElements + 1.
 *
 * @param gain Set to channelFrequency / (2π * channelWidth)
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQuadFmDemod(
    const cuComplex* input,
    float* output,
    float gain,
    size_t numOutputElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrQuadAmDemod(
    const cuComplex* input,
    float* output,
    size_t numOutputElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

#endif  // GPUSDR_QUAD_DEMOD_H
