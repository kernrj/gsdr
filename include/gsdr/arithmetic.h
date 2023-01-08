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

#ifndef GPUSDR_ARITHMETIC_H
#define GPUSDR_ARITHMETIC_H

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "gsdr_export.h"
#include "util.h"

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrAddConstFF(
    const float* input,
    float addConst,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream);

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrAddConstCC(
    const cuComplex* input,
    cuComplex addConst,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream);

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrAddConstCF(
    const cuComplex* input,
    float addConst,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream);

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrAddConstFC(
    const float* input,
    cuComplex addConst,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream);

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrMultiplyCC(
    const cuComplex* in1,
    const cuComplex* in2,
    cuComplex* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream);

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrMultiplyFF(
    const float* in1,
    const float* in2,
    float* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream);

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrMultiplyCF(
    const cuComplex* in1,
    const float* in2,
    cuComplex* out,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream);

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrAddToMagnitude(
    const cuComplex* input,
    float addToMagnitude,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream);

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t
gsdrMagnitude(const cuComplex* in, float* out, size_t numElements, int32_t cudaDevice, cudaStream_t cudaStream);

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t
gsdrAbs(const float* in, float* out, size_t numElements, int32_t cudaDevice, cudaStream_t cudaStream);

#endif  // GPUSDR_ARITHMETIC_H
