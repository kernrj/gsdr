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

#ifndef GPUSDR_CONVERSION_H
#define GPUSDR_CONVERSION_H

#include <gsdr/cuda_util.h>
#include <gsdr/gsdr_export.h>
#include <gsdr/util.h>
#include <stdint.h>

/**
 * Converts int8_t values to float values in the range -1 <= output value <= 1
 * Input values of -128 and -127 produce an output value of -1.0.
 * An input value of 127 produces an output value 1.0.
 * An input value of 0 produces an output value of 0.0.
 *
 * @param input
 * @param output
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t
gsdrInt8ToNormFloat(const int8_t* input, float* output, size_t numElements, int32_t cudaDevice, cudaStream_t cudaStream)
    GSDR_NO_EXCEPT;

#endif  // GPUSDR_CONVERSION_H
