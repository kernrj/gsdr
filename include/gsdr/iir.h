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

#ifndef GPUSDR_IIR_H
#define GPUSDR_IIR_H

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gsdr/gsdr_export.h>
#include <gsdr/util.h>
#include <stdint.h>

/**
 * @file iir.h
 * @brief IIR (Infinite Impulse Response) filter functions for GPU acceleration
 *
 * This file provides high-performance IIR filter implementations that exploit
 * instruction-level parallelism (ILP) and utilize shared memory for optimal
 * performance on CUDA devices.
 *
 * KEY FEATURES:
 * - Race-condition-free implementation using thread-private history
 * - ILP exploitation through multi-sample processing per thread
 * - Shared memory utilization for coefficient storage
 * - Coalesced memory access patterns
 * - Support for both float and complex data types
 *
 * ALGORITHM:
 * The implementation uses a direct form II IIR structure:
 * y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] + ... - a1*y[n-1] - a2*y[n-2] - ...
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Each thread processes 8 consecutive samples for ILP
 * - Thread-private history arrays eliminate synchronization overhead
 * - Coefficients stored in shared memory for fast access
 * - Memory coalescing for efficient global memory access
 *
 * USAGE NOTES:
 * - History buffers are maintained internally by each thread
 * - The inputHistory and outputHistory parameters are kept for API compatibility
 * - Maximum filter order is limited by available register/shared memory
 * - Performance is optimal for filter orders 2-8
 * - Default configuration processes 8 samples per thread for optimal ILP
 * - Shared memory usage is optimized to use only required space for coefficients
 */

/**
 * High-performance IIR filter for float data with optimized implementation
 *
 * This function implements a race-condition-free IIR filter that exploits
 * instruction-level parallelism by processing multiple samples per thread.
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Each thread processes 8 consecutive samples concurrently
 * - Thread-private history eliminates race conditions
 * - Shared memory used for coefficient storage
 * - Coalesced memory access patterns
 *
 * @param bCoeffs Feedforward coefficients [b0, b1, b2, ...] in GPU memory
 * @param aCoeffs Feedback coefficients [1.0, -a1, -a2, ...] in GPU memory
 * @param coeffCount Number of coefficients (same for b and a arrays)
 * @param inputHistory IGNORED - Legacy parameter kept for API compatibility
 * @param outputHistory IGNORED - Legacy parameter kept for API compatibility
 * @param input Input float array in GPU memory
 * @param output Output float array in GPU memory
 * @param numElements Number of elements to process
 * @param cudaDevice CUDA device ID
 * @param cudaStream CUDA stream for execution
 * @return cudaError_t CUDA error code
 *
 * @note The history buffers are managed internally by each thread
 * @note Optimal performance for filter orders 2-8
 * @note Maximum filter order limited by register availability
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrIirFF(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    float* inputHistory,
    float* outputHistory,
    const float* input,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * High-performance IIR filter for complex data with optimized implementation
 *
 * This function implements a race-condition-free IIR filter that exploits
 * instruction-level parallelism by processing multiple samples per thread.
 *
 * PERFORMANCE CHARACTERISTICS:
 * - Each thread processes 8 consecutive samples concurrently
 * - Thread-private history eliminates race conditions
 * - Shared memory used for coefficient storage
 * - Coalesced memory access patterns
 *
 * @param bCoeffs Feedforward coefficients [b0, b1, b2, ...] in GPU memory
 * @param aCoeffs Feedback coefficients [1.0, -a1, -a2, ...] in GPU memory
 * @param coeffCount Number of coefficients (same for b and a arrays)
 * @param inputHistory IGNORED - Legacy parameter kept for API compatibility
 * @param outputHistory IGNORED - Legacy parameter kept for API compatibility
 * @param input Input complex array in GPU memory
 * @param output Output complex array in GPU memory
 * @param numElements Number of elements to process
 * @param cudaDevice CUDA device ID
 * @param cudaStream CUDA stream for execution
 * @return cudaError_t CUDA error code
 *
 * @note The history buffers are managed internally by each thread
 * @note Optimal performance for filter orders 2-8
 * @note Maximum filter order limited by register availability
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrIirCC(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    cuComplex* inputHistory,
    cuComplex* outputHistory,
    const cuComplex* input,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * IIR filter with configurable samples per thread for performance tuning
 *
 * This advanced function allows tuning the number of samples each thread processes,
 * which can be used to optimize performance for specific filter orders and data sizes.
 *
 * PERFORMANCE TUNING:
 * - Higher samplesPerThread: Better ILP, more registers used
 * - Lower samplesPerThread: More threads, better occupancy
 * - Typical values: 4, 8, 16 depending on filter order and GPU architecture
 *
 * @param bCoeffs Feedforward coefficients [b0, b1, b2, ...] in GPU memory
 * @param aCoeffs Feedback coefficients [1.0, -a1, -a2, ...] in GPU memory
 * @param coeffCount Number of coefficients (same for b and a arrays)
 * @param inputHistory IGNORED - Legacy parameter kept for API compatibility
 * @param outputHistory IGNORED - Legacy parameter kept for API compatibility
 * @param input Input array in GPU memory
 * @param output Output array in GPU memory
 * @param numElements Number of elements to process
 * @param samplesPerThread Number of consecutive samples each thread processes
 * @param cudaDevice CUDA device ID
 * @param cudaStream CUDA stream for execution
 * @return cudaError_t CUDA error code
 *
 * @note Higher samplesPerThread values require more registers per thread
 * @note Lower samplesPerThread values may improve occupancy
 * @note Test different values to find optimal performance for your use case
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrIirFFCustom(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    float* inputHistory,
    float* outputHistory,
    const float* input,
    float* output,
    size_t numElements,
    size_t samplesPerThread,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrIirCCCustom(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    cuComplex* inputHistory,
    cuComplex* outputHistory,
    const cuComplex* input,
    cuComplex* output,
    size_t numElements,
    size_t samplesPerThread,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

/**
 * DEPRECATED: Legacy IIR filter implementation with race conditions
 *
 * This function is kept only for backward compatibility with existing code.
 * It contains serious race conditions where multiple threads write to the same
 * history buffer locations simultaneously, leading to incorrect results.
 *
 * @deprecated Use gsdrIirFF or gsdrIirCC instead
 * @param bCoeffs Feedforward coefficients [b0, b1, b2, ...] in GPU memory
 * @param aCoeffs Feedback coefficients [1.0, -a1, -a2, ...] in GPU memory
 * @param coeffCount Number of coefficients (same for b and a arrays)
 * @param inputHistory Input history buffer (size: coeffCount-1) - RACE CONDITION RISK
 * @param outputHistory Output history buffer (size: coeffCount-1) - RACE CONDITION RISK
 * @param input Input array in GPU memory
 * @param output Output array in GPU memory
 * @param numElements Number of elements to process
 * @param cudaDevice CUDA device ID
 * @param cudaStream CUDA stream for execution
 * @return cudaError_t CUDA error code
 *
 * @warning This implementation has race conditions and should not be used
 * @warning Results will be incorrect for parallel execution
 */
GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrIirFFLegacy(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    float* inputHistory,
    float* outputHistory,
    const float* input,
    float* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

GSDR_C_LINKAGE GSDR_PUBLIC cudaError_t gsdrIirCCLegacy(
    const float* bCoeffs,
    const float* aCoeffs,
    size_t coeffCount,
    cuComplex* inputHistory,
    cuComplex* outputHistory,
    const cuComplex* input,
    cuComplex* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) GSDR_NO_EXCEPT;

#endif  // GPUSDR_IIR_H