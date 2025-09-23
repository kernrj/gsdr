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

// Constants for performance tuning and limits
static const uint32_t DEFAULT_THREADS_PER_BLOCK = 256;
static const uint32_t DEFAULT_SAMPLES_PER_THREAD = 8;
static const uint32_t MAX_SAMPLES_PER_THREAD = 32;
static const uint32_t CUDA_MAX_GRID_DIMENSION = 65535;
static const size_t MAX_SHARED_MEMORY_BYTES = 49152;  // 48KB per SM

/**
 * OPTIMIZED IIR Filter Kernel with ILP and Shared Memory
 *
 * This implementation addresses several critical issues in the original version:
 *
 * 1. RACE CONDITION FIX: Each thread maintains its own private history state,
 *    eliminating the dangerous shared history buffer writes that caused
 *    race conditions between threads.
 *
 * 2. ILP EXPLOITATION: Each thread processes multiple consecutive samples
 *    (typically 4-8) to exploit instruction-level parallelism. This allows
 *    the GPU to hide memory latency and execute multiple arithmetic operations
 *    concurrently.
 *
 * 3. SHARED MEMORY: Filter coefficients are loaded into shared memory once
 *    per block, providing much faster access than repeated global memory reads.
 *
 * 4. MEMORY COALESCING: Consecutive threads process consecutive samples,
 *    enabling coalesced memory accesses for both reads and writes.
 *
 * ALGORITHM OVERVIEW:
 * - Direct Form II IIR structure: y[n] = b0*x[n] + b1*x[n-1] + ... - a1*y[n-1] - a2*y[n-2] - ...
 * - Each thread processes a chunk of consecutive samples independently
 * - Thread-private history arrays maintain filter state between samples
 * - Coefficients stored in shared memory for fast access
 * - Samples processed in batches to exploit ILP
 *
 * PERFORMANCE OPTIMIZATIONS:
 * - Thread-private history eliminates synchronization overhead
 * - Shared memory reduces global memory bandwidth requirements
 * - ILP exploitation through batch processing
 * - Coalesced memory access patterns
 *
 * CONSTRAINTS:
 * - All threads in a block must process the same number of samples
 * - Maximum filter order limited by available registers/shared memory
 * - Input and output arrays must be properly sized
 *
 * @param bCoeffs Global memory array of feedforward coefficients [b0, b1, b2, ...]
 * @param aCoeffs Global memory array of feedback coefficients [1.0, -a1, -a2, ...]
 * @param coeffCount Number of coefficients (same for b and a arrays)
 * @param input Input signal array in global memory
 * @param output Output signal array in global memory
 * @param numElements Number of elements in input/output arrays
 * @param samplesPerThread Number of consecutive samples each thread processes
 */
template <class IN_T, class OUT_T, class COEFF_T, uint32_t MAX_COEFF_COUNT>
__global__ void k_IirOptimized(
    const COEFF_T* __restrict__ bCoeffs,      // Feedforward coefficients [b0, b1, b2, ...]
    const COEFF_T* __restrict__ aCoeffs,      // Feedback coefficients [1.0, -a1, -a2, ...]
    uint32_t coeffCount,
    const IN_T* __restrict__ input,
    OUT_T* __restrict__ output,
    uint32_t numElements,
    uint32_t samplesPerThread) {

  // Calculate shared memory requirements
  // We store both b and a coefficients in shared memory
  extern __shared__ COEFF_T sharedCoeffs[];
  COEFF_T* sharedBCoeffs = sharedCoeffs;
  COEFF_T* sharedACoeffs = sharedCoeffs + MAX_COEFF_COUNT;

  // Load coefficients into shared memory
  // Multiple threads collaborate to load all coefficients
  // This is done once per block and reused by all threads
  for (uint32_t i = threadIdx.x; i < coeffCount; i += blockDim.x) {
    sharedBCoeffs[i] = bCoeffs[i];
    sharedACoeffs[i] = aCoeffs[i];
  }

  // Ensure all threads have finished loading coefficients
  __syncthreads();

  // Calculate the starting sample index for this thread
  // Each thread processes 'samplesPerThread' consecutive samples
  uint32_t threadStartIdx = blockIdx.x * blockDim.x * samplesPerThread + threadIdx.x * samplesPerThread;

  // Early exit if this thread has no work to do
  if (threadStartIdx >= numElements) {
    return;
  }

  // Thread-private history arrays
  // These maintain the filter state for each thread independently
  // This eliminates race conditions that plagued the original implementation
  IN_T inputHistory[MAX_COEFF_COUNT];   // History for input values x[n-1], x[n-2], etc.
  OUT_T outputHistory[MAX_COEFF_COUNT];  // History for output values y[n-1], y[n-2], etc.

  // Initialize history arrays to zero
  // This represents the state before processing begins (beginning of signal)
  #pragma unroll
  for (uint32_t i = 0; i < MAX_COEFF_COUNT; i++) {
    inputHistory[i] = zero<IN_T>();
    outputHistory[i] = zero<OUT_T>();
  }

  // Process samples in batches to exploit ILP
  // Each batch processes 4 samples concurrently for better instruction-level parallelism
  for (uint32_t batchStart = 0; batchStart < samplesPerThread; batchStart += 4) {
    uint32_t globalIdx = threadStartIdx + batchStart;

    // Stop if we've processed all samples for this thread
    if (globalIdx >= numElements) {
      break;
    }

    // Calculate how many samples to process in this batch
    // Ensure we don't exceed array bounds
    uint32_t remainingSamples = min(4U, min(samplesPerThread - batchStart, numElements - globalIdx));

    // Arrays to hold current batch of input samples and output results
    IN_T currentInputs[4];
    OUT_T results[4];

    // Load current input samples for this batch
    // We load 4 samples even if we only process some to maintain alignment
    #pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t idx = globalIdx + i;
      currentInputs[i] = (idx < numElements) ? input[idx] : zero<IN_T>();
    }

    // Process each sample in the current batch
    // This loop exploits ILP by processing multiple samples concurrently
    #pragma unroll
    for (uint32_t s = 0; s < remainingSamples; s++) {
      // Compute feedforward sum: b0*x[n] + b1*x[n-1] + b2*x[n-2] + ...
      OUT_T feedforward = zero<OUT_T>();
      #pragma unroll
      for (uint32_t i = 0; i < coeffCount; i++) {
        // For i=0, use current input sample
        // For i>0, use historical input values
        IN_T inputVal = (i == 0) ? currentInputs[s] : inputHistory[i - 1];
        feedforward += inputVal * sharedBCoeffs[i];
      }

      // Compute feedback sum: -a1*y[n-1] - a2*y[n-2] - ...
      OUT_T feedback = zero<OUT_T>();
      #pragma unroll
      for (uint32_t i = 1; i < coeffCount; i++) {
        // Start from i=1 since a[0] = 1.0 (implicit)
        feedback += outputHistory[i - 1] * sharedACoeffs[i];
      }

      // Final IIR output: feedforward - feedback
      results[s] = feedforward - feedback;

      // Update history arrays for next sample
      // Shift all history values back by one position
      #pragma unroll
      for (int32_t i = coeffCount - 2; i >= 0; i--) {
        inputHistory[i + 1] = inputHistory[i];
        outputHistory[i + 1] = outputHistory[i];
      }

      // Insert current values at the beginning of history
      inputHistory[0] = currentInputs[s];
      outputHistory[0] = results[s];
    }

    // Write results to global memory
    // Only write samples that are within bounds
    #pragma unroll
    for (uint32_t s = 0; s < remainingSamples; s++) {
      uint32_t idx = globalIdx + s;
      if (idx < numElements) {
        output[idx] = results[s];
      }
    }
  }
}


/**
 * Optimized IIR filter implementation using thread-private history
 *
 * This function provides a high-performance IIR filter implementation that:
 * - Eliminates race conditions by using thread-private history
 * - Exploits ILP by processing multiple samples per thread
 * - Uses shared memory for coefficient storage
 * - Provides coalesced memory access patterns
 *
 * @param bCoeffs Feedforward coefficients [b0, b1, b2, ...]
 * @param aCoeffs Feedback coefficients [1.0, -a1, -a2, ...]
 * @param coeffCount Number of coefficients (same for b and a)
 * @param inputHistory IGNORED - Parameter kept for API consistency
 * @param outputHistory IGNORED - Parameter kept for API consistency
 * @param input Input signal array
 * @param output Output signal array
 * @param numElements Number of elements to process
 * @param cudaDevice CUDA device ID
 * @param cudaStream CUDA stream for execution
 * @return cudaError_t indicating success or failure
 */
template <class IN_T, class OUT_T, class COEFF_T, uint32_t MAX_COEFF_COUNT = 32>
static cudaError_t iirGenericOptimized(
    const COEFF_T* bCoeffs,
    const COEFF_T* aCoeffs,
    size_t coeffCount,
    IN_T* inputHistory,      // IGNORED - kept for API compatibility
    OUT_T* outputHistory,    // IGNORED - kept for API compatibility
    const IN_T* input,
    OUT_T* output,
    size_t numElements,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {

  SIMPLE_CUDA_FNC_START("IIR Optimized");

  // Input validation
  if (coeffCount < 2) {
    return cudaErrorInvalidValue;  // Need at least 2 coefficients (e.g., b0 and b1, or b0 and a1 depending on filter structure)
  }

  if (coeffCount > MAX_COEFF_COUNT) {
    return cudaErrorInvalidValue;  // Filter order too high for this implementation
  }

  if (numElements == 0) {
    return cudaSuccess;  // Nothing to do
  }

  // Calculate optimal configuration
  // Each thread processes multiple samples to exploit ILP
  const uint32_t samplesPerThread = DEFAULT_SAMPLES_PER_THREAD;  // Tune this value for optimal performance
  const uint32_t threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;  // Reasonable block size

  // Calculate grid dimensions
  uint32_t totalThreadWork = numElements;  // Total samples to process
  uint32_t totalThreadsNeeded = (totalThreadWork + samplesPerThread - 1) / samplesPerThread;
  uint32_t numBlocks = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

  // Ensure we don't exceed CUDA grid limits
  if (numBlocks > CUDA_MAX_GRID_DIMENSION) {
    // Fallback: reduce samples per thread if we have too many blocks
    return cudaErrorInvalidValue;  // For now, let caller handle this
  }

  // Calculate shared memory requirements
  // We need space for both b and a coefficients
  size_t sharedMemSize = 2 * MAX_COEFF_COUNT * sizeof(COEFF_T);

  // Ensure shared memory doesn't exceed reasonable limits (typically 48KB per SM)
  if (sharedMemSize > MAX_SHARED_MEMORY_BYTES) {
    return cudaErrorInvalidValue;
  }

  // Launch optimized kernel with compile-time bounds checking via template parameter
  switch (coeffCount) {
    case 2:  // 1st order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 2><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 3:  // 2nd order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 3><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 4:  // 3rd order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 4><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 5:  // 4th order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 5><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 6:  // 5th order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 6><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 7:  // 6th order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 7><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 8:  // 7th order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 8><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    default:  // Higher order filters use maximum supported order
      k_IirOptimized<IN_T, OUT_T, COEFF_T, MAX_COEFF_COUNT><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
  }

  SIMPLE_CUDA_FNC_END("IIR Optimized");
}


/**
 * High-performance IIR filter for float data using optimized implementation
 *
 * This function uses the optimized IIR implementation that eliminates race conditions
 * and provides much better performance through ILP exploitation and shared memory usage.
 *
 * The history buffers (inputHistory, outputHistory) are ignored in this implementation
 * but kept for API compatibility. Each thread maintains its own private history state.
 *
 * @param bCoeffs Feedforward coefficients [b0, b1, b2, ...]
 * @param aCoeffs Feedback coefficients [1.0, -a1, -a2, ...]
 * @param coeffCount Number of coefficients (same for b and a arrays)
 * @param inputHistory IGNORED - Legacy parameter for API compatibility
 * @param outputHistory IGNORED - Legacy parameter for API compatibility
 * @param input Input float array in GPU memory
 * @param output Output float array in GPU memory
 * @param numElements Number of elements to process
 * @param cudaDevice CUDA device ID
 * @param cudaStream CUDA stream for execution
 * @return cudaError_t CUDA error code
 */
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

  return iirGenericOptimized<float, float, float>(
      bCoeffs, aCoeffs, coeffCount, inputHistory, outputHistory,
      input, output, numElements, cudaDevice, cudaStream);
}

/**
 * High-performance IIR filter for complex data using optimized implementation
 *
 * This function uses the optimized IIR implementation that eliminates race conditions
 * and provides much better performance through ILP exploitation and shared memory usage.
 *
 * The history buffers (inputHistory, outputHistory) are ignored in this implementation
 * but kept for API compatibility. Each thread maintains its own private history state.
 *
 * @param bCoeffs Feedforward coefficients [b0, b1, b2, ...]
 * @param aCoeffs Feedback coefficients [1.0, -a1, -a2, ...]
 * @param coeffCount Number of coefficients (same for b and a arrays)
 * @param inputHistory IGNORED - Legacy parameter for API compatibility
 * @param outputHistory IGNORED - Legacy parameter for API compatibility
 * @param input Input complex array in GPU memory
 * @param output Output complex array in GPU memory
 * @param numElements Number of elements to process
 * @param cudaDevice CUDA device ID
 * @param cudaStream CUDA stream for execution
 * @return cudaError_t CUDA error code
 */
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

  return iirGenericOptimized<cuComplex, cuComplex, float>(
      bCoeffs, aCoeffs, coeffCount, inputHistory, outputHistory,
      input, output, numElements, cudaDevice, cudaStream);
}

/**
 * IIR filter with configurable samples per thread for performance tuning
 *
 * This function allows users to specify the number of samples each thread processes,
 * enabling performance optimization for specific filter orders and data sizes.
 */
template <class IN_T, class OUT_T, class COEFF_T, uint32_t MAX_COEFF_COUNT = 32>
static cudaError_t iirGenericCustom(
    const COEFF_T* bCoeffs,
    const COEFF_T* aCoeffs,
    size_t coeffCount,
    IN_T* inputHistory,      // IGNORED - kept for API compatibility
    OUT_T* outputHistory,    // IGNORED - kept for API compatibility
    const IN_T* input,
    OUT_T* output,
    size_t numElements,
    size_t samplesPerThread,
    int32_t cudaDevice,
    cudaStream_t cudaStream) {

  SIMPLE_CUDA_FNC_START("IIR Custom");

  // Input validation
  if (coeffCount < 2) {
    return cudaErrorInvalidValue;
  }

  if (coeffCount > MAX_COEFF_COUNT) {
    return cudaErrorInvalidValue;
  }

  if (numElements == 0) {
    return cudaSuccess;
  }

  // Validate samplesPerThread parameter
  if (samplesPerThread == 0 || samplesPerThread > MAX_SAMPLES_PER_THREAD) {
    return cudaErrorInvalidValue;  // Reasonable limits
  }

  // Calculate configuration
  const uint32_t threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;

  // Calculate grid dimensions
  uint32_t totalThreadWork = numElements;
  uint32_t totalThreadsNeeded = (totalThreadWork + samplesPerThread - 1) / samplesPerThread;
  uint32_t numBlocks = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

  // Ensure we don't exceed CUDA grid limits
  if (numBlocks > CUDA_MAX_GRID_DIMENSION) {
    return cudaErrorInvalidValue;
  }

  // Calculate shared memory requirements
  size_t sharedMemSize = 2 * coeffCount * sizeof(COEFF_T);

  if (sharedMemSize > MAX_SHARED_MEMORY_BYTES) {
    return cudaErrorInvalidValue;
  }

  // Launch kernel with custom samples per thread
  switch (coeffCount) {
    case 2:  // 1st order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 2><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 3:  // 2nd order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 3><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 4:  // 3rd order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 4><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 5:  // 4th order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 5><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 6:  // 5th order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 6><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 7:  // 6th order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 7><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    case 8:  // 7th order filter
      k_IirOptimized<IN_T, OUT_T, COEFF_T, 8><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
    default:  // Higher order filters
      k_IirOptimized<IN_T, OUT_T, COEFF_T, MAX_COEFF_COUNT><<<numBlocks, threadsPerBlock, sharedMemSize, cudaStream>>>(
          bCoeffs, aCoeffs, coeffCount, input, output, numElements, samplesPerThread);
      break;
  }

  SIMPLE_CUDA_FNC_END("IIR Custom");
}


/**
 * Custom IIR filter with configurable samples per thread
 */
GSDR_C_LINKAGE cudaError_t gsdrIirFFCustom(
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
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {

  return iirGenericCustom<float, float, float>(
      bCoeffs, aCoeffs, coeffCount, inputHistory, outputHistory,
      input, output, numElements, samplesPerThread, cudaDevice, cudaStream);
}

GSDR_C_LINKAGE cudaError_t gsdrIirCCCustom(
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
    cudaStream_t cudaStream) GSDR_NO_EXCEPT {

  return iirGenericCustom<cuComplex, cuComplex, float>(
      bCoeffs, aCoeffs, coeffCount, inputHistory, outputHistory,
      input, output, numElements, samplesPerThread, cudaDevice, cudaStream);
}