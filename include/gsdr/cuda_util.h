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

#ifndef SDRTEST_SRC_CUDA_UTIL_H_
#define SDRTEST_SRC_CUDA_UTIL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

/*
 * CLion shows a syntax error at the '<<<>>>' when invoking a kernel because
 * it doesn't have a declaration for this function. The error is cosmetic
 * because it compiles fine, but it's distracting.
 */
extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = nullptr);

#ifdef DEBUG
#define CHECK_CUDA_RET(__cmdDescriptionCStr)                        \
  do {                                                              \
    cudaError_t __checkCudaRetStatus = cudaDeviceSynchronize();     \
    if (__checkCudaRetStatus != cudaSuccess) {                      \
      const char* errName = cudaGetErrorName(__checkCudaRetStatus); \
      char __description[1024];                                     \
      snprintf(                                                     \
          __description,                                            \
          sizeof(__description) - 1,                                \
          "%s:%d - Error %d - %s - %s",                             \
          __FILE__,                                                 \
          __LINE__,                                                 \
          __checkCudaRetStatus,                                     \
          errName,                                                  \
          (__cmdDescriptionCStr));                                  \
      __description[sizeof(__description) - 1] = 0;                 \
                                                                    \
      fprintf(stderr, "%s\n", __description);                       \
                                                                    \
      return __checkCudaRetStatus;                                  \
    }                                                               \
  } while (false)
#else  // #ifdef DEBUG
#define CHECK_CUDA_RET(__cmdDescriptionCStr) (void)0
#endif  // #ifdef DEBUG

#define SAFE_CUDA_RET(__cmd)                                       \
  do {                                                             \
    CHECK_CUDA_RET("Before: " #__cmd);                             \
    cudaError_t __safeCudaRetStatus = (__cmd);                     \
    if (__safeCudaRetStatus != cudaSuccess) {                      \
      const char* errName = cudaGetErrorName(__safeCudaRetStatus); \
      char __description[1024];                                    \
      snprintf(                                                    \
          __description,                                           \
          sizeof(__description) - 1,                               \
          "%s:%d - Error %d - %s - %s",                            \
          __FILE__,                                                \
          __LINE__,                                                \
          __safeCudaRetStatus,                                     \
          errName,                                                 \
          #__cmd);                                                 \
      __description[sizeof(__description) - 1] = 0;                \
                                                                   \
      fprintf(stderr, "%s\n", __description);                      \
                                                                   \
      return __safeCudaRetStatus;                                  \
    }                                                              \
    CHECK_CUDA_RET("After: " #__cmd);                              \
  } while (false)

/**
 * Gets the current gsdr device, or returns a negative error code.
 * @return
 */
inline int32_t getCurrentCudaDevice() {
  int32_t device = -1;
  const cudaError_t status = cudaGetDevice(&device);

  if (status != cudaSuccess) {
    return -status;
  }

  return device;
}

#endif  // SDRTEST_SRC_CUDA_UTIL_H_
