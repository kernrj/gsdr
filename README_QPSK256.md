# QPSK256 Implementation for GSDR

This document describes the QPSK256 (256-ary Quadrature Phase Shift Keying) modulation and demodulation implementation for the GSDR library.

## Overview

QPSK256 is a high-order digital modulation scheme where each symbol represents 8 bits of information (256 constellation points). This provides 8x the spectral efficiency of standard QPSK. The implementation provides:

- **256-ary modulation**: Maps 8-bit values (0-255) to complex constellation points
- **Two constellation types**: Rectangular (16x16 grid) and Circular (optimal circle packing)
- **Single-stream processing**: Standard QPSK256 modulation/demodulation for one stream
- **Multi-stream processing**: Optimized kernels for processing 4 streams simultaneously
- **Efficient demodulation**: Nearest-neighbor search using constant memory for fast lookup

## Constellation Types

### Rectangular Constellation (Type 0)
- **Arrangement**: 16x16 grid (256-QAM like)
- **Advantages**: Easy to implement, uniform spacing
- **Peak-to-average power ratio**: Moderate
- **Use case**: AWGN channels, systems requiring regular spacing

### Circular Constellation (Type 1)
- **Arrangement**: Points arranged in concentric circles with optimal spacing
- **Advantages**: Better peak-to-average power ratio, optimal for nonlinear channels
- **Spacing**: Based on circle packing theory for maximum minimum distance
- **Use case**: Nonlinear channels, power amplifier limited systems

## Constellation Point Distribution (Circular)

The circular constellation uses the following distribution:

| Circle | Points | Radius | Total Points |
|--------|--------|--------|--------------|
| 0      | 1      | 0.0    | 1            |
| 1      | 8      | 0.3    | 9            |
| 2      | 16     | 0.6    | 25           |
| 3      | 24     | 0.85   | 49           |
| 4      | 32     | 1.1    | 81           |
| 5      | 40     | 1.35   | 121          |
| 6      | 48     | 1.6    | 169          |
| 7      | 56     | 1.85   | 225          |

Remaining points (up to 256) are placed at radius 1.95 with uniform angular spacing.

## API Functions

### Initialization (Required)

```cpp
// Initialize constellation points in device constant memory
cudaError_t gsdrQpsk256InitConstellation(
    uint32_t constellationType,  // 0=rectangular, 1=circular
    float amplitude,             // Amplitude scaling factor
    int32_t cudaDevice,          // CUDA device index
    cudaStream_t cudaStream);    // CUDA stream
```

**Note**: Must be called once before using QPSK256 functions.

### Basic QPSK256 Functions

```cpp
// Single stream modulation
cudaError_t gsdrQpsk256Modulate(
    const uint8_t* inputBytes,   // Input bytes (0-255), each is one symbol
    cuComplex* output,           // Complex output samples
    uint32_t numSymbols,         // Number of symbols to generate
    float amplitude,             // Amplitude scaling factor
    uint32_t constellationType,  // 0=rectangular, 1=circular
    int32_t cudaDevice,          // CUDA device index
    cudaStream_t cudaStream);    // CUDA stream

// Single stream demodulation
cudaError_t gsdrQpsk256Demodulate(
    const cuComplex* input,      // Complex input samples
    uint8_t* outputBytes,        // Output bytes (0-255)
    uint32_t numSymbols,         // Number of symbols to demodulate
    uint32_t constellationType,  // 0=rectangular, 1=circular
    int32_t cudaDevice,          // CUDA device index
    cudaStream_t cudaStream);    // CUDA stream
```

### Multi-Stream Functions (4 streams)

```cpp
// 4-stream modulation
cudaError_t gsdrQpsk256Modulate4x(
    const uint8_t* inputBytes0, const uint8_t* inputBytes1,
    const uint8_t* inputBytes2, const uint8_t* inputBytes3,
    cuComplex* output0, cuComplex* output1,
    cuComplex* output2, cuComplex* output3,
    uint32_t numSymbols, float amplitude, uint32_t constellationType,
    int32_t cudaDevice, cudaStream_t cudaStream);

// 4-stream demodulation
cudaError_t gsdrQpsk256Demodulate4x(
    const cuComplex* input0, const cuComplex* input1,
    const cuComplex* input2, const cuComplex* input3,
    uint8_t* outputBytes0, uint8_t* outputBytes1,
    uint8_t* outputBytes2, uint8_t* outputBytes3,
    uint32_t numSymbols, uint32_t constellationType,
    int32_t cudaDevice, cudaStream_t cudaStream);
```

## Performance Characteristics

### Spectral Efficiency
- **8 bits per symbol** (vs 2 bits for QPSK)
- **4x higher spectral efficiency** than standard QPSK
- **256 constellation points** for maximum likelihood detection

### Computational Complexity
- **Modulation**: O(1) - Direct table lookup
- **Demodulation**: O(256) - Nearest neighbor search per symbol
- **Memory**: 256 complex values in constant memory

### Error Performance
- **Rectangular**: Better performance in AWGN channels
- **Circular**: Better performance in nonlinear channels
- **Requires higher SNR** than lower-order modulation schemes

## Usage Example

```cpp
#include <gsdr/qpsk256.h>

// Example: QPSK256 modulation and demodulation
const uint32_t numSymbols = 1024;
const float amplitude = 1.0f;
const uint32_t constellationType = 1; // Circular constellation

// Input data (1024 bytes = 1024 symbols)
uint8_t* inputData = new uint8_t[numSymbols];
for (int i = 0; i < numSymbols; ++i) {
    inputData[i] = i % 256; // Example data
}

// Device memory pointers
uint8_t* d_input;
cuComplex* d_modulated;
uint8_t* d_demodulated;

// Allocate device memory...
cudaMalloc(&d_input, numSymbols * sizeof(uint8_t));
cudaMalloc(&d_modulated, numSymbols * sizeof(cuComplex));
cudaMalloc(&d_demodulated, numSymbols * sizeof(uint8_t));

// Copy input data to device...
cudaMemcpy(d_input, inputData, numSymbols * sizeof(uint8_t), cudaMemcpyHostToDevice);

// Step 1: Initialize constellation
cudaError_t error = gsdrQpsk256InitConstellation(
    constellationType, amplitude, 0, 0);
if (error != cudaSuccess) {
    // Handle error...
}

// Step 2: Modulate
error = gsdrQpsk256Modulate(
    d_input, d_modulated, numSymbols, amplitude,
    constellationType, 0, 0);
if (error != cudaSuccess) {
    // Handle error...
}

// Step 3: Demodulate (in real applications, this would be done on received signal)
error = gsdrQpsk256Demodulate(
    d_modulated, d_demodulated, numSymbols,
    constellationType, 0, 0);
if (error != cudaSuccess) {
    // Handle error...
}

// Copy results back to host...
uint8_t* outputData = new uint8_t[numSymbols];
cudaMemcpy(outputData, d_demodulated, numSymbols * sizeof(uint8_t), cudaMemcpyDeviceToHost);

// Clean up...
cudaFree(d_input);
cudaFree(d_modulated);
cudaFree(d_demodulated);
delete[] inputData;
delete[] outputData;
```

## Implementation Details

### CUDA Kernel Architecture

The QPSK256 kernels follow optimized patterns:

```cpp
__global__ void k_Qpsk256Modulate(
    const uint8_t* inputBytes,
    cuComplex* output,
    uint32_t numSymbols,
    float amplitude,
    uint32_t constellationType) {

  uint32_t symbolIndex = blockDim.x * blockIdx.x + threadIdx.x;

  if (symbolIndex >= numSymbols) {
    return;
  }

  // Direct lookup from constant memory
  uint8_t symbol = inputBytes[symbolIndex];
  cuComplex point = (constellationType == 0) ?
      c_qpsk256_rectangular[symbol] : c_qpsk256_circular[symbol];

  output[symbolIndex] = point;
}
```

### Demodulation Strategy

Demodulation uses exhaustive nearest-neighbor search:

1. **Load received symbol** from global memory
2. **Compute distance** to all 256 constellation points
3. **Track minimum distance** and corresponding symbol
4. **Output best symbol** (8-bit value)

### Memory Organization

- **Constant Memory**: 256 constellation points (2 × 256 × 8 = 4KB)
- **Global Memory**: Input/output arrays
- **Thread Organization**: 32 threads per warp for optimal memory coalescing

## Performance Optimizations

### Instruction-Level Parallelism
- **SIMD Operations**: Vectorized distance calculations
- **Warp Optimization**: 32-thread blocks for efficient GPU utilization
- **Memory Coalescing**: Optimized global memory access patterns

### Constant Memory Usage
- **Fast Lookup**: Constellation points stored in constant memory
- **Broadcast**: Single read serves entire warp
- **Cache Efficiency**: High hit rate for constellation lookups

### Multi-Stream Processing
- **4x Throughput**: Simultaneous processing of 4 independent streams
- **Resource Sharing**: Shared constellation memory across streams
- **Independent Operation**: No inter-stream dependencies

## Error Handling

All functions return `cudaError_t` and follow GSDR patterns:

- **Device Management**: Proper CUDA context handling
- **Memory Validation**: Input parameter validation
- **Stream Synchronization**: Proper stream ordering
- **Error Propagation**: Detailed error reporting

## Build Integration

The QPSK256 implementation is integrated into the GSDR build system:

1. **Header file**: `include/gsdr/qpsk256.h`
2. **Implementation**: `src/qpsk256.cu`
3. **Build integration**: Added to `CMakeLists.txt`
4. **Export headers**: Automatic generation by build system

## Applications

QPSK256 is suitable for:

- **High-speed data transmission** requiring spectral efficiency
- **Cable modems** and broadband systems
- **Satellite communications** with high SNR
- **Systems requiring** 8 bits per symbol modulation
- **Research applications** studying high-order modulation

## Notes

- **Initialization Required**: Must call `gsdrQpsk256InitConstellation()` before use
- **Higher SNR Required**: 256-ary modulation needs better channel conditions
- **Computational Cost**: Demodulation has O(256) complexity per symbol
- **Memory Efficient**: Only 4KB of constant memory for constellation data