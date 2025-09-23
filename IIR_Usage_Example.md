# IIR Filter Usage Example

This document explains how to use the new IIR (Infinite Impulse Response) filter functions in the GSDR library.

## Overview

The IIR filter implementation uses a direct form II structure, which is more numerically stable than direct form I. The filter processes input samples and produces output samples based on feedforward coefficients (b) and feedback coefficients (a).

## Function Signatures

### Float IIR Filter
```cpp
cudaError_t gsdrIirFF(
    const float* bCoeffs,        // Feedforward coefficients (b0, b1, b2, ...)
    const float* aCoeffs,        // Feedback coefficients (1.0, -a1, -a2, ...)
    size_t coeffCount,           // Number of coefficients
    float* inputHistory,         // Input history buffer (size: coeffCount-1)
    float* outputHistory,        // Output history buffer (size: coeffCount-1)
    const float* input,          // Input samples
    float* output,               // Output samples
    size_t numElements,          // Number of input samples
    int32_t cudaDevice,          // CUDA device ID
    cudaStream_t cudaStream      // CUDA stream
);
```

### Complex IIR Filter
```cpp
cudaError_t gsdrIirCC(
    const float* bCoeffs,        // Feedforward coefficients (b0, b1, b2, ...)
    const float* aCoeffs,        // Feedback coefficients (1.0, -a1, -a2, ...)
    size_t coeffCount,           // Number of coefficients
    cuComplex* inputHistory,     // Input history buffer (size: coeffCount-1)
    cuComplex* outputHistory,    // Output history buffer (size: coeffCount-1)
    const cuComplex* input,      // Input samples
    cuComplex* output,           // Output samples
    size_t numElements,          // Number of input samples
    int32_t cudaDevice,          // CUDA device ID
    cudaStream_t cudaStream      // CUDA stream
);
```

## Coefficient Format

For an IIR filter of the form:
```
y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] + ... + bm*x[n-m] - a1*y[n-1] - a2*y[n-2] - ... - an*y[n-n]
```

- `bCoeffs` should contain: `[b0, b1, b2, ..., bm]`
- `aCoeffs` should contain: `[1.0, -a1, -a2, ..., -an]` (note a0 is always 1.0)
- `coeffCount` should be `max(m, n) + 1`

## Example: Second-Order Low-Pass Filter

```cpp
#include <gsdr/gsdr.h>
#include <cuda_runtime.h>
#include <cmath>

// Design a 2nd order Butterworth low-pass filter
// Cutoff frequency: 0.1 * sampling_rate
float cutoff = 0.1f;
float samplingRate = 1.0f;  // Normalized
float nyquist = samplingRate / 2.0f;
float normalizedCutoff = cutoff / nyquist;

// Butterworth filter coefficients
float bCoeffs[3] = {1.0f, 2.0f, 1.0f};  // Will be normalized
float aCoeffs[3] = {1.0f, -1.561f, 0.6414f};

// Normalize b coefficients
float normFactor = 0.0985f;  // Depends on cutoff frequency
bCoeffs[0] *= normFactor;
bCoeffs[1] *= normFactor;
bCoeffs[2] *= normFactor;

size_t coeffCount = 3;
size_t historySize = coeffCount - 1;  // 2

// Allocate host memory
float* h_input = new float[numSamples];
float* h_output = new float[numSamples];
float* h_bCoeffs = new float[coeffCount];
float* h_aCoeffs = new float[coeffCount];

// Initialize coefficients
memcpy(h_bCoeffs, bCoeffs, coeffCount * sizeof(float));
memcpy(h_aCoeffs, aCoeffs, coeffCount * sizeof(float));

// Allocate device memory
float *d_input, *d_output, *d_bCoeffs, *d_aCoeffs;
float *d_inputHistory, *d_outputHistory;

cudaMalloc(&d_input, numSamples * sizeof(float));
cudaMalloc(&d_output, numSamples * sizeof(float));
cudaMalloc(&d_bCoeffs, coeffCount * sizeof(float));
cudaMalloc(&d_aCoeffs, coeffCount * sizeof(float));
cudaMalloc(&d_inputHistory, historySize * sizeof(float));
cudaMalloc(&d_outputHistory, historySize * sizeof(float));

// Copy data to device
cudaMemcpy(d_input, h_input, numSamples * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_bCoeffs, h_bCoeffs, coeffCount * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_aCoeffs, h_aCoeffs, coeffCount * sizeof(float), cudaMemcpyHostToDevice);

// Initialize history buffers to zero
cudaMemset(d_inputHistory, 0, historySize * sizeof(float));
cudaMemset(d_outputHistory, 0, historySize * sizeof(float));

// Apply IIR filter
cudaError_t error = gsdrIirFF(
    d_bCoeffs,
    d_aCoeffs,
    coeffCount,
    d_inputHistory,
    d_outputHistory,
    d_input,
    d_output,
    numSamples,
    0,  // device 0
    0   // default stream
);

// Copy result back
cudaMemcpy(h_output, d_output, numSamples * sizeof(float), cudaMemcpyDeviceToHost);

// Cleanup
cudaFree(d_input);
cudaFree(d_output);
cudaFree(d_bCoeffs);
cudaFree(d_aCoeffs);
cudaFree(d_inputHistory);
cudaFree(d_outputHistory);
delete[] h_input;
delete[] h_output;
delete[] h_bCoeffs;
delete[] h_aCoeffs;
```

## Important Notes

1. **History Buffers**: Must be pre-allocated with size `coeffCount - 1` and initialized to zero for the first run.

2. **Memory Layout**: All input/output data and coefficients must be in GPU memory.

3. **Thread Safety**: The current implementation processes elements sequentially to maintain history consistency. For high-performance applications, consider batching or using multiple filter instances.

4. **Coefficient Normalization**: Ensure your filter coefficients are properly normalized for the desired frequency response.

5. **Filter Stability**: Make sure your filter coefficients represent a stable filter (all poles inside the unit circle).

## Performance Considerations

- For best performance, process large batches of data at once
- The filter maintains internal state, so consecutive calls will continue from where the previous call left off
- Consider using multiple CUDA streams for concurrent processing