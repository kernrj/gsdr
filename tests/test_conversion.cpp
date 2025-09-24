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

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>

#include "test_main.cpp"
#include <gsdr/conversion.h>

class ConversionTest : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();
    }

    cudaError_t error = cudaSuccess;
    const size_t testSize = 1024;

    std::vector<int8_t> generateInt8TestData(size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int8_t> dis(-128, 127);

        std::vector<int8_t> data(size);
        std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
        return data;
    }
};

TEST_F(ConversionTest, Int8ToFloatTest) {
    auto input = generateInt8TestData(testSize);
    auto output = std::vector<float>(testSize);

    error = gsdrInt8ToNormFloat(input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Int8ToNormFloat failed";

    // Verify results - int8 values should be normalized to [-1, 1]
    for (size_t i = 0; i < testSize; ++i) {
        float expected = input[i] / 127.0f; // Normalize to [-1, 1] range
        EXPECT_NEAR(output[i], expected, 1e-6f) << "Incorrect conversion at index " << i;
        EXPECT_GE(output[i], -1.0f) << "Value too negative at index " << i;
        EXPECT_LE(output[i], 1.0f) << "Value too positive at index " << i;
    }
}

TEST_F(ConversionTest, Int8ToFloatRangeTest) {
    // Test edge cases
    std::vector<int8_t> edgeCases = {-128, -127, -1, 0, 1, 126, 127};
    auto output = std::vector<float>(edgeCases.size());

    error = gsdrInt8ToNormFloat(edgeCases.data(), output.data(), edgeCases.size(), 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Int8ToNormFloat failed for edge cases";

    // Check specific values
    EXPECT_NEAR(output[0], -1.0f, 1e-6f) << "Minimum int8 should map to -1.0f";
    EXPECT_NEAR(output[1], -1.0f, 1e-6f) << "int8 -127 should map to -1.0f";
    EXPECT_EQ(output[3], 0.0f) << "Zero should remain zero";
    EXPECT_NEAR(output[5], 1.0f, 1e-6f) << "int8 126 should map to 1.0f";
    EXPECT_NEAR(output[6], 1.0f, 1e-6f) << "Maximum int8 should map to 1.0f";
}

TEST_F(ConversionTest, ZeroLengthTest) {
    std::vector<int8_t> input;
    std::vector<float> output;

    // Test with zero length (should not crash)
    error = gsdrInt8ToNormFloat(input.data(), output.data(), 0, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Int8ToNormFloat failed with zero length";
}

TEST_F(ConversionTest, LargeArrayTest) {
    const size_t largeSize = 65536; // 64K elements
    auto input = generateInt8TestData(largeSize);
    auto output = std::vector<float>(largeSize);

    error = gsdrInt8ToNormFloat(input.data(), output.data(), largeSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Int8ToNormFloat failed for large array";

    // Verify a few random samples
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < 100; ++i) {
        size_t randomIndex = gen() % largeSize;
        float expected = input[randomIndex] / 127.0f;
        EXPECT_NEAR(output[randomIndex], expected, 1e-6f)
            << "Incorrect conversion at random index " << randomIndex;
    }
}

TEST_F(ConversionTest, PrecisionTest) {
    // Test precision with specific values
    std::vector<int8_t> preciseInput = {0, 1, -1, 64, -64, 127, -128};
    auto output = std::vector<float>(preciseInput.size());

    error = gsdrInt8ToNormFloat(preciseInput.data(), output.data(), preciseInput.size(), 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Int8ToNormFloat failed for precision test";

    // Check precise values
    EXPECT_EQ(output[0], 0.0f);
    EXPECT_NEAR(output[1], 1.0f / 127.0f, 1e-6f);
    EXPECT_NEAR(output[2], -1.0f / 127.0f, 1e-6f);
    EXPECT_NEAR(output[3], 64.0f / 127.0f, 1e-6f);
    EXPECT_NEAR(output[4], -64.0f / 127.0f, 1e-6f);
    EXPECT_NEAR(output[5], 126.0f / 127.0f, 1e-6f);
    EXPECT_EQ(output[6], -1.0f);
}

TEST_F(ConversionTest, StatisticalTest) {
    const size_t statSize = 10000;
    auto input = generateInt8TestData(statSize);
    auto output = std::vector<float>(statSize);

    error = gsdrInt8ToNormFloat(input.data(), output.data(), statSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Int8ToNormFloat failed for statistical test";

    // Calculate statistics
    float sum = 0.0f, sumSq = 0.0f;
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();

    for (float val : output) {
        sum += val;
        sumSq += val * val;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    float mean = sum / statSize;
    float variance = (sumSq / statSize) - (mean * mean);
    float stddev = sqrtf(variance);

    // For uniformly distributed int8 values, the normalized float values
    // should have mean ≈ 0 and stddev ≈ 1/sqrt(3) ≈ 0.577
    EXPECT_NEAR(mean, 0.0f, 0.01f) << "Mean should be approximately 0, got " << mean;
    EXPECT_NEAR(stddev, 1.0f / sqrtf(3.0f), 0.05f)
        << "Stddev should be approximately 0.577, got " << stddev;

    // Range should be approximately [-1, 1]
    EXPECT_GE(minVal, -1.0f) << "Minimum value should be >= -1.0f";
    EXPECT_LE(maxVal, 1.0f) << "Maximum value should be <= 1.0f";
}

TEST_F(ConversionTest, DeterministicTest) {
    // Test that the same input always produces the same output
    auto input = generateInt8TestData(testSize);
    auto output1 = std::vector<float>(testSize);
    auto output2 = std::vector<float>(testSize);

    // Run conversion twice
    error = gsdrInt8ToNormFloat(input.data(), output1.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "First Int8ToNormFloat failed";

    error = gsdrInt8ToNormFloat(input.data(), output2.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Second Int8ToNormFloat failed";

    // Results should be identical
    for (size_t i = 0; i < testSize; ++i) {
        EXPECT_EQ(output1[i], output2[i]) << "Non-deterministic result at index " << i;
    }
}

TEST_F(ConversionTest, BoundaryTest) {
    // Test boundary conditions
    const std::vector<size_t> testSizes = {1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025};

    for (size_t size : testSizes) {
        auto input = generateInt8TestData(size);
        auto output = std::vector<float>(size);

        error = gsdrInt8ToNormFloat(input.data(), output.data(), size, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "Int8ToNormFloat failed for size " << size;

        // Verify range for this size
        for (size_t i = 0; i < size; ++i) {
            float expected = input[i] / 127.0f;
            EXPECT_NEAR(output[i], expected, 1e-6f) << "Incorrect conversion at index " << i;
            EXPECT_GE(output[i], -1.0f) << "Value too negative at index " << i;
            EXPECT_LE(output[i], 1.0f) << "Value too positive at index " << i;
        }
    }
}

TEST_F(ConversionTest, PerformanceTest) {
    // Performance test with larger arrays
    const size_t perfSize = 1048576; // 1M elements
    auto input = generateInt8TestData(perfSize);
    auto output = std::vector<float>(perfSize);

    // Time the conversion (this is a basic performance check)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    error = gsdrInt8ToNormFloat(input.data(), output.data(), perfSize, 0, stream);
    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);
    EXPECT_EQ(error, cudaSuccess) << "Int8ToNormFloat failed for performance test";

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Should complete in reasonable time (less than 100ms for 1M elements)
    EXPECT_LT(milliseconds, 100.0f) << "Performance test took too long: " << milliseconds << "ms";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}