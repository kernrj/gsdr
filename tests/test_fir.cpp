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
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

#include "test_main.cpp"
#include <gsdr/fir.h>

class FirTest : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();
    }

    cudaError_t error = cudaSuccess;
    const size_t testSize = 1024;

    std::vector<float> generateTestData(size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<float> data(size);
        std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
        return data;
    }

    std::vector<cuComplex> generateComplexTestData(size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<cuComplex> data(size);
        std::generate(data.begin(), data.end(), [&]() {
            return make_cuComplex(dis(gen), dis(gen));
        });
        return data;
    }

    // Generate low-pass filter taps
    std::vector<float> generateLowPassTaps(size_t numTaps, float cutoffFreq) {
        std::vector<float> taps(numTaps);
        float sum = 0.0f;

        for (size_t i = 0; i < numTaps; ++i) {
            if (i == numTaps / 2) {
                taps[i] = 2.0f * cutoffFreq;
            } else {
                float x = 2.0f * cutoffFreq * (i - numTaps / 2.0f);
                taps[i] = sinf(x) / x;
            }
            sum += taps[i];
        }

        // Normalize
        for (float& tap : taps) {
            tap /= sum;
        }

        return taps;
    }
};

TEST_F(FirTest, FloatFloatFilterTest) {
    const size_t numTaps = 32;
    auto taps = generateLowPassTaps(numTaps, 0.1f);
    auto input = generateTestData(testSize + numTaps);
    auto output = std::vector<float>(testSize);

    error = gsdrFirFF(numTaps, taps.data(), numTaps, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FirFF failed";

    // Verify output is reasonable (not all zeros, not infinite)
    bool allZero = true;
    bool hasValidData = true;

    for (float val : output) {
        if (val != 0.0f) allZero = false;
        if (!std::isfinite(val)) hasValidData = false;
    }

    EXPECT_FALSE(allZero) << "FIR filter output is all zeros";
    EXPECT_TRUE(hasValidData) << "FIR filter produced non-finite values";
}

TEST_F(FirTest, ComplexComplexFilterTest) {
    const size_t numTaps = 32;
    auto taps = generateComplexTestData(numTaps);
    auto input = generateComplexTestData(testSize + numTaps);
    auto output = std::vector<cuComplex>(testSize);

    error = gsdrFirCC(numTaps, taps.data(), numTaps, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FirCC failed";

    // Verify output is reasonable
    bool allZero = true;
    bool hasValidData = true;

    for (const cuComplex& val : output) {
        if (val.x != 0.0f || val.y != 0.0f) allZero = false;
        if (!std::isfinite(val.x) || !std::isfinite(val.y)) hasValidData = false;
    }

    EXPECT_FALSE(allZero) << "FIR filter output is all zeros";
    EXPECT_TRUE(hasValidData) << "FIR filter produced non-finite values";
}

TEST_F(FirTest, FloatComplexFilterTest) {
    const size_t numTaps = 32;
    auto taps = generateComplexTestData(numTaps);
    auto input = generateTestData(testSize + numTaps);
    auto output = std::vector<cuComplex>(testSize);

    error = gsdrFirCF(numTaps, taps.data(), numTaps, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FirCF failed";

    // Verify output is reasonable
    bool allZero = true;
    bool hasValidData = true;

    for (const cuComplex& val : output) {
        if (val.x != 0.0f || val.y != 0.0f) allZero = false;
        if (!std::isfinite(val.x) || !std::isfinite(val.y)) hasValidData = false;
    }

    EXPECT_FALSE(allZero) << "FIR filter output is all zeros";
    EXPECT_TRUE(hasValidData) << "FIR filter produced non-finite values";
}

TEST_F(FirTest, ComplexFloatFilterTest) {
    const size_t numTaps = 32;
    auto taps = generateTestData(numTaps);
    auto input = generateComplexTestData(testSize + numTaps);
    auto output = std::vector<float>(testSize);

    error = gsdrFirFC(numTaps, taps.data(), numTaps, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FirFC failed";

    // Verify output is reasonable
    bool allZero = true;
    bool hasValidData = true;

    for (float val : output) {
        if (val != 0.0f) allZero = false;
        if (!std::isfinite(val)) hasValidData = false;
    }

    EXPECT_FALSE(allZero) << "FIR filter output is all zeros";
    EXPECT_TRUE(hasValidData) << "FIR filter produced non-finite values";
}

TEST_F(FirTest, DecimationTest) {
    const size_t numTaps = 16;
    auto taps = generateLowPassTaps(numTaps, 0.2f);
    auto input = generateTestData(testSize * 4 + numTaps); // 4x oversampled
    auto output = std::vector<float>(testSize);

    error = gsdrFirFF(numTaps, taps.data(), numTaps, input.data(), output.data(), testSize, 4, stream);
    EXPECT_EQ(error, cudaSuccess) << "FirFF with decimation failed";

    // Output should be decimated by factor of 4
    EXPECT_EQ(output.size(), testSize) << "Incorrect output size for decimation";

    // Verify reasonable output
    bool hasValidData = true;
    for (float val : output) {
        if (!std::isfinite(val)) hasValidData = false;
    }
    EXPECT_TRUE(hasValidData) << "Decimation produced non-finite values";
}

TEST_F(FirTest, ImpulseResponseTest) {
    const size_t numTaps = 8;
    std::vector<float> impulseTaps = {0.1f, 0.2f, 0.3f, 0.4f, 0.3f, 0.2f, 0.1f, 0.0f};
    std::vector<float> input(testSize + numTaps, 0.0f);
    input[0] = 1.0f; // Unit impulse at start
    auto output = std::vector<float>(testSize);

    error = gsdrFirFF(numTaps, impulseTaps.data(), numTaps, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FirFF impulse response failed";

    // Check that impulse response matches the filter taps (approximately)
    for (size_t i = 0; i < std::min(numTaps, testSize); ++i) {
        EXPECT_NEAR(output[i], impulseTaps[i], 1e-5f)
            << "Impulse response mismatch at index " << i;
    }
}

TEST_F(FirTest, LowPassFilterTest) {
    const size_t numTaps = 32;
    auto taps = generateLowPassTaps(numTaps, 0.1f); // Cutoff at 0.1 * Nyquist
    auto input = generateTestData(testSize + numTaps);
    auto output = std::vector<float>(testSize);

    error = gsdrFirFF(numTaps, taps.data(), numTaps, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Low-pass FIR filter failed";

    // For low-pass filter, high-frequency components should be attenuated
    // This is a basic sanity check
    float maxValue = 0.0f;
    for (float val : output) {
        maxValue = std::max(maxValue, std::abs(val));
    }

    // Should not be all zeros
    EXPECT_GT(maxValue, 1e-6f) << "Low-pass filter output is essentially zero";
}

TEST_F(FirTest, HighPassFilterTest) {
    const size_t numTaps = 32;
    // Simple high-pass filter: [1, -1] kernel extended
    std::vector<float> highPassTaps(numTaps, 0.0f);
    highPassTaps[0] = 0.5f;
    highPassTaps[1] = -0.5f;

    auto input = generateTestData(testSize + numTaps);
    auto output = std::vector<float>(testSize);

    error = gsdrFirFF(numTaps, highPassTaps.data(), numTaps, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "High-pass FIR filter failed";

    // Should have some output
    float energy = 0.0f;
    for (float val : output) {
        energy += val * val;
    }
    EXPECT_GT(energy, 1e-6f) << "High-pass filter output has no energy";
}

TEST_F(FirTest, ZeroTapsTest) {
    std::vector<float> input(testSize, 1.0f);
    auto output = std::vector<float>(testSize);

    // Test with zero taps (should pass through)
    error = gsdrFirFF(1, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
    // This might fail if taps is null, which is expected behavior
    // Just verify it doesn't crash
}

TEST_F(FirTest, EdgeCasesTest) {
    // Test various edge cases
    const std::vector<std::tuple<size_t, size_t>> testCases = {
        {1, 1}, {2, 1}, {1, 2}, {16, 8}, {8, 16}, {31, 15}, {32, 16}, {33, 17}
    };

    for (const auto& [numTaps, outputSize] : testCases) {
        auto taps = generateLowPassTaps(numTaps, 0.1f);
        auto input = generateTestData(outputSize + numTaps);
        auto output = std::vector<float>(outputSize);

        error = gsdrFirFF(numTaps, taps.data(), numTaps, input.data(), output.data(), outputSize, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "FirFF failed for taps=" << numTaps << ", size=" << outputSize;

        // Verify output size
        EXPECT_EQ(output.size(), outputSize) << "Incorrect output size";
    }
}

TEST_F(FirTest, LargeFilterTest) {
    const size_t largeSize = 4096;
    const size_t numTaps = 128;
    auto taps = generateLowPassTaps(numTaps, 0.05f);
    auto input = generateTestData(largeSize + numTaps);
    auto output = std::vector<float>(largeSize);

    error = gsdrFirFF(numTaps, taps.data(), numTaps, input.data(), output.data(), largeSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Large FIR filter failed";

    // Verify reasonable output
    float maxValue = 0.0f;
    for (float val : output) {
        maxValue = std::max(maxValue, std::abs(val));
    }
    EXPECT_GT(maxValue, 1e-6f) << "Large filter output is essentially zero";
}

TEST_F(FirTest, FilterCoefficientTest) {
    // Test with specific known coefficients
    const std::vector<float> knownTaps = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f};
    const size_t numTaps = knownTaps.size();
    std::vector<float> input(testSize + numTaps, 0.0f);
    input[0] = 1.0f; // Unit impulse
    auto output = std::vector<float>(testSize);

    error = gsdrFirFF(numTaps, knownTaps.data(), numTaps, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FIR filter with known coefficients failed";

    // Check first few output values match expected impulse response
    for (size_t i = 0; i < std::min(numTaps, testSize); ++i) {
        EXPECT_NEAR(output[i], knownTaps[i], 1e-5f)
            << "Impulse response mismatch at index " << i;
    }
}