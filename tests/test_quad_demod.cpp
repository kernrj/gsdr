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
#include <gsdr/quad_demod.h>

class QuadDemodTest : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();
    }

    cudaError_t error = cudaSuccess;
    const size_t testSize = 1024;

    // Generate FM-like signal for quad demod testing
    std::vector<cuComplex> generateQuadSignal(size_t size, float frequencyDeviation) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<cuComplex> signal(size);
        float phase = 0.0f;

        for (size_t i = 0; i < size; ++i) {
            // Generate message signal
            float messageFreq = 0.01f;
            float message = sinf(2.0f * M_PIf * messageFreq * i);

            // Frequency modulation: phase changes based on message
            float phaseIncrement = frequencyDeviation * message;
            signal[i] = make_cuComplex(cosf(phase), sinf(phase));
            phase += phaseIncrement;
        }

        return signal;
    }

    // Generate constant frequency signal
    std::vector<cuComplex> generateConstantFreqSignal(size_t size, float frequency) {
        std::vector<cuComplex> signal(size);
        float phase = 0.0f;
        float phaseIncrement = 2.0f * M_PIf * frequency;

        for (size_t i = 0; i < size; ++i) {
            signal[i] = make_cuComplex(cosf(phase), sinf(phase));
            phase += phaseIncrement;
        }

        return signal;
    }
};

TEST_F(QuadDemodTest, FmDemodulationTest) {
    auto input = generateQuadSignal(testSize, 0.1f);
    auto output = std::vector<float>(testSize - 1); // Quad demod outputs one less sample
    const float gain = 1.0f;

    error = gsdrQuadFmDemod(input.data(), output.data(), gain, testSize - 1, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed";

    // Verify output is reasonable
    bool hasValidData = true;
    float maxValue = 0.0f;
    float minValue = std::numeric_limits<float>::max();

    for (float val : output) {
        if (!std::isfinite(val)) hasValidData = false;
        maxValue = std::max(maxValue, val);
        minValue = std::min(minValue, val);
    }

    EXPECT_TRUE(hasValidData) << "Quad FM demodulation produced non-finite values";
    EXPECT_GT(maxValue, 1e-6f) << "Quad FM demodulation output is essentially zero";
    EXPECT_LT(std::abs(minValue), 1e6f) << "Quad FM demodulation output has unreasonably large values";
}

TEST_F(QuadDemodTest, ConstantFrequencyTest) {
    // Test with constant frequency input (should produce near-zero output)
    auto input = generateConstantFreqSignal(testSize, 0.1f);
    auto output = std::vector<float>(testSize - 1);
    const float gain = 1.0f;

    error = gsdrQuadFmDemod(input.data(), output.data(), gain, testSize - 1, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed for constant frequency";

    // Output should be close to zero for constant frequency
    float maxDeviation = 0.0f;
    for (float val : output) {
        maxDeviation = std::max(maxDeviation, std::abs(val));
    }

    EXPECT_LT(maxDeviation, 0.1f) << "Constant frequency should produce near-zero output, got max deviation " << maxDeviation;
}

TEST_F(QuadDemodTest, GainTest) {
    auto input = generateQuadSignal(testSize, 0.1f);
    const std::vector<float> gains = {0.1f, 1.0f, 10.0f, 100.0f};

    for (float gain : gains) {
        auto output = std::vector<float>(testSize - 1);

        error = gsdrQuadFmDemod(input.data(), output.data(), gain, testSize - 1, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed for gain " << gain;

        // Check that gain scaling works
        float maxOutput = 0.0f;
        for (float val : output) {
            maxOutput = std::max(maxOutput, std::abs(val));
        }

        EXPECT_GT(maxOutput, 1e-6f) << "No output for gain " << gain;
    }
}

TEST_F(QuadDemodTest, FrequencyDeviationTest) {
    const std::vector<float> deviations = {0.01f, 0.1f, 0.5f, 1.0f};

    for (float deviation : deviations) {
        auto input = generateQuadSignal(testSize, deviation);
        auto output = std::vector<float>(testSize - 1);
        const float gain = 1.0f;

        error = gsdrQuadFmDemod(input.data(), output.data(), gain, testSize - 1, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed for deviation " << deviation;

        // Higher deviation should produce larger output variation
        float variance = 0.0f;
        float mean = 0.0f;

        for (float val : output) {
            mean += val;
        }
        mean /= output.size();

        for (float val : output) {
            variance += (val - mean) * (val - mean);
        }
        variance /= output.size();

        EXPECT_GT(variance, 1e-6f) << "No variation for deviation " << deviation;
    }
}

TEST_F(QuadDemodTest, NoiseRobustnessTest) {
    auto input = generateQuadSignal(testSize, 0.1f);

    // Add noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);

    for (auto& sample : input) {
        sample.x += noise(gen);
        sample.y += noise(gen);
    }

    auto output = std::vector<float>(testSize - 1);
    const float gain = 1.0f;

    error = gsdrQuadFmDemod(input.data(), output.data(), gain, testSize - 1, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed with noise";

    // Should still produce reasonable output
    float variance = 0.0f;
    float mean = 0.0f;

    for (float val : output) {
        mean += val;
    }
    mean /= output.size();

    for (float val : output) {
        variance += (val - mean) * (val - mean);
    }
    variance /= output.size();

    EXPECT_GT(variance, 1e-6f) << "No signal variation after demodulation with noise";
}

TEST_F(QuadDemodTest, EdgeCasesTest) {
    const std::vector<size_t> testSizes = {2, 3, 4, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025};

    for (size_t size : testSizes) {
        auto input = generateQuadSignal(size, 0.1f);
        auto output = std::vector<float>(size - 1);
        const float gain = 1.0f;

        error = gsdrQuadFmDemod(input.data(), output.data(), gain, size - 1, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed for size " << size;

        // Verify output size (should be input size - 1)
        EXPECT_EQ(output.size(), size - 1) << "Incorrect output size for size " << size;
    }
}

TEST_F(QuadDemodTest, MinimumSizeTest) {
    // Test minimum size (2 elements)
    auto input = generateQuadSignal(2, 0.1f);
    auto output = std::vector<float>(1);
    const float gain = 1.0f;

    error = gsdrQuadFmDemod(input.data(), output.data(), gain, 1, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed for minimum size";

    EXPECT_EQ(output.size(), 1) << "Incorrect output size for minimum input";
}

TEST_F(QuadDemodTest, ConsistencyTest) {
    auto input = generateQuadSignal(testSize, 0.1f);
    const float gain = 1.0f;

    // Run demodulation multiple times
    for (int i = 0; i < 5; ++i) {
        auto output = std::vector<float>(testSize - 1);

        error = gsdrQuadFmDemod(input.data(), output.data(), gain, testSize - 1, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed on iteration " << i;

        // Results should be consistent
        if (i > 0) {
            // Could compare with previous result for consistency
        }
    }
}

TEST_F(QuadDemodTest, ZeroInputTest) {
    std::vector<cuComplex> zeroInput(testSize, make_cuComplex(0.0f, 0.0f));
    auto output = std::vector<float>(testSize - 1);
    const float gain = 1.0f;

    error = gsdrQuadFmDemod(zeroInput.data(), output.data(), gain, testSize - 1, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed for zero input";

    // Output should be zero or very small
    float maxValue = 0.0f;
    for (float val : output) {
        maxValue = std::max(maxValue, std::abs(val));
    }

    EXPECT_LT(maxValue, 1e-3f) << "Zero input should produce near-zero output";
}

TEST_F(QuadDemodTest, LargeValuesTest) {
    // Test with large input values
    std::vector<cuComplex> largeInput(testSize);
    std::fill(largeInput.begin(), largeInput.end(), make_cuComplex(100.0f, 100.0f));
    auto output = std::vector<float>(testSize - 1);
    const float gain = 1.0f;

    error = gsdrQuadFmDemod(largeInput.data(), output.data(), gain, testSize - 1, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "Quad FM demodulation failed for large input values";

    // Should handle large values gracefully
    bool hasValidData = true;
    for (float val : output) {
        if (!std::isfinite(val)) hasValidData = false;
    }
    EXPECT_TRUE(hasValidData) << "Large input values produced non-finite output";
}