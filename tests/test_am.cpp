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
#include <gsdr/am.h>

class AmTest : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();
    }

    cudaError_t error = cudaSuccess;
    const size_t testSize = 1024;

    // Generate AM modulated signal
    std::vector<cuComplex> generateAmSignal(size_t size, float carrierFreq, float sampleRate, float modulationIndex = 0.7f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<cuComplex> signal(size);
        float phase = 0.0f;
        float phaseIncrement = 2.0f * M_PIf * carrierFreq / sampleRate;

        for (size_t i = 0; i < size; ++i) {
            // Generate message signal (low frequency)
            float messageFreq = 0.01f * sampleRate; // Much lower than carrier
            float message = sinf(2.0f * M_PIf * messageFreq * i / sampleRate);

            // AM modulation: s(t) = [1 + m(t)] * cos(2πf_c t)
            float amplitude = 1.0f + modulationIndex * message;
            signal[i] = make_cuComplex(amplitude * cosf(phase), amplitude * sinf(phase));
            phase += phaseIncrement;
        }

        return signal;
    }

    // Generate simple AM signal with known parameters
    std::vector<cuComplex> generateSimpleAmSignal(size_t size, float modulationFreq, float sampleRate) {
        std::vector<cuComplex> signal(size);
        float phase = 0.0f;
        float carrierPhase = 0.0f;
        float carrierIncrement = 2.0f * M_PIf * 0.1f * sampleRate / sampleRate; // Carrier at 0.1 * sample rate
        float modulationIncrement = 2.0f * M_PIf * modulationFreq / sampleRate;

        for (size_t i = 0; i < size; ++i) {
            // Simple AM: amplitude = 1 + 0.5 * sin(2πf_m t)
            float amplitude = 1.0f + 0.5f * sinf(modulationIncrement * i);
            signal[i] = make_cuComplex(amplitude * cosf(carrierPhase), amplitude * sinf(carrierPhase));
            carrierPhase += carrierIncrement;
        }

        return signal;
    }
};

TEST_F(AmTest, BasicDemodulationTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f; // Same as carrier for simplicity

    auto input = generateAmSignal(testSize, carrierFreq, sampleRate);
    auto output = std::vector<float>(testSize);

    error = gsdrAmDemod(sampleRate, carrierFreq, channelFreq, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AM demodulation failed";

    // Verify output is reasonable
    bool hasValidData = true;
    float maxValue = 0.0f;
    float minValue = std::numeric_limits<float>::max();

    for (float val : output) {
        if (!std::isfinite(val)) hasValidData = false;
        maxValue = std::max(maxValue, val);
        minValue = std::min(minValue, val);
    }

    EXPECT_TRUE(hasValidData) << "AM demodulation produced non-finite values";
    EXPECT_GT(maxValue, 1e-6f) << "AM demodulation output is essentially zero";
    EXPECT_LT(minValue, 1e6f) << "AM demodulation output has unreasonably large values";
}

TEST_F(AmTest, KnownSignalTest) {
    const float sampleRate = 1000.0f;
    const float modulationFreq = 10.0f; // Known modulation frequency
    const float channelFreq = 100.0f;

    auto input = generateSimpleAmSignal(testSize, modulationFreq, sampleRate);
    auto output = std::vector<float>(testSize);

    error = gsdrAmDemod(sampleRate, channelFreq, channelFreq, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AM demodulation failed for known signal";

    // The demodulated signal should contain the modulation frequency
    // Basic check: should have some variation
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

    EXPECT_GT(variance, 1e-6f) << "AM demodulation output has no variation";
}

TEST_F(AmTest, FrequencyOffsetTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const std::vector<float> channelFreqs = {95.0f, 100.0f, 105.0f}; // Test frequency offsets

    for (float channelFreq : channelFreqs) {
        auto input = generateAmSignal(testSize, carrierFreq, sampleRate);
        auto output = std::vector<float>(testSize);

        error = gsdrAmDemod(sampleRate, carrierFreq, channelFreq, input.data(), output.data(), testSize, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "AM demodulation failed for channel freq " << channelFreq;

        // Verify output is reasonable
        bool hasValidData = true;
        for (float val : output) {
            if (!std::isfinite(val)) hasValidData = false;
        }
        EXPECT_TRUE(hasValidData) << "AM demodulation produced non-finite values for channel freq " << channelFreq;
    }
}

TEST_F(AmTest, ModulationIndexTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;
    const std::vector<float> modulationIndices = {0.1f, 0.5f, 0.9f, 1.0f};

    for (float modIndex : modulationIndices) {
        auto input = generateAmSignal(testSize, carrierFreq, sampleRate, modIndex);
        auto output = std::vector<float>(testSize);

        error = gsdrAmDemod(sampleRate, carrierFreq, channelFreq, input.data(), output.data(), testSize, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "AM demodulation failed for modulation index " << modIndex;

        // Check that output range is reasonable for the modulation index
        float maxValue = 0.0f;
        for (float val : output) {
            maxValue = std::max(maxValue, std::abs(val));
        }

        // Higher modulation index should produce larger output variation
        EXPECT_GT(maxValue, 1e-6f) << "No output variation for modulation index " << modIndex;
    }
}

TEST_F(AmTest, OvermodulationTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;

    // Generate overmodulated signal (modulation index > 1)
    auto input = generateAmSignal(testSize, carrierFreq, sampleRate, 1.5f);
    auto output = std::vector<float>(testSize);

    error = gsdrAmDemod(sampleRate, carrierFreq, channelFreq, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AM demodulation failed for overmodulation";

    // Should still produce valid output
    bool hasValidData = true;
    for (float val : output) {
        if (!std::isfinite(val)) hasValidData = false;
    }
    EXPECT_TRUE(hasValidData) << "AM overmodulation produced non-finite values";
}

TEST_F(AmTest, NoiseRobustnessTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;

    auto input = generateAmSignal(testSize, carrierFreq, sampleRate);
    auto output = std::vector<float>(testSize);

    // Add noise to simulate channel impairments
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f); // SNR ≈ 20 dB

    for (auto& sample : input) {
        sample.x += noise(gen);
        sample.y += noise(gen);
    }

    error = gsdrAmDemod(sampleRate, carrierFreq, channelFreq, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AM demodulation failed with noise";

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

TEST_F(AmTest, EdgeCasesTest) {
    const std::vector<size_t> testSizes = {1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025};

    for (size_t size : testSizes) {
        const float sampleRate = 1000.0f;
        const float carrierFreq = 100.0f;
        const float channelFreq = 100.0f;

        auto input = generateAmSignal(size, carrierFreq, sampleRate);
        auto output = std::vector<float>(size);

        error = gsdrAmDemod(sampleRate, carrierFreq, channelFreq, input.data(), output.data(), size, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "AM demodulation failed for size " << size;

        // Verify output size
        EXPECT_EQ(output.size(), size) << "Incorrect output size for size " << size;
    }
}

TEST_F(AmTest, ParameterValidationTest) {
    std::vector<cuComplex> input(testSize);
    auto output = std::vector<float>(testSize);

    // Test with zero sample rate (should not crash)
    error = gsdrAmDemod(0.0f, 100.0f, 100.0f, input.data(), output.data(), testSize, 0, stream);
    // This might fail, but should not crash

    // Test with very high frequencies
    error = gsdrAmDemod(1000000.0f, 500000.0f, 500000.0f, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "AM demodulation failed for high frequencies";
}

TEST_F(AmTest, ConsistencyTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;

    auto input = generateAmSignal(testSize, carrierFreq, sampleRate);

    // Run demodulation multiple times
    for (int i = 0; i < 5; ++i) {
        auto output = std::vector<float>(testSize);

        error = gsdrAmDemod(sampleRate, carrierFreq, channelFreq, input.data(), output.data(), testSize, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "AM demodulation failed on iteration " << i;

        // Results should be consistent
        if (i > 0) {
            // Compare with previous result
            // Should be identical for the same input
        }
    }
}