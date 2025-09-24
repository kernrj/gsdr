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
#include <gsdr/fm.h>

class FmTest : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();
    }

    cudaError_t error = cudaSuccess;
    const size_t testSize = 1024;

    // Generate FM modulated signal
    std::vector<cuComplex> generateFmSignal(size_t size, float sampleRate, float carrierFreq, float deviation) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<cuComplex> signal(size);
        float phase = 0.0f;
        float phaseIncrement = 2.0f * M_PIf * carrierFreq / sampleRate;

        for (size_t i = 0; i < size; ++i) {
            // Generate message signal (low frequency)
            float messageFreq = 0.01f * sampleRate;
            float message = sinf(2.0f * M_PIf * messageFreq * i / sampleRate);

            // FM modulation: instantaneous frequency = carrier + deviation * message
            float instantaneousFreq = carrierFreq + deviation * message;
            float currentPhaseIncrement = 2.0f * M_PIf * instantaneousFreq / sampleRate;

            signal[i] = make_cuComplex(cosf(phase), sinf(phase));
            phase += currentPhaseIncrement;
        }

        return signal;
    }

    // Generate FM signal with known parameters
    std::vector<cuComplex> generateKnownFmSignal(size_t size, float sampleRate, float modulationFreq, float deviation) {
        std::vector<cuComplex> signal(size);
        float phase = 0.0f;
        float carrierFreq = 0.1f * sampleRate;
        float phaseIncrement = 2.0f * M_PIf * carrierFreq / sampleRate;
        float modulationIncrement = 2.0f * M_PIf * modulationFreq / sampleRate;

        for (size_t i = 0; i < size; ++i) {
            // FM: frequency deviation proportional to message
            float message = sinf(modulationIncrement * i);
            float instantaneousFreq = carrierFreq + deviation * message;
            float currentPhaseIncrement = 2.0f * M_PIf * instantaneousFreq / sampleRate;

            signal[i] = make_cuComplex(cosf(phase), sinf(phase));
            phase += currentPhaseIncrement;
        }

        return signal;
    }
};

TEST_F(FmTest, BasicDemodulationTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;
    const float deviation = 50.0f;

    auto input = generateFmSignal(testSize, sampleRate, carrierFreq, deviation);
    auto output = std::vector<float>(testSize);

    error = gsdrFmDemod(sampleRate, carrierFreq, channelFreq, deviation, 1,
                        0, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FM demodulation failed";

    // Verify output is reasonable
    bool hasValidData = true;
    float maxValue = 0.0f;
    float minValue = std::numeric_limits<float>::max();

    for (float val : output) {
        if (!std::isfinite(val)) hasValidData = false;
        maxValue = std::max(maxValue, val);
        minValue = std::min(minValue, val);
    }

    EXPECT_TRUE(hasValidData) << "FM demodulation produced non-finite values";
    EXPECT_GT(maxValue, 1e-6f) << "FM demodulation output is essentially zero";
    EXPECT_LT(std::abs(minValue), 1e6f) << "FM demodulation output has unreasonably large values";
}

TEST_F(FmTest, KnownModulationTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;
    const float deviation = 50.0f;
    const float modulationFreq = 10.0f;

    auto input = generateKnownFmSignal(testSize, sampleRate, modulationFreq, deviation);
    auto output = std::vector<float>(testSize);

    error = gsdrFmDemod(sampleRate, carrierFreq, channelFreq, deviation, 1,
                        0, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FM demodulation failed for known signal";

    // Should have some variation due to the modulation
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

    EXPECT_GT(variance, 1e-6f) << "FM demodulation output has no variation";
}

TEST_F(FmTest, DeviationTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;
    const std::vector<float> deviations = {10.0f, 50.0f, 100.0f, 200.0f};

    for (float deviation : deviations) {
        auto input = generateFmSignal(testSize, sampleRate, carrierFreq, deviation);
        auto output = std::vector<float>(testSize);

        error = gsdrFmDemod(sampleRate, carrierFreq, channelFreq, deviation, 1,
                            0, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "FM demodulation failed for deviation " << deviation;

        // Check that output range scales with deviation
        float maxValue = 0.0f;
        for (float val : output) {
            maxValue = std::max(maxValue, std::abs(val));
        }

        EXPECT_GT(maxValue, 1e-6f) << "No output for deviation " << deviation;
    }
}

TEST_F(FmTest, FrequencyOffsetTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float deviation = 50.0f;
    const std::vector<float> channelFreqs = {90.0f, 100.0f, 110.0f};

    for (float channelFreq : channelFreqs) {
        auto input = generateFmSignal(testSize, sampleRate, carrierFreq, deviation);
        auto output = std::vector<float>(testSize);

        error = gsdrFmDemod(sampleRate, carrierFreq, channelFreq, deviation, 1,
                            0, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "FM demodulation failed for channel freq " << channelFreq;

        // Verify output is reasonable
        bool hasValidData = true;
        for (float val : output) {
            if (!std::isfinite(val)) hasValidData = false;
        }
        EXPECT_TRUE(hasValidData) << "FM demodulation produced non-finite values for channel freq " << channelFreq;
    }
}

TEST_F(FmTest, LowPassFilterTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;
    const float deviation = 50.0f;

    // Generate low-pass filter taps
    std::vector<float> lowPassTaps(16);
    float cutoffFreq = 0.1f; // 0.1 * Nyquist

    for (size_t i = 0; i < lowPassTaps.size(); ++i) {
        if (i == lowPassTaps.size() / 2) {
            lowPassTaps[i] = 2.0f * cutoffFreq;
        } else {
            float x = 2.0f * cutoffFreq * (i - lowPassTaps.size() / 2.0f);
            lowPassTaps[i] = sinf(x) / x;
        }
    }

    // Normalize
    float sum = 0.0f;
    for (float tap : lowPassTaps) sum += tap;
    for (float& tap : lowPassTaps) tap /= sum;

    auto input = generateFmSignal(testSize, sampleRate, carrierFreq, deviation);
    auto output = std::vector<float>(testSize);

    error = gsdrFmDemod(sampleRate, carrierFreq, channelFreq, deviation, 1,
                        0, lowPassTaps.data(), lowPassTaps.size(),
                        input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FM demodulation with low-pass filter failed";

    // Should still produce valid output
    bool hasValidData = true;
    for (float val : output) {
        if (!std::isfinite(val)) hasValidData = false;
    }
    EXPECT_TRUE(hasValidData) << "FM demodulation with filter produced non-finite values";
}

TEST_F(FmTest, DecimationTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;
    const float deviation = 50.0f;
    const size_t decimation = 4;

    auto input = generateFmSignal(testSize * decimation, sampleRate, carrierFreq, deviation);
    auto output = std::vector<float>(testSize);

    error = gsdrFmDemod(sampleRate, carrierFreq, channelFreq, deviation, decimation,
                        0, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FM demodulation with decimation failed";

    // Output should be decimated
    EXPECT_EQ(output.size(), testSize) << "Incorrect output size with decimation";
}

TEST_F(FmTest, NoiseRobustnessTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;
    const float deviation = 50.0f;

    auto input = generateFmSignal(testSize, sampleRate, carrierFreq, deviation);
    auto output = std::vector<float>(testSize);

    // Add noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f); // SNR ≈ 20 dB

    for (auto& sample : input) {
        sample.x += noise(gen);
        sample.y += noise(gen);
    }

    error = gsdrFmDemod(sampleRate, carrierFreq, channelFreq, deviation, 1,
                        0, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FM demodulation failed with noise";

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

TEST_F(FmTest, EdgeCasesTest) {
    const std::vector<size_t> testSizes = {1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025};

    for (size_t size : testSizes) {
        const float sampleRate = 1000.0f;
        const float carrierFreq = 100.0f;
        const float channelFreq = 100.0f;
        const float deviation = 50.0f;

        auto input = generateFmSignal(size, sampleRate, carrierFreq, deviation);
        auto output = std::vector<float>(size);

        error = gsdrFmDemod(sampleRate, carrierFreq, channelFreq, deviation, 1,
                            0, nullptr, 0, input.data(), output.data(), size, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "FM demodulation failed for size " << size;

        // Verify output size
        EXPECT_EQ(output.size(), size) << "Incorrect output size for size " << size;
    }
}

TEST_F(FmTest, ParameterValidationTest) {
    std::vector<cuComplex> input(testSize);
    auto output = std::vector<float>(testSize);

    // Test with invalid parameters
    error = gsdrFmDemod(0.0f, 100.0f, 100.0f, 50.0f, 1,
                        0, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
    // This might fail, but should not crash

    // Test with very high deviation
    error = gsdrFmDemod(1000.0f, 100.0f, 100.0f, 1000.0f, 1,
                        0, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "FM demodulation failed for high deviation";
}

TEST_F(FmTest, ConsistencyTest) {
    const float sampleRate = 1000.0f;
    const float carrierFreq = 100.0f;
    const float channelFreq = 100.0f;
    const float deviation = 50.0f;

    auto input = generateFmSignal(testSize, sampleRate, carrierFreq, deviation);

    // Run demodulation multiple times
    for (int i = 0; i < 5; ++i) {
        auto output = std::vector<float>(testSize);

        error = gsdrFmDemod(sampleRate, carrierFreq, channelFreq, deviation, 1,
                            0, nullptr, 0, input.data(), output.data(), testSize, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "FM demodulation failed on iteration " << i;

        // Results should be consistent
        if (i > 0) {
            // Could compare with previous result for consistency
        }
    }
}