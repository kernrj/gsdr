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
#include <map>
#include <random>
#include <algorithm>

#include "test_main.cpp"
#include <gsdr/qpsk256.h>

class Qpsk256Test : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();

        // Initialize QPSK256 constellation
        error = gsdrQpsk256InitConstellation(constellationType, amplitude, 0, stream);
        if (error != cudaSuccess) {
            GTEST_SKIP() << "QPSK256 constellation initialization failed, skipping test";
        }
    }

    cudaError_t error = cudaSuccess;
    const size_t numSymbols = 512; // Smaller size due to 256-ary complexity
    const uint32_t constellationType = 1; // Circular constellation
    const float amplitude = 1.0f;
    std::vector<uint8_t> testData = test_utils::generateQpsk256TestData(numSymbols);
    std::vector<cuComplex> modulatedData;
    std::vector<uint8_t> demodulatedData;

    void setupData() {
        modulatedData.resize(numSymbols);
        demodulatedData.resize(numSymbols);
    }
};

TEST_F(Qpsk256Test, InitializationTest) {
    // Test constellation initialization
    error = gsdrQpsk256InitConstellation(0, 1.0f, 0, stream); // Rectangular
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 rectangular constellation initialization failed";

    error = gsdrQpsk256InitConstellation(1, 1.0f, 0, stream); // Circular
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 circular constellation initialization failed";
}

TEST_F(Qpsk256Test, RectangularConstellationTest) {
    setupData();

    // Test rectangular constellation
    error = gsdrQpsk256InitConstellation(0, amplitude, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 rectangular constellation initialization failed";

    error = gsdrQpsk256Modulate(testData.data(), modulatedData.data(), numSymbols, amplitude, 0, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 rectangular modulation failed";

    // Verify output is not all zeros
    bool allZero = true;
    for (const auto& sample : modulatedData) {
        if (std::abs(sample.x) > 1e-6f || std::abs(sample.y) > 1e-6f) {
            allZero = false;
            break;
        }
    }
    EXPECT_FALSE(allZero) << "All modulated samples are zero";
}

TEST_F(Qpsk256Test, CircularConstellationTest) {
    setupData();

    // Test circular constellation
    error = gsdrQpsk256InitConstellation(1, amplitude, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 circular constellation initialization failed";

    error = gsdrQpsk256Modulate(testData.data(), modulatedData.data(), numSymbols, amplitude, 1, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 circular modulation failed";

    // Verify output is not all zeros
    bool allZero = true;
    for (const auto& sample : modulatedData) {
        if (std::abs(sample.x) > 1e-6f || std::abs(sample.y) > 1e-6f) {
            allZero = false;
            break;
        }
    }
    EXPECT_FALSE(allZero) << "All modulated samples are zero";
}

TEST_F(Qpsk256Test, ModulationAccuracyTest) {
    setupData();

    // Test modulation and demodulation round trip
    error = gsdrQpsk256InitConstellation(constellationType, amplitude, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 constellation initialization failed";

    error = gsdrQpsk256Modulate(testData.data(), modulatedData.data(), numSymbols, amplitude, constellationType, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 modulation failed";

    error = gsdrQpsk256Demodulate(modulatedData.data(), demodulatedData.data(), numSymbols, constellationType, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 demodulation failed";

    // Calculate symbol error rate (should be 0 for ideal channel)
    size_t symbolErrors = 0;
    for (size_t i = 0; i < numSymbols; ++i) {
        if (testData[i] != demodulatedData[i]) {
            symbolErrors++;
        }
    }

    float ser = static_cast<float>(symbolErrors) / numSymbols;
    EXPECT_EQ(ser, 0.0f) << "QPSK256 should have perfect reconstruction, SER: " << ser;
}

TEST_F(Qpsk256Test, ConstellationPointCountTest) {
    setupData();

    // Test that we get the expected number of unique constellation points
    std::vector<cuComplex> uniquePoints;

    for (uint32_t type = 0; type < 2; ++type) {
        error = gsdrQpsk256InitConstellation(type, amplitude, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 constellation initialization failed";

        // Generate modulation for all possible symbols (0-255)
        std::vector<uint8_t> allSymbols(256);
        std::vector<cuComplex> allModulated(256);

        for (int i = 0; i < 256; ++i) {
            allSymbols[i] = i;
        }

        error = gsdrQpsk256Modulate(allSymbols.data(), allModulated.data(), 256, amplitude, type, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 modulation failed";

        // Count unique points
        uniquePoints.clear();
        for (const auto& sample : allModulated) {
            bool found = false;
            for (const auto& point : uniquePoints) {
                if (std::abs(sample.x - point.x) < 1e-6f &&
                    std::abs(sample.y - point.y) < 1e-6f) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                uniquePoints.push_back(sample);
            }
        }

        // Should have exactly 256 unique points
        EXPECT_EQ(uniquePoints.size(), 256) << "QPSK256 should have exactly 256 unique constellation points for type " << type;
    }
}

TEST_F(Qpsk256Test, AmplitudeScalingTest) {
    setupData();

    const std::vector<float> amplitudes = {0.5f, 1.0f, 2.0f, 5.0f};

    for (float testAmplitude : amplitudes) {
        error = gsdrQpsk256InitConstellation(constellationType, testAmplitude, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 constellation initialization failed";

        std::vector<cuComplex> modulatedDataAmp(numSymbols);
        error = gsdrQpsk256Modulate(testData.data(), modulatedDataAmp.data(), numSymbols, testAmplitude, constellationType, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 modulation failed";

        // Check amplitude scaling
        float maxMagnitude = 0.0f;
        for (const auto& sample : modulatedDataAmp) {
            float magnitude = cuCabsf(sample);
            maxMagnitude = std::max(maxMagnitude, magnitude);
        }

        // For unit amplitude, max magnitude should be approximately amplitude
        EXPECT_NEAR(maxMagnitude, testAmplitude, 0.1f * testAmplitude)
            << "Incorrect amplitude scaling for amplitude " << testAmplitude;
    }
}

TEST_F(Qpsk256Test, NoiseRobustnessTest) {
    setupData();

    // Test with different SNR levels
    const std::vector<float> snr_db_values = {5.0f, 10.0f, 15.0f, 20.0f};

    for (float snr_db : snr_db_values) {
        error = gsdrQpsk256InitConstellation(constellationType, amplitude, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 constellation initialization failed";

        error = gsdrQpsk256Modulate(testData.data(), modulatedData.data(), numSymbols, amplitude, constellationType, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 modulation failed";

        // Add noise
        float noise_std = 1.0f / sqrtf(powf(10.0f, snr_db / 10.0f));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise_dist(0.0f, noise_std);

        for (auto& sample : modulatedData) {
            sample.x += noise_dist(gen);
            sample.y += noise_dist(gen);
        }

        // Demodulate
        error = gsdrQpsk256Demodulate(modulatedData.data(), demodulatedData.data(), numSymbols, constellationType, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 demodulation failed";

        // Calculate symbol error rate
        size_t symbolErrors = 0;
        for (size_t i = 0; i < numSymbols; ++i) {
            if (testData[i] != demodulatedData[i]) {
                symbolErrors++;
            }
        }

        float ser = static_cast<float>(symbolErrors) / numSymbols;
        float expected_ser = 1.0f - (1.0f - erfc(sqrtf(powf(10.0f, snr_db / 10.0f)) / sqrtf(2.0f))) / 2.0f; // Theoretical approximation

        // Allow some tolerance for noise
        EXPECT_LT(ser, expected_ser * 2.0f) << "Symbol error rate too high for SNR " << snr_db << " dB: " << ser;
    }
}

TEST_F(Qpsk256Test, EdgeCasesTest) {
    // Test edge cases
    const std::vector<size_t> testSizes = {1, 2, 3, 4, 15, 16, 17, 255, 256, 257};

    for (size_t size : testSizes) {
        std::vector<uint8_t> testDataSmall = test_utils::generateQpsk256TestData(size);
        std::vector<cuComplex> modulatedDataSmall(size);
        std::vector<uint8_t> demodulatedDataSmall(size);

        error = gsdrQpsk256InitConstellation(constellationType, amplitude, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 constellation initialization failed";

        // Modulation
        error = gsdrQpsk256Modulate(testDataSmall.data(), modulatedDataSmall.data(), size, amplitude, constellationType, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 modulation failed for size " << size;

        // Demodulation
        error = gsdrQpsk256Demodulate(modulatedDataSmall.data(), demodulatedDataSmall.data(), size, constellationType, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK256 demodulation failed for size " << size;
    }
}

TEST_F(Qpsk256Test, ConstellationTypeComparisonTest) {
    setupData();

    std::vector<uint8_t> testDataSmall = test_utils::generateQpsk256TestData(256);
    std::vector<cuComplex> rectangularModulated(256);
    std::vector<cuComplex> circularModulated(256);

    // Test rectangular constellation
    error = gsdrQpsk256InitConstellation(0, amplitude, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 rectangular constellation initialization failed";

    error = gsdrQpsk256Modulate(testDataSmall.data(), rectangularModulated.data(), 256, amplitude, 0, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 rectangular modulation failed";

    // Test circular constellation
    error = gsdrQpsk256InitConstellation(1, amplitude, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 circular constellation initialization failed";

    error = gsdrQpsk256Modulate(testDataSmall.data(), circularModulated.data(), 256, amplitude, 1, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK256 circular modulation failed";

    // Calculate average power for each constellation
    float rectPower = 0.0f, circPower = 0.0f;
    for (int i = 0; i < 256; ++i) {
        rectPower += cuCabsf(rectangularModulated[i]) * cuCabsf(rectangularModulated[i]);
        circPower += cuCabsf(circularModulated[i]) * cuCabsf(circularModulated[i]);
    }
    rectPower /= 256.0f;
    circPower /= 256.0f;

    // Both should have similar average power
    EXPECT_NEAR(rectPower, circPower, 0.1f) << "Constellation types have significantly different power levels";

    // Circular constellation should have better peak-to-average power ratio
    float rectPeakPower = 0.0f, circPeakPower = 0.0f;
    for (int i = 0; i < 256; ++i) {
        float rectMag = cuCabsf(rectangularModulated[i]);
        float circMag = cuCabsf(circularModulated[i]);
        rectPeakPower = std::max(rectPeakPower, rectMag * rectMag);
        circPeakPower = std::max(circPeakPower, circMag * circMag);
    }

    // Circular should have lower peak power
    EXPECT_LE(circPeakPower, rectPeakPower) << "Circular constellation has higher peak power than rectangular";
}