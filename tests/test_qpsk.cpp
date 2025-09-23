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

#include "test_main.cpp"
#include <gsdr/qpsk.h>

class QpskTest : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();

        // Initialize QPSK constellation points
        error = gsdrQpskModulate(testData.data(), modulatedData.data(), testData.size(), 1.0f, 0, stream);
        if (error != cudaSuccess) {
            GTEST_SKIP() << "QPSK modulation failed, skipping test";
        }
    }

    cudaError_t error = cudaSuccess;
    const size_t numSymbols = 1024;
    std::vector<uint8_t> testData = test_utils::generateQpskTestData(numSymbols);
    std::vector<cuComplex> modulatedData;
    std::vector<uint8_t> demodulatedData;

    void setupData() {
        modulatedData.resize(numSymbols);
        demodulatedData.resize(numSymbols / 4 + 1, 0);
    }
};

TEST_F(QpskTest, ModulationTest) {
    setupData();

    // Test basic modulation
    error = gsdrQpskModulate(testData.data(), modulatedData.data(), numSymbols, 1.0f, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK modulation failed";

    // Verify that output data is not all zeros (basic sanity check)
    bool allZero = true;
    for (const auto& sample : modulatedData) {
        if (sample.x != 0.0f || sample.y != 0.0f) {
            allZero = false;
            break;
        }
    }
    EXPECT_FALSE(allZero) << "All modulated samples are zero";

    // Check that we get the expected constellation points
    // QPSK should produce points at (1,1), (-1,1), (-1,-1), (1,-1)
    std::vector<cuComplex> uniquePoints;
    for (const auto& sample : modulatedData) {
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

    // Should have exactly 4 unique constellation points
    EXPECT_EQ(uniquePoints.size(), 4) << "QPSK should have exactly 4 constellation points";
}

TEST_F(QpskTest, DemodulationTest) {
    setupData();

    // First modulate the data
    error = gsdrQpskModulate(testData.data(), modulatedData.data(), numSymbols, 1.0f, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK modulation failed";

    // Then demodulate
    error = gsdrQpskDemodulate(modulatedData.data(), demodulatedData.data(), numSymbols, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK demodulation failed";

    // Verify demodulation accuracy (should be perfect for ideal channel)
    bool perfectDemod = test_utils::verifyQpskDemodulation(testData, demodulatedData, numSymbols);
    EXPECT_TRUE(perfectDemod) << "QPSK demodulation should be perfect for ideal channel";
}

TEST_F(QpskTest, RoundTripTest) {
    setupData();

    // Perform modulation-demodulation round trip
    error = gsdrQpskModulate(testData.data(), modulatedData.data(), numSymbols, 1.0f, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK modulation failed";

    error = gsdrQpskDemodulate(modulatedData.data(), demodulatedData.data(), numSymbols, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK demodulation failed";

    // Verify perfect reconstruction
    bool perfectRoundTrip = test_utils::verifyQpskDemodulation(testData, demodulatedData, numSymbols);
    EXPECT_TRUE(perfectRoundTrip) << "QPSK round-trip should be perfect";
}

TEST_F(QpskTest, AmplitudeTest) {
    setupData();

    // Test with different amplitudes
    const std::vector<float> amplitudes = {0.5f, 1.0f, 2.0f, 10.0f};

    for (float amplitude : amplitudes) {
        std::vector<cuComplex> modulatedDataAmp(numSymbols);

        error = gsdrQpskModulate(testData.data(), modulatedDataAmp.data(), numSymbols, amplitude, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK modulation failed for amplitude " << amplitude;

        // Check that the amplitude scaling is correct
        float maxMagnitude = 0.0f;
        for (const auto& sample : modulatedDataAmp) {
            float magnitude = cuCabsf(sample);
            maxMagnitude = std::max(maxMagnitude, magnitude);
        }

        // For unit amplitude, max magnitude should be sqrt(2) ≈ 1.414
        float expectedMaxMag = amplitude * sqrtf(2.0f);
        EXPECT_NEAR(maxMagnitude, expectedMaxMag, 0.01f)
            << "Incorrect amplitude scaling for amplitude " << amplitude;
    }
}

TEST_F(QpskTest, ConstellationPointsTest) {
    setupData();

    // Test that all constellation points are correct
    error = gsdrQpskModulate(testData.data(), modulatedData.data(), numSymbols, 1.0f, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK modulation failed";

    // Count occurrences of each constellation point
    std::map<std::pair<float, float>, int> pointCounts;
    for (const auto& sample : modulatedData) {
        float real = roundf(sample.x * 1000) / 1000; // Round to avoid floating point precision issues
        float imag = roundf(sample.y * 1000) / 1000;
        pointCounts[{real, imag}]++;
    }

    // Should have 4 distinct points
    EXPECT_EQ(pointCounts.size(), 4) << "QPSK should produce exactly 4 distinct constellation points";

    // Each point should be one of: (1,1), (-1,1), (-1,-1), (1,-1)
    const std::vector<std::pair<float, float>> expectedPoints = {
        {1.0f, 1.0f}, {-1.0f, 1.0f}, {-1.0f, -1.0f}, {1.0f, -1.0f}
    };

    for (const auto& [point, count] : pointCounts) {
        bool found = false;
        for (const auto& expected : expectedPoints) {
            if (std::abs(point.first - expected.first) < 1e-6f &&
                std::abs(point.second - expected.second) < 1e-6f) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Unexpected constellation point: (" << point.first << ", " << point.second << ")";
    }
}

TEST_F(QpskTest, BitErrorRateTest) {
    setupData();

    // Test with added noise to simulate channel impairments
    const float snr_db = 10.0f; // 10 dB SNR
    const float noise_std = 1.0f / sqrtf(powf(10.0f, snr_db / 10.0f));

    // Modulate
    error = gsdrQpskModulate(testData.data(), modulatedData.data(), numSymbols, 1.0f, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK modulation failed";

    // Add noise to simulate channel
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise_dist(0.0f, noise_std);

    for (auto& sample : modulatedData) {
        sample.x += noise_dist(gen);
        sample.y += noise_dist(gen);
    }

    // Demodulate
    error = gsdrQpskDemodulate(modulatedData.data(), demodulatedData.data(), numSymbols, 0, stream);
    EXPECT_EQ(error, cudaSuccess) << "QPSK demodulation failed";

    // Calculate bit error rate
    size_t bitErrors = 0;
    for (size_t i = 0; i < numSymbols; ++i) {
        size_t byteIndex = i / 4;
        size_t bitOffset = (i % 4) * 2;
        uint8_t originalSymbol = (testData[byteIndex] >> bitOffset) & 0x3;
        uint8_t demodSymbol = (demodulatedData[byteIndex] >> bitOffset) & 0x3;

        if (originalSymbol != demodSymbol) {
            bitErrors++;
        }
    }

    float ber = static_cast<float>(bitErrors) / (numSymbols * 2); // 2 bits per symbol
    EXPECT_LT(ber, 0.01f) << "Bit error rate too high: " << ber; // Should be < 1% at 10 dB SNR
}

TEST_F(QpskTest, EdgeCasesTest) {
    // Test edge cases
    const std::vector<size_t> testSizes = {1, 2, 3, 4, 31, 32, 33, 1023, 1024, 1025};

    for (size_t size : testSizes) {
        std::vector<uint8_t> testDataSmall = test_utils::generateQpskTestData(size);
        std::vector<cuComplex> modulatedDataSmall(size);
        std::vector<uint8_t> demodulatedDataSmall(size / 4 + 1, 0);

        // Modulation
        error = gsdrQpskModulate(testDataSmall.data(), modulatedDataSmall.data(), size, 1.0f, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK modulation failed for size " << size;

        // Demodulation
        error = gsdrQpskDemodulate(modulatedDataSmall.data(), demodulatedDataSmall.data(), size, 0, stream);
        EXPECT_EQ(error, cudaSuccess) << "QPSK demodulation failed for size " << size;
    }
}