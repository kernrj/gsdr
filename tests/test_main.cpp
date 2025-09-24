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
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#include <gsdr/gsdr.h>

// Test utilities and common functions
class CudaTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }

        // Create CUDA stream
        err = cudaStreamCreate(&stream);
        EXPECT_EQ(err, cudaSuccess) << "Failed to create CUDA stream";
    }

    void TearDown() override {
        if (stream != nullptr) {
            cudaError_t err = cudaStreamDestroy(stream);
            EXPECT_EQ(err, cudaSuccess) << "Failed to destroy CUDA stream";
        }
    }

    cudaStream_t stream = nullptr;

    // Helper function to generate random data
    std::vector<uint8_t> generateRandomBytes(size_t count) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        std::vector<uint8_t> data(count);
        std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
        return data;
    }

    // Helper function to generate random complex data
    std::vector<cuComplex> generateRandomComplex(size_t count) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<cuComplex> data(count);
        std::generate(data.begin(), data.end(), [&]() {
            return make_cuComplex(dis(gen), dis(gen));
        });
        return data;
    }

    // Helper function to generate random float data
    std::vector<float> generateRandomFloats(size_t count) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<float> data(count);
        std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
        return data;
    }

    // Helper function to check if two complex vectors are approximately equal
    void expectComplexVectorsNear(const std::vector<cuComplex>& a,
                                  const std::vector<cuComplex>& b,
                                  float tolerance = 1e-5f) {
        ASSERT_EQ(a.size(), b.size()) << "Vector sizes don't match";

        for (size_t i = 0; i < a.size(); ++i) {
            float real_diff = std::abs(a[i].x - b[i].x);
            float imag_diff = std::abs(a[i].y - b[i].y);
            EXPECT_LE(real_diff, tolerance) << "Real part mismatch at index " << i;
            EXPECT_LE(imag_diff, tolerance) << "Imaginary part mismatch at index " << i;
        }
    }

    // Helper function to check if two float vectors are approximately equal
    void expectFloatVectorsNear(const std::vector<float>& a,
                                const std::vector<float>& b,
                                float tolerance = 1e-5f) {
        ASSERT_EQ(a.size(), b.size()) << "Vector sizes don't match";

        for (size_t i = 0; i < a.size(); ++i) {
            float diff = std::abs(a[i] - b[i]);
            EXPECT_LE(diff, tolerance) << "Float mismatch at index " << i;
        }
    }
};

// Global test utilities
namespace test_utils {

// Generate test data for QPSK (2 bits per symbol)
std::vector<uint8_t> generateQpskTestData(size_t numSymbols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 3); // 2 bits: 0-3

    std::vector<uint8_t> data(numSymbols / 4 + 1, 0);
    for (size_t i = 0; i < numSymbols; ++i) {
        uint8_t symbol = dis(gen);
        size_t byteIndex = i / 4;
        size_t bitOffset = (i % 4) * 2;
        data[byteIndex] |= (symbol << bitOffset);
    }
    return data;
}

// Generate test data for QPSK256 (8 bits per symbol)
std::vector<uint8_t> generateQpsk256TestData(size_t numSymbols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255); // 8 bits: 0-255

    std::vector<uint8_t> data(numSymbols);
    std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
    return data;
}

// Verify QPSK demodulation results
bool verifyQpskDemodulation(const std::vector<uint8_t>& original,
                           const std::vector<uint8_t>& demodulated,
                           size_t numSymbols) {
    for (size_t i = 0; i < numSymbols; ++i) {
        size_t byteIndex = i / 4;
        size_t bitOffset = (i % 4) * 2;
        uint8_t originalSymbol = (original[byteIndex] >> bitOffset) & 0x3;
        uint8_t demodSymbol = (demodulated[byteIndex] >> bitOffset) & 0x3;

        if (originalSymbol != demodSymbol) {
            return false;
        }
    }
    return true;
}

} // namespace test_utils

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}