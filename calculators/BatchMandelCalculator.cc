/**
 * @file BatchMandelCalculator.cc
 * @author Alexander Polok <xpolok03@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "BatchMandelCalculator.h"

#define AVX512_IN_BYTES 64
#define blockSizeL3 512
#define blockSizeL2 128
#define blockSizeL1 64

BatchMandelCalculator::BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit) :
        BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator") {
    data = (int *) (aligned_alloc(AVX512_IN_BYTES, height * width * sizeof(int)));

    // Array of arranged Xs in range(x_start, x_start + width * dx)
    xInitValA = static_cast<float *>(aligned_alloc(AVX512_IN_BYTES, sizeof(float) * width));
    yInitValA = static_cast<float *>(aligned_alloc(AVX512_IN_BYTES, sizeof(float) * height));

    // Arrays keeping imaginary and real values of each complex number on current row
    zRealA = static_cast<float *>(aligned_alloc(AVX512_IN_BYTES, sizeof(float) * blockSizeL1 * blockSizeL1));
    zImagA = static_cast<float *>(aligned_alloc(AVX512_IN_BYTES, sizeof(float) * blockSizeL1 * blockSizeL1));
    conditionAchievedA = static_cast<int *>(aligned_alloc(AVX512_IN_BYTES, sizeof(int) * blockSizeL1 * blockSizeL1));

    // Cast doubles to floats
    auto xStart = static_cast<float>(x_start);
    auto dxF = static_cast<float>(dx);
    auto yStart = static_cast<float>(y_start);
    auto dyF = static_cast<float>(dy);

    float *xInitValAP = xInitValA;
    float *yInitValAP = yInitValA;
    // Arrange Xs
#pragma omp simd aligned(xInitValAP:64) simdlen(64)
    for (size_t point = 0; point < width; point++) {
        xInitValAP[point] = xStart + static_cast<float>(point) * dxF;
    }
#pragma omp simd aligned(yInitValAP:64) simdlen(64)
    for (size_t point = 0; point < height; point++) {
        yInitValAP[point] = yStart + static_cast<float>(point) * dyF;
    }
}

BatchMandelCalculator::~BatchMandelCalculator() {
    free(data);
    free(xInitValA);
    free(zRealA);
    free(zImagA);
    free(conditionAchievedA);
    data = nullptr;
    xInitValA = nullptr;
    zRealA = nullptr;
    zImagA = nullptr;
    conditionAchievedA = nullptr;
}


int *BatchMandelCalculator::calculateMandelbrot() {
    // copy reference to use in pragma omp
    int *dataP = data;
    float *xInitValAP = xInitValA;
    float *yInitValAP = yInitValA;
    float *zRealAP = zRealA;
    float *zImagAP = zImagA;
    int *conditionAchievedAP = conditionAchievedA;

    // Cast doubles to floats
    int sum;

    size_t batch_size = blockSizeL1 * blockSizeL1;
    // L3 blocking
    // Split the number of rows of the data into a few blocks
    for (size_t blockNL3 = 0; blockNL3 < width / blockSizeL3; blockNL3++) {
        // Split the number of cols of the data into a few blocks
        size_t L3prefixN = blockNL3 * blockSizeL3;
        for (size_t blockML3 = 0; blockML3 < height / blockSizeL3; blockML3++) {
            size_t L3prefixM = blockML3 * blockSizeL3;
            // L2 blocking
            // Split the number of rows of the data into a few blocks
            for (size_t blockNL2 = 0; blockNL2 < blockSizeL3 / blockSizeL2; blockNL2++) {
                size_t L2prefixN = L3prefixN + blockNL2 * blockSizeL2;
                // Split the number of cols of the data into a few blocks
                for (size_t blockML2 = 0; blockML2 < blockSizeL3 / blockSizeL2; blockML2++) {
                    size_t L2prefixM = L3prefixM + blockML2 * blockSizeL2;

                    for (size_t blockNL1 = 0; blockNL1 < blockSizeL2 / blockSizeL1; blockNL1++) {
                        size_t L1prefixN = L2prefixN + blockNL1 * blockSizeL1;
                        // Split the number of cols of the data into a few blocks
                        for (size_t blockML1 = 0; blockML1 < blockSizeL2 / blockSizeL1; blockML1++) {
                            size_t L1prefixM = L2prefixM + blockML1 * blockSizeL1;

                            // Copy starting x's to current row, clean up limits, copy imaginary value
                            #pragma omp simd aligned(xInitValAP, yInitValAP, zImagAP, zRealAP, conditionAchievedAP:64) linear(row:1) collapse(2)
                            for (size_t row = 0; row < blockSizeL1; row++) {
                                size_t prefix = row * blockSizeL1;
                                for (size_t col = 0; col < blockSizeL1; col++) {
                                    size_t curr_ind = prefix + col;
                                    zImagAP[curr_ind] = yInitValAP[L1prefixM + col];
                                    zRealAP[curr_ind] = xInitValAP[L1prefixN + row];
                                    conditionAchievedAP[curr_ind] = 0;
                                }
                            }
                            // Auxiliary variable to stop loop
                            // Iterate until every point threshold condition is met or iteration limit is reached
                            for (int current_iteration = 0; current_iteration < limit; current_iteration++) {
                                sum = 0;
                                // Iterate over each complex number and
                                #pragma omp simd aligned(xInitValAP, yInitValAP, zImagAP, zRealAP, conditionAchievedAP:64) simdlen(64) reduction(+:sum) linear(current_iteration:1) collapse(2)
                                for (size_t row = 0; row < blockSizeL1; row++) {
                                    size_t prefix = row * blockSizeL1;
                                    for (size_t col = 0; col < blockSizeL1; col++) {
                                        size_t curr_ind = prefix + col;

                                        // Real and imaginary component on pow2
                                        float r2 = zRealAP[curr_ind] * zRealAP[curr_ind];
                                        float i2 = zImagAP[curr_ind] * zImagAP[curr_ind];

                                        // Threshold condition cast to int for more optimal computation
                                        auto conditionAchieved = static_cast<int>(r2 + i2 > 4.0f);

                                        // Single cast to float (optimization purpose)
                                        auto conditionAchievedF = static_cast<float>(conditionAchieved);

                                        // Negation of condition (optimization purpose
                                        auto conditionNotAchieved = static_cast<float>(1 - conditionAchieved);

                                        // Counter of condition achievements by each point
                                        // Subtracting it from max iteration reached we get on which iteration condition was firstly satisfied
                                        conditionAchievedAP[curr_ind] += conditionAchieved;

                                        // Sum up condition achievements
                                        sum += conditionAchieved;


                                        // Calculate new imaginary and real component of current point
                                        // We need to stop incrementing to not overflow -> ternary operator is slower than next lines
                                        zImagAP[curr_ind] = zImagAP[curr_ind] *
                                                            ((conditionNotAchieved * 2.0f * zRealAP[curr_ind]) +
                                                             conditionAchievedF) +
                                                            (conditionNotAchieved * yInitValAP[L1prefixM + col]);
                                        zRealAP[curr_ind] = (conditionAchievedF * zRealAP[curr_ind]) +
                                                            (conditionNotAchieved *
                                                             (r2 - i2 + xInitValAP[L1prefixN + row]));

                                    }
                                }
                                // Break limit loop if all points meets ending condition
                                if (sum == batch_size || current_iteration == limit - 1) {
                                    // Calculate output data shift
                                    int finished_iterations = current_iteration + 1;


                                    // Save every point first condition met iteration
                                    #pragma omp simd aligned(dataP, conditionAchievedAP:64) simdlen(64) linear(current_iteration:1) collapse(2)
                                    for (size_t row = 0; row < blockSizeL1; row++) {
                                        size_t prefix = row * blockSizeL1;
                                        for (size_t col = 0; col < blockSizeL1; col++) {
                                            const size_t rowGlobal =
                                                    blockNL3 * blockSizeL3 + blockNL2 * blockSizeL2 +
                                                    blockNL1 * blockSizeL1 +
                                                    row;
                                            auto row_shift = static_cast<long>(rowGlobal * width);
                                            const size_t colGlobal =
                                                    blockML3 * blockSizeL3 + blockML2 * blockSizeL2 +
                                                    blockML1 * blockSizeL1 +
                                                    col;
                                            dataP[row_shift + colGlobal] =
                                                    finished_iterations - conditionAchievedAP[prefix + col];
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return data;
}