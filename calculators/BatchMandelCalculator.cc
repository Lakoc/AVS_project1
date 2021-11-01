/**
 * @file BatchMandelCalculator.cc
 * @author Alexander Polok <xpolok03@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 1.11.2021
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "BatchMandelCalculator.h"

#define AVX512_IN_BYTES 64
#define blockSizeL1 32

BatchMandelCalculator::BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit) :
        BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator") {
    data = (int *) (aligned_alloc(AVX512_IN_BYTES, width * height * sizeof(int)));

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

    // Arrange Ys
    #pragma omp simd aligned(yInitValAP:64) simdlen(64)
    for (size_t point = 0; point < height; point++) {
        yInitValAP[point] = yStart + static_cast<float>(point) * dyF;
    }
}

BatchMandelCalculator::~BatchMandelCalculator() {
    free(data);
    free(xInitValA);
    free(yInitValA);
    free(zRealA);
    free(zImagA);
    free(conditionAchievedA);
    data = nullptr;
    xInitValA = nullptr;
    yInitValA = nullptr;
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

    // Condition counter
    int sum;

    size_t batch_size = blockSizeL1 * blockSizeL1;

    // Blocking
    for (size_t blockN = 0; blockN < height / blockSizeL1; blockN++) {
        size_t prefixN = blockN * blockSizeL1;
        for (size_t blockM = 0; blockM < width / blockSizeL1; blockM++) {
            size_t prefixM = blockM * blockSizeL1;

            // Copy starting x's and y's to current batch, clean up limits
            for (size_t row = 0; row < blockSizeL1; row++) {
                #pragma omp simd aligned(xInitValAP, yInitValAP, zImagAP, zRealAP:64) simdlen(64) linear(row)
                for (size_t col = 0; col < blockSizeL1; col++) {
                    size_t curr_ind = row * blockSizeL1 + col;
                    zImagAP[curr_ind] = yInitValAP[prefixN + col];
                    zRealAP[curr_ind] = xInitValAP[prefixM + row];
                    conditionAchievedAP[curr_ind] = 0;
                }
            }

            // Iterate until every point threshold condition is met or iteration limit is reached
            for (int current_iteration = 0; current_iteration < limit; current_iteration++) {
                sum = 0;
                // Iterate over batch
                for (size_t row = 0; row < blockSizeL1; row++) {
                    #pragma omp simd aligned(xInitValAP, yInitValAP, zImagAP, zRealAP, conditionAchievedAP:64) simdlen(64) reduction(+:sum) linear(row:1)
                    for (size_t col = 0; col < blockSizeL1; col++) {
                        size_t curr_ind = row * blockSizeL1 + col;

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
                    for (size_t row = 0; row < blockSizeL1; row++) {
                        #pragma omp simd aligned(dataP, conditionAchievedAP:64) simdlen(64) linear(row)
                        for (size_t col = 0; col < blockSizeL1; col++) {
                            const size_t rowGlobal = prefixN + row;
                            const size_t colGlobal = prefixM + col;
                            dataP[rowGlobal * width + colGlobal] =
                                    finished_iterations - conditionAchievedAP[col * blockSizeL1 + row];
                        }
                    }
                    break;
                }
            }
        }
    }
    return data;
}