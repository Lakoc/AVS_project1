/**
 * @file LineMandelCalculator.cc
 * @author Alexander Polok <xpolok03@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include "LineMandelCalculator.h"
#define AVX512_IN_BYTES 64

LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize, unsigned limit) :
        BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator") {
    data = (int *) (aligned_alloc(AVX512_IN_BYTES,width * height * sizeof(int)));

    // Array of arranged Xs in range(x_start, x_start + width * dx)
    xInitValA = static_cast<float *>(aligned_alloc(AVX512_IN_BYTES, sizeof(float) * width));

    // Arrays keeping imaginary and real values of each complex number on current row
    zRealA = static_cast<float *>(aligned_alloc(AVX512_IN_BYTES, sizeof(float) * width));
    zImagA = static_cast<float *>(aligned_alloc(AVX512_IN_BYTES, sizeof(float) * width));
    conditionAchievedA = static_cast<int *>(aligned_alloc(AVX512_IN_BYTES, sizeof(int) * width));

    // Cast doubles to floats
    auto xStart = static_cast<float>(x_start);
    auto dxF = static_cast<float>(dx);

    float *xInitValAP = xInitValA;
    // Arrange Xs
    #pragma omp simd aligned(xInitValAP:64) simdlen(64)
    for (size_t point = 0; point < width; point++) {
        xInitValAP[point] = xStart + static_cast<float>(point) * dxF;
    }
}

LineMandelCalculator::~LineMandelCalculator() {
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


int *LineMandelCalculator::calculateMandelbrot() {
    // copy reference to use in pragma omp
    int *dataP = data;
    float *xInitValAP = xInitValA;
    float *zRealAP = zRealA;
    float *zImagAP = zImagA;
    int *conditionAchievedAP = conditionAchievedA;

    // Cast doubles to floats
    auto yStart = static_cast<float>(y_start);
    auto dyF = static_cast<float>(dy);
    int sum;
    // Iterate over all rows
    for (size_t row = 0; row < height; row++) {
        // Calculate imaginary value of selected row
        auto imag = yStart + static_cast<float>(row) * dyF;

        // Copy starting x's to current row, clean up limits, copy imaginary value
        #pragma omp simd aligned(xInitValAP, zImagAP, zRealAP, conditionAchievedAP:64) linear(row:1)
        for (size_t point = 0; point < width; point++) {
            zImagAP[point] = imag;
            zRealAP[point] = xInitValAP[point];
            conditionAchievedAP[point] = 0;
        }

        // Iterate until every point threshold condition is met or iteration limit is reached
        for (int current_iteration = 0; current_iteration < limit; current_iteration++) {

            // Auxiliary variable to stop loop
            sum = 0;

            // Iterate over each complex number and
            #pragma omp simd aligned(xInitValAP, zImagAP, zRealAP, conditionAchievedAP:64) simdlen(64) reduction(+:sum) linear(current_iteration:1)
            for (int point = 0; point < width; point++) {

                // Real and imaginary component on pow2
                float r2 = zRealAP[point] * zRealAP[point];
                float i2 = zImagAP[point] * zImagAP[point];

                // Threshold condition cast to int for more optimal computation
                auto conditionAchieved = static_cast<int>(r2 + i2 > 4.0f);

                // Single cast to float (optimization purpose)
                auto conditionAchievedF = static_cast<float>(conditionAchieved);

                // Negation of condition (optimization purpose
                auto conditionNotAchieved = static_cast<float>(1 - conditionAchieved);

                // Counter of condition achievements by each point
                // Subtracting it from max iteration reached we get on which iteration condition was firstly satisfied
                conditionAchievedAP[point] += conditionAchieved;

                // Sum up condition achievements
                sum += conditionAchieved;


                // Calculate new imaginary and real component of current point
                // We need to stop incrementing to not overflow -> ternary operator is slower than next lines
                zImagAP[point] = zImagAP[point] * ((conditionNotAchieved * 2.0f * zRealAP[point]) + conditionAchievedF) +
                                (conditionNotAchieved * imag);
                zRealAP[point] = (conditionAchievedF * zRealAP[point]) +
                                (conditionNotAchieved * (r2 - i2 + xInitValAP[point]));

            }
            // Break limit loop if all points meets ending condition
            if (sum == width || current_iteration == limit - 1) {
                // Calculate output data shift
                auto row_shift = static_cast<long>(row * width);
                int finished_iterations = current_iteration + 1;

                // Save every point first condition met iteration
                #pragma omp simd aligned(dataP, conditionAchievedAP:64) simdlen(64) linear(current_iteration:1)
                for (int point = 0; point < width; point++) {
                    dataP[row_shift + point] = finished_iterations - conditionAchievedAP[point];
                }
                break;

            }
        }
    }
    return data;
}
