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


LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize, unsigned limit) :
        BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator") {
    data = (int *) (malloc(height * width * sizeof(int)));
}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    data = nullptr;
}


int *LineMandelCalculator::calculateMandelbrot() {
    int *pdata = data;
    // Array of arranged Xs in range(x_start, x_start + width * dx)
    alignas(128) float xInitValA[width];

    // Arrays keeping imaginary and real values of each complex number on current row
    alignas(128) float zRealA[width];
    alignas(128) float zImagA[width];

    // Array of counts of limit achievements of each complex number on current row
    alignas(128) int conditionAchievedA[width];

    // Cast floats to doubles
    auto xStart = static_cast<float>(x_start);
    auto yStart = static_cast<float>(y_start);
    auto dxF = static_cast<float>(dx);
    auto dyF = static_cast<float>(dy);

    // Arrange Xs
    #pragma omp simd aligned(xInitValA:128)
    for (size_t point = 0; point < width; point++) {
        xInitValA[point] = xStart + static_cast<float>(point) * dxF;
    }

    // Iterate over all rows
    for (size_t row = 0; row < height; row++) {
        // Calculate imaginary value of selected row
        auto imag = yStart + static_cast<float>(row) * dyF;


        // Copy starting x's to current row, clean up limits, copy imaginary value
        #pragma omp simd aligned(xInitValA, zImagA, zRealA, conditionAchievedA:128)
        for (size_t point = 0; point < width; point++) {
            zImagA[point] = imag;
            zRealA[point] = xInitValA[point];
            conditionAchievedA[point] = 0;
        }

        // Iterate until every point threshold condition is met or iteration limit is reached
        for (int current_iteration = 0; current_iteration < limit; current_iteration++) {

            // Auxiliary variable to stop loop
            int sum = 0;

            // Iterate over each complex number and
            #pragma omp simd aligned(pdata, xInitValA, zImagA, zRealA, conditionAchievedA:128) reduction(+:sum)
            for (int point = 0; point < width; point++) {

                // Real and imaginary component on pow2
                float r2 = zRealA[point] * zRealA[point];
                float i2 = zImagA[point] * zImagA[point];

                // Threshold condition cast to int for more optimal computation
                auto conditionAchieved = static_cast<int>(r2 + i2 > 4.0f);
                // Negation of condition (optimization purpose
                auto conditionNotAchieved = static_cast<float>(1 - conditionAchieved);

                // Counter of condition achievements by each point
                // Subtracting it from max iteration reached we get on which iteration condition was firstly satisfied
                conditionAchievedA[point] += conditionAchieved;

                // Sum up condition achievements
                sum += conditionAchieved;

                // Single cast to float (optimization purpose)
                auto conditionAchievedF = static_cast<float>(conditionAchieved);

                // Calculate new imaginary and real component of current point
                // We need to stop incrementing to not overflow -> ternary operator is slower than next lines
                zImagA[point] = zImagA[point] * ((conditionNotAchieved * 2.0f * zRealA[point]) + conditionAchievedF) +
                                (conditionNotAchieved * imag);
                zRealA[point] = (conditionAchievedF * zRealA[point]) +
                                (conditionNotAchieved * (r2 - i2 + xInitValA[point]));

            }
            // Break limit loop if all points meets ending condition
            if (sum == width || current_iteration == limit - 1) {
                // Calculate output data shift
                auto row_shift = static_cast<long>(row * width);
                int finished_iterations = current_iteration + 1;

                // Save every point first condition met iteration
                #pragma omp simd aligned(pdata, conditionAchievedA:128)
                for (int point = 0; point < width; point++) {
                    pdata[row_shift + point] = finished_iterations - conditionAchievedA[point];
                }
                break;


            }
        }
    }
    return data;
}
