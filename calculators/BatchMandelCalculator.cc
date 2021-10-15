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

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
    data = (int *) (malloc(height * width * sizeof(int)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
    free(data);
    data = nullptr;
}


int * BatchMandelCalculator::calculateMandelbrot () {
    int *pdata = data;
    alignas(64) float x_init_valA[width];
    alignas(64) float zRealA[width];
    alignas(64) float zImagA[width];
    alignas(64) int limitsA[width];

#pragma omp simd
    for (int i = 0; i < width; i++) {
        auto val = static_cast<float>(x_start + i * dx);
        x_init_valA[i] = val;
    }


    for (int i = 0; i < height; i++) {
        auto imag = static_cast<float>(y_start + i * dy);
        memcpy(zRealA, x_init_valA, sizeof(float) * width);

#pragma omp simd linear(i:1)
        for (size_t j = 0; j < width; j++) {
            zImagA[j] = imag;
            limitsA[j] = 0;
        }

        for (int j = 0; j < limit; j++) {
            int sum = 0;
#pragma omp simd linear(i, j:1) aligned(pdata, x_init_valA, zImagA, zRealA, limitsA:64) reduction(+:sum)
            for (int k = 0; k < width; k++) {

                float r2 = zRealA[k] * zRealA[k];
                float i2 = zImagA[k] * zImagA[k];

                auto limit_reached = static_cast<int>(r2 + i2 > 4.0f);
                int limit_non_reached = 1- limit_reached;
                limitsA[k] += limit_reached;

                sum += limit_reached;

//                zImagA[k] = limit_reached ? zImagA[k] : 2.0f * zRealA[k] * zImagA[k] + imag;
                zImagA[k] = zImagA[k] * ((limit_non_reached * 2.0f * zRealA[k]) + limit_reached) + (limit_non_reached * imag);
//                zRealA[k] = limit_reached ? zRealA[k] : r2 - i2 + x_init_valA[k];
                zRealA[k] = (limit_reached * zRealA[k]) + (limit_non_reached *(r2 - i2 + x_init_valA[k]));

            }
            if (sum == width || j == limit - 1) {
                int row_shift = i * width;
                int normalization = j + 1;


#pragma omp simd
                for (int k = 0; k < width; k++) {
                    pdata[row_shift + k] = normalization - limitsA[k];
                }
                break;


            }
        }
    }
    return data;
}