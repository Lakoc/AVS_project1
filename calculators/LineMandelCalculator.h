/**
 * @file LineMandelCalculator.h
 * @author Alexander Polok <xpolok03@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    // @TODO add all internal parameters
    int *data;
    // Array of arranged Xs in range(x_start, x_start + width * dx)
//    alignas(64) float* xInitValA;
////
//    // Arrays keeping imaginary and real values of each complex number on current row
//    alignas(64) float* zRealA;
//    alignas(64) float* zImagA;
//
//    // Array of counts of limit achievements of each complex number on current row
//    alignas(64) int* conditionAchievedA;
//
//    // Cast floats to doubles
//    float xStart;
//    float yStart;
//    float dxF;
//    float dyF;

//    // Array of arranged Xs in range(x_start, x_start + width * dx)
//    xInitValA = static_cast<float *>(aligned_alloc(64 * sizeof(float), width * sizeof(float)));
//
//    // Arrays keeping imaginary and real values of each complex number on current row
//    zRealA = static_cast<float *>(aligned_alloc(64 * sizeof(float), width * sizeof(float)));
//    zImagA = static_cast<float *>(aligned_alloc(64 * sizeof(float), width * sizeof(float)));
//
//    // Array of counts of limit achievements of each complex number on current row
//    conditionAchievedA = static_cast<int *>(aligned_alloc(64 * sizeof(int), width * sizeof(int)));
//
//    // Cast floats to doubles
//    xStart = static_cast<float>(x_start);
//    yStart = static_cast<float>(y_start);
//    dxF = static_cast<float>(dx);
//    dyF = static_cast<float>(dy);
//
//    // Arrange Xs
//#pragma omp simd
//    for (size_t point = 0; point < width; point++) {
//        xInitValA[point] = xStart + static_cast<float>(point) * dxF;
//    }
};