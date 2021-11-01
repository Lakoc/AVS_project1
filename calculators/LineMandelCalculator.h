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
    int *data;
    // Array of arranged Xs in range(x_start, x_start + width * dx)
    float* xInitValA;

    // Arrays keeping imaginary and real values of each complex number on current row
    float* zRealA;
    float* zImagA;

    // Array of counts of limit achievements of each complex number on current row
    int* conditionAchievedA;
};