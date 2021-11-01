/**
 * @file BatchMandelCalculator.h
 * @author Alexander Polok <xpolok03@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator {
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);

    ~BatchMandelCalculator();

    int *calculateMandelbrot();

private:
    int *data;
    // Array of arranged Xs in range(x_start, x_start + width * dx)
    float *xInitValA;
    float *yInitValA;

    // Arrays keeping imaginary and real values of each complex number on current row
    float *zRealA;
    float *zImagA;

    // Array of counts of limit achievements of each complex number on current row
    int *conditionAchievedA;
};

#endif