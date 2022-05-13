#ifndef UTILS_H
#define UTILS_H

#include <random>


void MakeDenseMatrix(int rows, int columns, float *matrix,
                     std::default_random_engine generator, float lb=-1.0, float ub=1.0)
{
    std::uniform_real_distribution<float> distribution(lb, ub);
    
    for (int64_t i = 0; i < static_cast<int64_t>(rows) * columns; ++i){
        matrix[i] = distribution(generator);
    }
}

#endif