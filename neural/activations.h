#pragma once

#include "../matrix/matrix.h"
#include <omp.h>

double sigmoid(double input);
Matrix* sigmoidPrime(Matrix* m);
Matrix* softmax(Matrix* m);