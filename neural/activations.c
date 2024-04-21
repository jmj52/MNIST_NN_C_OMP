#include "activations.h"

#include <math.h>
#include "../matrix/ops.h"


// Applies the sigmoid function to a given input value
double sigmoid(double input) {
	return 1.0 / (1 + exp(-1 * input));
}

// Applies the derivate of the sigmoid function to a given matrix. 
// Assumes sigmoid has already been applied to matrix m
Matrix* sigmoidPrime(Matrix* m) {
	Matrix* ones = matrix_create(m->rows, m->cols);
	matrix_fill(ones, 1);
	Matrix* subtracted = subtract(ones, m);
	Matrix* multiplied = multiply(m, subtracted);
	matrix_free(ones);
	matrix_free(subtracted);
	return multiplied;
}


// Normalizes range of output prediction values
Matrix* softmax(Matrix* m) {
	double total = 0;
// First calculate cumulative sum of exponents value
#pragma omp parallel for collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			total += exp(m->entries[i][j]);
		}
	}
// Next normalize each value
	Matrix* mat = matrix_create(m->rows, m->cols);
#pragma omp parallel for collapse(2)	
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			mat->entries[i][j] = exp(m->entries[i][j]) / total;
		}
	}
	return mat;
}