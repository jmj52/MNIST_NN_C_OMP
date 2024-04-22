#include "ops.h"
#include <stdlib.h>
#include <stdio.h>


// Verify row and column count of provided matrices are the same
int check_dimensions(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}


// Element-wise multiplication of provided matrices (m1 + m2) 
Matrix* multiply(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
#pragma omp parallel for collapse(2)
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] * m2->entries[i][j];
			}
		}
		return m;
	} else {
		printf("Dimension mismatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}


// Element-wise addition of provided matrices (m1 + m2) 
Matrix* add(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
#pragma omp parallel for collapse(2)
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
			}
		}
		return m;
	} else {
		printf("Dimension mismatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}


// Element-wise subtraction of provided matrices (m1 - m2) 
Matrix* subtract(Matrix *m1, Matrix *m2) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
#pragma omp parallel for collapse(2)
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
			}
		}
		return m;
	} else {
		printf("Dimension mismatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}


// For all elements of matrix, apply the passed in function
Matrix* apply(double (*func)(double), Matrix* m) {
	Matrix *mat = matrix_copy(m);
#pragma omp parallel for collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = (*func)(m->entries[i][j]);
		}
	}
	return mat;
}


// Returns the dot-product of the provided matrices 
Matrix* dot(Matrix *m1, Matrix *m2) {
	if (m1->cols == m2->rows) {
		Matrix *m = matrix_create(m1->rows, m2->cols);
#pragma omp parallel for collapse(2)
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				double sum = 0;
				for (int k = 0; k < m2->rows; k++) {
					sum += m1->entries[i][k] * m2->entries[k][j];
				}
				m->entries[i][j] = sum;
			}
		}
		return m;
	} else {
		printf("Dimension mismatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}


// For all elements of matrix, multiply by a fixed value
Matrix* scale(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
#pragma omp parallel for collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] *= n;
		}
	}
	return mat;
}


// For all elements of matrix, add a fixed value
Matrix* addScalar(double n, Matrix* m) {
	Matrix* mat = matrix_copy(m);
#pragma omp parallel for collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] += n;
		}
	}
	return mat;
}


// Flip m x n matrix into n x m matrix
Matrix* transpose(Matrix* m) {
	Matrix* mat = matrix_create(m->cols, m->rows);
#pragma omp parallel for collapse(2)
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[j][i] = m->entries[i][j];
		}
	}
	return mat;
}