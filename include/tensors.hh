#pragma once

#include <iostream>

// This should run on the GPU
namespace matrix_ops {
typedef enum {
  OK = 0,
} EXIT_CODE;

typedef struct {
  std::uint32_t rows, cols;
  std::float_t data;
} matrix;

matrix *mat_create(void *mem, std::uint32_t rows, std::uint32_t cols);
void matrix_copy(matrix *dst, matrix *src);
void matrix_clear(matrix *matrix);
void matrix_fill(matrix *matrix, std::float_t data);
void matrix_scale(matrix *matrix, std::float_t scale);
EXIT_CODE matrix_add(matrix *matrix, std::float_t scale);
EXIT_CODE matrix_sub(matrix *matrix, std::float_t scale);
EXIT_CODE matrix_mult(matrix *out, const matrix *x, const matrix *y,
                      std::atomic_bool zero_out, std::atomic_bool transpose_x,
                      std::atomic_bool transpose_y);
EXIT_CODE matrix_relu(matrix *out, matrix *in);
EXIT_CODE matrix_softmax(matrix *out, matrix *in);
EXIT_CODE matrix_cross_entropy(matrix *out, const matrix *p, const matrix *q);
EXIT_CODE matrix_relu_grad(matrix *out, const matrix *in);
EXIT_CODE matrix_softmax_add_gradient(matrix *out, const matrix *softmax_out);
EXIT_CODE matrix_corss_entropy_add_gradient(matrix *out, const matrix *p,
                                            const matrix *q);

}; // namespace matrix_ops
