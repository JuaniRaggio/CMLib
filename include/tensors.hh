#pragma once

#include <iostream>

class Tensor {
private:
  std::uint32_t rows, cols;
  std::float_t data;

public:
  Tensor(std::uint32_t rows, std::uint32_t cols);
  ~Tensor();

  void matrix_copy(Tensor *src);
  void matrix_clear();
  void matrix_fill(std::float_t data);
  void matrix_scale(std::float_t scale);
  std::uint8_t matrix_add(std::float_t scale);
  std::uint8_t matrix_sub(Tensor other);
  Tensor matrix_mult(const Tensor x, const Tensor y, std::atomic_bool zero_out,
                     std::atomic_bool transpose_x,
                     std::atomic_bool transpose_y);
  Tensor matrix_relu(Tensor in);
  Tensor matrix_softmax(Tensor in);
  Tensor matrix_cross_entropy(const Tensor p, const Tensor q);
  Tensor matrix_relu_grad(const Tensor in);
  Tensor matrix_softmax_add_gradient(Tensor out, const Tensor softmax_out);
  Tensor matrix_corss_entropy_add_gradient(const Tensor p, const Tensor q);
};
