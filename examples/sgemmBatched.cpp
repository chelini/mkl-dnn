#include "dnnl.hpp"
#include "example_utils.hpp"
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace dnnl;

void init_vector(std::vector<float> &v) {
  std::mt19937 gen;
  std::uniform_real_distribution<float> u(-1, 1);
  for (auto &e : v)
    e = /*u(gen)*/ 5;
}

// run C(m, n, [p]) += B(k, [p], m) * A(k, n)
void test4() {
  
  const size_t M = 32;
  const size_t N = 32;
  const size_t K = 64;
  const size_t P = 3;

  std::vector<float> A(K * N);
  init_vector(A);
  std::vector<float> B(K * P * M);
  init_vector(B);
  std::vector<float> C(M * N * P);
  init_vector(C);

  std::vector<float> CGold(M * N * P);
  CGold = C;

  std::cout << C[0] << " -- " << CGold[0] << std::endl;

  for (size_t m = 0; m < M; m++)
    for (size_t n = 0; n < N; n++)
      for (size_t p = 0; p < P; p++)
        for (size_t k = 0; k < K; k++)
          CGold[m * N * P + n * P + p] += B[k * P * M + p * M + m] * A[k * N + n];

  int64_t lda = N;
  int64_t ldb = K;
  int64_t ldc = N;

  for (size_t p = 0; p < P; p++)
    dnnl_sgemm('T', 'N', M, N, K, 1.0, B.data() + p * M, ldb, A.data(), lda, 1.0, C.data() + p * N, ldc);

  std::cout << C[0] << " -- " << CGold[0] << std::endl;
}

// run C(m, [n], p) += A(m, k) * B(k, [n], p) 
void test1() {

  const size_t M = 32;
  const size_t N = 3; // batch
  const size_t K = 64;
  const size_t P = 32;

  std::vector<float> A(M * K);
  init_vector(A);
  std::vector<float> B(K * N * P);
  init_vector(B);
  std::vector<float> C(M * N * P);
  init_vector(C);

  std::vector<float> CGold(M * N * P);
  CGold = C;

  std::cout << C[0] << " -- " << CGold[0] << std::endl;

  for (size_t m = 0; m < M; m++)
    for (size_t n = 0; n < N; n++)
      for (size_t p = 0; p < P; p++)
        for (size_t k = 0; k < K; k++)
          CGold[m * N * P + n * P + p] += A[m * K + k] * B[k * N * P + n * P + p];

  int64_t lda = K;
  int64_t ldb = P;
  int64_t ldc = P;
  
  for (size_t n = 0; n < N; n++)
    dnnl_sgemm('N', 'N', M, P, K, 1.0, A.data(), lda, B.data() + n * P, ldb, 1.0, C.data() + n * P, ldc);   


  std::cout << C[0] << " -- " << CGold[0] << std::endl; 
}

// gemm
void test2() {
  const size_t I = 32;
  const size_t J = 32;
  const size_t K = 64;

  std::vector<float> A(I * K);
  init_vector(A);
  std::vector<float> B(K * J);
  init_vector(B);
  std::vector<float> C(I * J);
  init_vector(C);
  std::vector<float> CGold(I * J);
  CGold = C;

  std::cout << C[0] << " -- " << CGold[0] << std::endl;

  for (size_t i = 0; i < I; i++)
    for (size_t j = 0; j < J; j++)
      for (size_t k = 0; k < K; k++)
        CGold[i * J + j] += A[i * K + k] * B[k * J + j];

  dnnl_sgemm('N', 'N', I, J, K, 1.0, A.data(), K, B.data(), J, 1.0, C.data(), J);

  std::cout << C[0] << " -- " << CGold[0] << std::endl;
}

// C(m, (np)) = A(m, k) * B(k, (np))
void test3() {
  const size_t M = 32;
  const size_t N = 32;
  const size_t P = 32;
  const size_t K = 64;

  std::vector<float> A(M * K);
  init_vector(A);
  std::vector<float> B(K * N *P);
  init_vector(B);
  std::vector<float> C(M * N * P);
  init_vector(C);
  std::vector<float> CGold(M * N * P);
  CGold = C;

  std::cout << C[0] << " -- " << CGold[0] << std::endl;

  for (size_t m = 0; m < M; m++)
    for (size_t n = 0; n < N; n++)
      for (size_t p = 0; p < P; p++)
        for (size_t k = 0; k < K; k++)
          CGold[m * N * P + n * P + p] += A[m * K + k] * B[k * N * P + n * P + p];

  dnnl_sgemm('N', 'N', M, N * P, K, 1.0, A.data(), K, B.data(), N * P, 0.0, C.data(), N * P);
      
  std::cout << C[0] << " -- " << CGold[0] << std::endl;
}

int main() {
  test4();
  return 0;
}
