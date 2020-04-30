/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "dnnl.hpp"
#include "example_utils.hpp"
#include <chrono>
using namespace dnnl;

#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER  start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << ": " << \
    std::chrono::duration_cast<std::chrono::milliseconds>( \
            std::chrono::high_resolution_clock::now()-start \
    ).count() << " ms " << std::endl; 

namespace {
void init_vector(std::vector<float> &v) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u(-1, 1);
    for (auto &e : v)
        e = u(gen);
}
int compare_vectors(const std::vector<float> &v1, const std::vector<float> &v2,
        int64_t K, const char *message) {
    double v1_l2 = 0, diff_l2 = 0;
    for (size_t n = 0; n < v1.size(); ++n) {
        float diff = v1[n] - v2[n];
        v1_l2 += v1[n] * v1[n];
        diff_l2 += diff * diff;
    }
    v1_l2 = std::sqrt(v1_l2);
    diff_l2 = std::sqrt(diff_l2);
    // Finding the reasonable (tight and accurate) threshold is quite difficult
    // problem.
    // The implementation testing might also use special data filling to
    // alleviate issues related to the finite precision arithmetic.
    // However, in simple cases the machine epsilon multiplied by log(K) should
    // work reasonably well.
    const double threshold = std::numeric_limits<float>::epsilon()
            * std::log(std::max(2., (double)K));
    bool ok = diff_l2 <= threshold * v1_l2;
    printf("%s\n\tL2 Norms"
           "\n\t\tReference matrix:%g\n\t\tError:%g\n\t\tRelative_error:%g\n"
           "\tAccuracy check: %s\n",
            message, v1_l2, diff_l2, diff_l2 / v1_l2, ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
} // namespace
int number_of_runs = 5;
float fixed_beta = 0.f;
engine eng(engine::kind::cpu, 0); // We create a global engine for simplicity
// Create a _dynamic_ MatMul primitive that can work with arbitrary shapes
// and alpha parameters.
// Warning: current limitation is that beta parameter should be known in
// advance (use fixed_beta).
matmul dynamic_matmul_create() {
    // We assume that beta is known at the primitive creation time
    float beta = fixed_beta;
    memory::dims a_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
    memory::dims b_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
    memory::dims c_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
    memory::dims a_strides = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
    memory::dims b_strides = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
    memory::dims c_strides = {DNNL_RUNTIME_DIM_VAL, 1};
    memory::desc a_md(a_shape, memory::data_type::f32, a_strides);
    memory::desc b_md(b_shape, memory::data_type::f32, b_strides);
    memory::desc c_md(c_shape, memory::data_type::f32, c_strides);
    // Create attributes (to handle alpha dynamically and beta if necessary)
    primitive_attr attr;
    attr.set_output_scales(/* mask */ 0, {DNNL_RUNTIME_F32_VAL});
    if (beta != 0.f) {
        post_ops po;
        po.append_sum(beta);
        attr.set_post_ops(po);
    }
    // Create a MatMul primitive
    matmul::desc matmul_d(a_md, b_md, c_md);
    matmul::primitive_desc matmul_pd(matmul_d, attr, eng);
    return matmul(matmul_pd);
}
// Execute a _dynamic_ MatMul primitive created earlier. All the parameters are
// passed at a run-time (except for beta which has to be specified at the
// primitive creation time due to the current limitation).
void dynamic_matmul_execute(matmul &matmul_p, char transA, char transB,
        int64_t M, int64_t N, int64_t K, float alpha, const float *A,
        int64_t lda, const float *B, int64_t ldb, float beta, float *C,
        int64_t ldc) {
    using dims = memory::dims;
    if (beta != fixed_beta)
        throw std::logic_error("Run-time beta is not yet supported.");
    // Translate transA and transB
    dims a_strides = tolower(transA) == 'n' ? dims {lda, 1} : dims {1, lda};
    dims b_strides = tolower(transB) == 'n' ? dims {ldb, 1} : dims {1, ldb};
    // Wrap raw pointers into DNNL memories (with proper shapes)
    memory A_m({{M, K}, memory::data_type::f32, a_strides}, eng, (void *)A);
    memory B_m({{K, N}, memory::data_type::f32, b_strides}, eng, (void *)B);
    memory C_m({{M, N}, memory::data_type::f32, {ldc, 1}}, eng, (void *)C);
    // Prepare DNNL memory for alpha
    memory alpha_m({{1}, memory::data_type::f32, {1}}, eng, &alpha);
    // Execute the MatMul primitive
    stream s(eng);
    matmul_p.execute(s,
            {{DNNL_ARG_SRC, A_m}, {DNNL_ARG_WEIGHTS, B_m}, {DNNL_ARG_DST, C_m},
                    {DNNL_ARG_ATTR_OUTPUT_SCALES, alpha_m}});
    s.wait();
}
// Create and execute a _static_ MatMul primitive. All shapes and parameters
// are hard-coded in the primitive and cannot be changed later.
void static_matmul_create_and_execute(char transA, char transB, int64_t M,
        int64_t N, int64_t K, float alpha, const float *A, int64_t lda,
        const float *B, int64_t ldb, float beta, float *C, int64_t ldc) {
    using dims = memory::dims;
    // Prepare strides based on the transA and transB flags: transposed
    // matrices have strides swapped
    dims a_strides = tolower(transA) == 'n' ? dims {lda, 1} : dims {1, lda};
    dims b_strides = tolower(transB) == 'n' ? dims {ldb, 1} : dims {1, ldb};
    // Prepare memory descriptors
    memory::desc a_md({M, K}, memory::data_type::f32, a_strides);
    memory::desc b_md({K, N}, memory::data_type::f32, b_strides);
    memory::desc c_md({M, N}, memory::data_type::f32, {ldc, 1});
    // Create attributes (to handle alpha and beta if necessary)
    primitive_attr attr;
    if (alpha != 1.f) attr.set_output_scales(/* mask */ 0, {alpha});
    if (beta != 0.f) {
        post_ops po;
        po.append_sum(beta);
        attr.set_post_ops(po);
    }
    // Create a MatMul primitive
    matmul::desc matmul_d(a_md, b_md, c_md);
    matmul::primitive_desc matmul_pd(matmul_d, attr, eng);
    matmul matmul_p(matmul_pd);
    // Wrap raw pointers into DNNL memory objects
    memory A_m(a_md, eng, (void *)A);
    memory B_m(b_md, eng, (void *)B);
    memory C_m(c_md, eng, (void *)C);
    // Execute the MatMul primitive.
    // Since here all shapes and parameters are static, please note that we
    // don't need to pass alpha (scales) again, as they are already hard-coded
    // in the primitive descriptor. Also, we are not allowed to change the
    // shapes of matrices A, B, and C -- they should exactly match
    // the memory descriptors passed to MatMul operation descriptor.
    stream s(eng);
    matmul_p.execute(s,
            {{DNNL_ARG_SRC, A_m}, {DNNL_ARG_WEIGHTS, B_m},
                    {DNNL_ARG_DST, C_m}});
    s.wait();
}
void sgemm_and_matmul_with_params(char transA, char transB, int64_t M,
        int64_t N, int64_t K, float alpha, float beta) {
    if (beta != fixed_beta)
        throw std::logic_error("Run-time beta is not yet supported.");
    // Allocate and initialize matrices
    std::vector<float> A(M * K);
    init_vector(A);
    std::vector<float> B(K * N);
    init_vector(B);
    std::vector<float> C_sgemm(M * N);
    init_vector(C_sgemm);
    std::vector<float> C_dynamic_matmul = C_sgemm;
    std::vector<float> C_static_matmul = C_sgemm;
    // Prepare leading dimensions
    int64_t lda = tolower(transA) == 'n' ? K : M;
    int64_t ldb = tolower(transB) == 'n' ? N : K;
    int64_t ldc = N;
    // 1. Execute sgemm
    for (int run = 0; run < number_of_runs; ++run) {
        INIT_TIMER
        START_TIMER
        dnnl_sgemm(transA, transB, M, N, K, alpha, A.data(), lda, B.data(), ldb,
                beta, C_sgemm.data(), ldc);
        STOP_TIMER("sgemm")
    }
    // 2.a Create dynamic MatMul
    auto dynamic_matmul = dynamic_matmul_create();
    // 2.b Execute
    for (int run = 0; run < number_of_runs; ++run)
        dynamic_matmul_execute(dynamic_matmul, transA, transB, M, N, K, alpha,
                A.data(), lda, B.data(), ldb, beta, C_dynamic_matmul.data(),
                ldc);
    // 3. Execute static MatMul
    for (int run = 0; run < number_of_runs; ++run)
        static_matmul_create_and_execute(transA, transB, M, N, K, alpha,
                A.data(), lda, B.data(), ldb, beta, C_static_matmul.data(),
                ldc);
    int rc = 0;
    rc |= compare_vectors(
            C_sgemm, C_dynamic_matmul, K, "Compare SGEMM vs dynamic MatMul");
    if (rc) throw std::logic_error("The resulting matrices diverged too much.");
    rc |= compare_vectors(
            C_sgemm, C_static_matmul, K, "Compare SGEMM vs static MatMul");
    if (rc) throw std::logic_error("The resulting matrices diverged too much.");
}
void sgemm_and_matmul() {
    sgemm_and_matmul_with_params('N', 'N', 1024, 1024, 1024, 1.1f, fixed_beta);
}
int main(int argc, char **argv) {
    return handle_example_errors({engine::kind::cpu}, sgemm_and_matmul);
}
