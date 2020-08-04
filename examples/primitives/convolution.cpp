/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/// @example convolution.cpp
/// > Annotated version: @ref convolution_example_cpp
///
/// @page convolution_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Convolution](@ref dev_guide_convolution) primitive in forward propagation
/// mode.
///
/// Key optimizations included in this example:
/// - Creation of optimized memory format from the primitive descriptor;
/// - Primitive attributes with fused post-ops.
///
/// @page convolution_example_cpp Convolution Primitive Example
/// @copydetails convolution_example_cpp_short
///
/// @include convolution.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER  start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "RUNTIME of " << name << ": " << \
    std::chrono::duration_cast<std::chrono::milliseconds>( \
            std::chrono::high_resolution_clock::now()-start \
    ).count() << " ms " << std::endl; 

using tag = memory::format_tag;
using dt = memory::data_type;

void convolution_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 1, // batch size
            IC = 1, // input channels
            IH = 1024, // input height
            IW = 1024, // input width
            OC = 1, // output channels
            KH = 32, // weights height
            KW = 32, // weights width
            PH_L = 0, // height padding: left
            PH_R = 0, // height padding: right
            PW_L = 0, // width padding: left
            PW_R = 0, // width padding: right
            SH = 2, // height-wise stride
            SW = 2, // width-wise stride
            OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
            OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims weights_dims = {OC, IC, KH, KW};
    memory::dims dst_dims = {N, OC, OH, OW};

    // Strides, padding dimensions.
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights, and dst tensors.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    }); 

    // Create memory objects for tensor data (src, weights, dst). In this
    // example, NCHW layout is assumed for src and dst, and OIHW for weights.
    auto user_src_mem = memory({src_dims, dt::f32, tag::nchw}, engine);
    auto user_weights_mem = memory({weights_dims, dt::f32, tag::oihw}, engine);
    auto user_dst_mem = memory({dst_dims, dt::f32, tag::nchw}, engine);

    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
    auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), user_src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);

    // Create operation descriptor.
    auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            conv_dst_md, strides_dims, padding_dims_l,
            padding_dims_r);

    // Create primitive descriptor.
    auto conv_pd
            = convolution_forward::primitive_desc(conv_desc, engine);

    // For now, assume that the src, weights, and dst memory layouts generated
    // by the primitive and the ones provided by the user are identical.
    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    // Reorder the data in case the src and weights memory layouts generated by
    // the primitive and the ones provided by the user are different. In this
    // case, we create additional memory objects with internal buffers that will
    // contain the reordered data. The data in dst will be reordered after the
    // convolution computation has finalized.
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), engine);
        reorder(user_src_mem, conv_src_mem)
                .execute(engine_stream, user_src_mem, conv_src_mem);
    }

    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), engine);
        reorder(user_weights_mem, conv_weights_mem)
                .execute(engine_stream, user_weights_mem, conv_weights_mem);
    }

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_pd.dst_desc(), engine);
    }

    // Create the primitive.
    auto conv_prim = convolution_forward(conv_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

    // Primitive execution: convolution with ReLU.
    INIT_TIMER
    START_TIMER
    conv_prim.execute(engine_stream, conv_args);
    STOP_TIMER("convolution")

    // Reorder the data in case the dst memory descriptor generated by the
    // primitive and the one provided by the user are different.
    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        reorder(conv_dst_mem, user_dst_mem)
                .execute(engine_stream, conv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = conv_dst_mem;

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            convolution_example, parse_engine_kind(argc, argv));
}
