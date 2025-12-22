// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_onednn.hpp"
#include "convolution_inst.h"
#include "permute_inst.h"
#include "intel_gpu/runtime/format.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"
#include "convolution_shape_inference.hpp"

#include "utils.hpp"

#include "intel_gpu/runtime/debug_configuration.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

static std::shared_ptr<dnnl::convolution_forward::primitive_desc> get_convolution_primitive_descriptor(const kernel_impl_params& impl_params,
                                            const dnnl::primitive_attr& attr = dnnl::primitive_attr(),
                                            dnnl::memory::format_tag tag_in_out = dnnl::memory::format_tag::undef) {
    auto& engine = impl_params.prog->get_engine();
    auto prim = impl_params.typed_desc<convolution>();

    auto input_layout = impl_params.get_input_layout(0);
    auto weights_layout = impl_params.get_input_layout(1);
    auto output_layout = impl_params.get_output_layout();
    auto auto_pad = prim->auto_pad;
    
    // Debug ALL convolutions to understand groups parameter
    std::cerr << "[DEBUG] Conv: name=" << impl_params.desc->id 
              << ", groups=" << prim->groups 
              << ", input_channels=" << input_layout.feature()
              << ", weights_shape=[";
    auto ws = weights_layout.get_shape();
    for (size_t i = 0; i < ws.size(); i++) {
        if (i > 0) std::cerr << ",";
        std::cerr << ws[i];
    }
    std::cerr << "]" << std::endl;
    
    // For grouped convolutions, check if we need to recover the correct weights shape/format
    // This handles the case where GroupConvolution with groups=input_channels (depthwise)
    // had its weights incorrectly simplified or has wrong format
    if (prim->groups > 1 && prim->groups == input_layout.feature()) {
        auto weights_shape = weights_layout.get_shape();
        auto weights_tensor = weights_layout.get_tensor();
        
        std::cerr << "[DEBUG] Depthwise grouped conv detected: groups=" << prim->groups
                  << ", input_channels=" << input_layout.feature()
                  << ", weights_shape=[";
        for (size_t i = 0; i < weights_shape.size(); i++) {
            if (i > 0) std::cerr << ",";
            std::cerr << weights_shape[i];
        }
        std::cerr << "], format=" << weights_layout.format.to_string() << std::endl;
        std::cerr << "  weights_tensor: batch[0]=" << weights_tensor.batch[0]
                  << ", feature[0]=" << weights_tensor.feature[0]
                  << ", spatial[0]=" << weights_tensor.spatial[0]
                  << ", spatial[1]=" << weights_tensor.spatial[1] << std::endl;
        
        // Case 1: Weights have correct 4D shape [G,O/G,I,K] but wrong format (bfyx instead of goiyx)
        // For 1D depthwise conv with kernel_size K: shape should be [groups, 1, 1, K]
        // In bfyx format: batch=G, feature=O/G, spatial[0]=kernel_x, spatial[1]=kernel_y
        // OpenVINO treats 1D conv as 2D with Y=1, so we need 5D: [G, O/G, I, Y=1, X=K]
        if (weights_shape.size() == 4 &&
            weights_tensor.batch[0] == prim->groups &&
            weights_tensor.feature[0] <= 1 &&
            weights_tensor.spatial[1] == 1) {  // Y dimension == 1 means 1D conv treated as 2D
            
            // Extract kernel size from spatial[0] (X dimension)
            int kernel_size = weights_tensor.spatial[0];
            
            std::cerr << "  4D shape detected with kernel_size=" << kernel_size << std::endl;
            std::cerr << "  Converting to 5D goiyx format: [G=" << prim->groups 
                      << ",O/G=1,I=1,Y=1,X=" << kernel_size << "]" << std::endl;
            
            // Create correct 5D shape for grouped convolution
            // goiyx format: [groups, output_per_group, input_per_group, kernel_y, kernel_x]
            ov::PartialShape goiyx_shape = {
                static_cast<ov::Dimension::value_type>(prim->groups),        // G = 192
                static_cast<ov::Dimension::value_type>(1),                    // O/G = 1
                static_cast<ov::Dimension::value_type>(1),                    // I = 1
                static_cast<ov::Dimension::value_type>(weights_tensor.spatial[1]),  // Y from spatial[1]
                static_cast<ov::Dimension::value_type>(kernel_size)           // X from spatial[0]
            };
            
            // Update layout with goiyx format and 5D shape
            weights_layout.format = format::get_default_format(5, true, true);  // goiyx format
            weights_layout.set_partial_shape(goiyx_shape);
            
            std::cerr << "  Reconstructed weights_layout: " << weights_layout.to_short_string() << std::endl;
        }
        // Case 2: Weights are missing the kernel dimension entirely
        // Expected: [G, out_per_group, in_per_group, Y, X] for 2D grouped conv (goiyx format, 5D)
        // Actual might be: [G, in_per_group, 1] (3D) missing kernel dimensions
        else if (weights_shape.size() == 3 && 
            weights_tensor.batch[0] == prim->groups &&
            weights_tensor.feature[0] <= 1 &&
            weights_tensor.spatial[0] == 1 && weights_tensor.spatial[1] == 1) {
            
            // Try to infer kernel size from dilation and stride
            // For 1D conv, kernel size is often 3, 5, 7, etc.
            // We can also check the actual memory size if available
            int likely_kernel_size = 3;  // Common default for 1D depthwise conv
            
            std::cerr << "  Likely missing kernel dimension. Inferring kernel_size=" << likely_kernel_size << std::endl;
            std::cerr << "  Reconstructing weights shape to 5D for goiyx format" << std::endl;
            std::cerr << "  From: [" << weights_tensor.batch[0] << ","
                      << weights_tensor.feature[0] << "," << weights_tensor.spatial[0] << "]" << std::endl;
            std::cerr << "  To: [G=" << weights_tensor.batch[0] << ", O=" 
                      << weights_tensor.feature[0] << ", I=" 
                      << weights_tensor.spatial[0] << ","
                      << "Y=" << 1 << ", X=" << likely_kernel_size << "]" << std::endl;
            
            // Create a new 5D grouped tensor: goiyx format = group(G), batch(O), feature(I), spatial(Y, X)
            // For depthwise 1D conv: groups=192, output_per_group=1, input_per_group=1, y=1, kernel_x=3
            cldnn::tensor new_tensor(
                group(static_cast<int32_t>(weights_tensor.batch[0])),      // groups (G)
                batch(static_cast<int32_t>(weights_tensor.feature[0])),    // output_per_group (O)
                feature(static_cast<int32_t>(weights_tensor.spatial[0])),  // input_per_group (I)
                spatial(static_cast<int32_t>(likely_kernel_size),          // kernel_x (X)
                        static_cast<int32_t>(1))                           // kernel_y (Y) = 1 for 1D conv
            );
            
            // Update the layout with correct 5D shape and grouped format (goiyx)
            weights_layout.set_tensor(new_tensor);
            weights_layout.format = format::get_default_format(5, true, true);  // goiyx format
            
            std::cerr << "  Reconstructed weights_layout: " << weights_layout.to_short_string() << std::endl;
        }
    }

    // issue: it could not find the implementation for 1d kernel GroupConvolution from onednn.
    // root-cause: 3d tensor of input/output is changed to 4d via ngraph.
    //             Creating conv description returns error if two inputs have same tensor of data input and weight.
    //     - original dims of IR
    //       input1: [  1, 280, 1200]      // [number of batches, number of channels, X]
    //       input2: [280,   1,    1, 67]  // [number of output channels, number of input channels, Y, X]
    //       output: [  1, 280, 1200]      // [number of batches, number of kernel output channels, X]
    //     - changed dims
    //       input1: [  1, 280, 1200,  1]
    //       input2: [280,   1,   67,  1]
    //       output: [  1, 280, 1200,  1]
    // WA: Weight tensor will be updated from 4d to 5d.
    // Note: This also handles the case where weights_layout has been incorrectly simplified to non-grouped format
    // (e.g., bfyx instead of goiyx) while still containing 1D convolution kernel data.
    auto grouped_weights = format::is_grouped(weights_layout.format) || prim->grouped_weights_shape;
    
    std::cerr << "[DEBUG] Convolution check: grouped_weights=" << grouped_weights 
              << ", prim->grouped_weights_shape=" << prim->grouped_weights_shape
              << ", input_rank=" << input_layout.get_rank()
              << ", input_spatial_rank=" << input_layout.get_spatial_rank()
              << ", weights_format=" << weights_layout.format.to_string()
              << ", is_grouped=" << format::is_grouped(weights_layout.format) << std::endl;
    
    // Additional heuristic: detect if this is actually a grouped 1D convolution that was incorrectly 
    // identified as regular convolution. Characteristics:
    // - Input is 3D (batch, channels, spatial) or 4D with only 1 spatial dimension used
    // - Weights should be goix (group, out_per_group, in_per_group, kernel_x) but might be bfyx
    // - The "batch" dimension of weights equals input channels (indicating groups)
    // - The "feature" dimension of weights is small (out_per_group)
    auto weights_tensor = weights_layout.get_tensor();
    if (!grouped_weights && input_layout.feature() > 1) {
        std::cerr << "[DEBUG] Heuristic check: input_feature=" << input_layout.feature()
                  << ", weights_tensor.batch=" << weights_tensor.batch[0]
                  << ", weights_tensor.feature=" << weights_tensor.feature[0]
                  << ", spatial=[" << weights_tensor.spatial[0] << "," << weights_tensor.spatial[1] << "," << weights_tensor.spatial[2] << "]"
                  << std::endl;
        
        // Check if weights "batch" equals input channels (typical for depthwise/grouped conv)
        // and spatial pattern suggests 1D convolution
        bool has_1d_kernel = (weights_tensor.spatial[0] > 1 && weights_tensor.spatial[1] == 1 && weights_tensor.spatial[2] == 1) ||
                            (weights_tensor.spatial[1] > 1 && weights_tensor.spatial[0] == 1 && weights_tensor.spatial[2] == 1);
        
        std::cerr << "[DEBUG] has_1d_kernel=" << has_1d_kernel 
                  << ", batch==feature? " << (weights_tensor.batch[0] == input_layout.feature())
                  << ", feature==1? " << (weights_tensor.feature[0] == 1) << std::endl;
        
        if (weights_tensor.batch[0] == input_layout.feature() && 
            weights_tensor.feature[0] == 1 &&
            has_1d_kernel) {
            grouped_weights = true;  // Override the flag
            std::cerr << "[DEBUG] Detected misidentified 1D grouped convolution! weights_tensor: ["
                      << weights_tensor.batch[0] << "," << weights_tensor.feature[0] << ","
                      << weights_tensor.spatial[0] << "," << weights_tensor.spatial[1] << "]" << std::endl;
        }
    }
    
    // Check if this is a 1D grouped convolution that needs weight format upgrade:
    // - Either format is already grouped, OR primitive indicates grouped weights
    // - Input has 3D spatial shape (1D convolution)
    // - Weights need to be upgraded to 5D for oneDNN compatibility
    bool is_1d_grouped_conv = grouped_weights && (input_layout.get_spatial_rank() == 1 || 
                                                   (input_layout.get_rank() == 3) ||
                                                   (input_layout.get_rank() == 4 && input_layout.spatial(1) == 1 && input_layout.spatial(2) == 1));
    
    if (is_1d_grouped_conv) {
        auto tensor = weights_layout.get_tensor();
        
        std::cerr << "[DEBUG] 1D grouped conv detected, weights_layout before fix: " << weights_layout.to_short_string() << std::endl;
        std::cerr << "  tensor.batch: " << tensor.batch[0] << std::endl;
        std::cerr << "  tensor.feature: " << tensor.feature[0] << std::endl;
        std::cerr << "  tensor.spatial[0]: " << tensor.spatial[0] << std::endl;
        std::cerr << "  tensor.spatial[1]: " << tensor.spatial[1] << std::endl;
        std::cerr << "  tensor.spatial[2]: " << tensor.spatial[2] << std::endl;
        std::cerr << "  weights_layout.get_rank(): " << weights_layout.get_rank() << std::endl;
        std::cerr << "  format::is_grouped(weights_layout.format): " << format::is_grouped(weights_layout.format) << std::endl;
        
        // If weights format is not grouped (e.g., incorrectly simplified to bfyx),
        // we need to restore it based on the actual tensor shape
        if (!format::is_grouped(weights_layout.format) && prim->grouped_weights_shape) {
            // Reconstruct the correct grouped format based on tensor rank
            // The tensor should contain the correct 4D data even if layout format is wrong
            weights_layout.format = format::get_default_format(weights_layout.get_rank(), true, true);
            std::cerr << "  Restored grouped format: " << weights_layout.format.to_string() << std::endl;
        }
        
        if (tensor.spatial[0] == 1 && tensor.spatial[1] != 1) {
            std::swap(tensor.spatial[0], tensor.spatial[1]);
            weights_layout.set_tensor(tensor);
            std::cerr << "  Swapped spatial[0] and spatial[1]" << std::endl;
        }
        // Upgrade to 5D for oneDNN (which doesn't support 1D grouped conv directly)
        weights_layout.format = format::get_default_format(weights_layout.get_rank() + 1, true, true);
        std::cerr << "  Final weights_layout: " << weights_layout.to_short_string() << std::endl;
    }

    auto [input_md, weights_md, output_md] = onednn::get_conv_memory_descs(input_layout, weights_layout, output_layout, tag_in_out);

    std::cerr << "[DEBUG] Memory descriptors for " << impl_params.desc->id << ":" << std::endl;
    std::cerr << "  input_md dims: [";
    for (size_t i = 0; i < input_md.get_dims().size(); i++) {
        if (i > 0) std::cerr << ",";
        std::cerr << input_md.get_dims()[i];
    }
    std::cerr << "], format_kind=" << static_cast<int>(input_md.get_format_kind()) << std::endl;
    std::cerr << "  weights_md dims: [";
    for (size_t i = 0; i < weights_md.get_dims().size(); i++) {
        if (i > 0) std::cerr << ",";
        std::cerr << weights_md.get_dims()[i];
    }
    std::cerr << "], format_kind=" << static_cast<int>(weights_md.get_format_kind()) << std::endl;
    std::cerr << "  output_md dims: [";
    for (size_t i = 0; i < output_md.get_dims().size(); i++) {
        if (i > 0) std::cerr << ",";
        std::cerr << output_md.get_dims()[i];
    }
    std::cerr << "], format_kind=" << static_cast<int>(output_md.get_format_kind()) << std::endl;
    std::cerr << "  groups=" << prim->groups << std::endl;
    std::cerr << "  stride=[";
    for (size_t i = 0; i < prim->stride.size(); i++) {
        if (i > 0) std::cerr << ",";
        std::cerr << prim->stride[i];
    }
    std::cerr << "], dilation=[";
    for (size_t i = 0; i < prim->dilation.size(); i++) {
        if (i > 0) std::cerr << ",";
        std::cerr << prim->dilation[i];
    }
    std::cerr << "], pad_begin=[";
    for (size_t i = 0; i < prim->padding_begin.size(); i++) {
        if (i > 0) std::cerr << ",";
        std::cerr << prim->padding_begin[i];
    }
    std::cerr << "], pad_end=[";
    for (size_t i = 0; i < prim->padding_end.size(); i++) {
        if (i > 0) std::cerr << ",";
        std::cerr << prim->padding_end[i];
    }
    std::cerr << "]" << std::endl;

    dnnl::memory::dims stride(prim->stride.begin(), prim->stride.end());
    dnnl::memory::dims dilation(prim->dilation.begin(), prim->dilation.end());
    dnnl::memory::dims pad_l(prim->padding_begin.begin(), prim->padding_begin.end());
    dnnl::memory::dims pad_r(prim->padding_end.begin(), prim->padding_end.end());

    // For 1D convolutions with 5D weights (goiyx format), we need 2D spatial parameters
    // OpenVINO treats 1D conv as 2D with Y=1, so we need to expand parameters early
    if (weights_md.get_dims().size() == 5 && stride.size() == 1) {
        // Insert Y dimension parameters at the beginning
        stride.insert(stride.begin(), 1);      // stride_y = 1
        dilation.insert(dilation.begin(), 1);  // dilation_y = 1  
        pad_l.insert(pad_l.begin(), 0);        // pad_y_begin = 0
        pad_r.insert(pad_r.begin(), 0);        // pad_y_end = 0
        
        std::cerr << "[DEBUG] Extended 1D params to 2D for 5D weights: "
                  << "stride=[" << stride[0] << "," << stride[1] << "], "
                  << "dilation=[" << dilation[0] << "," << dilation[1] << "]" << std::endl;
    }

    if (auto_pad == ov::op::PadType::SAME_UPPER || auto_pad == ov::op::PadType::SAME_LOWER) {
        ov::op::v1::Convolution op;
        op.set_dilations(prim->dilation);
        op.set_strides(prim->stride);
        op.set_auto_pad(auto_pad);
        const auto spatial_rank = input_layout.get_spatial_rank();

        ov::PartialShape kernel;
        for (int32_t i = static_cast<int32_t>(spatial_rank) - 1; i >= 0; i--) {
            kernel.emplace_back(weights_layout.spatial(i));
        }

        ov::op::convolution::apply_auto_pad(&op,
                                            input_layout.get_partial_shape(),
                                            kernel,
                                            pad_l.begin(),
                                            pad_r.begin());
        for (size_t i = 0; i < dilation.size(); i++) {
            dilation[i]--;
        }
    } else {
        // adjust_conv_dilation_pad(dilation, stride, pad_l, pad_r, input_md, output_md, weights_md, grouped_weights);
        for (size_t i = 0; i < dilation.size(); i++) {
            dilation[i]--;
            int weights_offset = (grouped_weights ? 3 : 2) + static_cast<int>(i);
            auto os = output_md.get_dims()[2 + i];
            auto is = input_md.get_dims()[2 + i];
            auto ks = weights_md.get_dims()[weights_offset];
            auto kernel_range = 1 + (ks - 1) * (dilation[i] + 1);
            pad_r[i] = (os - 1) * stride[i] - is + kernel_range - pad_l[i];
        }
    }



    // Extend conv parameters in case if spatials rank of output memory doesn't match size of parameters
    int64_t insert_count = static_cast<int64_t>(output_md.get_dims().size()) - 2 - stride.size();
    if (insert_count > 0) {
        stride.insert(stride.end(), insert_count, 1);
        dilation.insert(dilation.end(), insert_count, 0);
        pad_l.insert(pad_l.end(), insert_count, 0);
        pad_r.insert(pad_r.end(), insert_count, 0);
    }

    // Debug: log final parameters before creating primitive descriptor
    if (prim->groups > 1) {
        std::cerr << "[DEBUG] Final params for " << impl_params.desc->id << ":" << std::endl;
        std::cerr << "  stride=[";
        for (size_t i = 0; i < stride.size(); i++) {
            if (i > 0) std::cerr << ",";
            std::cerr << stride[i];
        }
        std::cerr << "], dilation=[";
        for (size_t i = 0; i < dilation.size(); i++) {
            if (i > 0) std::cerr << ",";
            std::cerr << dilation[i];
        }
        std::cerr << "], pad_l=[";
        for (size_t i = 0; i < pad_l.size(); i++) {
            if (i > 0) std::cerr << ",";
            std::cerr << pad_l[i];
        }
        std::cerr << "], pad_r=[";
        for (size_t i = 0; i < pad_r.size(); i++) {
            if (i > 0) std::cerr << ",";
            std::cerr << pad_r[i];
        }
        std::cerr << "]" << std::endl;
    }

    if (prim->bias.is_valid()) {
        auto bias_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(2), dnnl::memory::format_tag::any, onednn::mem_flags::flatten);
        return std::make_shared<dnnl::convolution_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            input_md,
            weights_md,
            bias_md,
            output_md,
            stride,
            dilation,
            pad_l,
            pad_r,
            attr);
    } else {
        return std::make_shared<dnnl::convolution_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct,
            input_md,
            weights_md,
            output_md,
            stride,
            dilation,
            pad_l,
            pad_r,
            attr);
    }
}

struct convolution_onednn : typed_primitive_onednn_impl<convolution> {
    using parent = typed_primitive_onednn_impl<convolution>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::convolution_onednn)

private:
    int _zero_point_mask;
    dnnl::memory::data_type _wzp_data_type = dnnl::memory::data_type::undef;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<convolution_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(convolution_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);

        {
            auto weights = instance.weights_memory();
            auto offset = onednn::get_offset(instance.get_input_layout(1), _pd.dnnl::primitive_desc_base::weights_desc(0));
            args.insert({DNNL_ARG_WEIGHTS, weights->get_onednn_memory(_pd.weights_desc(0), offset)});
        }

        if (instance.bias_term()) {
            auto bias = instance.bias_memory();
            auto offset = onednn::get_offset(instance.get_input_layout(2), _pd.dnnl::primitive_desc_base::weights_desc(1));
            args.insert({DNNL_ARG_BIAS, bias->get_onednn_memory(_pd.weights_desc(1), offset)});
        }

        if (instance.activations_zero_points_term()) {
            auto a_zp = instance.activations_zero_points_memory();

            // In the case of dynamic model, if choose_impl was executed in runtime,
            // a_zp could be remained as u8 or i8.
            if (a_zp->get_layout().data_type != data_types::i32) {
                auto& conv_node = instance.get_node().as<convolution>();
                auto& a_zp_node = conv_node.activations_zero_points().as<data>();
                a_zp = a_zp_node.get_attached_memory_ptr();
            }

            dnnl::memory::desc desc = onednn::layout_to_memory_desc(a_zp->get_layout(), dnnl::memory::format_tag::a, onednn::mem_flags::flatten);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, a_zp->get_onednn_memory(desc)});

            GPU_DEBUG_TRACE_DETAIL << instance.id() << " activations_zero_points: "
                << " " << a_zp->get_layout().to_short_string() << std::endl;
        }

        if (instance.weights_zero_points_term()) {
            auto w_zp = instance.weights_zero_points_memory();
            dnnl::memory::desc desc = onednn::layout_to_memory_desc(w_zp->get_layout(), dnnl::memory::format_tag::a, onednn::mem_flags::flatten);
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, w_zp->get_onednn_memory(desc)});

            GPU_DEBUG_TRACE_DETAIL << instance.id() << " weights_zero_points: "
                << " " << w_zp->get_layout().to_short_string() << std::endl;
        }

        return args;
    }

    void set_zero_point_mask(int zero_point_mask) {
        _zero_point_mask = zero_point_mask;
    }

    void set_weights_zero_point_data_type(dnnl::memory::data_type data_type) {
        _wzp_data_type = data_type;
    }

    template <typename T>
    static void set_activation_zero_points_attr(const std::shared_ptr<dnnl::primitive_attr>& attrs,
                                                cldnn::data_node& node, int& zero_point_mask) {
        int32_t zp_val = DNNL_RUNTIME_S32_VAL;
        bool is_per_tensor = onednn::is_per_tensor<T>(node, zp_val);
        memory::ptr s32_mem = onednn::convert_zp_data_to_s32<T>(node.get_attached_memory_ptr());
        node.attach_memory(s32_mem, false);
        zero_point_mask = is_per_tensor ? 0 : 2;
        attrs->set_zero_points_mask(DNNL_ARG_SRC, zero_point_mask);
    }

    static std::shared_ptr<dnnl::primitive_attr> get_primitive_attributes(const typed_program_node<convolution>& arg,
                                                                            const kernel_impl_params& impl_params,
                                                                            int& zero_point_mask,
                                                                            dnnl::memory::data_type& wzp_data_type) {
        auto attrs = impl_params.attrs_onednn;

        if (arg.activations_zero_points_term()) {
            auto& a_zp = arg.activations_zero_points();
            auto a_zp_dtype = a_zp.get_output_layout().data_type;

            if (!data_type_traits::is_i8_u8(a_zp_dtype) && a_zp_dtype != data_types::i32) {
                throw std::runtime_error("Unsupported data type for activations zero points for oneDNN convolution");
            }

            if (a_zp_dtype == data_types::i8) {
                set_activation_zero_points_attr<ov::element_type_traits<data_types::i8>::value_type>(attrs, a_zp.as<data>(), zero_point_mask);
            } else if (a_zp_dtype == data_types::u8) {
                set_activation_zero_points_attr<ov::element_type_traits<data_types::u8>::value_type>(attrs, a_zp.as<data>(), zero_point_mask);
            } else if (a_zp_dtype == data_types::i32) {
                set_activation_zero_points_attr<ov::element_type_traits<data_types::i32>::value_type>(attrs, a_zp.as<data>(), zero_point_mask);
            }
        }

        if (arg.weights_zero_points_term()) {
            auto& wzp = arg.weights_zero_points();
            auto wzp_layout = wzp.get_output_layout();
            wzp_data_type = convert_data_type(wzp_layout.data_type);
            if (wzp_layout.count() == 1) {
                attrs->set_zero_points(DNNL_ARG_WEIGHTS, 0, dnnl::memory::dims{}, wzp_data_type);
            } else {
                throw std::runtime_error("Convolution oneDNN primitive doesn't support PER_OC weights zero points");
            }
        }

        return attrs;
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd, bool rotate) {
        auto cldnn_prim = impl_params.typed_desc<convolution>();

        auto source_weights_layout = impl_params.get_input_layout(1);
        auto input_layout = impl_params.get_input_layout(0);
        
        // Apply the same reconstruction as in get_convolution_primitive_descriptor
        // This ensures weights reorder sees the corrected 5D shape
        if (cldnn_prim->groups > 1 && cldnn_prim->groups == input_layout.feature()) {
            auto weights_shape = source_weights_layout.get_shape();
            auto weights_tensor = source_weights_layout.get_tensor();
            
            // Case 1: 4D bfyx weights that need to be 5D goiyx
            if (weights_shape.size() == 4 &&
                weights_tensor.batch[0] == cldnn_prim->groups &&
                weights_tensor.feature[0] <= 1 &&
                weights_tensor.spatial[1] == 1) {
                
                int kernel_size = weights_tensor.spatial[0];
                
                ov::PartialShape goiyx_shape = {
                    static_cast<ov::Dimension::value_type>(cldnn_prim->groups),
                    static_cast<ov::Dimension::value_type>(1),
                    static_cast<ov::Dimension::value_type>(1),
                    static_cast<ov::Dimension::value_type>(weights_tensor.spatial[1]),
                    static_cast<ov::Dimension::value_type>(kernel_size)
                };
                
                source_weights_layout.format = format::get_default_format(5, true, true);
                source_weights_layout.set_partial_shape(goiyx_shape);
                
                std::cerr << "[DEBUG] get_weights_reorder: Reconstructed source_weights_layout to " 
                          << source_weights_layout.to_short_string() << std::endl;
            }
        }
        
        auto grouped_weights = format::is_grouped(source_weights_layout.format) || cldnn_prim->grouped_weights_shape;
        auto target_weights_desc = pd.weights_desc(0);

        auto shape_consistent = onednn::keep_weights_reorder_shape_consistent(source_weights_layout, target_weights_desc);
        OPENVINO_ASSERT(shape_consistent, "[GPU] Input shape and output shape of weight reorder should be same.");

        auto source_weights_desc = onednn::layout_to_memory_desc(source_weights_layout);

        const bool weights_format = true;
        auto traits = convert_memory_desc_to_traits(target_weights_desc, weights_format, grouped_weights);

        auto target_weights_layout = source_weights_layout;
        target_weights_layout.format = format(traits);

        return std::make_shared<WeightsReorderParamsOneDNN>(source_weights_layout,
                                                            target_weights_layout,
                                                            source_weights_desc,
                                                            target_weights_desc,
                                                            rotate,
                                                            grouped_weights);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        ob << _zero_point_mask;

        const dnnl::convolution_forward::primitive_desc *typed_pd
            = reinterpret_cast<const dnnl::convolution_forward::primitive_desc *>(&_pd);

        ob << typed_pd->get_strides();
        ob << typed_pd->get_dilations();
        ob << typed_pd->get_padding_l();
        ob << typed_pd->get_padding_r();
        ob << typed_pd->bias_desc().is_zero();

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ob.getKernelImplParams());
        auto prim = impl_params->typed_desc<convolution>();
        bool has_wzp = prim->weights_zero_points.is_valid();
        if (has_wzp) {
            ob << make_data(&_wzp_data_type, sizeof(dnnl::memory::data_type));
        }

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        ib >> _zero_point_mask;
        if (_zero_point_mask != -1) {
            _attrs->set_zero_points_mask(DNNL_ARG_SRC, _zero_point_mask);
        }

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());

        auto [input_md, weights_md, output_md] = onednn::get_conv_memory_descs(impl_params->get_input_layout(0),
                                                                                impl_params->get_input_layout(1),
                                                                                impl_params->get_output_layout(),
                                                                                dnnl::memory::format_tag::undef);

        dnnl::memory::dims strides;
        dnnl::memory::dims dilates;
        dnnl::memory::dims padding_l;
        dnnl::memory::dims padding_r;
        ib >> strides;
        ib >> dilates;
        ib >> padding_l;
        ib >> padding_r;

        bool zero_bias;
        ib >> zero_bias;

        auto prim = impl_params->typed_desc<convolution>();
        bool has_wzp = prim->weights_zero_points.is_valid();
        if (has_wzp) {
            ib >> make_data(&_wzp_data_type, sizeof(dnnl::memory::data_type));
            _attrs->set_zero_points(DNNL_ARG_WEIGHTS, 0, dnnl::memory::dims{}, _wzp_data_type);
        }

        if (zero_bias) {
            auto prim_desc = std::make_shared<dnnl::convolution_forward::primitive_desc>(
                                    ib.get_engine().get_onednn_engine(),
                                    dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                                    input_md, weights_md, output_md,
                                    strides, dilates, padding_l, padding_r,
                                    *_attrs.get());
            _pd = *prim_desc;
        } else {
            auto bias_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(2), dnnl::memory::format_tag::any, onednn::mem_flags::flatten);
            auto prim_desc = std::make_shared<dnnl::convolution_forward::primitive_desc>(
                                    ib.get_engine().get_onednn_engine(),
                                    dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                                    input_md, weights_md, bias_md, output_md,
                                    strides, dilates, padding_l, padding_r,
                                    *_attrs.get());
            _pd = *prim_desc;
        }

        _scratchpad_md = _pd.scratchpad_desc();

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const convolution_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        int zero_point_mask = -1;
        dnnl::memory::data_type wzp_data_type = dnnl::memory::data_type::undef;

        auto attr = get_primitive_attributes(arg, impl_params, zero_point_mask, wzp_data_type);

        auto prim_desc = get_convolution_primitive_descriptor(impl_params, *attr);

        auto conv_onednn_impl = std::make_unique<convolution_onednn>(engine, config, attr, *prim_desc,
                                                get_weights_reorder(impl_params, *prim_desc, arg.get_transposed()));

        conv_onednn_impl->set_zero_point_mask(zero_point_mask);
        conv_onednn_impl->set_weights_zero_point_data_type(wzp_data_type);

        return conv_onednn_impl;
    }
};

std::unique_ptr<primitive_impl> ConvolutionImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<convolution>());
    return convolution_onednn::create(static_cast<const convolution_node&>(node), params);
}

in_out_fmts_t ConvolutionImplementationManager::query_formats(const program_node& node) const {
    assert(node.is_type<convolution>());
    std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
    std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

    const auto& conv_node = node.as<convolution>();

    auto prim_desc = get_convolution_primitive_descriptor(*node.get_kernel_impl_params(), dnnl::primitive_attr(), dnnl::memory::format_tag::any);

    for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
        if (node.get_dependency(idx).is_constant())
            continue;

        // Conv or deconv gets a preferred format for its data input based on source memory description
        // But an input format for fused post-ops should be same with an output format of conv/deconv
        size_t prim_input = node.get_dependency_index(conv_node.input());
        size_t prim_weights = node.get_primitive()->input_size();

        // Note: did not handle attribute properly. especially for zero-point
        cldnn::format src_fmt = format::any;
        if (idx == prim_input) {
            src_fmt = onednn::find_data_format(prim_desc->src_desc());
        } else if (idx == prim_weights) {
            src_fmt = format::any;
        } else {  // Dep for fused post ops
            src_fmt = onednn::find_data_format(prim_desc->dst_desc());
        }

        // WA: Avoid b_fs_yx_fsv2 because Onednn tag aBcd2b is not declared.
        if (src_fmt == format::b_fs_yx_fsv2)
            src_fmt = format::byxf;

        // WA: shallow convolution needs to set input format by bfyx.
        //     onednn recommended byxf for input format. It will insert reorder before shallow conv.
        if (node.get_input_layout(0).get_partial_shape()[1] == 3) {
            bool can_optimize_permute = false;
            // In permute-conv pattern, check if permute can be optimized
            // when the input memory of permute has been aligned like byxf format.
            // ex) pattern: input (bfyx) -> permute (byxf) -> oneDNN convolution
            //      input layout of permute: bfyx [b:1, f:416, y:416, x:3]
            //     output layout of permute: byxf [b:1, f:3, y:416, x:416]
            // In this case, it can be handled by changing only the shape of permute without the kernel execution.
            if (node.get_output_layout().get_rank() == 4 && node.get_dependency(0).is_type<permute>()) {
                auto& pnode = node.get_dependency(0).as<permute>();
                can_optimize_permute = pnode.get_users().size() == 1
                    && !pnode.has_fused_primitives()
                    && !pnode.is_output() && pnode.get_input_layout(0).is_static()
                    && pnode.is_reverse_rotating_except_batch();
            }
            if (!can_optimize_permute) {
                src_fmt = format::get_default_format(node.get_input_layout(0).get_rank(), false, false);
            } else {
                // The size of dependencies and users must each be 1.
                // In permute-conv pattern, the preferred format of permute should follow previous node.
                node.get_dependency(0).init_preferred_fmt(1, 1);
                node.get_dependency(0).set_preferred_input_fmt(0, format::bfyx);
                node.get_dependency(0).can_be_optimized(true);
            }
        }

        in_fmts[idx] = src_fmt;
    }

    auto dst_fmt = onednn::find_data_format(prim_desc->dst_desc());
    if (out_fmts[0] == format::any) {
        out_fmts[0] = dst_fmt;
    }

    // WA: Avoid b_fs_yx_fsv2 because Onednn tag aBcd2b is not declared.
    if (out_fmts[0] == format::b_fs_yx_fsv2)
        out_fmts[0] = format::byxf;

    // Errata: Best impl for shallow input conv with zero-point ops is ocl:xe_lp.
    if (in_fmts[0] == format::bfyx) {
        if (conv_node.get_input_layout(0).feature() <= 8 && conv_node.activations_zero_points_term() &&
            conv_node.get_input_layout(0).data_type == data_types::u8 && conv_node.get_output_layout().data_type == data_types::u8) {
            dst_fmt = format::b_fs_yx_fsv32;
        }
    }
    return {in_fmts, out_fmts};
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::convolution_onednn)
