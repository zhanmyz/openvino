// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/concat.hpp"

#include "intel_gpu/primitives/concatenation.hpp"

namespace ov::intel_gpu {

static void CreateConcatOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Concat>& op) {
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    // std::cout << "***layerName: " << layerName << std::endl;
    // if (layerName == "concat:__module.update_block.motion_encoder/aten::cat/Concat") {
    //     std::cout << "Inputs names for " << layerName << ":" << std::endl;
    //     for (size_t i = 0; i < inputs.size(); ++i) {
    //         const auto& input = inputs[i];
    //         // 获取输入节点的详细信息
    //         auto input_node = op->get_input_node_ptr(i);
    //         auto input_output = op->input_value(i);

    //         std::cout << "  Input[" << i << "]: pid=" << input.pid
    //                   << ", idx=" << input.idx;

    //         // 尝试获取输入节点的名称
    //         if (input_node) {
    //             std::cout << ", node_name=" << input_node->get_friendly_name()
    //                       << ", node_type=" << input_node->get_type_name();

    //             // 获取输入张量的名称
    //             auto tensor_names = input_output.get_names();
    //             if (!tensor_names.empty()) {
    //                 std::cout << ", tensor_names=[";
    //                 bool first = true;
    //                 for (const auto& name : tensor_names) {
    //                     if (!first) std::cout << ", ";
    //                     std::cout << name;
    //                     first = false;
    //                 }
    //                 std::cout << "]";
    //             }

    //             // 获取形状信息
    //             std::cout << ", shape=" << input_output.get_partial_shape();
    //         }

    //         std::cout << std::endl;
    //     }

    //     // 输出concat操作本身的信息
    //     std::cout << "  Concat output shape: " << op->get_output_partial_shape(0) << std::endl;
    //     std::cout << "  Concat axis: " << op->get_axis() << std::endl;
    // }
    int64_t axis = op->get_axis();
    if (axis < 0)
        axis = axis + static_cast<int64_t>(op->get_input_partial_shape(0).rank().get_length());

    auto concatPrim = cldnn::concatenation(
        layerName,
        inputs,
        axis,
        cldnn::element_type_to_data_type(op->get_output_element_type(0)));

    p.add_primitive(*op, concatPrim);
}

REGISTER_FACTORY_IMPL(v0, Concat);

}  // namespace ov::intel_gpu
