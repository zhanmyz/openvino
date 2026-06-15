// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_for_rms_norm_block.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/core/rt_info.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu {

bool DisableFP16CompForRMSNormBlock::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool is_changed = false;

    for (const auto& node : model->get_ordered_ops()) {
        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(node);
        if (!rms)
            continue;

        // Mark the RMS node to stay in FP32
        ov::disable_conversion(rms, ov::element::f16);
        is_changed = true;

        // Mark all direct MatMul consumers to stay in FP32
        for (const auto& target_input : rms->output(0).get_target_inputs()) {
            auto consumer = target_input.get_node()->shared_from_this();
            if (ov::is_type<ov::op::v0::MatMul>(consumer)) {
                ov::disable_conversion(consumer, ov::element::f16);
            }
        }
    }

    return is_changed;
}

}  // namespace ov::intel_gpu
