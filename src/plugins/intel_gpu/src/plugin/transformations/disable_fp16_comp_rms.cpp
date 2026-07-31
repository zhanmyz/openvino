// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_rms.hpp"

#include "ov_ops/rms.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

#include <memory>

namespace ov::intel_gpu {

DisableFP16CompForGemma3RMSPattern::DisableFP16CompForGemma3RMSPattern() {
    using namespace ov::pass::pattern;

    auto const_or_convert = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{
        wrap_type<ov::op::v0::Constant>(),
        wrap_type<ov::op::v0::Convert>({wrap_type<ov::op::v0::Constant>()})
    });
 
    auto add_m = wrap_type<ov::op::v1::Add>({any_input(), any_input()}, type_matches(element::f32));
    auto rms_post_m = wrap_type<ov::op::internal::RMS>({any_input(), const_or_convert}, type_matches(element::f32));
    auto add_1_m = wrap_type<ov::op::v1::Add>({add_m, rms_post_m}, type_matches(element::f32));
    auto rms_m = wrap_type<ov::op::internal::RMS>({add_1_m, const_or_convert}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(pattern_map.at(rms_m).get_node_shared_ptr());
        if (!rms || transformation_callback(rms)) {
            return false;
        }

        auto rms_post = pattern_map.at(rms_post_m).get_node_shared_ptr();
        if (rms_post) {
            ov::disable_conversion(rms_post, element::f16);
        }

        ov::disable_conversion(rms, element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rms_m, "DisableFP16CompForGemma3RMSPattern");
    this->register_matcher(m, callback);
}

namespace {
// Determines whether the upstream MatMul's weight matrix poses FP16 overflow risk.
//
// Mathematical derivation (Cauchy-Schwarz inequality on dot product):
// For Y = X @ W^T (MatMul), each output element Y_j = dot(X, W_j).
// By Cauchy-Schwarz: |Y_j| ≤ ||X||_2 × ||W_j||_2
//
// After RMS normalization (which produces unit-RMS output scaled by gamma ≈ 1):
//   ||X||_2 = sqrt(sum(x_i²)) = sqrt(K × mean(x²)) = sqrt(K × 1) = sqrt(K)
//
// For weight row W_j with K elements:
//   ||W_j||_2 ≤ sqrt(K) × max|W_element|
//
// Combining: |Y_j| ≤ sqrt(K) × sqrt(K) × max|W| = K × max|W|
//
// Overflow occurs when: K × max|W| > FP16_MAX (65504)
// Equivalently: max|W| > FP16_MAX / K
//
// Constants:
//   FP16_MAX = 65504 (IEEE 754 half-precision maximum finite value)
//   K = reduction dimension of the weight constant (from shape)
constexpr float FP16_MAX_VALUE = 65504.0f;

// Trace through the weight input chain to find the root Constant.
// Returns true if the weights pose overflow risk based on the derived criterion.
bool has_overflow_risk_weights(const std::shared_ptr<ov::Node>& matmul_node) {
    // Weight is typically input 1 for MatMul
    auto weight_output = matmul_node->input_value(1);
    auto node = weight_output.get_node_shared_ptr();

    // Trace back through Convert, Reshape, Multiply, Subtract, etc. to find root Constant
    for (int depth = 0; depth < 10 && node; ++depth) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            const auto& et = constant->get_element_type();
            if (!et.is_real()) {
                return false;  // Quantized (integer) weights — bounded by quantization range
            }

            // Determine reduction dimension K from weight shape.
            // For MatMul Y = X @ W^T: weight shape is [N, K] (transposed=true) or [K, N] (transposed=false).
            // The reduction dim is the dimension that gets summed over.
            const auto& shape = constant->get_shape();
            if (shape.size() < 2) {
                return false;
            }
            bool transpose_b = false;
            if (auto matmul_op = ov::as_type_ptr<ov::op::v0::MatMul>(matmul_node)) {
                transpose_b = matmul_op->get_transpose_b();
            }
            // If transpose_b=true: weight is [N, K], reduction dim = shape[last]
            // If transpose_b=false: weight is [K, N], reduction dim = shape[second_to_last]
            const size_t reduction_dim = transpose_b ? shape.back() : shape[shape.size() - 2];
            if (reduction_dim == 0) {
                return false;
            }

            // Compute max|W| over all elements
            float max_abs = 0.0f;
            const auto num_elements = ov::shape_size(shape);
            if (et == ov::element::f16) {
                const auto* data = constant->get_data_ptr<ov::float16>();
                for (size_t i = 0; i < num_elements; ++i) {
                    float val = std::abs(static_cast<float>(data[i]));
                    if (val > max_abs) max_abs = val;
                }
            } else if (et == ov::element::f32) {
                const auto* data = constant->get_data_ptr<float>();
                for (size_t i = 0; i < num_elements; ++i) {
                    float val = std::abs(data[i]);
                    if (val > max_abs) max_abs = val;
                }
            } else {
                return false;
            }

            // Apply derived overflow criterion: max|W| > FP16_MAX / K
            const float k_f = static_cast<float>(reduction_dim);
            const float overflow_bound = FP16_MAX_VALUE / k_f;
            return max_abs > overflow_bound;
        }
        if (node->inputs().empty()) {
            break;
        }
        // Follow the first input (main data path of the weight decompression chain)
        node = node->input_value(0).get_node_shared_ptr();
    }
    return false;  // couldn't determine, don't mark
}
}  // namespace

DisableFP16CompForRMSWithFullPrecisionWeights::DisableFP16CompForRMSWithFullPrecisionWeights() {
    using namespace ov::pass::pattern;

    auto const_or_convert = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{
        wrap_type<ov::op::v0::Constant>(),
        wrap_type<ov::op::v0::Convert>({wrap_type<ov::op::v0::Constant>()})
    });

    auto rms_m = wrap_type<ov::op::internal::RMS>({any_input(), const_or_convert},
        [](const ov::Output<ov::Node>& output) -> bool {
            return output.get_element_type() == ov::element::f32 &&
                   !ov::is_conversion_disabled(output.get_node_shared_ptr(), ov::element::f16);
        });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto rms = m.get_match_root();
        if (!rms || transformation_callback(rms)) {
            return false;
        }

        // Trace upstream from RMS data input through Add (residual) to find MatMul
        auto data_input = rms->input_value(0).get_node_shared_ptr();

        // The typical pattern is: MatMul → Add (residual) → RMS
        // or directly: MatMul → RMS
        std::shared_ptr<ov::Node> upstream_matmul = nullptr;

        // Check if direct input is MatMul
        if (ov::as_type_ptr<ov::op::v0::MatMul>(data_input)) {
            upstream_matmul = data_input;
        }
        // Check if input is Add (residual) with a MatMul feeding into it
        else if (auto add = ov::as_type_ptr<ov::op::v1::Add>(data_input)) {
            for (size_t i = 0; i < add->get_input_size(); ++i) {
                auto add_input = add->input_value(i).get_node_shared_ptr();
                if (ov::as_type_ptr<ov::op::v0::MatMul>(add_input)) {
                    upstream_matmul = add_input;
                    break;
                }
            }
        }

        if (!upstream_matmul) {
            return false;
        }

        // Mark RMS only if the upstream MatMul has weights with overflow risk
        if (!has_overflow_risk_weights(upstream_matmul)) {
            return false;
        }

        ov::disable_conversion(rms, ov::element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rms_m, "DisableFP16CompForRMSWithFullPrecisionWeights");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
