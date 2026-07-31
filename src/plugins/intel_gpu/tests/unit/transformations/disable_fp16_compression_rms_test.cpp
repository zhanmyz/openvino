// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/rt_info/disable_precision_conversion.hpp>

#include "plugin/transformations/disable_fp16_comp_rms.hpp"
#include "ov_ops/rms.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/add.hpp"

using namespace testing;
using namespace ov::intel_gpu;

static const std::string name_rms_1 = "rms_1";
static const std::string name_rms_2 = "rms_2";

// This model creates the exact pattern that DisableFP16CompForGemma3RMSPattern is looking for.
// (Add, RMS) -> Add -> RMS
static std::shared_ptr<ov::Model> create_model_to_match(bool use_convert = false) {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});
    auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});
    auto input3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});

    // Pattern part 1: add_m
    auto add_m = std::make_shared<ov::op::v1::Add>(input1, input2);

    // Pattern part 2: rms_post_m
    std::shared_ptr<ov::Node> rms_const_or_convert_1;
    if (use_convert) {
        auto const_node_1 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{128}, {1.0f});
        rms_const_or_convert_1 = std::make_shared<ov::op::v0::Convert>(const_node_1, ov::element::f32);
    } else {
        rms_const_or_convert_1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    }
    auto rms_post_m = std::make_shared<ov::op::internal::RMS>(input3, rms_const_or_convert_1, 1e-5);
    rms_post_m->set_friendly_name(name_rms_1);

    // Pattern part 3: add_1_m
    auto add_1_m = std::make_shared<ov::op::v1::Add>(add_m, rms_post_m);

    // Pattern part 4: rms_m
    std::shared_ptr<ov::Node> rms_const_or_convert_2;
    if (use_convert) {
        auto const_node_2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{128}, {1.0f});
        rms_const_or_convert_2 = std::make_shared<ov::op::v0::Convert>(const_node_2, ov::element::f32);
    } else {
        rms_const_or_convert_2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    }
    auto rms_m = std::make_shared<ov::op::internal::RMS>(add_1_m, rms_const_or_convert_2, 1e-5);
    rms_m->set_friendly_name(name_rms_2);

    return std::make_shared<ov::Model>(ov::OutputVector{rms_m}, ov::ParameterVector{input1, input2, input3});
}

// This model has a similar structure but doesn't match the specific pattern.
static std::shared_ptr<ov::Model> create_model_not_to_match() {
    auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 32, 128});

    auto rms_const_1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    auto rms_1 = std::make_shared<ov::op::internal::RMS>(input1, rms_const_1, 1e-5);
    rms_1->set_friendly_name(name_rms_1);

    auto some_other_op = std::make_shared<ov::op::v1::Add>(rms_1, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}));

    auto rms_const_2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128}, {1.0f});
    auto rms_2 = std::make_shared<ov::op::internal::RMS>(some_other_op, rms_const_2, 1e-5);
    rms_2->set_friendly_name(name_rms_2);

    return std::make_shared<ov::Model>(ov::OutputVector{rms_2}, ov::ParameterVector{input1});
}

static void run_test(std::shared_ptr<ov::Model> model,
                     const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForGemma3RMSPattern>();

    precisions_map fp_convert_precision_map = {
        {ov::element::f32, ov::element::f16}
    };
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map);

    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it != expected_fp16_disabled_status.end()) {
            bool expected_status = it->second;
            if (expected_status) {
                ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f16))
                    << "FP16 compression is not disabled for node: " << op->get_friendly_name();
            } else {
                ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f16))
                    << "FP16 compression is unexpectedly disabled for node: " << op->get_friendly_name();
            }
        }
    }
}

TEST(TransformationTests, DisableFP16CompForRMS_Positive) {
    auto model = create_model_to_match();
    // In the matching pattern, both rms_1 (rms_post_m) and rms_2 (rms_m) should have FP16 compression disabled.
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_1, true},
        {name_rms_2, true}
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompForRMS_PositiveConvert) {
    auto model = create_model_to_match(true);
    // In the matching pattern, both rms_1 (rms_post_m) and rms_2 (rms_m) should have FP16 compression disabled.
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_1, true},
        {name_rms_2, true}
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompForRMS_Negative) {
    auto model = create_model_not_to_match();
    // In the non-matching model, no RMS node should have FP16 compression disabled by the pass.
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_1, false},
        {name_rms_2, false}
    };
    run_test(model, expected_status);
}

// ============================================================================
// Tests for DisableFP16CompForRMSWithFullPrecisionWeights
// ============================================================================

static const std::string name_rms_fp_weights = "rms_full_precision_weights";

// Creates a model with large-magnitude (f16) weights feeding into MatMul → Add → RMS.
// This simulates T5/ParlerTTS where max|W| exceeds FP16_MAX / K.
// For K=768: threshold = 65504 / 768 ≈ 85.3. Using val=130.0 >> 85.3.
static std::shared_ptr<ov::Model> create_model_with_large_fp_weights() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 39, 768});
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 39, 768});

    // Large weight (val=130.0 > threshold≈85.3): Constant(f16) → Convert(f32) → MatMul
    auto weight_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{768, 768}, {130.0f});
    auto weight_convert = std::make_shared<ov::op::v0::Convert>(weight_const, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight_convert, false, true);

    // Residual Add → RMS
    auto add = std::make_shared<ov::op::v1::Add>(matmul, residual);
    auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{768}, {1.0f});
    auto rms = std::make_shared<ov::op::internal::RMS>(add, gamma, 1e-6);
    rms->set_friendly_name(name_rms_fp_weights);

    return std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{input, residual});
}

// Creates a model with quantized (u8) weights feeding into MatMul → Add → RMS.
// This simulates quantized LLMs (gemma-3, qwen3) where weights are 4-bit/8-bit.
static std::shared_ptr<ov::Model> create_model_with_quantized_weights() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 256});
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 256});

    // Quantized weight: Constant(u8) → Convert(f16) → Subtract(zp) → Multiply(scale) → Convert(f32) → MatMul
    auto weight_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{256, 256}, {128});
    auto weight_convert_f16 = std::make_shared<ov::op::v0::Convert>(weight_const, ov::element::f16);
    auto zero_point = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {128.0f});
    auto subtract = std::make_shared<ov::op::v1::Subtract>(weight_convert_f16, zero_point);
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {0.01f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    auto weight_convert_f32 = std::make_shared<ov::op::v0::Convert>(multiply, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight_convert_f32, false, true);

    // Residual Add → RMS
    auto add = std::make_shared<ov::op::v1::Add>(matmul, residual);
    auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256}, {1.0f});
    auto rms = std::make_shared<ov::op::internal::RMS>(add, gamma, 1e-6);
    rms->set_friendly_name(name_rms_fp_weights);

    return std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{input, residual});
}

// Creates a model with MatMul → RMS (no Add in between).
// Large weight magnitude — should still be matched.
// For K=768: threshold = 65504 / 768^1.5 ≈ 3.08. Using val=100.0 >> 3.08.
static std::shared_ptr<ov::Model> create_model_matmul_direct_to_rms() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 39, 768});

    // Full-precision weight directly to MatMul → RMS (val=100.0 > threshold)
    auto weight_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{768, 768}, {100.0f});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight_const, false, true);

    auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{768}, {1.0f});
    auto rms = std::make_shared<ov::op::internal::RMS>(matmul, gamma, 1e-6);
    rms->set_friendly_name(name_rms_fp_weights);

    return std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{input});
}

// Creates a model with small-magnitude f16 weights → should NOT be marked.
// This simulates gemma-3 FP16 where weights are bounded (trained with BF16 awareness).
// For K=256: threshold = 65504 / 256^1.5 ≈ 15.98. Using val=1.5 << 15.98.
static std::shared_ptr<ov::Model> create_model_with_small_fp_weights() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 256});
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 256});

    // Small weight (val=1.5 < threshold≈15.98): Constant(f16) → Convert(f32) → MatMul
    auto weight_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{256, 256}, {1.5f});
    auto weight_convert = std::make_shared<ov::op::v0::Convert>(weight_const, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight_convert, false, true);

    // Residual Add → RMS
    auto add = std::make_shared<ov::op::v1::Add>(matmul, residual);
    auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256}, {1.0f});
    auto rms = std::make_shared<ov::op::internal::RMS>(add, gamma, 1e-6);
    rms->set_friendly_name(name_rms_fp_weights);

    return std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{input, residual});
}

static void run_test_full_precision_weights(std::shared_ptr<ov::Model> model,
                                            const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompForRMSWithFullPrecisionWeights>();

    precisions_map fp_convert_precision_map = {
        {ov::element::f32, ov::element::f16}
    };
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map);

    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it != expected_fp16_disabled_status.end()) {
            bool expected_status = it->second;
            if (expected_status) {
                ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f16))
                    << "FP16 compression should be disabled for node: " << op->get_friendly_name()
                    << " (full-precision weights detected upstream)";
            } else {
                ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f16))
                    << "FP16 compression should NOT be disabled for node: " << op->get_friendly_name()
                    << " (quantized weights detected upstream — no overflow risk)";
            }
        }
    }
}

// Positive: RMS fed by MatMul with large (f16, val=50) weights through Add → should be marked
TEST(TransformationTests, DisableFP16CompForRMSFullPrecWeights_Positive_AddPath) {
    auto model = create_model_with_large_fp_weights();
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_fp_weights, true}
    };
    run_test_full_precision_weights(model, expected_status);
}

// Positive: RMS fed directly by MatMul with large (f32, val=100) weights → should be marked
TEST(TransformationTests, DisableFP16CompForRMSFullPrecWeights_Positive_DirectMatMul) {
    auto model = create_model_matmul_direct_to_rms();
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_fp_weights, true}
    };
    run_test_full_precision_weights(model, expected_status);
}

// Negative: RMS fed by MatMul with quantized (u8) weights → should NOT be marked
TEST(TransformationTests, DisableFP16CompForRMSFullPrecWeights_Negative_QuantizedWeights) {
    auto model = create_model_with_quantized_weights();
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_fp_weights, false}
    };
    run_test_full_precision_weights(model, expected_status);
}

// Negative: RMS fed by MatMul with small f16 weights (val=1.5 < threshold 5.0) → should NOT be marked
// This simulates gemma-3 FP16 where weights are bounded (trained with BF16 awareness).
TEST(TransformationTests, DisableFP16CompForRMSFullPrecWeights_Negative_SmallFPWeights) {
    auto model = create_model_with_small_fp_weights();
    std::unordered_map<std::string, bool> expected_status = {
        {name_rms_fp_weights, false}
    };
    run_test_full_precision_weights(model, expected_status);
}
