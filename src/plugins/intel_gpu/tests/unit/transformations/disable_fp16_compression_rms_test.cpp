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
// Single comprehensive test for CVS-187992: DisableFP16CompForRMSWithFullPrecisionWeights
//
// Covers all pass behaviours using SCOPED_TRACE for per-case attribution:
//   A – Baseline: without the pass, large-weight RMS is NOT marked (A/B anchor)
//   B – Positive: K=768, val=130 > threshold=85.3 → RMS marked
//   C – Positive: direct MatMul→RMS (no Add), K=768, val=100 → RMS marked
//   D – Positive: K=2048, val=50 > threshold=31.98 → RMS marked
//   E – Negative: quantized u8 weights → RMS NOT marked
//   F – Negative: K=256, val=1.5 < threshold=255.9 → NOT marked (gemma-3 FP16)
//   G – Negative: K=2048, val=10.6 < threshold=31.98 → NOT marked (T5 wo/MatMul)
// ============================================================================
TEST(TransformationTests, CVS187992_DisableFP16CompForRMSWithFullPrecisionWeights) {
    const std::string rms_name = "test_rms";
    const precisions_map fp_map = {{ov::element::f32, ov::element::f16}};

    // Helper: build model with f16/f32 weight Constant → [optional Convert] → MatMul → [Add] → RMS
    auto build_fp_model = [&](float weight_val,
                               ov::Shape weight_shape,
                               ov::element::Type weight_et,
                               bool has_add,
                               bool transpose_b = true) -> std::shared_ptr<ov::Model> {
        const ov::Dimension feat_in  = static_cast<ov::Dimension::value_type>(has_add ? weight_shape[transpose_b ? 1 : 0] : weight_shape[transpose_b ? 1 : 0]);
        const ov::Dimension feat_out = static_cast<ov::Dimension::value_type>(transpose_b ? weight_shape[0] : weight_shape[1]);
        auto input    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 39, feat_in});
        auto w_const  = ov::op::v0::Constant::create(weight_et, weight_shape, {weight_val});
        std::shared_ptr<ov::Node> w_node = w_const;
        if (weight_et != ov::element::f32)
            w_node = std::make_shared<ov::op::v0::Convert>(w_const, ov::element::f32);
        auto matmul   = std::make_shared<ov::op::v0::MatMul>(input, w_node, false, transpose_b);
        auto gamma    = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(transpose_b ? weight_shape[0] : weight_shape[1])}, {1.0f});
        std::shared_ptr<ov::Node> rms_in = matmul;
        ov::ParameterVector params{input};
        if (has_add) {
            auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 39, feat_out});
            params.push_back(residual);
            rms_in = std::make_shared<ov::op::v1::Add>(matmul, residual);
        }
        auto rms = std::make_shared<ov::op::internal::RMS>(rms_in, gamma, 1e-6);
        rms->set_friendly_name(rms_name);
        return std::make_shared<ov::Model>(ov::OutputVector{rms}, params);
    };

    // Helper: build model with quantized (u8) weights (dequant chain)
    auto build_quantized_model = [&]() -> std::shared_ptr<ov::Model> {
        auto input    = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 256});
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32, 256});
        auto wc       = ov::op::v0::Constant::create(ov::element::u8,  ov::Shape{256, 256}, {128});
        auto zp       = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1},        {128.f});
        auto sc       = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1},        {0.01f});
        auto cvt_f16  = std::make_shared<ov::op::v0::Convert>(wc, ov::element::f16);
        auto sub      = std::make_shared<ov::op::v1::Subtract>(cvt_f16, zp);
        auto mul      = std::make_shared<ov::op::v1::Multiply>(sub, sc);
        auto cvt_f32  = std::make_shared<ov::op::v0::Convert>(mul, ov::element::f32);
        auto matmul   = std::make_shared<ov::op::v0::MatMul>(input, cvt_f32, false, true);
        auto gamma    = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256}, {1.f});
        auto rms_in   = std::make_shared<ov::op::v1::Add>(matmul, residual);
        auto rms      = std::make_shared<ov::op::internal::RMS>(rms_in, gamma, 1e-6);
        rms->set_friendly_name(rms_name);
        return std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{input, residual});
    };

    // Helper: run passes and return whether the RMS node is marked
    auto is_marked = [&](std::shared_ptr<ov::Model> model, bool register_pass) -> bool {
        ov::pass::Manager manager;
        if (register_pass)
            manager.register_pass<DisableFP16CompForRMSWithFullPrecisionWeights>();
        manager.register_pass<ov::pass::ConvertPrecision>(fp_map);
        manager.run_passes(model);
        for (const auto& op : model->get_ops())
            if (op->get_friendly_name() == rms_name)
                return ov::is_conversion_disabled(op, ov::element::f16);
        return false;
    };

    // ── Case A: Baseline ───────────────────────────────────────────────────
    // Without the pass, large-weight RMS is NOT marked.
    // Paired with case B this proves the pass is what causes marking (A/B anchor).
    {
        SCOPED_TRACE("A: Baseline — K=768 val=130, WITHOUT pass → NOT marked");
        EXPECT_FALSE(is_marked(build_fp_model(130.f, {768, 768}, ov::element::f16, true), false));
    }

    // ── Case B: MatMul → Add → RMS, f16 weight, K=768, val=130 > threshold=85.3 ──
    {
        SCOPED_TRACE("B: K=768 val=130 > threshold=85.3, f16 weight, MatMul+Add → RMS marked");
        EXPECT_TRUE(is_marked(build_fp_model(130.f, {768, 768}, ov::element::f16, true), true));
    }

    // ── Case C: MatMul → RMS (no Add), f32 weight, K=768, val=100 > threshold=85.3 ──
    {
        SCOPED_TRACE("C: K=768 val=100 > threshold=85.3, f32 weight, direct MatMul → RMS marked");
        EXPECT_TRUE(is_marked(build_fp_model(100.f, {768, 768}, ov::element::f32, false), true));
    }

    // ── Case D: K=2048, val=50 > threshold=31.98 → marked ────────────────
    {
        SCOPED_TRACE("D: K=2048 val=50 > threshold=31.98 → RMS marked");
        EXPECT_TRUE(is_marked(build_fp_model(50.f, {768, 2048}, ov::element::f16, true), true));
    }

    // ── Case E: Quantized (u8) weights → NOT marked ───────────────────────
    {
        SCOPED_TRACE("E: Quantized u8 weights (integer, bounded) → RMS NOT marked");
        EXPECT_FALSE(is_marked(build_quantized_model(), true));
    }

    // ── Case F: f16 weight, K=256, val=1.5 < threshold=255.9 → NOT marked ─
    // Simulates gemma-3 FP16 (well-trained, weight magnitudes far below threshold).
    {
        SCOPED_TRACE("F: K=256 val=1.5 < threshold=255.9 → NOT marked (gemma-3 FP16)");
        EXPECT_FALSE(is_marked(build_fp_model(1.5f, {256, 256}, ov::element::f16, true), true));
    }

    // ── Case G: T5 wo/MatMul (K=2048, val=10.6 < threshold=31.98) → NOT marked ─
    // The Cauchy-Schwarz criterion assumes unit-RMS activations; T5's FFN gated
    // intermediate has ||X||_2 ≈ 79,872 (>> sqrt(2048)=45), so overflow occurs
    // despite criterion saying "safe". The RMS kernel INF clamp handles actual NaN.
    {
        SCOPED_TRACE("G: K=2048 val=10.6 < threshold=31.98 → NOT marked (T5 wo/MatMul; kernel clamp handles NaN)");
        EXPECT_FALSE(is_marked(build_fp_model(10.6f, {768, 2048}, ov::element::f16, true), true));
    }
}
