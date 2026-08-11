// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief This transformation disables fp16 compression for RMS nodes in a specific pattern
 * to prevent precision loss.
 *
 * The targeted pattern is:
 *
 *     ...               ...
 *      |                 |
 *   Add (f32)        RMS (f32)
 * (add_m)          (rms_post_m)
 *      \              /
 *       \            /
 *         Add (f32)
 *        (add_1_m)
 *            |
 *            |
 *         RMS (f32)
 *         (rms_m)
 *
 * This pass finds the final RMS node (rms_m) in this chain and disables fp16 compression
 * for both itself and the preceding RMS node (rms_post_m). This is done to maintain
 * higher precision, as the result of the intermediate `add_1_m` operation can exceed
 * the representable range of fp16, leading to significant precision loss.
 * By keeping this pattern in fp32, numerical stability is preserved.
 */
class DisableFP16CompForGemma3RMSPattern: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForGemma3RMSPattern");
    DisableFP16CompForGemma3RMSPattern();
};

/**
 * @brief Disables fp16 compression for RMS nodes whose upstream MatMul weights
 * can produce outputs exceeding the fp16 representable range.
 *
 * Overflow criterion (Cauchy-Schwarz inequality on dot product):
 *   After RMS normalization, ||X||_2 = sqrt(K) (unit-RMS, K = feature dim).
 *   For weight row W_j: ||W_j||_2 ≤ sqrt(K) × max|W|.
 *   By Cauchy-Schwarz: |Y_j| = |X · W_j| ≤ ||X||_2 × ||W_j||_2 ≤ K × max|W|.
 *   Overflow occurs when: K × max|W| > FP16_MAX (65504).
 *   Equivalently: max|W| > FP16_MAX / K.
 *
 * Models trained in pure FP32 (e.g. T5/ParlerTTS) can have large weight values
 * (max|W| > FP16_MAX / K), producing MatMul outputs that overflow fp16.
 * INF propagates through residual Add into RMS: rsqrt(INF)=0, INF×0=NaN.
 *
 * Models trained with FP16/BF16 awareness (e.g. gemma-3, qwen3) keep weights
 * bounded (max|W| << FP16_MAX / K), making overflow impossible.
 *
 * Detection: traces from RMS → Add (residual) → MatMul → weight Constant.
 * Evaluates the derived overflow criterion against the actual weight values.
 */
class DisableFP16CompForRMSWithFullPrecisionWeights : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForRMSWithFullPrecisionWeights");
    DisableFP16CompForRMSWithFullPrecisionWeights();
};

}   // namespace ov::intel_gpu
