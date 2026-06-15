// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Disables FP16 conversion for RMS nodes and their downstream MatMul consumers
 * within encoder/decoder transformer blocks.
 *
 * Models like T5 (used in ParlerTTS, etc.) have FFN layers whose MatMul outputs
 * can exceed the FP16 representable range (65504). When combined with residual
 * connections, these overflows (INF) propagate and produce NaN in subsequent layers.
 *
 * This transformation identifies RMS nodes that feed into MatMul operations and
 * marks the entire path to stay in FP32, preventing FP16 overflow in the dense layers.
 *
 * Pattern matched:
 *   RMS → (optional Reshape/Convert) → MatMul
 *
 * Both the RMS node and the MatMul nodes are marked with disable_conversion(f16).
 */
class DisableFP16CompForRMSNormBlock : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("DisableFP16CompForRMSNormBlock");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
