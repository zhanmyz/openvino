// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/matrix_nms.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/runtime/memory.hpp>

#include "matrix_nms_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace tests;

namespace {

#define PAD       -1.0
#define PADI      -1
#define THRESHOLD 1e-3f

template <class T>
std::vector<T> convert(const std::vector<float>& v) {
    return {v.begin(), v.end()};
}

struct matrix_nms_test_inputs {
    int num_butches;
    int num_boxes;
    int num_classes;
    int num_selected_boxes;
    bool sort_result_across_batch;
    float score_threshold;
    int nms_top_k;
    int keep_top_k;
    int background_class;
    float gaussian_sigma;
    float post_threshold;
    bool normalized;
    std::vector<float> boxes_values;
    std::vector<float> scores_values;
    std::vector<float> expected_output;
    std::vector<int> expected_selected_boxes;
    std::vector<int> expected_valid_outputs;
    ov::op::v8::MatrixNms::SortResultType sort_result_type;
    ov::op::v8::MatrixNms::DecayFunction decay_function;
    std::string test_name;
};

using matrix_nms_test_params = std::tuple<matrix_nms_test_inputs, format::type, bool>;

template <class T>
struct matrix_nms_gpu_test : public testing::TestWithParam<matrix_nms_test_params> {
public:
    void test() {
        const auto& [test_inputs, blocked_format, is_caching_test] = testing::TestWithParam<matrix_nms_test_params>::GetParam();

        const auto data_type = ov::element::from<T>();
        const auto plain_format = format::bfyx;

        auto& engine = get_test_engine();

        auto boxes = engine.allocate_memory(
            {data_type, plain_format, tensor{test_inputs.num_butches, test_inputs.num_boxes, 1, 4}});
        auto scores = engine.allocate_memory(
            {data_type,
             plain_format,
             tensor{test_inputs.num_butches, test_inputs.num_classes, 1, test_inputs.num_boxes}});

        auto selected_boxes =
            engine.allocate_memory({data_types::i32, plain_format, tensor{test_inputs.num_selected_boxes, 1, 1, 1}});
        auto valid_outputs =
            engine.allocate_memory({data_types::i32, plain_format, tensor{test_inputs.num_butches, 1, 1, 1}});

        set_values(boxes, convert<T>(test_inputs.boxes_values));
        set_values(scores, convert<T>(test_inputs.scores_values));

        const ov::op::v8::MatrixNms::Attributes attrs(test_inputs.sort_result_type,
                                                      test_inputs.sort_result_across_batch,
                                                      ov::element::i32,
                                                      test_inputs.score_threshold,
                                                      test_inputs.nms_top_k,
                                                      test_inputs.keep_top_k,
                                                      test_inputs.background_class,
                                                      test_inputs.decay_function,
                                                      test_inputs.gaussian_sigma,
                                                      test_inputs.post_threshold,
                                                      test_inputs.normalized);

        topology topology;
        topology.add(input_layout("boxes", boxes->get_layout()));
        topology.add(input_layout("scores", scores->get_layout()));
        topology.add(mutable_data("selected_boxes", selected_boxes));
        topology.add(mutable_data("valid_outputs", valid_outputs));

        topology.add(reorder("reordered_boxes", input_info("boxes"), blocked_format, data_type));
        topology.add(reorder("reordered_scores", input_info("scores"), blocked_format, data_type));

        topology.add(matrix_nms("reordered_matrix_nms",
                                input_info("reordered_boxes"),
                                input_info("reordered_scores"),
                                input_info("selected_boxes"),
                                input_info("valid_outputs"),
                                attrs));
        topology.add(reorder("matrix_nms", input_info("reordered_matrix_nms"), plain_format, data_type));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("boxes", boxes);
        network->set_input_data("scores", scores);

        auto outputs = network->execute();

        auto output = outputs.at("matrix_nms").get_memory();
        cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());

        cldnn::mem_lock<int> selected_boxes_ptr(selected_boxes, get_test_stream());
        cldnn::mem_lock<int> valid_outputs_ptr(valid_outputs, get_test_stream());

        const auto expected_output = convert<T>(test_inputs.expected_output);
        ASSERT_EQ(expected_output.size(), output_ptr.size());
        for (size_t i = 0; i < expected_output.size(); ++i) {
            ASSERT_NEAR(expected_output[i], output_ptr[i], THRESHOLD);
        }

        if (!is_caching_test) {
            ASSERT_EQ(test_inputs.expected_selected_boxes.size(), selected_boxes_ptr.size());
            for (size_t i = 0; i < test_inputs.expected_selected_boxes.size(); ++i) {
                ASSERT_EQ(test_inputs.expected_selected_boxes[i], selected_boxes_ptr[i]);
            }

            ASSERT_EQ(test_inputs.expected_valid_outputs.size(), valid_outputs_ptr.size());
            for (size_t i = 0; i < test_inputs.expected_valid_outputs.size(); ++i) {
                ASSERT_EQ(test_inputs.expected_valid_outputs[i], valid_outputs_ptr[i]);
            }
        }
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<matrix_nms_test_params>& info) {
        auto& test_inputs = std::get<0>(info.param);
        std::ostringstream result;

        auto sort_res_type_str =
            test_inputs.sort_result_type == ov::op::v8::MatrixNms::SortResultType::SCORE
                ? "score"
                : test_inputs.sort_result_type == ov::op::v8::MatrixNms::SortResultType::CLASSID ? "class_id" : "none";
        auto decay_function_str =
            test_inputs.decay_function == ov::op::v8::MatrixNms::DecayFunction::LINEAR
                ? "linear"
                : test_inputs.decay_function == ov::op::v8::MatrixNms::DecayFunction::GAUSSIAN ? "gaussian" : "none";

        result << "SortResultAcrossBatch=" << bool_to_str(test_inputs.sort_result_across_batch) << "_";
        result << "ScoreThreshold=" << test_inputs.score_threshold << "_";
        result << "NmsTopK=" << test_inputs.nms_top_k << "_";
        result << "KeepTopK=" << test_inputs.keep_top_k << "_";
        result << "BackgroundClass=" << test_inputs.background_class << "_";
        result << "GaussianSigma=" << test_inputs.gaussian_sigma << "_";
        result << "PostThreshold=" << test_inputs.post_threshold << "_";
        result << "Normalized=" << bool_to_str(test_inputs.normalized) << "_";
        result << "sort_result_type=" << sort_res_type_str << "_";
        result << "decay_function=" << decay_function_str << "_";
        result << "Format=" << fmt_to_str(std::get<1>(info.param)) << "_";
        result << "Cached=" << bool_to_str(std::get<2>(info.param));

        if (!test_inputs.test_name.empty())
            result << "_TN=" << test_inputs.test_name;

        return result.str();
    }
};

matrix_nms_test_inputs get_matrix_nms_smoke_inputs() {
    return {1,      // num_butches
            6,      // num_boxes
            2,      // num_classes
            3,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.0f,   // score_threshold
            3,      // nms_top_k
            -1,     // keep_top_k
            0,      // background_class
            2.0f,   // gaussian_sigma
            0.0f,   // post_threshold
            true,   // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3},  // scores
            std::vector<float>{1.00,                                                              // expected_output
                               0.95,
                               0.00,
                               0.00,
                               1.00,
                               1.00,
                               1.00,
                               0.8,
                               0.00,
                               10.00,
                               1.00,
                               11.00,
                               1.00,
                               0.13636364,
                               0.0,
                               0.1,
                               1.0,
                               1.1},
            std::vector<int>{0, 3, 1},          // expected_selected_boxes
            std::vector<int>{3},                // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
            "smoke"};
}

matrix_nms_test_inputs get_matrix_nms_gaussian_inputs() {
    return {1,      // num_butches
            6,      // num_boxes
            2,      // num_classes
            3,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.0f,   // score_threshold
            3,      // nms_top_k
            -1,     // keep_top_k
            0,      // background_class
            2.0f,   // gaussian_sigma
            0.0f,   // post_threshold
            true,   // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3},  // scores
            std::vector<float>{1.00,                                                              // expected_output
                               0.95,
                               0.00,
                               0.00,
                               1.00,
                               1.00,
                               1.00,
                               0.8,
                               0.00,
                               10.00,
                               1.00,
                               11.00,
                               1.00,
                               0.1966116,
                               0.0,
                               0.1,
                               1.0,
                               1.1},
            std::vector<int>{0, 3, 1},            // expected_selected_boxes
            std::vector<int>{3},                  // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,    // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::GAUSSIAN,  // decay_function
            "gaussian"};
}

matrix_nms_test_inputs get_matrix_nms_two_batches_two_classes_inputs() {
    return {2,      // num_butches
            6,      // num_boxes
            2,      // num_classes
            6,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.0f,   // score_threshold
            3,      // nms_top_k
            -1,     // keep_top_k
            0,      // background_class
            2.0f,   // gaussian_sigma
            0.0f,   // post_threshold
            true,   // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // scores
                               0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3},
            std::vector<float>{1.00, 0.95,  0.00, 0.00,  1.00, 1.00,  // expected_output
                               1.00, 0.8,   0.00, 10.00, 1.00, 11.00,      1.00, 0.13636364, 0.0,  0.1,
                               1.0,  1.1,   1.00, 0.95,  0.00, 0.00,       1.00, 1.00,       1.00, 0.8,
                               0.00, 10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0,  0.1,        1.0,  1.1},
            std::vector<int>{0, 3, 1, 6, 9, 7},  // expected_selected_boxes
            std::vector<int>{3, 3},              // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            "two_batches_two_classes"};
}

matrix_nms_test_inputs get_matrix_nms_two_batches_two_classes_by_score_cross_batch_inputs() {
    return {2,     // num_butches
            6,     // num_boxes
            2,     // num_classes
            12,    // num_selected_boxes
            true,  // sort_result_across_bch
            0.0f,  // score_threshold
            3,     // nms_top_k
            -1,    // keep_top_k
            -1,    // background_class
            2.0f,  // gaussian_sigma
            0.5f,  // post_threshold
            true,  // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // scores
                               0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3},
            std::vector<float>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00,  // expected_output
                               1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95,
                               0.00, 0.00, 1.00, 1.00,  PAD,  PAD,   PAD,  PAD,  PAD,  PAD,   PAD,  PAD,   PAD,  PAD,
                               PAD,  PAD,  0.00, 0.90,  0.00, 0.00,  1.00, 1.00, 0.00, 0.90,  0.00, 0.00,  1.00, 1.00,
                               1.00, 0.80, 0.00, 10.00, 1.00, 11.00, 1.00, 0.80, 0.00, 10.00, 1.00, 11.00, PAD,  PAD,
                               PAD,  PAD,  PAD,  PAD,   PAD,  PAD,   PAD,  PAD,  PAD,  PAD},
            std::vector<int>{3, 0, 9, 6, PADI, PADI, 0, 6, 3, 9, PADI, PADI},  // expected_selected_boxes
            std::vector<int>{4, 4},                                            // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,                                 // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,                                 // decay_function
            "two_batches_two_classes_by_score_cross_batch"};
}

matrix_nms_test_inputs get_matrix_nms_two_batches_two_classes_by_classid_cross_batch_inputs() {
    return {2,     // num_butches
            6,     // num_boxes
            2,     // num_classes
            12,    // num_selected_boxes
            true,  // sort_result_across_bch
            0.0f,  // score_threshold
            3,     // nms_top_k
            -1,    // keep_top_k
            -1,    // background_class
            2.0f,  // gaussian_sigma
            0.5f,  // post_threshold
            true,  // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // scores
                               0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3},
            std::vector<float>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00,  // expected_output
                               0.00, 0.90, 0.00, 0.00,  1.00, 1.00,  0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90,
                               0.00, 0.00, 1.00, 1.00,  PAD,  PAD,   PAD,  PAD,  PAD,  PAD,   PAD,  PAD,   PAD,  PAD,
                               PAD,  PAD,  1.00, 0.95,  0.00, 0.00,  1.00, 1.00, 1.00, 0.80,  0.00, 10.00, 1.00, 11.00,
                               1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00, PAD,  PAD,
                               PAD,  PAD,  PAD,  PAD,   PAD,  PAD,   PAD,  PAD,  PAD,  PAD},
            std::vector<int>{3, 0, 9, 6, PADI, PADI, 0, 3, 6, 9, PADI, PADI},  // expected_selected_boxes
            std::vector<int>{4, 4},                                            // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::CLASSID,                              // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,                                 // decay_function
            "matrix_nms_two_batches_two_classes_by_classid_cross_batch"};
}

matrix_nms_test_inputs get_matrix_nms_by_keep_top_k_inputs() {
    return {2,      // num_butches
            6,      // num_boxes
            2,      // num_classes
            6,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.0f,   // score_threshold
            3,      // nms_top_k
            3,      // keep_top_k
            0,      // background_class
            2.0f,   // gaussian_sigma
            0.0f,   // post_threshold
            true,   // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // scores
                               0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3},
            std::vector<float>{1.00, 0.95,  0.00, 0.00,  1.00, 1.00,  // expected_output
                               1.00, 0.8,   0.00, 10.00, 1.00, 11.00,      1.00, 0.13636364, 0.0,  0.1,
                               1.0,  1.1,   1.00, 0.95,  0.00, 0.00,       1.00, 1.00,       1.00, 0.8,
                               0.00, 10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0,  0.1,        1.0,  1.1},
            std::vector<int>{0, 3, 1, 6, 9, 7},    // expected_selected_boxes
            std::vector<int>{3, 3},                // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::CLASSID,  // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,     // decay_function
            "matrix_nms_by_keep_top_k"};
}

matrix_nms_test_inputs get_matrix_nms_background_inputs() {
    return {1,      // num_butches
            6,      // num_boxes
            2,      // num_classes
            6,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.0f,   // score_threshold
            3,      // nms_top_k
            -1,     // keep_top_k
            -1,     // background_class
            2.0f,   // gaussian_sigma
            0.0f,   // post_threshold
            true,   // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3},  // scores
            std::vector<float>{0.00, 0.95, 0.0,  10.0, 1.0,  11.0,                                // expected_output
                               1.00, 0.95, 0.0,  0.0,  1.0,  1.0,        0.00, 0.9,  0.0,  0.0,
                               1.0,  1.0,  1.00, 0.8,  0.0,  10.0,       1.0,  11.0, 0.00, 0.13636364,
                               0.0,  0.1,  1.0,  1.1,  1.00, 0.13636364, 0.0,  0.1,  1.0,  1.1},
            std::vector<int>{3, 0, 0, 3, 1, 1},  // expected_selected_boxes
            std::vector<int>{6},                 // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            "matrix_nms_background"};
}

matrix_nms_test_inputs get_matrix_nms_flipped_coordinates_inputs() {
    return {1,      // num_butches
            6,      // num_boxes
            1,      // num_classes
            3,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.0f,   // score_threshold
            3,      // nms_top_k
            -1,     // keep_top_k
            -1,     // background_class
            2.0f,   // gaussian_sigma
            0.0f,   // post_threshold
            true,   // normalized
            std::vector<float>{1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,  // boxes
                               0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3},  // scores
            std::vector<float>{0.00,
                               0.95,
                               0.0,
                               10.0,
                               1.0,
                               11.0,  // expected_output
                               0.00,
                               0.9,
                               1.0,
                               1.0,
                               0.0,
                               0.0,
                               0.00,
                               0.75,
                               0.0,
                               0.1,
                               1.0,
                               1.1},
            std::vector<int>{3, 0, 1},          // expected_selected_boxes
            std::vector<int>{3},                // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
            "flipped_coordinates"};
}

matrix_nms_test_inputs get_matrix_nms_post_threshold_inputs() {
    return {1,      // num_butches
            6,      // num_boxes
            1,      // num_classes
            3,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.0f,   // score_threshold
            3,      // nms_top_k
            -1,     // keep_top_k
            -1,     // background_class
            2.0f,   // gaussian_sigma
            0.8f,   // post_threshold
            true,   // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3},  // scores
            std::vector<float>{0.00,
                               0.95,
                               0.00,
                               10.00,
                               1.00,
                               11.00,  // expected_output
                               0.00,
                               0.9,
                               0.00,
                               0.00,
                               1.00,
                               1.00,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD},
            std::vector<int>{3, 0, PADI},       // expected_selected_boxes
            std::vector<int>{2},                // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
            "post_threshold"};
}

matrix_nms_test_inputs get_matrix_nms_identical_boxes_inputs() {
    return {1,      // num_butches
            10,     // num_boxes
            1,      // num_classes
            3,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.0f,   // score_threshold
            3,      // nms_top_k
            -1,     // keep_top_k
            -1,     // background_class
            2.0f,   // gaussian_sigma
            0.3f,   // post_threshold
            true,   // normalized
            std::vector<float>{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,  // boxes
                               1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                               0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0},
            std::vector<float>{0.4, 0.01, 0.2, 0.09, 0.15, 0.05, 0.02, 0.03, 0.05, 0.0},  // scores
            std::vector<float>{0.00,
                               0.40,
                               0.00,
                               0.00,
                               1.00,
                               1.00,  // expected_output
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD},
            std::vector<int>{0, PADI, PADI},    // expected_selected_boxes
            std::vector<int>{1},                // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
            "identical_boxes"};
};

matrix_nms_test_inputs get_matrix_nms_top_k_inputs() {
    return {1,      // num_butches
            6,      // num_boxes
            1,      // num_classes
            2,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.0f,   // score_threshold
            2,      // nms_top_k
            -1,     // keep_top_k
            -1,     // background_class
            2.0f,   // gaussian_sigma
            0.0f,   // post_threshold
            true,   // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3},  // scores
            std::vector<float>{0.00,
                               0.95,
                               0.00,
                               10.00,
                               1.00,
                               11.00,  // expected_output
                               0.00,
                               0.90,
                               0.00,
                               0.00,
                               1.00,
                               1.00},
            std::vector<int>{3, 0},             // expected_selected_boxes
            std::vector<int>{2},                // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
            "matrix_nms_nms_top_k"};
}

matrix_nms_test_inputs get_matrix_nms_single_box_inputs() {
    return {1,                                                       // num_butches
            1,                                                       // num_boxes
            1,                                                       // num_classes
            1,                                                       // num_selected_boxes
            false,                                                   // sort_result_across_bch
            0.0f,                                                    // score_threshold
            3,                                                       // nms_top_k
            -1,                                                      // keep_top_k
            -1,                                                      // background_class
            2.0f,                                                    // gaussian_sigma
            0.0f,                                                    // post_threshold
            true,                                                    // normalized
            std::vector<float>{0.0, 0.0, 1.0, 1.0},                  // boxes
            std::vector<float>{0.9},                                 // scores
            std::vector<float>{0.00, 0.90, 0.00, 0.00, 1.00, 1.00},  // expected_output
            std::vector<int>{0},                                     // expected_selected_boxes
            std::vector<int>{1},                                     // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,                       // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,                       // decay_function
            "matrix_nms_single_box"};
}

matrix_nms_test_inputs get_matrix_nms_no_output_inputs() {
    return {1,      // num_butches
            6,      // num_boxes
            1,      // num_classes
            3,      // num_selected_boxes
            false,  // sort_result_across_bch
            2.0f,   // score_threshold
            3,      // nms_top_k
            -1,     // keep_top_k
            -1,     // background_class
            2.0f,   // gaussian_sigma
            0.0f,   // post_threshold
            true,   // normalized
            std::vector<float>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0},
            std::vector<float>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3},  // scores
            std::vector<float>{PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,  // expected_output
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD,
                               PAD},
            std::vector<int>{PADI, PADI, PADI},  // expected_selected_boxes
            std::vector<int>{0},                 // expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            "matrix_nms_no_output"};
}

matrix_nms_test_inputs get_matrix_nms_large_value_of_max_boxes_per_class() {
    const int num_boxes = 22743;
    const int num_classes = 2;

    // [batch, boxes, 1, 4]
    std::vector<float> boxes = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};
    boxes.resize(num_boxes * 4, PAD);

    // [batch, classes, 1, boxes]
    std::vector<float> scores = {
        0.9,  0.75, 0.6, 0.95, 0.5, 0.3};
    scores.resize(num_boxes * num_classes, PAD);
    scores[num_boxes * (num_classes - 1)] = 0.95;
    scores[num_boxes * (num_classes - 1) + 1] = 0.75;
    scores[num_boxes * (num_classes - 1) + 2] = 0.6;
    scores[num_boxes * (num_classes - 1) + 3] = 0.80;
    scores[num_boxes * (num_classes - 1) + 4] = 0.5;
    scores[num_boxes * (num_classes - 1) + 5] = 0.3;

    std::vector<float> expected_output = {
	1.00, 0.95, 0.00, 0.00, 1.00, 1.00,  1.00, 0.8, 0.00, 10.00, 1.00, 11.00,
        1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1};

    return {
            1,      // num_butches
            num_boxes,		// num_boxes
            num_classes,	// num_classes
            3,      // num_selected_boxes
            false,  // sort_result_across_bch
            0.01f,  // score_threshold
            -1,     // nms_top_k
            3,      // keep_top_k
            0,      // background_class
            2.0f,   // gaussian_sigma
            0.01f,  // post_threshold
            true,   // normalized
            boxes,
            scores,
            expected_output,		// expected_output
            std::vector<int>{0, 3, 1},	// expected_selected_boxes
            std::vector<int>{3},	// expected_valid_output
            ov::op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
            ov::op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
            "large_value_of_max_boxes_per_class"};
}

// Regression test for CVS-186753 -- targets the typed-pointer over-stride bug
// on the iou_matrix / iou_max / min_decays internal buffers of matrix_nms_ref.
//
// Pre-fix code in matrix_nms_ref.cl did:
//     __global INPUT1_TYPE* iou_matrix =
//         input_iou_matrix + offset * MAX_BOXES_PER_CLASS * sizeof(INPUT1_TYPE);
// OpenCL typed-pointer arithmetic already strides by sizeof(element), so the
// extra `* sizeof(INPUT1_TYPE)` over-multiplied the offset by 4x for f32.
// For offset > 0, every IOU/decay write therefore landed past the end of the
// allocated 4-byte * batches * classes * MAX_BOXES_PER_CLASS region; on
// Windows GPU the OOB writes corrupted neighbouring driver memory and
// triggered a TDR. The post-fix kernel removes the sizeof().
//
// On Linux / some IGC versions the corruption was hidden because reads were
// symmetrically over-strided in the same kernel, so the kernel's *output*
// values stayed consistent end-to-end. This regression test therefore does
// NOT rely on output values -- it inspects the iou_matrix internal buffer
// directly after execute() and asserts that the in-bounds slice for the
// active class (offset = 1) actually received the IOU values that the kernel
// is supposed to write there.
//
// A/B behaviour:
//   * Post-fix kernel: writes go to the correct iou_matrix[offset*K + j],
//     so the in-bounds slice contains non-zero IOU values for the active
//     class -- this test PASSES.
//   * Pre-fix kernel: writes go to iou_matrix[offset*K*sizeof(T) + j] which
//     is way past the buffer end; the in-bounds slice retains the zero from
//     the allocator's reset=true initial value -- this test FAILS.
TEST(matrix_nms_gpu_test, iou_matrix_buffer_offset_regression_cvs_186753) {
    auto& engine = get_test_engine();
    const auto data_type = data_types::f32;
    const auto plain_format = format::bfyx;

    // Configuration: 1 batch, 2 classes (class 0 = background), 4 boxes.
    // background_class = 0 ensures the kernel's BACKGROUND_CLASS early-return
    // path runs for offset = 0 and the over-strided write path runs only for
    // offset = batchId * NUM_CLASSES + classId = 0 * 2 + 1 = 1 -- exactly the
    // offset value where Bug C produces the largest absolute mis-stride for
    // small MAX_BOXES_PER_CLASS.
    const int num_batches = 1;
    const int num_classes = 2;
    const int num_boxes = 4;
    const int keep_top_k = 4;  // becomes MAX_BOXES_PER_CLASS in the kernel
    const int background_class = 0;

    // Boxes with overlap so Stage 0 Phase 4 actually populates iou_matrix
    // with non-zero values for multiple (i, j) pairs.
    const std::vector<float> boxes_values = {
        0.0f, 0.0f,   1.0f, 1.0f,
        0.0f, 0.1f,   1.0f, 1.1f,
        0.0f, 0.2f,   1.0f, 1.2f,
        0.0f, 0.3f,   1.0f, 1.3f,
    };
    // Per-class scores: class 0 (background) gets zeros; class 1 gets a
    // descending ladder that produces 4 valid boxes after the score_threshold
    // filter so Stage 0 has work to do.
    std::vector<float> scores_values(num_classes * num_boxes, 0.0f);
    scores_values[num_boxes * 1 + 0] = 0.95f;
    scores_values[num_boxes * 1 + 1] = 0.85f;
    scores_values[num_boxes * 1 + 2] = 0.75f;
    scores_values[num_boxes * 1 + 3] = 0.65f;

    auto boxes = engine.allocate_memory(
        {data_type, plain_format, tensor{num_batches, num_boxes, 1, 4}});
    auto scores = engine.allocate_memory(
        {data_type, plain_format, tensor{num_batches, num_classes, 1, num_boxes}});
    auto selected_boxes = engine.allocate_memory(
        {data_types::i32, plain_format, tensor{keep_top_k, 1, 1, 1}});
    auto valid_outputs = engine.allocate_memory(
        {data_types::i32, plain_format, tensor{num_batches, 1, 1, 1}});
    set_values(boxes, boxes_values);
    set_values(scores, scores_values);

    const ov::op::v8::MatrixNms::Attributes attrs(
        ov::op::v8::MatrixNms::SortResultType::CLASSID,
        false,                  // sort_result_across_batch
        ov::element::i32,
        0.0f,                   // score_threshold (keep all 4)
        -1,                     // nms_top_k (no cap)
        keep_top_k,             // keep_top_k (-> MAX_BOXES_PER_CLASS)
        background_class,
        ov::op::v8::MatrixNms::DecayFunction::LINEAR,
        2.0f,                   // gaussian_sigma
        0.0f,                   // post_threshold (don't filter on decay)
        true);                  // normalized

    topology topology;
    topology.add(input_layout("boxes", boxes->get_layout()));
    topology.add(input_layout("scores", scores->get_layout()));
    topology.add(mutable_data("selected_boxes", selected_boxes));
    topology.add(mutable_data("valid_outputs", valid_outputs));
    topology.add(reorder("reordered_boxes", input_info("boxes"), plain_format, data_type));
    topology.add(reorder("reordered_scores", input_info("scores"), plain_format, data_type));
    topology.add(matrix_nms("matrix_nms_prim",
                            input_info("reordered_boxes"),
                            input_info("reordered_scores"),
                            input_info("selected_boxes"),
                            input_info("valid_outputs"),
                            attrs));
    topology.add(reorder("matrix_nms_out", input_info("matrix_nms_prim"), plain_format, data_type));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("boxes", boxes);
    network.set_input_data("scores", scores);

    auto outputs = network.execute();

    // Synchronize GPU execution by reading the output before inspecting
    // internal buffers, ensuring Stage 0 kernel has completed.
    auto output = outputs.at("matrix_nms_out").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_GT(output_ptr.size(), 0u);  // sanity: kernel produced output

    // Internal buffer layout from matrix_nms_kernel_ref.cpp:
    //   [0] box_info       (BOX_INFO * batches * classes * MAX_BOXES_PER_CLASS)
    //   [1] sel_boxes_num  (int      * batches * classes)
    //   [2] iou_matrix     (float    * batches * classes * MAX_BOXES_PER_CLASS)
    //   [3] iou_max        (float    * batches * classes * MAX_BOXES_PER_CLASS)
    //   [4] min_decays     (float    * batches * classes * MAX_BOXES_PER_CLASS)
    auto inst = network.get_primitive("matrix_nms_prim");
    const auto& internals = inst->get_intermediates_memories();
    ASSERT_GE(internals.size(), 5u);

    const size_t expected_floats =
        static_cast<size_t>(num_batches) * num_classes * keep_top_k;
    auto iou_matrix_mem = internals[2];
    ASSERT_EQ(iou_matrix_mem->size(), expected_floats * sizeof(float));

    cldnn::mem_lock<float, mem_lock_type::read> iou_matrix_lock(iou_matrix_mem, get_test_stream());

    // offset for the active (non-background) class:
    //   offset = batchId * NUM_CLASSES + classId = 0 * 2 + 1 = 1
    // The kernel writes iou_matrix[offset * MAX_BOXES_PER_CLASS + j] for
    // j in [0, valid_boxes_num). With 4 above-threshold boxes and
    // MAX_BOXES_PER_CLASS = 4 these writes cover j = 0, 1, 2 (Phase 4 loop
    // skips j == i, so each i in [1, 4) writes (i-1) entries before
    // resetting iou_max[i] to 0 -- the cumulative effect is that
    // iou_matrix[offset*K + 0], [offset*K + 1], [offset*K + 2] all receive
    // non-zero IOU values from the overlapping boxes above).
    const int offset = 0 * num_classes + 1;
    const size_t base = static_cast<size_t>(offset) * keep_top_k;
    ASSERT_LE(base + keep_top_k, expected_floats);

    // Post-fix: at least one of the in-bounds iou_matrix slots for the
    // active class must have been overwritten (non-zero IOU value) by a
    // real IOU value from Stage 0 Phase 4.
    // Pre-fix: the writes are over-strided past the buffer end, so the
    // entire in-bounds slice for the active class retains the allocator's
    // initial zero value and this assertion fails.
    bool any_overwritten = false;
    for (int j = 0; j < keep_top_k; ++j) {
        if (iou_matrix_lock[base + j] != 0.0f) {
            any_overwritten = true;
            break;
        }
    }
    ASSERT_TRUE(any_overwritten)
        << "iou_matrix slice for offset=" << offset
        << " (base=" << base << ", K=" << keep_top_k
        << ") is all-zero after execute(). "
           "This indicates that Stage 0's writes were over-strided past "
           "the buffer end -- the typed-pointer * sizeof(INPUT1_TYPE) bug "
           "in matrix_nms_ref.cl is back. See CVS-186753.";
}

const std::vector<format::type> layout_formats = {format::bfyx,
                                                  format::b_fs_yx_fsv16,
                                                  format::b_fs_yx_fsv32,
                                                  format::bs_fs_yx_bsv16_fsv16,
                                                  format::bs_fs_yx_bsv32_fsv32,
                                                  format::bs_fs_yx_bsv32_fsv16};

#ifdef RUN_ALL_MODEL_CACHING_TESTS
const std::vector<bool> run_caching_test = {false, true};
#else
const std::vector<bool> run_caching_test = {false};
#endif

#define INSTANTIATE_MATRIX_NMS_TEST_SUITE(input_type, func)                                                \
    using matrix_nms_gpu_test_##input_type##func = matrix_nms_gpu_test<input_type>;                        \
    TEST_P(matrix_nms_gpu_test_##input_type##func, test) {                                                 \
        test();                                                                                            \
    }                                                                                                      \
    INSTANTIATE_TEST_SUITE_P(matrix_nms_test_##input_type##func,                                           \
                             matrix_nms_gpu_test_##input_type##func,                                       \
                             testing::Combine(testing::Values(func()), testing::ValuesIn(layout_formats),  \
                                              testing::ValuesIn(run_caching_test)),                        \
                             matrix_nms_gpu_test_##input_type##func::PrintToStringParamName);

INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_smoke_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_gaussian_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_two_batches_two_classes_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_two_batches_two_classes_by_classid_cross_batch_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_two_batches_two_classes_by_score_cross_batch_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_by_keep_top_k_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_background_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_flipped_coordinates_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_post_threshold_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_identical_boxes_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_top_k_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_single_box_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_no_output_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float, get_matrix_nms_large_value_of_max_boxes_per_class)

using ov::float16;
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_smoke_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_gaussian_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_two_batches_two_classes_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_by_keep_top_k_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_two_batches_two_classes_by_classid_cross_batch_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_two_batches_two_classes_by_score_cross_batch_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_background_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_flipped_coordinates_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_post_threshold_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_identical_boxes_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_top_k_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_single_box_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_no_output_inputs)
INSTANTIATE_MATRIX_NMS_TEST_SUITE(float16, get_matrix_nms_large_value_of_max_boxes_per_class)

#ifndef RUN_ALL_MODEL_CACHING_TESTS
INSTANTIATE_TEST_SUITE_P(matrix_nms_test_float16get_matrix_nms_smoke_inputs_cached,
                         matrix_nms_gpu_test_float16get_matrix_nms_smoke_inputs,
                         testing::Combine(testing::Values(get_matrix_nms_smoke_inputs()), testing::ValuesIn(layout_formats),
                                          testing::Values(true)),
                         matrix_nms_gpu_test_float16get_matrix_nms_smoke_inputs::PrintToStringParamName);
#endif

#undef INSTANTIATE_MATRIX_NMS_TEST_SUITE

}  // namespace
