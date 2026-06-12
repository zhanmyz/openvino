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

// Regression test for CVS-186753 -- targets Bug A (private memory overflow) and
// Bug B (O(N^2) bubble sort TDR) by using the exact pp-yolo model configuration:
//   22743 boxes, 80 classes, nms_top_k=-1, keep_top_k=100, score_threshold=0.01
// with ALL boxes having scores above the threshold so valid_boxes_num = 22743.
//
// Pre-fix behaviour (this test FAILs by crashing / timing out):
//   Bug A: Stage 0 declared `int sorted_score_indices[NUM_BOXES]` as private
//          memory.  With NUM_BOXES=22743 that's 91KB per work-item; on GPUs
//          with limited private/scratch memory the clCreateKernel call fails
//          with CL_OUT_OF_RESOURCES. On Intel Arc, the IGC JIT silently
//          spills to scratch memory and the kernel becomes extremely slow.
//   Bug B: After collecting all 22743 above-threshold box indices, the old
//          code ran an O(N^2) bubble sort (sortIterative): 22743^2 ≈ 517 M
//          iterations per class × 80 classes overwhelms the 2-second Windows
//          GPU TDR deadline.
//
// Post-fix behaviour (this test PASSES):
//   The fix bounds the sorted private array to MAX_BOXES_PER_CLASS elements
//   (= min(num_boxes, batches * keep_top_k) = 100 for this topology) and uses
//   a bounded O(NUM_BOXES * MAX_BOXES_PER_CLASS) insertion sort. The full
//   pp-yolo scale executes in <1 second.
TEST(matrix_nms_gpu_test, ppyolo_full_scale_no_tdr_regression_cvs_186753) {
    auto& engine = get_test_engine();
    const auto data_type = data_types::f32;
    const auto plain_format = format::bfyx;

    // Exact pp-yolo MatrixNms configuration from the model XML.
    const int num_batches = 1;
    const int num_boxes = 22743;
    const int num_classes = 80;
    const int nms_top_k = -1;
    const int keep_top_k = 100;
    const int background_class = -1;
    const float score_threshold = 0.01f;
    const float post_threshold = 0.01f;
    const float gaussian_sigma = 2.0f;
    const bool normalized = false;

    // MAX_BOXES_PER_CLASS = min(num_boxes, batches * min(num_boxes * classes, keep_top_k))
    //                     = min(22743, 1 * min(22743*80, 100)) = 100
    // MAX_BOXES_PER_BATCH = keep_top_k = 100
    const int max_boxes_per_batch = keep_top_k;

    // Generate deterministic boxes spread across a grid so nearby boxes overlap.
    // This ensures IOU computation is non-trivial (not all-zero IOU).
    std::vector<float> boxes_values(num_boxes * 4);
    {
        const float grid_step = 5.0f;
        const float box_size = 20.0f;
        const int grid_cols = 200;
        for (int i = 0; i < num_boxes; ++i) {
            float x = static_cast<float>(i % grid_cols) * grid_step;
            float y = static_cast<float>(i / grid_cols) * grid_step;
            boxes_values[4 * i + 0] = x;
            boxes_values[4 * i + 1] = y;
            boxes_values[4 * i + 2] = x + box_size;
            boxes_values[4 * i + 3] = y + box_size;
        }
    }

    // Generate scores: ALL above score_threshold (0.01) so the kernel must
    // process all 22743 boxes per class. Uses LCG for determinism.
    std::vector<float> scores_values(num_classes * num_boxes);
    {
        uint32_t rng = 12345u;
        for (size_t i = 0; i < scores_values.size(); ++i) {
            rng = rng * 1103515245u + 12345u;
            // Map to [0.1, 1.0] -- well above score_threshold=0.01
            scores_values[i] = 0.1f + 0.9f * static_cast<float>((rng >> 16) & 0x7FFF) / 32767.0f;
        }
    }

    auto boxes = engine.allocate_memory(
        {data_type, plain_format, tensor{num_batches, num_boxes, 1, 4}});
    auto scores = engine.allocate_memory(
        {data_type, plain_format, tensor{num_batches, num_classes, 1, num_boxes}});
    auto selected_boxes = engine.allocate_memory(
        {data_types::i32, plain_format, tensor{max_boxes_per_batch, 1, 1, 1}});
    auto valid_outputs = engine.allocate_memory(
        {data_types::i32, plain_format, tensor{num_batches, 1, 1, 1}});
    set_values(boxes, boxes_values);
    set_values(scores, scores_values);

    const ov::op::v8::MatrixNms::Attributes attrs(
        ov::op::v8::MatrixNms::SortResultType::SCORE,
        false,                  // sort_result_across_batch
        ov::element::i32,
        score_threshold,
        nms_top_k,
        keep_top_k,
        background_class,
        ov::op::v8::MatrixNms::DecayFunction::LINEAR,
        gaussian_sigma,
        post_threshold,
        normalized);

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

    // If Bug A or Bug B are present, this call either:
    //   - throws CL_OUT_OF_RESOURCES during kernel compilation (Bug A), or
    //   - hangs until the Windows GPU TDR kills the device (Bug B), after which
    //     the GPU runtime throws an exception (e.g. CL_DEVICE_NOT_FOUND).
    auto outputs = network.execute();

    // --- Sanity checks: kernel produced reasonable output ---
    auto output = outputs.at("matrix_nms_out").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    // Output shape is [max_boxes_per_batch * num_batches, 6] = [100, 6] = 600 floats
    ASSERT_EQ(output_ptr.size(), static_cast<size_t>(max_boxes_per_batch * num_batches * 6));

    // valid_outputs must report > 0 detections (with 22743 boxes all above
    // threshold and 80 classes, there will be many surviving boxes).
    cldnn::mem_lock<int, mem_lock_type::read> valid_outputs_ptr(valid_outputs, get_test_stream());
    ASSERT_GT(valid_outputs_ptr[0], 0)
        << "valid_outputs is 0: kernel produced no detections despite all "
           "22743 boxes having scores > 0.01 across 80 classes.";

    // Verify at least some output rows are valid (not padding).
    // A valid row has class_idx >= 0 and finite score > 0.
    int valid_rows = 0;
    for (int i = 0; i < max_boxes_per_batch; ++i) {
        float cls = output_ptr[i * 6 + 0];
        float score = output_ptr[i * 6 + 1];
        if (cls >= 0.0f && score > 0.0f) {
            ++valid_rows;
            // Class index must be in [0, num_classes)
            ASSERT_GE(cls, 0.0f);
            ASSERT_LT(cls, static_cast<float>(num_classes));
            // Score must be finite and in (0, 1]
            ASSERT_TRUE(std::isfinite(score));
            ASSERT_LE(score, 1.0f);
            // Coordinates must be finite
            for (int j = 2; j < 6; ++j) {
                ASSERT_TRUE(std::isfinite(output_ptr[i * 6 + j]))
                    << "Non-finite coordinate at output row " << i << " col " << j;
            }
        }
    }
    ASSERT_GT(valid_rows, 0)
        << "All output rows are padding (-1). Kernel failed to produce detections.";

    // The test primarily validates that the kernel COMPLETES without TDR or
    // CL_OUT_OF_RESOURCES. If we reach this point, Bug A and Bug B are fixed.
}

// Regression test for CVS-186753 -- targets Bug D (stale box_info data across
// inferences due to unused tail not being zeroed).
//
// Pre-fix behaviour (this test FAILs):
//   Bug D has two aspects:
//   (1) The background class slice of box_info is never written (old code just
//       `return`s), retaining stale data from a previous inference.
//   (2) After Stage 0 writes valid entries to box_info, the unused tail slots
//       retain stale data from the previous inference.
//   (3) Stage 1's valid_outputs used `+=` (accumulation) instead of `=`
//       (assignment), so valid_outputs accumulates across inferences.
//
//   When executing a second inference with fewer/zero detections, Stage 1 sees
//   non-zero scores in stale slots and counts them as valid. The output of the
//   second inference contains ghost detections from the first.
//
// Post-fix behaviour (this test PASSES):
//   The fix zeroes unused box_info tail in Stage 0, zeroes the background class
//   slice, and Stage 1 uses assignment. A second inference with all-zero scores
//   correctly produces zero valid detections.
TEST(matrix_nms_gpu_test, stale_buffer_across_inferences_regression_cvs_186753) {
    auto& engine = get_test_engine();
    const auto data_type = data_types::f32;
    const auto plain_format = format::bfyx;

    // 1 batch, 3 classes (class 0 = background), 6 boxes.
    // keep_top_k = 6 so MAX_BOXES_PER_CLASS = 6.
    const int num_batches = 1;
    const int num_classes = 3;
    const int num_boxes = 6;
    const int keep_top_k = 6;
    const int background_class = 0;
    const int max_boxes_per_batch = keep_top_k;

    // Boxes with overlap to generate non-trivial detections.
    const std::vector<float> boxes_values = {
        0.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 0.1f, 1.0f, 1.1f,
        0.0f, 0.2f, 1.0f, 1.2f,
        0.0f, 10.0f, 1.0f, 11.0f,
        0.0f, 10.1f, 1.0f, 11.1f,
        0.0f, 100.0f, 1.0f, 101.0f,
    };

    // First inference: high scores for class 1 and 2 → produces detections.
    std::vector<float> scores_run1(num_classes * num_boxes, 0.0f);
    // class 1: all 6 boxes above threshold
    for (int i = 0; i < num_boxes; ++i)
        scores_run1[num_boxes * 1 + i] = 0.9f - 0.1f * i;
    // class 2: all 6 boxes above threshold
    for (int i = 0; i < num_boxes; ++i)
        scores_run1[num_boxes * 2 + i] = 0.85f - 0.1f * i;

    // Second inference: ALL scores at zero → zero detections expected.
    std::vector<float> scores_run2(num_classes * num_boxes, 0.0f);

    auto boxes = engine.allocate_memory(
        {data_type, plain_format, tensor{num_batches, num_boxes, 1, 4}});
    auto scores = engine.allocate_memory(
        {data_type, plain_format, tensor{num_batches, num_classes, 1, num_boxes}});
    auto selected_boxes = engine.allocate_memory(
        {data_types::i32, plain_format, tensor{max_boxes_per_batch, 1, 1, 1}});
    auto valid_outputs = engine.allocate_memory(
        {data_types::i32, plain_format, tensor{num_batches, 1, 1, 1}});

    set_values(boxes, boxes_values);

    const ov::op::v8::MatrixNms::Attributes attrs(
        ov::op::v8::MatrixNms::SortResultType::SCORE,
        false,
        ov::element::i32,
        0.0f,                   // score_threshold
        -1,                     // nms_top_k
        keep_top_k,
        background_class,
        ov::op::v8::MatrixNms::DecayFunction::LINEAR,
        2.0f,
        0.0f,                   // post_threshold
        true);

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

    // --- First inference: high scores → produces detections ---
    set_values(scores, scores_run1);
    network.set_input_data("scores", scores);
    auto outputs1 = network.execute();
    {
        auto output = outputs1.at("matrix_nms_out").get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> out_ptr(output, get_test_stream());
        // Confirm first inference actually produced detections.
        cldnn::mem_lock<int, mem_lock_type::read> vo_ptr(valid_outputs, get_test_stream());
        ASSERT_GT(vo_ptr[0], 0) << "First inference should produce detections.";
    }

    // --- Second inference: all-zero scores → zero detections expected ---
    set_values(scores, scores_run2);
    network.set_input_data("scores", scores);
    auto outputs2 = network.execute();
    {
        auto output = outputs2.at("matrix_nms_out").get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> out_ptr(output, get_test_stream());
        cldnn::mem_lock<int, mem_lock_type::read> vo_ptr(valid_outputs, get_test_stream());

        // Bug D check: valid_outputs must be 0 on the second inference.
        // Pre-fix: Stage 1 used += (accumulated from first run) OR stale
        // box_info entries from run 1 were sorted as valid detections.
        ASSERT_EQ(vo_ptr[0], 0)
            << "Second inference with all-zero scores reported "
            << vo_ptr[0] << " valid detections (expected 0). "
               "This indicates stale box_info data from the first inference "
               "was not cleared -- Bug D in matrix_nms_ref.cl is back. "
               "See CVS-186753.";

        // Additionally verify all output rows are padding.
        for (int i = 0; i < max_boxes_per_batch; ++i) {
            ASSERT_LT(out_ptr[i * 6 + 0], 0.0f)
                << "Output row " << i << " has class >= 0 on second inference "
                   "with zero scores. Ghost detection from stale buffer.";
        }
    }
}

// Regression test for CVS-186753 -- targets Bug F (O(N^2) cross-batch sort in
// Stage 2 causing Windows GPU TDR).
//
// Pre-fix behaviour (this test FAILs by TDR crash):
//   Stage 2 used `sortIterativeBoxesAcrossBatches(box_info)` which is an
//   O(N^2) bubble sort over NUM_BATCHES * NUM_CLASSES * MAX_BOXES_PER_CLASS
//   elements. With 2 batches × 80 classes × 200 boxes_per_class = 32000
//   entries, that's 32000^2 = 1 billion global memory comparisons → TDR.
//
// Post-fix behaviour (this test PASSES):
//   The fix uses a partial selection sort bounded by keep_top_k (100
//   iterations × 32000 = 3.2M operations) and exits on score==0 entries.
TEST(matrix_nms_gpu_test, cross_batch_sort_no_tdr_regression_cvs_186753) {
    auto& engine = get_test_engine();
    const auto data_type = data_types::f32;
    const auto plain_format = format::bfyx;

    // 2 batches, 80 classes, 200 boxes — triggers cross-batch sort in Stage 2.
    // sort_result_across_batch = TRUE activates the Stage 2 sort path.
    const int num_batches = 2;
    const int num_classes = 80;
    const int num_boxes = 200;
    const int nms_top_k = -1;
    const int keep_top_k = 100;
    const int background_class = -1;
    const float score_threshold = 0.01f;
    const float post_threshold = 0.01f;

    // MAX_BOXES_PER_CLASS = min(200, 2 * min(200*80, 100)) = min(200, 200) = 200
    // Stage 2 sort size = 2 * 80 * 200 = 32000 entries
    const int max_boxes_per_batch = keep_top_k;

    // Generate boxes on a grid with overlap.
    std::vector<float> boxes_values(num_batches * num_boxes * 4);
    {
        const float grid_step = 5.0f;
        const float box_size = 20.0f;
        const int grid_cols = 20;
        for (int b = 0; b < num_batches; ++b) {
            for (int i = 0; i < num_boxes; ++i) {
                int base = (b * num_boxes + i) * 4;
                float x = static_cast<float>(i % grid_cols) * grid_step;
                float y = static_cast<float>(i / grid_cols) * grid_step;
                boxes_values[base + 0] = x;
                boxes_values[base + 1] = y;
                boxes_values[base + 2] = x + box_size;
                boxes_values[base + 3] = y + box_size;
            }
        }
    }

    // Scores: all above threshold so many detections survive to Stage 2.
    std::vector<float> scores_values(num_batches * num_classes * num_boxes);
    {
        uint32_t rng = 67890u;
        for (size_t i = 0; i < scores_values.size(); ++i) {
            rng = rng * 1103515245u + 12345u;
            scores_values[i] = 0.1f + 0.9f * static_cast<float>((rng >> 16) & 0x7FFF) / 32767.0f;
        }
    }

    auto boxes = engine.allocate_memory(
        {data_type, plain_format, tensor{num_batches, num_boxes, 1, 4}});
    auto scores = engine.allocate_memory(
        {data_type, plain_format, tensor{num_batches, num_classes, 1, num_boxes}});
    auto selected_boxes = engine.allocate_memory(
        {data_types::i32, plain_format, tensor{max_boxes_per_batch * num_batches, 1, 1, 1}});
    auto valid_outputs = engine.allocate_memory(
        {data_types::i32, plain_format, tensor{num_batches, 1, 1, 1}});
    set_values(boxes, boxes_values);
    set_values(scores, scores_values);

    const ov::op::v8::MatrixNms::Attributes attrs(
        ov::op::v8::MatrixNms::SortResultType::SCORE,
        true,                   // sort_result_across_batch = TRUE → activates Stage 2 sort
        ov::element::i32,
        score_threshold,
        nms_top_k,
        keep_top_k,
        background_class,
        ov::op::v8::MatrixNms::DecayFunction::LINEAR,
        2.0f,
        post_threshold,
        false);                 // normalized=false (like pp-yolo)

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

    // If Bug F is present, Stage 2's O(N^2) cross-batch sort over 32000
    // entries would TDR (32000^2 ≈ 1 billion global memory ops).
    auto outputs = network.execute();

    auto output = outputs.at("matrix_nms_out").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    // Output shape = [max_boxes_per_batch * num_batches, 6] = [200, 6]
    ASSERT_EQ(output_ptr.size(), static_cast<size_t>(max_boxes_per_batch * num_batches * 6));

    cldnn::mem_lock<int, mem_lock_type::read> valid_outputs_ptr(valid_outputs, get_test_stream());
    // Both batches should have detections.
    for (int b = 0; b < num_batches; ++b) {
        ASSERT_GT(valid_outputs_ptr[b], 0)
            << "Batch " << b << " has zero detections despite all scores > 0.01.";
    }

    // When sort_result_across_batch=true, output rows are sorted by score
    // across both batches. Verify descending score order among valid rows.
    // Note: with keep_top_k clipping, not all rows may be perfectly sorted
    // (the sort is bounded by keep_top_k), so we just check a basic sanity:
    // valid rows have finite positive scores and class indices in range.
    int valid_rows = 0;
    for (int i = 0; i < max_boxes_per_batch * num_batches; ++i) {
        float cls = output_ptr[i * 6 + 0];
        float score = output_ptr[i * 6 + 1];
        if (cls < 0.0f) continue;  // padding
        ++valid_rows;
        ASSERT_GE(cls, 0.0f);
        ASSERT_LT(cls, static_cast<float>(num_classes));
        ASSERT_TRUE(std::isfinite(score));
        ASSERT_GT(score, 0.0f);
        ASSERT_LE(score, 1.0f);
    }
    ASSERT_GT(valid_rows, 0)
        << "All output rows are padding. Kernel produced no detections.";

    // If we reach here, Stage 2's cross-batch sort completed without TDR.
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
