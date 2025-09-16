// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/scatter_elements_update.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ScatterElementsUpdateLayerTest;
using ov::test::ScatterElementsUpdate12LayerTest;

// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape {
    {{10, 12, 15}, {{{1, 2, 4}, {0, 1, 2}}, {{2, 2, 2}, {-1, -2, -3}}}},
    {{15, 9, 8, 12}, {{{1, 2, 2, 2}, {0, 1, 2, 3}}, {{1, 2, 1, 4}, {-1, -2, -3, -4}}}},
    {{9, 9, 8, 8, 11, 10}, {{{1, 2, 1, 2, 1, 2}, {5, -3}}}},
};

// index value should not be random data
const std::vector<std::vector<size_t>> idxValue = {
        {1, 0, 4, 6, 2, 3, 7, 5}
};

const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

const std::vector<ov::element::Type> idxPrecisions = {
        ov::element::i32,
        ov::element::i64,
};

std::vector<ov::test::axisShapeInShape> combine_shapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::axisShapeInShape> res_vec;
    for (auto& input_shape : input_shapes) {
        for (auto& item : input_shape.second) {
            for (auto& elt : item.second) {
                res_vec.push_back(ov::test::axisShapeInShape{
                    ov::test::static_shapes_to_test_representation({input_shape.first, item.first}),
                    elt});
            }
        }
    }
    return res_vec;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterEltsUpdate,
    ScatterElementsUpdateLayerTest,
    ::testing::Combine(::testing::ValuesIn(combine_shapes(axesShapeInShape)),
                       ::testing::ValuesIn(idxValue),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterElementsUpdateLayerTest::getTestCaseName);


const std::vector<ov::op::v12::ScatterElementsUpdate::Reduction> reduceModes{
    // Reduction::NONE is omitted intentionally, because v12 with Reduction::NONE is converted to v3,
    // and v3 is already tested by smoke_ScatterEltsUpdate testsuite. It doesn't make sense to test the same code twice.
    // Don't forget to add Reduction::NONE when/if ConvertScatterElementsUpdate12ToScatterElementsUpdate3
    // transformation will be disabled (in common transforamtions pipeline or for GPU only).
    ov::op::v12::ScatterElementsUpdate::Reduction::SUM,
    ov::op::v12::ScatterElementsUpdate::Reduction::PROD,
    ov::op::v12::ScatterElementsUpdate::Reduction::MIN,
    ov::op::v12::ScatterElementsUpdate::Reduction::MAX,
    ov::op::v12::ScatterElementsUpdate::Reduction::MEAN
};

const std::vector<std::vector<int64_t>> idxWithNegativeValues = {
    {1, 0, 4, 6, 2, 3, 7, 5},
    {-1, 0, -4, -6, -2, -3, -7, -5},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterEltsUpdate12,
    ScatterElementsUpdate12LayerTest,
    ::testing::Combine(::testing::ValuesIn(combine_shapes(axesShapeInShape)),
                       ::testing::ValuesIn(idxWithNegativeValues),
                       ::testing::ValuesIn(reduceModes),
                       ::testing::ValuesIn({true, false}),
                       ::testing::Values(inputPrecisions[0]),
                       ::testing::Values(idxPrecisions[0]),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterElementsUpdate12LayerTest::getTestCaseName);

// ============================================================================
// Large batch size test cases for LWS constraint validation (CVS-171210)
// ============================================================================
// These test cases are SEPARATE from regular functional tests to:
// 1. Avoid impacting CI/CD performance (large batch tests are time-consuming)
// 2. Provide isolated testing for LWS constraint fixes
// 3. Allow independent debugging of batch size vs functional issues
// 4. Target batch_size > maxWorkGroupSize (1024) specifically

// Test case 1: batch_size=2048, feature axis - most common case
const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> batch2048Shapes {
    {{2048, 4, 1, 1}, {{{2048, 2, 1, 1}, {1}}}},
};

const std::vector<std::vector<int64_t>> batch2048Indices = {
    std::vector<int64_t>(2048 * 2, 0),  // All target index 0 for basic test
    std::vector<int64_t>(2048 * 2, 2),  // All target index 2 for SUM reduction test
};

// Test case 2: batch_size=1536, feature axis - reduction accuracy test
// This tests reduction mode accuracy when multiple indices target same location
const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> batch1536Shapes {
    {{1536, 4, 1, 1}, {{{1536, 2, 1, 1}, {1}}}},
};

const std::vector<std::vector<int64_t>> batch1536Indices = {
    std::vector<int64_t>(1536 * 2, 2),  // Both target same index 2 for reduction test (same as unit test)
};

// Test case 3: batch_size=1280, Y axis - different tensor dimension layout
// This tests LWS constraint on different tensor dimension layouts
const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> batch1280Shapes {
    {{1280, 1, 2, 2}, {{{1280, 1, 2, 1}, {2}}}},
};

const std::vector<std::vector<int64_t>> batch1280Indices = {
    []() {
        constexpr size_t total_size = 1280 * 1 * 2 * 1;
        std::vector<int64_t> indices;
        indices.reserve(total_size);

        // Only generate pattern for first few elements, then fill efficiently (like unit test)
        constexpr size_t pattern_limit = 100;  // Only compute first 100 elements
        for (size_t i = 0; i < std::min(total_size, pattern_limit); ++i) {
            indices.push_back(i % 2);  // Alternating 0,1 pattern for Y dimension
        }

        // Fill remaining elements with pattern[0] = 0 for efficiency
        if (total_size > pattern_limit) {
            indices.resize(total_size, 0);
        }

        return indices;
    }()
};

// Primary data types for accuracy validation
const std::vector<ov::element::Type> lwsTestPrecisions = {
    ov::element::f32,  // Primary precision
    ov::element::f16,  // GPU-optimized precision
};

// Separate test suites for each batch size to avoid parameter mismatch
INSTANTIATE_TEST_SUITE_P(
    lws_constraint_batch2048_ScatterEltsUpdate12,
    ScatterElementsUpdate12LayerTest,
    ::testing::Combine(::testing::ValuesIn(combine_shapes(batch2048Shapes)),
                       ::testing::ValuesIn(batch2048Indices),
                       ::testing::ValuesIn(reduceModes),  // Reuse existing full reduction modes
                       ::testing::Values(true),  // use_init_value = true for accuracy
                       ::testing::ValuesIn(lwsTestPrecisions),
                       ::testing::Values(idxPrecisions[0]),   // i32 indices only
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterElementsUpdate12LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    lws_constraint_batch1536_ScatterEltsUpdate12,
    ScatterElementsUpdate12LayerTest,
    ::testing::Combine(::testing::ValuesIn(combine_shapes(batch1536Shapes)),
                       ::testing::ValuesIn(batch1536Indices),
                       ::testing::ValuesIn(reduceModes),  // Reuse existing full reduction modes
                       ::testing::Values(true),  // use_init_value = true for accuracy
                       ::testing::ValuesIn(lwsTestPrecisions),
                       ::testing::Values(idxPrecisions[0]),   // i32 indices only
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterElementsUpdate12LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    lws_constraint_batch1280_ScatterEltsUpdate12,
    ScatterElementsUpdate12LayerTest,
    ::testing::Combine(::testing::ValuesIn(combine_shapes(batch1280Shapes)),
                       ::testing::ValuesIn(batch1280Indices),
                       ::testing::ValuesIn(reduceModes),  // Reuse existing full reduction modes
                       ::testing::Values(true),  // use_init_value = true for accuracy
                       ::testing::ValuesIn(lwsTestPrecisions),
                       ::testing::Values(idxPrecisions[0]),   // i32 indices only
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterElementsUpdate12LayerTest::getTestCaseName);
}  // namespace
