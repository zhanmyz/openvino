// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

// When input is f16, values exceeding 65504 overflow to INF. Clamping INF back
// to +/-HALF_MAX prevents NaN in RMS normalization (INF * rsqrt(INF) = INF*0 = NaN).
#if INPUT0_TYPE_SIZE == 2
#define RMS_CLAMP(val) clamp((val), (ACCUMULATOR_TYPE)(-65504.0f), (ACCUMULATOR_TYPE)(65504.0f))
#else
#define RMS_CLAMP(val) (val)
#endif

KERNEL(rms_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if ELEMENTWISE_AFFINE
    const __global INPUT1_TYPE* gamma,
#endif
    __global OUTPUT_TYPE* output
    #if HAS_FUSED_OPS_DECLS
        , FUSED_OPS_DECLS
    #endif
)
{
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);
    const uint w = 0;

    ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;
    for (uint z = 0; z < INPUT0_SIZE_Z; z++) {
        for (uint y = 0; y < INPUT0_SIZE_Y; y++) {
            for (uint x = 0; x < INPUT0_SIZE_X; x++) {
                const uint input_idx = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
                ACCUMULATOR_TYPE val = RMS_CLAMP(TO_ACCUMULATOR_TYPE(input[input_idx]));
                rms += pow(val, 2);
            }
        }
    }

    rms /= INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_SIZE_Z;
    rms = pow(sqrt(rms + TO_ACCUMULATOR_TYPE(EPSILON)), -1);

    for (uint z = 0; z < INPUT0_SIZE_Z; z++) {
        for (uint y = 0; y < INPUT0_SIZE_Y; y++) {
            for (uint x = 0; x < INPUT0_SIZE_X; x++) {
                const uint input_idx = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
                const uint output_idx = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
                ACCUMULATOR_TYPE input_val = RMS_CLAMP(TO_ACCUMULATOR_TYPE(input[input_idx]));
#if ELEMENTWISE_AFFINE
#if INPUT0_DIMS == 4
                const uint gamma_idx = y;
#elif INPUT0_DIMS == 5
                const uint gamma_idx = z;
#endif
                OUTPUT_TYPE result = TO_OUTPUT_TYPE(rms) * TO_OUTPUT_TYPE(input_val) * TO_OUTPUT_TYPE(gamma[gamma_idx]);
#else
                OUTPUT_TYPE result = TO_OUTPUT_TYPE(rms) * TO_OUTPUT_TYPE(input_val);
#endif
                #if HAS_FUSED_OPS
                    FUSED_OPS;
                    result = FUSED_OPS_RESULT;
                #endif
                output[output_idx] = result;
            }
        }
    }
}
