// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/utils.hpp"
#include "pooling_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils.hpp"

#include <memory>

namespace cldnn {
namespace onednn {

struct PoolingImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::pool")
    PoolingImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::onednn, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<pooling>());
        const auto& config = node.get_program().get_config();
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown || !config.get_use_onednn())
            return false;

        const auto& in_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        auto in_dt = in_layout.data_type;
        auto out_dt = out_layout.data_type;

        if (in_layout.data_padding || out_layout.data_padding)
            return false;

        static const std::vector<format::type> supported_formats = {
            format::any,
            format::bfyx,
            format::bfzyx,
            format::byxf,
            format::bzyxf,
            format::b_fs_yx_fsv8,
            format::b_fs_zyx_fsv8,
            format::b_fs_yx_fsv16,
            format::b_fs_zyx_fsv16,
            format::b_fs_yx_fsv32,
            format::b_fs_zyx_fsv32,
            format::bs_fs_yx_bsv4_fsv2,
            format::bs_fs_yx_bsv4_fsv4,
            format::bs_fs_yx_bsv8_fsv2,
            format::bs_fs_zyx_bsv8_fsv2,
            format::bs_fs_yx_bsv8_fsv4,
            format::bs_fs_zyx_bsv8_fsv4,
            format::bs_fs_yx_bsv16_fsv2,
            format::bs_fs_zyx_bsv16_fsv2,
            format::bs_fs_yx_bsv16_fsv4,
            format::bs_fs_zyx_bsv16_fsv4,
            format::bs_fs_yx_bsv16_fsv8,
            format::bs_fs_zyx_bsv16_fsv8,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::bs_fs_zyx_bsv32_fsv32,
        };

        bool fp_case = everyone_is(ov::element::f16, in_dt, out_dt);
        bool u8s8_case = one_of(in_dt, {ov::element::i8, ov::element::u8}) &&
                         one_of(out_dt, {ov::element::i8, ov::element::u8, ov::element::f32, ov::element::f16});

        if (!fp_case && !u8s8_case)
            return false;

        if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value, supported_formats))
            return false;

        if (!is_supported_post_ops(node))
            return false;

        // Reject pooling configurations that may cause register allocation failures in oneDNN JIT compiler.
        // oneDNN's JIT pooling kernel may fail with "out_of_registers_exception" for:
        // - Low precision types (FP16/BF16/INT8) with large channel counts
        // - This affects all formats, but blocked formats (fsv16/fsv32) are more susceptible
        //
        // Background:
        // - oneDNN has retry logic (cfg_.cut()) for register spills, but it may fail for extreme cases
        // - When oneDNN fails to create primitive, OpenVINO catches it as "Not Implemented"
        // - OCL kernel fallback provides reliable execution for unsupported configurations
        //
        // Register usage estimation:
        // - GPU architectures have limited General Register Files (GRF)
        // - Gen9-Gen12 (xe_lp): 128 GRF × 32 bytes = 4096 bytes
        // - Xe-HP+ with large_grf_mode: 256 GRF × 32 bytes = 8192 bytes
        // - Xe-HP+ without large_grf_mode: 128 GRF × 32 bytes = 4096 bytes
        // - Kernel overhead reduces available space to ~50% for data
        // - Blocked formats (fsv16/fsv32) have higher register pressure due to vectorized access
        const int64_t channels = in_layout.get_tensor().feature[0];
        const int type_size = static_cast<int>(ov::element::Type(in_dt).size());

        // Calculate a conservative channel threshold based on GPU architecture and data type
        // This prevents attempting primitive creation that will fail in oneDNN
        int64_t max_safe_channels = std::numeric_limits<int64_t>::max();

        // For low-precision types (2 bytes or less), estimate safe channel limit
        if (type_size <= 2) {
            // Determine GRF count based on GPU architecture
            // large_grf_mode can be inferred from num_threads_per_eu:
            // - xe_lp: 7 threads (always 128 GRF)
            // - xe_hp+: 4 threads = large_grf (256 GRF), 8 threads = normal mode (128 GRF)
            int grf_count = 128;  // Default for Gen9-Gen12 and Xe-HP+ without large_grf
            if (info.arch != gpu_arch::unknown) {
                // For Xe-HP and newer, check if large_grf_mode is active
                const bool is_xe_hp_or_newer = one_of(info.arch, {
                    gpu_arch::xe_hp, gpu_arch::xe_hpg, gpu_arch::xe_hpc,
                    gpu_arch::xe2, gpu_arch::xe3
                });

                if (is_xe_hp_or_newer && info.num_threads_per_eu == static_cast<uint32_t>(4)) {
                    grf_count = 256;  // large_grf_mode active
                }
            }

            const int64_t grf_size_bytes = 32;  // Each GRF is 32 bytes on all Intel GPUs
            const int64_t total_grf_bytes = grf_count * grf_size_bytes;

            // Reserve ~50% for kernel overhead (spills, temporaries, control flow)
            const int64_t usable_grf_bytes = total_grf_bytes / 2;

            // For blocked formats, account for higher vectorization overhead
            // fsv16 processes 16 channels, fsv32 processes 32 channels per SIMD lane
            bool is_blocked_format = one_of(in_layout.format.value, {
                format::b_fs_yx_fsv16, format::b_fs_zyx_fsv16,
                format::b_fs_yx_fsv32, format::b_fs_zyx_fsv32,
                format::bs_fs_yx_bsv16_fsv16, format::bs_fs_zyx_bsv16_fsv16,
                format::bs_fs_yx_bsv16_fsv32, format::bs_fs_zyx_bsv16_fsv32,
                format::bs_fs_yx_bsv32_fsv16, format::bs_fs_zyx_bsv32_fsv16,
                format::bs_fs_yx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv32
            });

            // Blocked formats require more registers for vectorized operations
            // Plain formats have lower register pressure but can still fail
            int vectorization_factor = is_blocked_format ? 4 : 2;
            max_safe_channels = usable_grf_bytes / (type_size * vectorization_factor);
        }

        if (channels > max_safe_channels) {
            // Reject to allow fallback to OCL kernel implementation
            return false;
        }

        return true;
    }
};

}  // namespace onednn
}  // namespace cldnn
