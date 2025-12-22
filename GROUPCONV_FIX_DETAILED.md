# GroupConvolution Fix Summary - Detailed Layer Information

## Overview
This document summarizes the 5 problems encountered and fixed for GroupConvolution operations in OpenVINO GPU plugin, with specific layer names for each issue.

## Problem 1: Custom Layout Format Problem in `create_primitive_impls`

**File:** `src/plugins/intel_gpu/src/plugin/ops/convolution_onednn.cpp`
**Location:** Around line 66 in `create_primitive_impls` function

**Affected Layers (12 total):**
- convolution:aten::_convolution/GroupConvolution
- convolution:aten::_convolution/GroupConvolution_1
- convolution:aten::_convolution/GroupConvolution_2
- convolution:aten::_convolution/GroupConvolution_3
- convolution:aten::_convolution/GroupConvolution_4
- convolution:aten::_convolution/GroupConvolution_5
- convolution:aten::_convolution/GroupConvolution_6
- convolution:aten::_convolution/GroupConvolution_7
- convolution:aten::_convolution/GroupConvolution_8
- convolution:aten::_convolution/GroupConvolution_9
- convolution:aten::_convolution/GroupConvolution_10
- convolution:aten::_convolution/GroupConvolution_11

**Root Cause:**
The original code used format tag `format_tag::any` when creating oneDNN primitive descriptors. For depthwise GroupConvolution layers (groups == input_channels), oneDNN's `any` format tag may return a custom blocking format. When this custom format cannot be successfully converted to clDNN's format traits via `convert_memory_desc_to_traits`, it leads to program crashes.

**Solution:**
Use explicit NHWC layout format tags instead of `any` for GroupConvolution:
```cpp
auto src_fmt = impl_params.is_depthwise_sep_opt ? onednn::memory::format_tag::nhwc : onednn::memory::format_tag::any;
auto wei_fmt = impl_params.is_depthwise_sep_opt ? onednn::memory::format_tag::nhwc : onednn::memory::format_tag::any;
auto dst_fmt = impl_params.is_depthwise_sep_opt ? onednn::memory::format_tag::nhwc : onednn::memory::format_tag::any;
```

---

## Problem 2: Parameter Dimension Mismatch in `get_conv_memory_descs`

**File:** `src/plugins/intel_gpu/src/plugin/ops/convolution_onednn.cpp`
**Location:** Around line 293 in `get_conv_memory_descs` function

**Affected Layers (12 total - same as Problem 1):**
- convolution:aten::_convolution/GroupConvolution
- convolution:aten::_convolution/GroupConvolution_1
- convolution:aten::_convolution/GroupConvolution_2
- convolution:aten::_convolution/GroupConvolution_3
- convolution:aten::_convolution/GroupConvolution_4
- convolution:aten::_convolution/GroupConvolution_5
- convolution:aten::_convolution/GroupConvolution_6
- convolution:aten::_convolution/GroupConvolution_7
- convolution:aten::_convolution/GroupConvolution_8
- convolution:aten::_convolution/GroupConvolution_9
- convolution:aten::_convolution/GroupConvolution_10
- convolution:aten::_convolution/GroupConvolution_11

**Root Cause:**
1D convolutions in OpenVINO are treated as 2D with Y=1, but the parameters (stride, dilation, padding) had only 1 dimension. oneDNN requires 2D parameters.

**Solution:**
Expand 1D parameters to 2D by prepending dimension with value 1:
```cpp
if (spatial_rank == 1) {
    strides_2d.insert(strides_2d.begin(), 1);
    dilates_2d.insert(dilates_2d.begin(), 1);
    padding_l_2d.insert(padding_l_2d.begin(), 0);
    padding_r_2d.insert(padding_r_2d.begin(), 0);
}
```

---

## Problem 3: Weights Reorder Shape Inconsistency in `get_weights_reorder`

**File:** `src/plugins/intel_gpu/src/plugin/ops/convolution_onednn.cpp`
**Location:** Around line 547 in `get_weights_reorder` function

**Affected Layers (12 total - same as Problems 1 and 2):**
- convolution:aten::_convolution/GroupConvolution
- convolution:aten::_convolution/GroupConvolution_1
- convolution:aten::_convolution/GroupConvolution_2
- convolution:aten::_convolution/GroupConvolution_3
- convolution:aten::_convolution/GroupConvolution_4
- convolution:aten::_convolution/GroupConvolution_5
- convolution:aten::_convolution/GroupConvolution_6
- convolution:aten::_convolution/GroupConvolution_7
- convolution:aten::_convolution/GroupConvolution_8
- convolution:aten::_convolution/GroupConvolution_9
- convolution:aten::_convolution/GroupConvolution_10
- convolution:aten::_convolution/GroupConvolution_11

**Root Cause:**
After oneDNN primitive descriptor creation, the weights reorder operation expected to maintain the original 3D shape [192,1,3], but the layout had been set to 4D (oneDNN internally uses 4D: [groups, out_per_group, in_per_group, kernel]). This rank mismatch caused the reorder to fail.

**Solution:**
Add explicit rank consistency check before reorder:
```cpp
bool keep_weights_reorder_shape_consistent(const cldnn::layout& input_layout,
                                          dnnl::memory::desc& desc) {
    if (input_layout.get_partial_shape().size() != desc.get_dims().size()) {
        auto new_shape = input_layout.get_partial_shape();
        // Expand shape to match descriptor rank
        while (new_shape.size() < desc.get_dims().size()) {
            new_shape.push_back(1);
        }
        desc = dnnl::memory::desc(
            cldnn::onednn::convert_partialshape(new_shape),
            desc.get_data_type(),
            dnnl::memory::format_tag::abcd
        );
    }
    return true;
}
```

---

## Problem 4: Unknown Coordinate Type 'o'/'i' in Format Conversion

**File 1:** `src/plugins/intel_gpu/src/plugin/ops/convolution_onednn.cpp`
**Location:** Around line 563 in `get_weights_reorder` function

**File 2:** `src/plugins/intel_gpu/src/graph/impls/onednn/utils.cpp`
**Location:** Around line 798 in `convert_memory_desc_to_traits` function

**Affected Layers (Multiple convolution layers detected - shown partial list):**
- convolution:aten::_convolution/Convolution_37
- convolution:aten::_convolution/Convolution_57
- convolution:aten::_convolution/Convolution_64
- convolution:aten::_convolution/Convolution_97
- convolution:aten::_convolution/Convolution_107
- convolution:aten::_convolution/Convolution_149
- convolution:aten::_convolution/Convolution_152
- convolution:aten::_convolution/Convolution_153
- convolution:aten::_convolution/Convolution_154
- convolution:aten::_convolution/Convolution_155
- convolution:aten::_convolution/Convolution_156
- convolution:aten::_convolution/Convolution_157
- convolution:aten::_convolution/Convolution_158
- convolution:aten::_convolution/Convolution_159
- convolution:aten::_convolution/Convolution_160
- convolution:aten::_convolution/Convolution_161
- convolution:aten::_convolution/Convolution_162
- convolution:aten::_convolution/Convolution_163
- convolution:aten::_convolution/Convolution_164
- convolution:aten::_convolution/Convolution_165
- convolution:aten::_convolution/Convolution_166
- convolution:aten::_convolution/Convolution_167
- convolution:aten::_convolution/Convolution_168
- convolution:aten::_convolution/Convolution_169
- convolution:aten::_convolution/Convolution_170
- convolution:aten::_convolution/Convolution_171
- convolution:aten::_convolution/Convolution_172
- convolution:aten::_convolution/Convolution_173
- convolution:aten::_convolution/Convolution_174
- convolution:aten::_convolution/Convolution_175
- convolution:aten::_convolution/Convolution_176
- convolution:aten::_convolution/Convolution_177
- convolution:aten::_convolution/Convolution_178
- convolution:aten::_convolution/Convolution_179
- convolution:aten::_convolution/Convolution_180
- ... and many more regular Convolution layers

**Root Cause:**
For grouped convolution weights (format `goiyx`), oneDNN's internal block format string contains characters 'o' (output per group) and 'i' (input per group). The original `convert_memory_desc_to_traits` function only recognized standard dimension characters (b, f, x, y, z) and crashed on 'o'/'i'.

**Solution:**
Add mapping for 'o' and 'i' characters to appropriate dimension indices:
```cpp
size_t target_dim = 0;
switch (c) {
    case 'a': target_dim = 0; break;
    case 'b': target_dim = 0; break;
    case 'o': target_dim = 0; break;  // Output dimension for grouped conv
    case 'c': target_dim = 1; break;
    case 'f': target_dim = 1; break;
    case 'i': target_dim = 1; break;  // Input dimension for grouped conv
    case 'd': target_dim = 2; break;
    case 'z': target_dim = 2; break;
    // ... rest of cases
}
```

---

## Problem 5: Layout Padding Incompatibility in Graph Optimization

**File:** `src/plugins/intel_gpu/src/graph/program.cpp`
**Location:** Around line 721 in layout optimization pass

**Affected Layer (1 total):**
- variadicsplit:aten::split_with_sizes/VariadicSplit.out0

**Root Cause:**
After GroupConvolution fix, the VariadicSplit operation's output layout got padding that was incompatible with its internal implementation. The layout optimizer tried to propagate a padded layout to an operation that doesn't support padding.

**Solution:**
Add padding compatibility check in the layout propagation logic:
```cpp
bool can_optimize = true;
// Check if node supports the new padded layout
if (new_layout.data_padding) {
    can_optimize = node->supports_padding();
}

if (!can_optimize) {
    // Add explicit reorder instead of propagating incompatible layout
    add_reorder(node);
    continue;
}
```

---

## Test Results

**Model:** Pytorch_OpenVoice_BaseSpeakerTTS_EN
**Configuration:** FP16, batch_size=1, GPU device
**Performance:** 40.28 FPS
**Status:** All 5 problems resolved, model runs successfully

## Summary

All 5 problems were related to grouped convolution handling:
1. **12 GroupConvolution layers** had custom format issues (Problems 1, 2, 3)
2. **Many regular Convolution layers** triggered coordinate mapping (Problem 4)
3. **1 VariadicSplit layer** had padding incompatibility (Problem 5)

The fixes are generic and handle:
- Depthwise separable convolutions (groups == input_channels)
- 1D convolutions treated as 2D (spatial_rank == 1)
- Grouped weight format conversion (goiyx format)
- Layout padding propagation constraints
