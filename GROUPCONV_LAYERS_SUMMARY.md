# GroupConvolution Fix - Layer Names Summary

## Quick Reference

### ERROR 1 & 2 & 3: GroupConvolution Layers (12 layers)
These 12 layers all had the same 3 problems:
- **ERROR 1**: Custom layout format problem
- **ERROR 2**: Parameter dimension mismatch  
- **ERROR 3**: Weights reorder shape inconsistency

**Layer Names:**
```
convolution:aten::_convolution/GroupConvolution
convolution:aten::_convolution/GroupConvolution_1
convolution:aten::_convolution/GroupConvolution_2
convolution:aten::_convolution/GroupConvolution_3
convolution:aten::_convolution/GroupConvolution_4
convolution:aten::_convolution/GroupConvolution_5
convolution:aten::_convolution/GroupConvolution_6
convolution:aten::_convolution/GroupConvolution_7
convolution:aten::_convolution/GroupConvolution_8
convolution:aten::_convolution/GroupConvolution_9
convolution:aten::_convolution/GroupConvolution_10
convolution:aten::_convolution/GroupConvolution_11
```

### ERROR 4: Unknown Coordinate 'o'/'i' (Many Convolution layers)
This error was triggered by many regular Convolution layers when oneDNN tried to convert grouped weight format.

**Sample Layer Names (partial list):**
```
convolution:aten::_convolution/Convolution_37
convolution:aten::_convolution/Convolution_57
convolution:aten::_convolution/Convolution_64
convolution:aten::_convolution/Convolution_97
convolution:aten::_convolution/Convolution_107
convolution:aten::_convolution/Convolution_149
convolution:aten::_convolution/Convolution_152
convolution:aten::_convolution/Convolution_153
convolution:aten::_convolution/Convolution_154
convolution:aten::_convolution/Convolution_155
convolution:aten::_convolution/Convolution_156
convolution:aten::_convolution/Convolution_157
convolution:aten::_convolution/Convolution_158
convolution:aten::_convolution/Convolution_159
convolution:aten::_convolution/Convolution_160
convolution:aten::_convolution/Convolution_161
convolution:aten::_convolution/Convolution_162
convolution:aten::_convolution/Convolution_163
convolution:aten::_convolution/Convolution_164
convolution:aten::_convolution/Convolution_165
convolution:aten::_convolution/Convolution_166
convolution:aten::_convolution/Convolution_167
convolution:aten::_convolution/Convolution_168
convolution:aten::_convolution/Convolution_169
convolution:aten::_convolution/Convolution_170
convolution:aten::_convolution/Convolution_171
convolution:aten::_convolution/Convolution_172
convolution:aten::_convolution/Convolution_173
convolution:aten::_convolution/Convolution_174
convolution:aten::_convolution/Convolution_175
convolution:aten::_convolution/Convolution_176
convolution:aten::_convolution/Convolution_177
convolution:aten::_convolution/Convolution_178
convolution:aten::_convolution/Convolution_179
convolution:aten::_convolution/Convolution_180
... and many more
```

### ERROR 5: Layout Padding Incompatibility (1 layer)
**Layer Name:**
```
variadicsplit:aten::split_with_sizes/VariadicSplit.out0
```

---

## How to Use This Information

### Search in Dumped Graph
When you dump the OpenVINO execution graph, you can search for these exact layer names to locate the problematic operations.

Example using `OV_GPU_Verbose=2`:
```bash
grep "GroupConvolution" graph_dump.txt
grep "VariadicSplit.out0" graph_dump.txt
```

### Correlation with Fixes
- **Errors 1, 2, 3**: All occur in the same 12 GroupConvolution layers
  - Fixed in: `convolution_onednn.cpp` (3 different locations)
  
- **Error 4**: Occurs in many regular Convolution layers  
  - Fixed in: `utils.cpp` (coordinate mapping)
  
- **Error 5**: Occurs in 1 VariadicSplit layer
  - Fixed in: `program.cpp` (layout optimizer)

---

## Model Information
- **Model**: Pytorch_OpenVoice_BaseSpeakerTTS_EN
- **Total GroupConvolution layers with issues**: 12
- **Total regular Convolution layers affected**: 40+
- **Other affected layers**: 1 VariadicSplit
- **Performance after fix**: 40.28 FPS (FP16, batch=1, GPU)
