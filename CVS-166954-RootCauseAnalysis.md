# CVS-166954 —— 根因分析与修复方案（中文详细版）

> **工单号：** CVS-166954
> **错误现象：** `[nightly][gpu] End input must be 1D (has rank: 0)`
> **受影响的模型：** `mask-rcnn-resnet50-fpn`
> **运行设备：** GPU
> **涉及组件：** `openvino_transformations`（公共图优化）
> **修复的文件：** `src/common/transformations/src/transformations/common_optimizations/move_eltwise_up_data_movement.cpp`

本文档面向**刚入门 OpenVINO 的同学**，把整个排查过程、debug 代码插在哪里、看到了什么输出、最终为什么这样改、以及修复是否安全通用，全部写清楚。每一步都有"小白解释 + 例子"。

---

## 0. 先用大白话讲清楚整个 bug

OpenVINO 有一种叫 **Pass（图优化）** 的东西，它会自动把模型里的算子做"等价变换"——比如把一个加法、乘法挪到更靠前的位置，让后面跑得更快。这次出问题的 Pass 叫：

> `MoveEltwiseUpThroughDataMovScalar`
> 直译：把"按元素运算"（如 Add / Mul / Min）**穿过**前面的"数据搬运算子"（如 Transpose / Reshape / Unsqueeze）**往前挪一格**。

它要挪算子的时候，需要把算子那个"标量常量"（比如一个数字 `800`）的形状从 `[1]` 改成 `[]`（真正的标量）。

**问题就出在这里**：原代码用的 API 是 **"全局替换"**——只要图里所有人引用了这个常量，**都会被改成新形状**。

打个比方🌰：

> **公司通讯录里有一个号码 `800`**，它被两个部门共用：
> - A 部门只想让这个号码"显示成 800"（不带区号），所以把通讯录里的 `[1]800` 改成了 `[]800`。
> - 但是 B 部门是"严格要求带区号才能拨号"的部门（比如 `StridedSlice` 算子要求 end 输入必须是 1 维向量）。
>
> A 部门一改全公司通讯录，B 部门拨号时就崩了：**"end 必须是 1D，但现在是 rank 0"**。

正确做法是：A 部门**自己抄一份**改成 `800`，**别动公司通讯录**。

---

## 1. 一句话总结

`MoveEltwiseUpThroughDataMovScalar` 用 `ov::replace_node_update_name(old_const, new_const)` 把一个 `[1]` 形状的常量改成了 `[]` 形状。这个 API 会**把所有用到这个常量的地方都替换**。当这个常量被另一个对秩（rank）敏感的消费者（比如 `Loop` body 里面的 `StridedSlice::end`）共享时，那个消费者的输入秩就被破坏了，后续 `ov::pass::Validate` 走到这里就报错：

```
End input must be 1D (has rank: 0)
```

修复方法是：**只把 eltwise 算子自己的那条输入边重新接到新标量常量上**，用 `eltwise->input(i).replace_source_output(new_constant->output(0))`，原来的 `[1]` 常量保持不动，其它消费者（StridedSlice）继续用原版。

---

## 2. 如何复现这个 bug

### 环境
- 分支：`cvs-166954-fix-end-input-issue`
- 工作目录：`/home/yazhan/zhan/openvino_gpu_plugin/openvino`
- Build 类型：本地 Debug 构建，输出在 `bin/intel64/Debug/`
- 模型：`models/mask-rcnn-resnet50-fpn/onnx/onnx/FP32/1/ov/mask-rcnn-resnet50-fpn.xml`

### 编译

```bash
cd /home/yazhan/zhan/openvino_gpu_plugin/openvino/build
make -j$(nproc) openvino_intel_gpu_plugin
```

### 复现命令

```bash
./bin/intel64/Debug/benchmark_app \
  -m models/mask-rcnn-resnet50-fpn/onnx/onnx/FP32/1/ov/mask-rcnn-resnet50-fpn.xml \
  -nstreams 1 -nireq 1 -b 1 -infer_precision f32 \
  -d GPU -hint none -inference_only=false -niter 1
```

修复前的报错（截取关键行）：

```
Check 'PartialShape::merge_into(...)' failed ...
End input must be 1D (has rank: 0)
```

修复后能正常加载、编译、推理结束。

---

## 3. 排查过程（debug 代码 + 看到的输出）

排查分四个阶段。**每一步都不靠猜，都有 debug 代码做证据**。所有 debug 代码都还**保留在仓库里**，由环境变量 `OV_DEBUG_CVS_166954` 控制开关，平时不会污染日志，方便你随时复现。

### 启用 debug 代码

```bash
export OV_DEBUG_CVS_166954=1
./bin/intel64/Debug/benchmark_app \
  -m models/mask-rcnn-resnet50-fpn/onnx/onnx/FP32/1/ov/mask-rcnn-resnet50-fpn.xml \
  -nstreams 1 -nireq 1 -b 1 -infer_precision f32 \
  -d GPU -hint none -inference_only=false -niter 1 \
  2>&1 | grep -E "DEBUG"
```

如果想跟踪别的节点（不是默认的 `Slice_2419`），可以再加一个变量：

```bash
export OV_DEBUG_CVS_166954_NODE=YourNodeFriendlyName
```

---

### 阶段 1：先找到"是哪一个 StridedSlice 出错了"

**问题：** 模型里有很多 `StridedSlice`，错误信息只说"end 不是 1D"，没说是哪一个。而且这个算子可能藏在 **Loop / TensorIterator 的 body 子图里**（小白解释：Loop 算子内部其实是一个独立的小模型，从外面遍历看不到里面）。

**Debug 代码插入位置：**
[src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp](src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp) — 在 `TransformationsPipeline::apply` 函数上方加了一个静态辅助函数：

```cpp
// CVS-166954 debug helper：根据 friendly_name 在主图和所有子图（Loop/TI/If body）
// 里找到目标节点，把它每个输入的 shape/rank 打出来。
// 用 OV_DEBUG_CVS_166954=1 启用，OV_DEBUG_CVS_166954_NODE=xxx 指定节点名。
static void debug_trace_node_by_name(const std::shared_ptr<ov::Model>& model,
                                     const std::string& stage) {
    if (!std::getenv("OV_DEBUG_CVS_166954"))
        return;
    const char* env_name = std::getenv("OV_DEBUG_CVS_166954_NODE");
    const std::string target = env_name ? env_name : "Slice_2419";
    std::function<void(const std::shared_ptr<ov::Model>&, const std::string&)> walk;
    walk = [&](const std::shared_ptr<ov::Model>& m, const std::string& scope) {
        for (const auto& node : m->get_ops()) {
            if (node->get_friendly_name() == target) {
                std::cerr << "[DEBUG][" << stage << "] '" << target
                          << "' type=" << node->get_type_name()
                          << " scope=" << scope << std::endl;
                for (size_t i = 0; i < node->get_input_size(); i++) {
                    auto in_node = node->input_value(i).get_node_shared_ptr();
                    std::cerr << "  input[" << i << "]: name='"
                              << in_node->get_friendly_name() << "' type="
                              << in_node->get_type_name()
                              << " et=" << node->get_input_element_type(i)
                              << " shape=" << node->get_input_partial_shape(i)
                              << std::endl;
                }
            }
            // 递归进 Loop / TensorIterator / If 的子图
            if (auto sg = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(node)) {
                for (size_t idx = 0; idx < sg->get_internal_subgraphs_size(); idx++) {
                    if (auto body = sg->get_function(static_cast<int>(idx)))
                        walk(body, scope + "/" + node->get_friendly_name() +
                                   "/body_" + std::to_string(idx));
                }
            }
        }
    };
    walk(model, "main");
}
```

然后在 `apply()` 函数里几个关键管线阶段后插入了调用（同样是 `OV_DEBUG_CVS_166954=1` 时才打印）。当时为了二分定位，依次插了：

```cpp
debug_trace_node_by_name(func, "INITIAL");                                   // 模型刚进来
// ... 主 manager.run_passes(func) 之后 ...
debug_trace_node_by_name(func, "AFTER_MAIN_MANAGER");
// ... UnrollTensorIterator manager 之后 ...
debug_trace_node_by_name(func, "AFTER_UnrollTensorIterator");
// ... PostUnrollConvertPrecision 之后 ...
debug_trace_node_by_name(func, "AFTER_PostUnrollConvertPrecision");
// ... ActivationsScaling manager 之后（在最后的 Validate 之前）...
debug_trace_node_by_name(func, "AFTER_ActivationsScaling_BEFORE_Validate");
```

> **当前仓库**：为了避免文件太脏，最终保留的是**入口处的 `INITIAL`** 一处调用，加上头部辅助函数。其它阶段的调用是临时排查用的，可以按上面格式自己加一两行就能复现。

**关键观察：**
- 出问题的 `StridedSlice` 实例 `friendly_name = "Slice_2419"`，**它在 `Loop_2299` 的 body 子图里**。
- 它的 `end`（input[2]）从 shape `[1]` 变成了 `[]`，**变化发生在 `ActivationsScaling` 这个 manager 阶段**。
- 喂给它 `end` 的常量名字是 `Unsqueeze_2377`，i32 类型，shape `[1]`，值是 `800`。

**小白解释：** 找到"嫌疑犯（Slice_2419）" 和"案发时间窗（ActivationsScaling 阶段）"。

---

### 阶段 2：在 ActivationsScaling 里二分到具体哪个 Pass

`ActivationsScaling` manager 里注册了一堆 Pass。我们用最朴素的办法：**逐个临时注释掉 Pass，看哪一个被注释掉以后，`Slice_2419` 的 `end` 就不变形了**。

最终锁定到这一行：

```cpp
manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMovScalar>(allowed_data_movement_ops);
```

**小白解释：** 二分查找是 debug 神器。100 个 Pass 也只用 7 步就能定位。

---

### 阶段 3：进入 Pass 内部，亲眼看到"凶器"动手

光知道是这个 Pass 还不够，要在 Pass 内部抓现行——看它在改哪个常量、原来的形状是什么、新形状是什么、这个常量被多少人共用。

**Debug 代码插入位置：**
[src/common/transformations/src/transformations/common_optimizations/move_eltwise_up_data_movement.cpp](src/common/transformations/src/transformations/common_optimizations/move_eltwise_up_data_movement.cpp) — 在那个修改 eltwise 常量形状的 `for` 循环里。**当前仓库已经保留了这段 debug 代码**（同样靠 `OV_DEBUG_CVS_166954` 开关），关键打印：

```cpp
if (std::getenv("OV_DEBUG_CVS_166954")) {
    std::cerr << "[DEBUG_MOVE_ELTWISE] Reshaping constant '"
              << old_eltwise_const->get_friendly_name()
              << "' from shape " << old_eltwise_const->get_shape()
              << " to scalar. Consumers: "
              << old_eltwise_const->get_output_target_inputs(0).size()
              << " eltwise='" << eltwise->get_friendly_name() << "'"
              << std::endl;
}
```

**抓到的"铁证"：**

```
[DEBUG_MOVE_ELTWISE] Reshaping constant 'Unsqueeze_2377'
    from shape [1] to scalar. Consumers: 2 eltwise='Minimum_20248'
```

这一行同时回答了所有问题：

| 问题 | 答案 |
|---|---|
| 改的是哪个常量？ | `Unsqueeze_2377` |
| 形状从什么改成什么？ | `[1]` → `[]`（标量） |
| 这个常量被几个算子共用？ | **2 个**（这是关键！） |
| 这个 Pass 当前匹配的 eltwise 是哪个？ | `Minimum_20248` |

**小白解释：** 一个常量被 2 个算子共享，但是 Pass 里用的 `replace_node_update_name` 会**把所有 2 个使用者都改掉**，结果"无辜的"那个 `Slice_2419` 就被牵连了。

---

### 阶段 4：去看 OpenVINO API 源码确认行为

不能光凭"我以为这个 API 是这样的"。直接去看了：

- **`ov::replace_node_update_name(old, new)`**：内部会调用 `replace_node(old, new)`，**遍历 old 节点每个 output 的每个 target_input，全部重新接到 new 的对应 output**。然后再把 friendly_name 转移过去。
  → **这是"全局替换"**。
- **`Input<Node>::replace_source_output(out)`**：**只重接当前这一条输入边**。
  → **这是"局部替换"**。

**小白比喻：**

> - `replace_node_update_name`：**把全公司通讯录里"800 这个号码"统一改成新的**——所有部门都受影响。
> - `replace_source_output`：**只在你自己手机里把"800"改名**——其他人手机不动。

我们这个场景：**只想给 eltwise 自己用一份新形状**，根本不想改其它共用者。所以应该用 `replace_source_output`，原代码用错了 API。

---

## 4. 根因（结论）

`MoveEltwiseUpThroughDataMovScalar` 为了把 eltwise 算子从 data-movement 算子（如 Transpose / Unsqueeze）下面挪到上面，需要把 eltwise 那个标量样的常量输入从 `[1]` 改成 `[]`。原代码用了**全局替换** `ov::replace_node_update_name`，但**新的标量形状只对 eltwise 自己合法**。

只要这个常量同时被另一个对输入秩有严格要求的节点共享（例如 `StridedSlice` 的 `begin`/`end`/`strides`、`Broadcast` 的 `target_shape`、`Reshape` 的 `shape_pattern`、`Gather` 的 `indices` 等），那个节点就会被悄悄破坏。

`mask-rcnn-resnet50-fpn` 这个模型的拓扑刚好踩到了这个组合：

```
                 Unsqueeze_2377  (i32, shape=[1], value=800)
                       │
       ┌───────────────┴────────────────────────────┐
       │                                            │
       ▼                                            ▼
   Minimum_20248                       Slice_2419::end (in Loop_2299 body)
   （eltwise，被 Pass 匹配，需要标量）   （StridedSlice，要求 end 是 1D）
```

这是一个**潜伏已久的 bug**，只是没有别的模型踩到这个共享拓扑而已。

---

## 5. 修复方案

**修改文件：** [src/common/transformations/src/transformations/common_optimizations/move_eltwise_up_data_movement.cpp](src/common/transformations/src/transformations/common_optimizations/move_eltwise_up_data_movement.cpp)

```cpp
// eltwise 的常量输入形状要和新的位置上游算子匹配
for (size_t i = 1; i < eltwise->get_input_size(); i++) {
    if (current->get_output_partial_shape(0).size() != eltwise->get_input_partial_shape(i).size()) {
        auto old_eltwise_const = ov::as_type_ptr<ov::opset8::Constant>(eltwise->get_input_node_shared_ptr(i));
        if (old_eltwise_const->get_shape().size() != 0) {
            auto new_constant = std::make_shared<ov::opset8::Constant>(*old_eltwise_const.get(), ov::Shape{});
            copy_runtime_info(old_eltwise_const, new_constant);
            // 只重接当前 eltwise 这条输入边，不要全局替换。
            // 因为这个常量可能被别的消费者共享（例如 Loop body 里
            // StridedSlice 的 'end' 输入），那些消费者要求保留原来的
            // 非标量形状。如果用全局替换，就会把它们的输入秩破坏掉，
            // 触发 "End input must be 1D (has rank: 0)" 这种校验失败。
            eltwise->input(i).replace_source_output(new_constant->output(0));
        }
    }
}
```

### 为什么这是"通用、专业"的修复（不是只为这一个模型 hack）

1. **改动范围最小化。** 只改 eltwise 自己那一条边，不动任何其它人。
2. **单消费者 / 多消费者两种场景都对。**
   - **单消费者：** 老常量没人用了，C++ 智能指针会自动回收。最终图等价于原来 `replace_node_update_name` 的效果（仅丢失一次"friendly_name 复制"的修饰，无功能影响）。
   - **多消费者：** 其它消费者继续用老常量（`[1]` 形状），完全符合预期。
3. **不会让其它模型变慢或出错。** 算子前移这件事还是照常完成（下面的 `replace_output_update_name` + `clone_with_new_inputs` 没动），**优化照样生效**——我们没有去削弱匹配条件、没有让 Pass 跳过这种情况。
4. **和姊妹 Pass 一致。** 同文件的 `MoveEltwiseUpThroughDataMovPerChannel` 在 matcher 里就用 `output.get_target_inputs().size() == 1` 拦掉了共享常量；标量版本因为可以"私下复制一份标量"，所以选择在 callback 里安全地处理共享场景——本修复正是这个思路的标准实现。
5. **内存代价可以忽略。** 多创建的常量只是**一个标量值（≤ 8 字节）**，跟把 eltwise 前移带来的运行时收益相比微不足道。
6. **保留 runtime info。** `copy_runtime_info(old, new)` 把原来的 attribute（去量化标记、解压标记等）都拷给新常量，不会丢信息。

### 其它备选方案（为什么没选）

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| **A. matcher 直接拒绝匹配多消费者常量** | 最简单 | **正常合法的优化场景也会被砍掉**，性能损失 | 太保守 |
| **B. 单消费者还用 `replace_node_update_name`，多消费者才走新分支** | 单消费者保留 friendly_name | 多了一个 if，认知负担大；行为和 C 几乎一样 | 被 C 替代 |
| **C. 统一用 `replace_source_output`（已选）** | 最小改动、统一逻辑、所有场景都对 | 单消费者场景下不再做 friendly_name 转移（纯装饰，无功能影响） | **采用** |
| **D. 把检查放进 matcher predicate** | 更早过滤 | matcher 阶段还无法判断"哪个 input 会被改成标量"，只能过严或过松 | 不可行 |

---

## 6. 回归测试

**测试文件：** [src/common/transformations/tests/common_optimizations/move_eltwise_up_data_movement_test.cpp](src/common/transformations/tests/common_optimizations/move_eltwise_up_data_movement_test.cpp)
**测试名：** `MoveEltwiseUpThroughDataMovTest.SharedConstantNotReshapedForOtherConsumers`

测试构造了**最小复现拓扑**：

```
              shared_const (i64, shape=[1], value=2)
                     │
       ┌─────────────┴──────────────────────┐
       │                                    │
       ▼                                    ▼
   Multiply (eltwise)             StridedSlice::end (要求 rank=1)
       ▲
   Unsqueeze ← Transpose ← Param
```

测试通过 `model_ref` 断言：
- eltwise 已经被挪到 `Transpose` 和 `Unsqueeze` 上面（优化生效）。
- eltwise 的常量输入是**新的 `[]` 形状常量**。
- `StridedSlice` 的 `end` 输入仍然是**原版 `[1]` 形状常量**。

> **复现前后对比：** 这个测试在原代码上**会失败**（StridedSlice 被破坏），在修复后代码上**通过**。这是"防止以后再写错"的护栏。

### 运行测试

```bash
cd /home/yazhan/zhan/openvino_gpu_plugin/openvino
./bin/intel64/Debug/ov_transformations_tests \
    --gtest_filter='MoveEltwiseUpThroughDataMovTest.*'
```

输出：

```
[ RUN      ] MoveEltwiseUpThroughDataMovTest.SharedConstantNotReshapedForOtherConsumers
[       OK ] MoveEltwiseUpThroughDataMovTest.SharedConstantNotReshapedForOtherConsumers (0 ms)
...
[==========] 18 tests from 1 test suite ran. (9 ms total)
[  PASSED  ] 18 tests.
```

整个 suite 18 个测试全过——**没有任何老场景被搞坏**。

---

## 7. 端到端模型验证

```bash
./bin/intel64/Debug/benchmark_app \
  -m models/mask-rcnn-resnet50-fpn/onnx/onnx/FP32/1/ov/mask-rcnn-resnet50-fpn.xml \
  -nstreams 1 -nireq 1 -b 1 -infer_precision f32 -d GPU \
  -hint none -inference_only=false -niter 1
```

修复后输出：

```
[ INFO ] First inference took 824.15 ms
[ INFO ] Count:    1 iterations
[ INFO ] Duration: 764.00 ms
[ INFO ] Throughput: 1.31 FPS
```

无异常，无校验失败。

---

## 8. 改动文件清单

| 文件 | 修改内容 |
|---|---|
| [src/common/transformations/src/transformations/common_optimizations/move_eltwise_up_data_movement.cpp](src/common/transformations/src/transformations/common_optimizations/move_eltwise_up_data_movement.cpp) | 用 `replace_source_output` + `copy_runtime_info` 替换原来的全局 `replace_node_update_name`；加注释说明原因；保留环境变量门控的 debug 打印用于复现。 |
| [src/common/transformations/tests/common_optimizations/move_eltwise_up_data_movement_test.cpp](src/common/transformations/tests/common_optimizations/move_eltwise_up_data_movement_test.cpp) | 加 `#include "openvino/op/strided_slice.hpp"`；新增回归测试 `SharedConstantNotReshapedForOtherConsumers`。 |
| [src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp](src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp) | 加 `debug_trace_node_by_name` 静态辅助函数和 `INITIAL` 阶段调用；用 `OV_DEBUG_CVS_166954` 环境变量门控，关闭时零开销，方便后人复现。 |

---

## 9. 经验总结（review 时也可以参考的点）

- **在 matcher callback 里，如果你的修改只想影响"被匹配的这个节点"，调用 `replace_node*` 之前先看 `get_output_target_inputs(0).size()`。** `replace_node` / `replace_node_update_name` **是全局替换**。
- **要"只换一条输入边"就用 `Input<Node>::replace_source_output`。** 这是 OpenVINO 里"局部重接"的标准用法。
- **Loop / TensorIterator / If 的 body 子图里藏着的消费者，从外层图一眼是看不到的。** debug 形状变化的时候，记得 `dynamic_pointer_cast<MultiSubGraphOp>` 然后递归进 body。被破坏的消费者经常就在那里。
- **"共享常量被静默改形状"是经典的潜伏 bug。** 给 Pass 写一个"显式共享常量给一个对秩敏感的节点（StridedSlice / Broadcast / Reshape / Gather）"的单元测试，是最便宜也最持久的护栏。
- **debug 代码不是用完就扔。** 把它们用环境变量门控保留下来（关闭时零代价），下次再出类似问题，新人一行命令就能复现，不用从头再"猜在哪打 log"。
