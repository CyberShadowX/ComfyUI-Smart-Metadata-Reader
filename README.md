# ComfyUI-Smart-Metadata-Reader

语言 / Language: 中文 | [English](README_EN.md)

`ComfyUI-Smart-Metadata-Reader` 是一个 Python-only 的 ComfyUI 自定义节点。它的主节点 `Smart Metadata Reader` 是图片加载器型节点，可以选择或上传图片，读取图片里已经保存的 AI 生成 metadata，并输出图片、遮罩、真实正负提示词和常用生成参数。

## 项目简介

这个节点不是“看图反推提示词”。它不会分析图片内容来猜 prompt，而是读取图片文件里已有的 metadata，尤其是 ComfyUI 生成 PNG 中保存的 `prompt` 和 `workflow` JSON。

它的核心目标是处理复杂 ComfyUI workflow：从最终 selected sampler 的 `positive` / `negative` conditioning 链路反向追踪，提取真正进入最终生图步骤的正面提示词、负面提示词、模型、VAE、LoRA、seed、steps、CFG、sampler、scheduler 和尺寸等信息。

## 解决什么问题

普通 prompt reader 往往只能读取简单的 A1111 参数，或者在 ComfyUI workflow 里直接查找 `CLIPTextEncode.inputs.text`。但复杂 workflow 里，`CLIPTextEncode.inputs.text` 可能是上游链接，例如字符串拼接节点、ShowText 缓存、翻译节点、ControlNet conditioning、detailer 或其他 custom node。

本项目把 ComfyUI metadata 当成 graph 解析，而不是把 workflow 里所有文本字段扫一遍。

## 与普通 Prompt Reader 的区别

- 从最终输出图片链路选择 sampler，而不是随便取最后一个节点。
- positive / negative 只从 selected sampler 的 conditioning 链路解析。
- 不扫描整个 workflow 的所有 `text` 字段。
- 不会把 ControlNet 的 image、strength、control_net、model_name 等参数混进 prompt。
- 不会把 Gemini / OpenAI / ChatGPT / Claude / LLM 节点的 `system_instruction`、`system_prompt`、`prompt`、`instruction`、`messages` 这类模板当成最终提示词。
- `ShowText|pysssss` 有 `text_0` 缓存结果时，会优先使用缓存结果。
- 不过滤 LoRA trigger words、角色词、风格词、NSFW 词、质量词、翻译文本或用户手写标签。

## 安装方式

进入你的 ComfyUI 安装目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CyberShadowX/ComfyUI-Smart-Metadata-Reader.git
cd ComfyUI-Smart-Metadata-Reader
pip install -r requirements.txt
```

安装后重启 ComfyUI。

## 使用方式

1. 在 ComfyUI 中添加节点 `Smart Metadata Reader`。
2. 从 ComfyUI `input` 目录选择图片，或通过图片控件上传图片。
3. 使用节点输出连接后续节点，或直接查看输出文本。

## 节点参数说明

- `parameter_index`：用于 A1111 / Forge 多组参数 fallback 时选择第几组参数。普通使用保持 `0`。
- `prefer_cached_text`：优先使用 `ShowText` 等节点保存的缓存结果，避免把 Gemini / OpenAI 的 `system_instruction`、`prompt template` 当成最终提示词。普通使用保持 `true`。
- `include_raw_json`：是否在内部解析结果中保留原始 `prompt` / `workflow` JSON。普通使用保持 `true`。
- `max_depth`：graph 反向追踪最大深度，用于防止循环。普通使用保持 `40`；极复杂工作流可以适当加大。

## 输出端口

主节点输出：

- `image`
- `mask`
- `positive`
- `negative`
- `seed`
- `steps`
- `cfg`
- `width`
- `height`
- `model_name`
- `filename`
- `setting`

`setting` 是给人看的整合参数文本，包含文件名、来源、模型、VAE、LoRA 列表、seed、steps、CFG、sampler、scheduler、尺寸、状态、置信度和 unresolved 信息。

## 支持的 Metadata 格式

- ComfyUI `prompt` metadata 优先，因为它更接近实际执行 graph。
- ComfyUI `workflow` metadata 用于补充 UI 缓存和 fallback，例如 `ShowText|pysssss` 的显示缓存。
- A1111 / Forge `parameters` 会在 ComfyUI graph 无法启动解析时作为 fallback。
- 通过 Pillow 读取 PNG、JPEG、WEBP 中的 metadata。

如果图片被平台压缩、聊天软件转发或图床处理后丢失 metadata，本节点无法恢复原始提示词。这种情况只能依赖其他图像反推工具。

## 支持的主要节点类型

Phase 1 初始支持包括：

- `KSampler`
- `KSamplerAdvanced`
- `SamplerCustom`
- `SamplerCustomAdvanced`
- `CLIPTextEncode`
- `CLIPTextEncodeSDXL`
- `StringFunction|pysssss`
- `ShowText|pysssss`
- `DeepTranslatorTextNode`
- `ControlNetApply`
- `ControlNetApplyAdvanced`
- `ConditioningCombine`
- `ConditioningConcat`
- `InpaintModelConditioning`
- `FaceDetailer` 和常见 detailer passthrough 模式
- `CheckpointLoaderSimple`
- `CheckpointLoader`
- `VAELoader`
- `LoraLoader`
- `LoraLoaderModelOnly`

## 重要解析原则

- positive 必须来自 selected sampler 的 `positive` conditioning 链路。
- negative 必须来自 selected sampler 的 `negative` conditioning 链路。
- 不扫描整个 workflow 的所有文本节点。
- 未连接到最终 sampler 的 Gemini、ShowText、StringFunction、PrimitiveString、备注文本或旧 CLIPTextEncode 不会污染结果。
- ControlNet、Detailer、Inpaint、ConditioningCombine、ConditioningConcat 只在最终 conditioning 链路上作为 passthrough 或 branch 解析。
- LoRA loader 的文件名和强度会单独进入 setting；但 LoRA trigger words 如果通过文本节点进入 prompt，就会保留在 positive / negative 中。
- 不做内容审查，不做提示词优化，不删 token。

## 已验证场景

Phase 1 实机和回归测试已覆盖：

- 普通 ComfyUI `prompt` / `workflow` metadata。
- Comfyroll `CR Apply ControlNet` positive conditioning passthrough。
- `ShowText|pysssss` 缓存文本。
- `StringFunction|pysssss` prompt 拼接。
- `DeepTranslatorTextNode` 文本来源。
- A1111 / Forge `parameters` fallback。
- PNG / JPEG / WEBP metadata 读取。

## 当前限制

- Phase 1 是 Python-only。节点内部三个大文本框和更像 SD Prompt Reader 的图片预览 UI 留到 Phase 2 JS 前端增强。
- 部分未知 custom conditioning 节点会返回 `PARTIAL` / unresolved，而不是猜测 prompt。
- 某些 custom node 的运行结果如果没有写入 metadata 缓存，可能无法还原。
- 被压缩或重新保存后丢失 metadata 的图片无法读取原始生成参数。

## 后续计划

- Phase 2：JS frontend enhancement，在节点内部显示 `positive` / `negative` / `setting` 三个文本框。
- Phase 3：更多 sampler adapter，例如 Efficient / Impact / custom sampler，基于真实失败样本补充。
- Phase 4：更多 metadata format compatibility。

## 本地验证

在插件目录运行：

```bash
python -m pytest -q
python -m compileall .
```

`tests/` 目录是开发和回归测试用的。普通 ComfyUI 用户不需要手动运行这些测试；仓库保留它们是为了在后续兼容更多 custom nodes 时，防止破坏已有功能。

## 手动 ComfyUI 验证

1. 重启 ComfyUI。
2. 搜索并添加 `Smart Metadata Reader`。
3. 选择或上传一张带有 metadata 的图片。
4. 确认节点输出 `image`、`mask`、`positive`、`negative` 和 `setting`。
5. 如果解析状态是 `PARTIAL`，查看 `setting` 中的 unresolved 信息，定位不支持的 custom node。
