# ComfyUI-Smart-Metadata-Reader

Language / 语言: [中文](README.md) | English

`ComfyUI-Smart-Metadata-Reader` is a Python-only ComfyUI custom node. Its main node, `Smart Metadata Reader`, works as an image loader: it lets you select or upload an image, reads the AI generation metadata already stored inside that image, and outputs the image, mask, real positive/negative prompts, and common generation settings.

## Overview

This node is not an image captioning or prompt guessing tool. It does not look at pixels and infer a prompt. It reads metadata that already exists in the image file, especially the `prompt` and `workflow` JSON saved by ComfyUI PNG outputs.

The core goal is to support complex ComfyUI workflows. The parser starts from the final selected sampler, walks backward through its `positive` and `negative` conditioning chains, and extracts the prompts and settings that actually reached the final generation step.

## What Problem It Solves

Many prompt readers can parse simple A1111 settings or basic ComfyUI images where `CLIPTextEncode.inputs.text` is a plain string. In complex workflows, that field may be a graph link to upstream string functions, cached `ShowText` nodes, translator nodes, ControlNet conditioning, detailers, or other custom node chains.

This project parses ComfyUI metadata as a graph instead of scanning every text field in the workflow.

## Difference From Regular Prompt Readers

- It selects the sampler connected to the final image output when possible.
- It resolves positive and negative prompts only from the selected sampler's conditioning chain.
- It does not scan every `text` field in the workflow.
- It does not mix ControlNet parameters such as image, strength, control_net, or model_name into the prompt.
- It does not treat Gemini / OpenAI / ChatGPT / Claude / LLM fields such as `system_instruction`, `system_prompt`, `prompt`, `instruction`, or `messages` as final prompts.
- When `ShowText|pysssss` has a `text_0` cached display result, that cached result is preferred.
- It does not filter LoRA trigger words, character tags, style tags, NSFW tags, quality tags, translated text, or user-written tags.

## Installation

From your ComfyUI installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CyberShadowX/ComfyUI-Smart-Metadata-Reader.git
cd ComfyUI-Smart-Metadata-Reader
pip install -r requirements.txt
```

Restart ComfyUI after installing.

## Usage

1. Add the `Smart Metadata Reader` node in ComfyUI.
2. Select an image from the ComfyUI `input` directory, or upload one through the image widget.
3. Use the node outputs in your workflow, or inspect the text outputs directly.

## Outputs

The main node outputs:

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

`setting` is human-readable text containing filename, source, model, VAE, LoRA list, seed, steps, CFG, sampler, scheduler, size, status, confidence, and unresolved information when available.

## Supported Metadata

- ComfyUI `prompt` metadata is preferred because it is closest to the executed graph.
- ComfyUI `workflow` metadata is used for UI cache and fallback values, such as `ShowText|pysssss` display cache.
- A1111 / Forge `parameters` metadata is used as fallback when ComfyUI graph parsing cannot start.
- PNG, JPEG, and WEBP metadata are read through Pillow.

If an image was compressed, forwarded through a chat app, or processed by a hosting platform that stripped metadata, this node cannot recover the original prompt. In that case, you need a separate image captioning or prompt inference tool.

## Supported Node Types

Initial Phase 1 support includes:

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
- `FaceDetailer` and common detailer passthrough patterns
- `CheckpointLoaderSimple`
- `CheckpointLoader`
- `VAELoader`
- `LoraLoader`
- `LoraLoaderModelOnly`

## Important Rules

- `positive` must come from the selected sampler's `positive` conditioning chain.
- `negative` must come from the selected sampler's `negative` conditioning chain.
- The parser does not scan every text node in the workflow.
- Unconnected Gemini, ShowText, StringFunction, PrimitiveString, note text, or old CLIPTextEncode nodes do not contaminate the result.
- ControlNet, Detailer, Inpaint, ConditioningCombine, and ConditioningConcat are treated as passthrough or branch nodes only when they are on the final conditioning chain.
- LoRA loader file names and strengths are extracted separately for settings; LoRA trigger words remain in the prompt if they entered the text chain.
- The parser does not perform content moderation, prompt optimization, or token deletion.

## Current Limitations

- Phase 1 is Python-only. In-node large text boxes and a richer SD Prompt Reader style preview UI are planned for the Phase 2 JavaScript enhancement.
- Some unknown custom conditioning nodes may return `PARTIAL` / unresolved instead of guessing.
- If a custom node did not save its generated result into metadata cache, the parser may not be able to recover it.
- Images with stripped metadata cannot provide original generation settings.

## Local Verification

Run these from the plugin directory:

```bash
python -m pytest -q
python -m compileall .
```

## Manual ComfyUI Verification

1. Restart ComfyUI.
2. Search for and add `Smart Metadata Reader`.
3. Select or upload an image with metadata.
4. Confirm the node outputs `image`, `mask`, `positive`, `negative`, and `setting`.
5. If the status is `PARTIAL`, inspect unresolved information in `setting` to identify unsupported custom nodes.
