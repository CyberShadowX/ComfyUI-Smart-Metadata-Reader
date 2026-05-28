# ComfyUI Smart Metadata Reader Specification

## 1. Project Positioning

`ComfyUI-Smart-Metadata-Reader` is an image-loader style ComfyUI custom node package. It is not only a parser library.

The primary node, `Smart Metadata Reader`, must behave like a practical image loader and metadata reader inside ComfyUI:

- Let the user select an image from ComfyUI's `input` directory.
- Let the user upload an image through the node UI.
- Load and output the image and mask.
- Read image metadata from PNG, JPEG, and WEBP files.
- Extract common generation values and prompt text from the metadata.
- Output common ports that can be wired directly into other ComfyUI nodes.

The core technical difference from ordinary prompt readers is graph-aware ComfyUI parsing. For ComfyUI images, the node must prefer the backend execution `prompt` JSON, select the final sampler that generated the image, and trace backward through the sampler's positive and negative conditioning inputs to recover the real text that entered generation.

The package may expose internal parser modules for testing and reuse, but the user-facing deliverable is the ComfyUI node.

## 2. Node Split

### 2.1 Smart Metadata Reader

`Smart Metadata Reader` is the primary node and must be implemented first.

Suggested category:

```text
metadata/image
```

Required inputs:

```text
image: COMBO[STRING], image_upload=True
parameter_index: INT, default 0
prefer_cached_text: BOOLEAN, default True
include_raw_json: BOOLEAN, default True
max_depth: INT, default 40
```

Required outputs:

```text
image: IMAGE
mask: MASK
positive: STRING
negative: STRING
seed: INT
steps: INT
cfg: FLOAT
width: INT
height: INT
model_name: STRING
filename: STRING
setting: STRING
```

The main node must be useful on its own. If development time is limited, this node takes priority over every debug or advanced node.

### 2.2 Smart Metadata Debug

`Smart Metadata Debug` is an enhanced node that can be added after the main node is stable.

Suggested category:

```text
metadata/debug
```

Enhanced outputs:

```text
vae_name: STRING
sampler_name: STRING
scheduler: STRING
loras_json: STRING
raw_prompt_json: STRING
raw_workflow_json: STRING
partial_result_json: STRING
debug_trace: STRING
confidence: FLOAT
status_message: STRING
```

The debug node should expose parser internals without making the primary node difficult to use.

## 3. UI Phases

### 3.1 Phase 1: Python-Only MVP

Phase 1 must deliver a working ComfyUI image loader node with:

- Image selection and upload input.
- Image and mask output.
- Metadata reading.
- ComfyUI `prompt` and `workflow` parsing.
- A1111/Forge `parameters` fallback.
- Graph traversal for positive and negative prompt extraction.
- Common numeric and string outputs.
- A human-readable `setting` output.
- Basic UI payload feedback when practical.

Phase 1 does not require a full SD Prompt Reader style frontend layout with three embedded large text boxes. The node may rely on output ports and simple ComfyUI UI payload feedback.

### 3.2 Phase 2: Frontend UI Enhancement

Phase 2 adds a ComfyUI JavaScript extension through:

```text
WEB_DIRECTORY = "./web/js"
```

The frontend enhancement should display these read-only areas inside the node:

- `positive` text box.
- `negative` text box.
- `setting` text box.
- Image preview.
- Image dimensions.
- OK, PARTIAL, or FAILED status.

JavaScript is only for presentation. Parser logic must remain in Python.

### 3.3 Phase 3: Compatibility Expansion

Phase 3 adds:

- More custom node adapters.
- Stronger A1111/Forge `parameters` fallback.
- More precise multi-sampler, refiner, and hires-fix handling.
- The `Smart Metadata Debug` node if it was not completed earlier.
- More machine-readable debug outputs.

## 4. Metadata Priority

For ComfyUI images, parsing priority is:

1. Use backend execution `prompt` JSON as the primary source because it is closest to the actual generation graph.
2. Use `workflow` JSON as supplemental data for UI cache, `widgets_values`, ShowText cache, preview context, and fallback.
3. Use A1111/Forge `parameters` when ComfyUI `prompt` data is absent or unusable.

The parser must not treat stale workflow widget text as the true execution prompt when execution `prompt` JSON provides a better source. If workflow fallback is used for a prompt segment, the trace and status must say so.

## 5. Prompt Parsing Principles

`positive` must be the merged text that truly entered the selected sampler's positive conditioning.

`negative` must be the merged text that truly entered the selected sampler's negative conditioning.

Hard rules:

- Do not hard-code node IDs.
- Do not hard-code a single test workflow shape.
- Do not assume a fixed node order.
- Parse by graph traversal, `class_type`, `inputs`, and link references such as `[node_id, output_index]`.
- Preserve all text that enters final conditioning.
- Do not remove LoRA trigger words.
- Do not remove character tags.
- Do not remove style tags.
- Do not remove NSFW terms.
- Do not remove quality tags.
- Do not remove translated text.
- Do not remove user-written prompt text.
- Do not rewrite words.
- Do not optimize prompts.
- Do not perform content moderation.

Allowed formatting cleanup:

- Remove empty segments.
- Trim obvious leading and trailing whitespace.
- Collapse clearly repeated commas produced by joining segments.
- Collapse excessive blank lines.
- Normalize obvious spacing around separators.

Formatting cleanup must never delete meaningful tokens.

## 6. LoRA Rules

LoRA loader nodes and LoRA trigger words are separate concerns.

LoRA loader nodes such as `LoraLoader`, `LoraLoaderModelOnly`, and compatible custom loaders should be extracted to `loras_json` with:

- Node ID.
- `class_type`.
- LoRA file name.
- `strength_model`.
- `strength_clip`.
- Connection path.

Example:

```json
[
  {
    "node_id": "17",
    "class_type": "LoraLoader",
    "lora_name": "character_lora.safetensors",
    "strength_model": 0.8,
    "strength_clip": 1.0,
    "path": "KSampler 72.model <- LoraLoader 17"
  }
]
```

If LoRA trigger words, character tags, or style tags enter `CLIPTextEncode` or final conditioning through text nodes, they must remain in `positive` or `negative`. The parser must not filter text because it looks like a LoRA trigger word.

## 7. Final Sampler Selection

Sampler selection must prefer the sampler that generated the loaded image.

Selection order:

1. Start from final image output nodes when possible, including `SaveImage`, `PreviewImage`, `SaveAnimatedWEBP`, and common image save nodes.
2. Walk backward through `image`, `images`, `samples`, `latent`, and VAE decode chains.
3. Select the nearest valid sampler upstream of the final output chain.
4. If no final output chain can be determined, choose a valid sampler through fallback heuristics.
5. If fallback heuristics are used, `setting`, `status_message`, and `debug_trace` must clearly state the reason and confidence.

Initial sampler candidates:

```text
KSampler
KSamplerAdvanced
SamplerCustom
SamplerCustomAdvanced
```

Additional sampler types should be added through adapters.

The parser must not pretend fallback selection is fully certain.

## 8. Conditioning Trace Rules

Conditioning resolution begins at:

```text
selected_sampler.inputs.positive
selected_sampler.inputs.negative
```

The resolver must follow links backward through passthrough and branch nodes, including:

```text
ControlNetApply
ControlNetApplyAdvanced
FaceDetailer
DetailerForEach
Impact Pack detailer nodes
InpaintModelConditioning
ConditioningCombine
ConditioningConcat
ConditioningSet
```

The resolver should also support adapter-driven matching for custom conditioning nodes that expose fields such as:

```text
conditioning
positive
negative
positive_conditioning
negative_conditioning
base_positive
base_negative
```

For branch nodes such as `ConditioningCombine` and `ConditioningConcat`, every actual branch that reaches the final sampler must be resolved and merged. `debug_trace` must show which prompt segment came from which node.

Unknown passthrough nodes must produce a partial result instead of fabricated prompt text.

## 9. Text Adapters

Initial text adapters must support:

```text
CLIPTextEncode
CLIPTextEncodeSDXL
StringFunction|pysssss
ShowText|pysssss
DeepTranslatorTextNode
Primitive/String/Text-like nodes
```

Common text fields:

```text
text
text_0
text_a
text_b
text_c
string
value
prompt
result
```

Adapter rules:

- `CLIPTextEncode`: resolve `inputs.text`; if it is a link, continue upstream.
- `CLIPTextEncodeSDXL`: resolve all prompt text fields that enter conditioning, such as `text`, `text_g`, and `text_l`.
- `StringFunction|pysssss`: for append-like actions, resolve `text_a`, `text_b`, and `text_c` in graph order and merge the segments.
- `ShowText|pysssss`: prefer `inputs.text_0` or workflow/cache display results. Only follow upstream when no cache exists. When upstream is used, reduce confidence and record the reason in trace.
- `DeepTranslatorTextNode`: resolve `inputs.text` for Phase 1; future adapters may prefer cached translated output when reliable metadata is available.
- Primitive/String/Text-like nodes: read common text fields without assuming exact node names.

`ShowText|pysssss` must avoid treating upstream `GeminiChatNode` prompt templates or system instructions as final image prompt text when cached displayed text is available.

## 10. Prompt Merge Strategy

Prompt segments should be represented internally with:

- Text value.
- Source node ID.
- Source class type.
- Source input field.
- Trace path.
- Confidence contribution.

Merging must follow actual graph path order where known. Branch nodes should preserve branch order from graph inputs.

The merger may clean separators and empty segments, but it must not classify, censor, or rewrite prompt content.

## 11. Setting Field

`setting` is human-readable text, not JSON.

Recommended format:

```text
Filename: 001.png
Source: ComfyUI prompt/workflow
Model: model.safetensors
VAE: vae.safetensors
LoRA:
- character_lora.safetensors (model 0.80, clip 1.00)
Seed: 123456789
Steps: 30
CFG: 7.0
Sampler: euler
Scheduler: normal
Size: 1200 x 1600
Status: OK
Confidence: 0.94
```

For partial results:

```text
Status: PARTIAL
Confidence: 0.52
Unresolved:
- Node 88 FooBarConditioning input positive_conditioning
```

Machine-readable data belongs in debug outputs such as `loras_json`, `partial_result_json`, and `debug_trace`.

## 12. Debug Trace

`debug_trace` should explain how the result was reached.

Example:

```text
SAMPLER_SELECTION:
SaveImage 91.images <- VAEDecode 89.samples <- KSampler 72.samples
Selected KSampler 72 because it is upstream of final SaveImage.

POSITIVE_TRACE:
KSampler 72.positive
 -> ControlNetApply 65.conditioning
 -> ConditioningCombine 61.conditioning_1
 -> CLIPTextEncode 40.text
 -> StringFunction|pysssss 22 [append]
    text_a: direct string
    text_b: ShowText|pysssss 143.text_0 cache
    text_c: DeepTranslatorTextNode 24.text

NEGATIVE_TRACE:
KSampler 72.negative
 -> CLIPTextEncode 41.text
 -> StringFunction|pysssss 48.text_a

LORA_TRACE:
KSampler 72.model <- LoraLoader 33 <- LoraLoader 28 <- CheckpointLoaderSimple 4
```

When fallback or workflow cache is used, trace must say so.

## 13. Failure and Partial Results

The parser must not guess when it cannot determine a value.

Failure behavior:

- Return all fields that were resolved.
- Return empty or sentinel values for fields that were not resolved.
- Set status to `PARTIAL` or `FAILED`.
- Lower confidence.
- Include unresolved node IDs, class types, and input fields in trace or partial result.

Examples:

```text
Status: PARTIAL
Unresolved:
- Node 88 UnknownConditioningRouter input positive
Reason: no adapter matched this conditioning passthrough node
```

## 14. A1111 and Forge Fallback

When ComfyUI `prompt` metadata is missing, invalid, or unusable, the parser should attempt A1111/Forge `parameters` parsing.

Phase 1 should parse common fields:

- Positive prompt.
- Negative prompt.
- Seed.
- Steps.
- CFG scale.
- Sampler.
- Size.
- Model name when present.

`parameter_index` selects among multiple parameter sets when the metadata contains more than one set.

## 15. Package Architecture

Planned layout:

```text
ComfyUI-Smart-Metadata-Reader/
  __init__.py
  nodes.py
  requirements.txt
  README.md
  pyproject.toml
  docs/
    spec.md
    implementation_plan.md
  smart_metadata_reader/
    __init__.py
    metadata_reader.py
    image_io.py
    models.py
    graph.py
    sampler_selector.py
    resolver.py
    prompt_merge.py
    lora_extractor.py
    settings_formatter.py
    a1111_parser.py
    trace.py
    adapters/
      __init__.py
      base.py
      clip_text.py
      text_sources.py
      pysssss.py
      conditioning.py
      samplers.py
      loaders.py
  web/
    js/
      smart_metadata_reader.js
  tests/
    fixtures/
    test_metadata_reader.py
    test_graph.py
    test_sampler_selector.py
    test_conditioning_resolver.py
    test_text_adapters.py
    test_lora_extractor.py
    test_node_id_independence.py
    test_node_contract.py
```

Parser modules must be testable without launching ComfyUI.

## 16. Test Requirements

Tests must cover:

- Simple `KSampler -> CLIPTextEncode`.
- `CLIPTextEncode.text` connected to upstream text node.
- `StringFunction|pysssss` three-part append.
- `ShowText|pysssss` `text_0` cache priority.
- `DeepTranslatorTextNode`.
- `ControlNetApply` passthrough.
- `ConditioningCombine` multi-branch merge.
- Multiple `CLIPTextEncode` nodes entering positive conditioning.
- LoRA trigger words retained in positive prompt.
- Multiple LoRA loader nodes exported to `loras_json`.
- Multiple samplers with final output sampler selection.
- Randomized node IDs with unchanged parse result.
- Unknown passthrough node returns partial result instead of invented prompt text.

## 17. Relationship to comfyui-prompt-reader-node

This project may use `receyuki/comfyui-prompt-reader-node` as a user experience reference:

- Image selection and upload.
- Positive prompt area.
- Negative prompt area.
- Setting area.
- Image preview.
- Common output ports.

This project must not copy its implementation and must not depend on its submodule. The parser must be owned by this project and designed around final sampler and conditioning graph traversal.
