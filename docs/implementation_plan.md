# Smart Metadata Reader Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python-only MVP for `Smart Metadata Reader`, a ComfyUI image-loader node that loads an image, reads metadata, extracts real positive and negative prompt text through ComfyUI graph traversal, and outputs common image and settings ports.

**Architecture:** The ComfyUI node in `nodes.py` will stay thin. It will call a testable parser package under `smart_metadata_reader/`. Metadata reading, graph indexing, sampler selection, conditioning resolution, prompt merging, LoRA extraction, settings formatting, and A1111 fallback will be separate modules with dataclass-based interfaces.

**Tech Stack:** Python 3, Pillow, torch tensors for ComfyUI `IMAGE` and `MASK` outputs, pytest for parser tests, standard-library `json`, `dataclasses`, `pathlib`, and `typing`.

---

## Phase 1 Scope

Phase 1 includes:

- Python-only `Smart Metadata Reader` node.
- Image selection and upload through ComfyUI `image_upload=True`.
- Metadata reading from PNG, JPEG, and WEBP.
- Image and mask tensor output.
- ComfyUI `prompt` and `workflow` parsing.
- Final sampler selection.
- Positive and negative conditioning graph traversal.
- Initial adapters for CLIP text, pysssss text nodes, generic text nodes, conditioning passthrough nodes, samplers, loaders, and LoRA loaders.
- Human-readable `setting` output.
- A1111/Forge `parameters` fallback for common fields.
- pytest coverage for the parser core and node contract.

Phase 1 excludes:

- Full frontend JavaScript node layout.
- Embedded large read-only positive, negative, and setting text boxes.
- Civitai hash calculation.
- Metadata writing.
- Exhaustive custom node support.

## File Map

Create these files:

```text
__init__.py
nodes.py
requirements.txt
pyproject.toml
smart_metadata_reader/__init__.py
smart_metadata_reader/models.py
smart_metadata_reader/image_io.py
smart_metadata_reader/metadata_reader.py
smart_metadata_reader/graph.py
smart_metadata_reader/sampler_selector.py
smart_metadata_reader/resolver.py
smart_metadata_reader/prompt_merge.py
smart_metadata_reader/lora_extractor.py
smart_metadata_reader/settings_formatter.py
smart_metadata_reader/a1111_parser.py
smart_metadata_reader/trace.py
smart_metadata_reader/adapters/__init__.py
smart_metadata_reader/adapters/base.py
smart_metadata_reader/adapters/clip_text.py
smart_metadata_reader/adapters/text_sources.py
smart_metadata_reader/adapters/pysssss.py
smart_metadata_reader/adapters/conditioning.py
smart_metadata_reader/adapters/samplers.py
smart_metadata_reader/adapters/loaders.py
tests/conftest.py
tests/fixtures.py
tests/test_metadata_reader.py
tests/test_graph.py
tests/test_sampler_selector.py
tests/test_conditioning_resolver.py
tests/test_text_adapters.py
tests/test_lora_extractor.py
tests/test_node_id_independence.py
tests/test_node_contract.py
```

## Core Interfaces

`smart_metadata_reader/models.py` will define:

```python
@dataclass
class MetadataBundle:
    filename: str
    width: int
    height: int
    prompt_raw: str | None
    workflow_raw: str | None
    parameters_raw: str | None
    source_format: str

@dataclass
class NodeRecord:
    node_id: str
    class_type: str
    inputs: dict[str, Any]
    widgets_values: list[Any] | dict[str, Any] | None = None

@dataclass
class PromptSegment:
    text: str
    node_id: str | None
    class_type: str | None
    field: str | None
    path: list[str]
    confidence: float = 1.0

@dataclass
class LoraInfo:
    node_id: str
    class_type: str
    lora_name: str
    strength_model: float | None
    strength_clip: float | None
    path: list[str]

@dataclass
class ParseResult:
    positive: str
    negative: str
    seed: int
    steps: int
    cfg: float
    width: int
    height: int
    model_name: str
    filename: str
    setting: str
    vae_name: str
    sampler_name: str
    scheduler: str
    loras: list[LoraInfo]
    raw_prompt_json: str
    raw_workflow_json: str
    partial_result: dict[str, Any]
    debug_trace: str
    confidence: float
    status_message: str
```

`smart_metadata_reader/graph.py` will define:

- `GraphIndex.__init__(prompt: dict[str, Any] | None, workflow: dict[str, Any] | None)`: normalize prompt and workflow nodes.
- `GraphIndex.get_node(node_id: str) -> NodeRecord | None`: return a normalized node by string ID.
- `GraphIndex.is_link(value: Any) -> bool`: detect ComfyUI link references such as `[node_id, output_index]`.
- `GraphIndex.link_target(value: Any) -> tuple[str, int] | None`: return the linked node ID and output index.
- `GraphIndex.upstream_link_inputs(node_id: str) -> list[tuple[str, Any]]`: list linked inputs for a node.
- `GraphIndex.downstream_nodes(node_id: str) -> list[NodeRecord]`: list nodes that consume the node.
- `GraphIndex.workflow_cache_for(node_id: str, field_names: list[str]) -> str | None`: read supplemental cache text from workflow metadata.

`smart_metadata_reader/resolver.py` will define:

- `ConditioningResolver.__init__(graph: GraphIndex, adapters: AdapterRegistry, max_depth: int, prefer_cached_text: bool)`: store resolver dependencies and traversal limits.
- `ConditioningResolver.resolve_positive(sampler: NodeRecord) -> list[PromptSegment]`: resolve `sampler.inputs.positive`.
- `ConditioningResolver.resolve_negative(sampler: NodeRecord) -> list[PromptSegment]`: resolve `sampler.inputs.negative`.
- `ConditioningResolver.resolve_value(value: Any, role: str, path: list[str], depth: int) -> list[PromptSegment]`: resolve strings, links, text nodes, and conditioning passthrough nodes.

`nodes.py` will define:

```python
class SmartMetadataReader:
    CATEGORY = "metadata/image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT", "FLOAT", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "positive", "negative", "seed", "steps", "cfg", "width", "height", "model_name", "filename", "setting")
    FUNCTION = "read_metadata"
```

## Task 1: Project Bootstrap

**Files:**

- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `__init__.py`
- Create: `smart_metadata_reader/__init__.py`
- Create: `smart_metadata_reader/adapters/__init__.py`

- [ ] **Step 1: Initialize Git when implementation starts**

Run from `F:\AI\AIProjects\ComfyUI-Smart-Metadata-Reader`:

```powershell
git init
git branch -M main
git remote add origin https://github.com/CyberShadowX/ComfyUI-Smart-Metadata-Reader.git
git status --short --branch
```

Expected: repository on branch `main` with untracked files only.

- [ ] **Step 2: Create package metadata**

`requirements.txt`:

```text
Pillow>=10.0.0
```

`pyproject.toml`:

```toml
[project]
name = "comfyui-smart-metadata-reader"
version = "0.1.0"
description = "ComfyUI image loader node that reads AI image metadata and traces real prompt conditioning."
requires-python = ">=3.10"
dependencies = ["Pillow>=10.0.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Register the node package**

`__init__.py` will export:

```python
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

- [ ] **Step 4: Run bootstrap check**

Run:

```powershell
python -m compileall .
```

Expected: Python files compile.

## Task 2: Data Models

**Files:**

- Create: `smart_metadata_reader/models.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write model import test**

Create a test that imports `MetadataBundle`, `NodeRecord`, `PromptSegment`, `LoraInfo`, and `ParseResult`.

Run:

```powershell
python -m pytest tests/test_graph.py -q
```

Expected before implementation: import failure.

- [ ] **Step 2: Add dataclasses**

Create the dataclasses listed in the Core Interfaces section. Use explicit defaults for unresolved scalar values:

```python
seed = -1
steps = 0
cfg = 0.0
width = 0
height = 0
```

- [ ] **Step 3: Run model test**

Run:

```powershell
python -m pytest tests/test_graph.py -q
```

Expected: imports pass.

## Task 3: Image and Metadata Reading

**Files:**

- Create: `smart_metadata_reader/image_io.py`
- Create: `smart_metadata_reader/metadata_reader.py`
- Test: `tests/test_metadata_reader.py`

- [ ] **Step 1: Write PNG metadata test**

Create a synthetic PNG in the test using Pillow with text chunks:

```python
pnginfo.add_text("prompt", json.dumps(prompt_graph))
pnginfo.add_text("workflow", json.dumps(workflow_graph))
pnginfo.add_text("parameters", "positive\nNegative prompt: negative\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 123, Size: 512x768")
```

Assert that `read_metadata(path)` returns `prompt_raw`, `workflow_raw`, `parameters_raw`, `width`, `height`, and `filename`.

- [ ] **Step 2: Implement `read_metadata`**

`metadata_reader.py` will expose:

Signature: `read_metadata(image_path: str | Path) -> MetadataBundle`.

Use Pillow `Image.open`. Read `image.info` keys `prompt`, `workflow`, and `parameters`.

- [ ] **Step 3: Implement image tensor conversion**

`image_io.py` will expose:

Signature: `load_image_and_mask(image_path: str | Path) -> tuple[Any, Any, int, int]`.

Return ComfyUI-compatible image tensor shape `[1, H, W, 3]` and mask shape `[1, H, W]`.

- [ ] **Step 4: Run metadata tests**

Run:

```powershell
python -m pytest tests/test_metadata_reader.py -q
```

Expected: metadata and dimensions are read from the synthetic image.

## Task 4: Graph Index

**Files:**

- Create: `smart_metadata_reader/graph.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write graph tests**

Cover:

- API prompt format with node IDs as dictionary keys.
- UI workflow format with `nodes` array.
- Link detection for `["6", 0]` and `[6, 0]`.
- `workflow_cache_for` reading cached `text_0` from workflow node inputs or widget values.

- [ ] **Step 2: Implement `GraphIndex`**

Normalize all node IDs to strings. Store prompt nodes as primary records. Attach workflow supplemental data by matching node ID.

- [ ] **Step 3: Run graph tests**

Run:

```powershell
python -m pytest tests/test_graph.py -q
```

Expected: graph lookup, link parsing, and workflow cache lookup pass.

## Task 5: Prompt Merge and Trace

**Files:**

- Create: `smart_metadata_reader/prompt_merge.py`
- Create: `smart_metadata_reader/trace.py`
- Test: `tests/test_text_adapters.py`

- [ ] **Step 1: Write prompt merge tests**

Assert:

- Empty segments are removed.
- Extra commas from joins are collapsed.
- LoRA trigger words and tags remain unchanged.
- Repeated meaningful tokens remain.

- [ ] **Step 2: Implement `merge_prompt_segments`**

Expose:

Signature: `merge_prompt_segments(segments: list[PromptSegment]) -> str`.

Only perform formatting cleanup allowed by `docs/spec.md`.

- [ ] **Step 3: Implement trace collector**

Expose:

`TraceCollector` methods:

- `add(section: str, line: str) -> None`
- `render() -> str`

- [ ] **Step 4: Run merge tests**

Run:

```powershell
python -m pytest tests/test_text_adapters.py -q
```

Expected: formatting cleanup passes without deleting trigger words.

## Task 6: Adapter Registry and Text Adapters

**Files:**

- Create: `smart_metadata_reader/adapters/base.py`
- Create: `smart_metadata_reader/adapters/clip_text.py`
- Create: `smart_metadata_reader/adapters/text_sources.py`
- Create: `smart_metadata_reader/adapters/pysssss.py`
- Test: `tests/test_text_adapters.py`

- [ ] **Step 1: Write adapter tests**

Cover:

- `CLIPTextEncode.inputs.text` direct string.
- `CLIPTextEncode.inputs.text` linked to generic text node.
- `CLIPTextEncodeSDXL` with multiple text fields.
- `StringFunction|pysssss` append from `text_a`, `text_b`, `text_c`.
- `ShowText|pysssss` prefers `text_0` cache.
- `DeepTranslatorTextNode.inputs.text`.

- [ ] **Step 2: Implement base adapter protocol**

Expose:

`AdapterRegistry` methods:

- `register(adapter: NodeAdapter) -> None`
- `adapter_for(node: NodeRecord) -> NodeAdapter | None`

`NodeAdapter` protocol methods:

- `matches(node: NodeRecord) -> bool`

- [ ] **Step 3: Implement text adapters**

Text adapters return `PromptSegment` instances and recursively call the resolver for linked input values.

- [ ] **Step 4: Run text adapter tests**

Run:

```powershell
python -m pytest tests/test_text_adapters.py -q
```

Expected: all text source cases pass.

## Task 7: Conditioning Resolver

**Files:**

- Create: `smart_metadata_reader/adapters/conditioning.py`
- Create: `smart_metadata_reader/resolver.py`
- Test: `tests/test_conditioning_resolver.py`

- [ ] **Step 1: Write resolver tests**

Cover:

- Simple `KSampler -> CLIPTextEncode`.
- `ControlNetApply` passthrough.
- `ControlNetApplyAdvanced` passthrough.
- `ConditioningCombine` two-branch merge.
- Multiple `CLIPTextEncode` nodes entering positive.
- Unknown passthrough produces partial result and trace entry.

- [ ] **Step 2: Implement conditioning adapter**

Support fields:

```python
[
    "conditioning",
    "positive",
    "negative",
    "positive_conditioning",
    "negative_conditioning",
    "base_positive",
    "base_negative",
]
```

Branch adapters must resolve all conditioning-like linked inputs.

- [ ] **Step 3: Implement depth and cycle guards**

`ConditioningResolver` must stop at `max_depth` and track visited `(node_id, field, role)` keys.

- [ ] **Step 4: Run resolver tests**

Run:

```powershell
python -m pytest tests/test_conditioning_resolver.py -q
```

Expected: passthrough and branch resolution pass, unknown nodes produce partial data.

## Task 8: Sampler Selection and Parameter Extraction

**Files:**

- Create: `smart_metadata_reader/adapters/samplers.py`
- Create: `smart_metadata_reader/sampler_selector.py`
- Test: `tests/test_sampler_selector.py`

- [ ] **Step 1: Write sampler selector tests**

Cover:

- `SaveImage <- VAEDecode <- KSampler`.
- `PreviewImage <- VAEDecode <- KSampler`.
- Multiple samplers where only one reaches final output.
- Fallback when no output node exists.

- [ ] **Step 2: Implement sampler candidates**

Initial candidate classes:

```python
{"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}
```

- [ ] **Step 3: Implement output-chain selection**

Walk backward from image output nodes through image, latent, sample, and VAE decode fields. Select nearest upstream sampler.

- [ ] **Step 4: Extract sampler values**

Expose:

Signature: `extract_sampler_settings(sampler: NodeRecord) -> dict[str, Any]`.

Return seed, steps, cfg, sampler_name, scheduler, and denoise when present.

- [ ] **Step 5: Run sampler tests**

Run:

```powershell
python -m pytest tests/test_sampler_selector.py -q
```

Expected: final output sampler is selected and fallback is flagged with reduced confidence.

## Task 9: Loader and LoRA Extraction

**Files:**

- Create: `smart_metadata_reader/adapters/loaders.py`
- Create: `smart_metadata_reader/lora_extractor.py`
- Test: `tests/test_lora_extractor.py`

- [ ] **Step 1: Write loader tests**

Cover:

- `CheckpointLoaderSimple` model name.
- `VAELoader` VAE name.
- One `LoraLoader`.
- Multiple chained `LoraLoader` nodes.
- LoRA trigger words in text remain in positive prompt.

- [ ] **Step 2: Implement loader extraction**

Expose:

Required functions:

- `extract_model_name(graph: GraphIndex, sampler: NodeRecord) -> str`
- `extract_vae_name(graph: GraphIndex) -> str`
- `extract_loras(graph: GraphIndex, sampler: NodeRecord) -> list[LoraInfo]`

LoRA extraction follows model and clip chains but never filters prompt text.

- [ ] **Step 3: Run LoRA tests**

Run:

```powershell
python -m pytest tests/test_lora_extractor.py -q
```

Expected: LoRA loader info is exported and trigger words remain in prompt output.

## Task 10: A1111 and Forge Fallback

**Files:**

- Create: `smart_metadata_reader/a1111_parser.py`
- Test: `tests/test_metadata_reader.py`

- [ ] **Step 1: Write fallback tests**

Input:

```text
masterpiece, character trigger
Negative prompt: blurry, low quality
Steps: 25, Sampler: Euler a, CFG scale: 7.5, Seed: 12345, Size: 768x1024, Model: demo_model
```

Assert positive, negative, seed, steps, cfg, sampler, size, and model.

- [ ] **Step 2: Implement `parse_a1111_parameters`**

Expose:

Signature: `parse_a1111_parameters(parameters: str, parameter_index: int = 0) -> ParseResult`.

- [ ] **Step 3: Run fallback tests**

Run:

```powershell
python -m pytest tests/test_metadata_reader.py -q
```

Expected: common A1111 fields parse correctly.

## Task 11: Settings Formatter

**Files:**

- Create: `smart_metadata_reader/settings_formatter.py`
- Test: `tests/test_node_contract.py`

- [ ] **Step 1: Write setting formatter tests**

Assert the setting text includes:

- Filename.
- Source.
- Model.
- VAE.
- LoRA list.
- Seed.
- Steps.
- CFG.
- Sampler.
- Scheduler.
- Size.
- Status.
- Confidence.
- Unresolved nodes for partial results.

- [ ] **Step 2: Implement `format_setting`**

Expose:

Signature: `format_setting(result: ParseResult) -> str`.

The output is human-readable text, not JSON.

- [ ] **Step 3: Run formatter tests**

Run:

```powershell
python -m pytest tests/test_node_contract.py -q
```

Expected: required setting lines are present.

## Task 12: Parser Orchestration

**Files:**

- Create: `smart_metadata_reader/metadata_reader.py`
- Modify: parser-facing modules from earlier tasks
- Test: `tests/test_conditioning_resolver.py`
- Test: `tests/test_node_id_independence.py`

- [ ] **Step 1: Write end-to-end parser tests**

Cover a ComfyUI prompt graph with:

- Final `KSampler`.
- Positive through `ControlNetApply`.
- `ConditioningCombine` with two branches.
- `StringFunction|pysssss` append.
- `ShowText|pysssss.text_0`.
- `DeepTranslatorTextNode`.
- Negative through a separate `CLIPTextEncode`.
- LoRA loaders.

- [ ] **Step 2: Write node ID independence test**

Create a helper that remaps all node IDs and link references to random strings. Assert that positive, negative, sampler values, and LoRA values match the original result.

- [ ] **Step 3: Implement orchestration**

Expose:

Signature:

```python
def parse_metadata_bundle(
    bundle: MetadataBundle,
    parameter_index: int = 0,
    prefer_cached_text: bool = True,
    include_raw_json: bool = True,
    max_depth: int = 40,
) -> ParseResult
```

Use ComfyUI graph parsing first. Use A1111 fallback when ComfyUI graph parsing cannot start.

- [ ] **Step 4: Run end-to-end tests**

Run:

```powershell
python -m pytest tests/test_conditioning_resolver.py tests/test_node_id_independence.py -q
```

Expected: graph traversal succeeds and random node IDs do not change results.

## Task 13: ComfyUI Node Contract

**Files:**

- Create: `nodes.py`
- Modify: `__init__.py`
- Test: `tests/test_node_contract.py`

- [ ] **Step 1: Write node contract tests**

Assert:

- `NODE_CLASS_MAPPINGS` contains `SmartMetadataReader`.
- `NODE_DISPLAY_NAME_MAPPINGS` contains `Smart Metadata Reader`.
- `INPUT_TYPES()` includes `image` with `image_upload=True`.
- `RETURN_TYPES` and `RETURN_NAMES` match the spec.

- [ ] **Step 2: Implement node class**

`SmartMetadataReader.INPUT_TYPES()` uses ComfyUI `folder_paths.get_input_directory()` and lists image files from the input directory.

`SmartMetadataReader.read_metadata(image, parameter_index, prefer_cached_text, include_raw_json, max_depth)`:

1. Resolves the selected image path.
2. Loads image and mask.
3. Reads metadata.
4. Parses metadata bundle.
5. Returns `(image, mask, positive, negative, seed, steps, cfg, width, height, model_name, filename, setting)`.

- [ ] **Step 3: Run node contract tests**

Run:

```powershell
python -m pytest tests/test_node_contract.py -q
```

Expected: node mappings, inputs, and outputs match the spec.

## Task 14: Phase 1 Verification

**Files:**

- All Phase 1 files

- [ ] **Step 1: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Compile all Python files**

Run:

```powershell
python -m compileall .
```

Expected: all Python files compile.

- [ ] **Step 3: Check Git status**

Run:

```powershell
git status --short --branch
```

Expected: only intended project files are changed or untracked.

- [ ] **Step 4: Commit Phase 1**

Run:

```powershell
git add .
git commit -m "feat: add python smart metadata reader mvp"
```

Expected: one commit containing the Python-only MVP.

## Test Execution Order

Run tests in this order during implementation:

```powershell
python -m pytest tests/test_graph.py -q
python -m pytest tests/test_metadata_reader.py -q
python -m pytest tests/test_text_adapters.py -q
python -m pytest tests/test_conditioning_resolver.py -q
python -m pytest tests/test_sampler_selector.py -q
python -m pytest tests/test_lora_extractor.py -q
python -m pytest tests/test_node_id_independence.py -q
python -m pytest tests/test_node_contract.py -q
python -m pytest -q
```

## Phase 1 Completion Criteria

Phase 1 is complete when:

- `Smart Metadata Reader` appears in ComfyUI node mappings.
- The node can select or upload an image.
- The node returns `IMAGE` and `MASK`.
- The node returns common ports defined in `docs/spec.md`.
- ComfyUI `prompt` metadata is parsed before `workflow` fallback.
- Positive and negative prompts are derived from selected sampler conditioning traversal.
- LoRA loader data is separated from prompt text.
- LoRA trigger words remain in prompt text when they enter conditioning.
- Unknown passthrough nodes create partial results instead of fabricated text.
- The full pytest suite passes.
