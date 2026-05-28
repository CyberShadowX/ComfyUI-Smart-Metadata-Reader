import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _save_jpeg_with_user_comment(path, comment: str, prefix: bytes = b"ASCII\x00\x00\x00"):
    exif = Image.Exif()
    if prefix.startswith(b"UNICODE"):
        payload = comment.encode("utf-16le")
    else:
        payload = comment.encode("utf-8")
    exif[37510] = prefix + payload
    Image.new("RGB", (32, 24), (20, 40, 60)).save(path, exif=exif)


def _save_png_with_prompt(path, prompt, workflow=None):
    pnginfo = PngInfo()
    pnginfo.add_text("prompt", json.dumps(prompt))
    if workflow is not None:
        pnginfo.add_text("workflow", json.dumps(workflow))
    Image.new("RGB", (32, 24), (20, 40, 60)).save(path, pnginfo=pnginfo)


def _simple_prompt_graph(positive="original positive prompt", negative="original negative prompt"):
    return {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "seed": 111,
                "steps": 12,
                "cfg": 5.5,
                "sampler_name": "euler",
                "scheduler": "normal",
            },
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": positive}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative}},
    }


def _upscale_prompt_graph(reader_image="003.png"):
    return {
        "22": {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": ["24", 0],
                "positive": ["26", 0],
                "negative": ["27", 0],
            },
        },
        "24": {
            "class_type": "UltimateSDUpscale",
            "inputs": {
                "image": ["40", 0],
                "positive": ["26", 0],
                "negative": ["27", 0],
                "seed": 222,
                "steps": 9,
                "cfg": 1.5,
                "sampler_name": "Euler",
                "scheduler": "normal",
                "tile_width": 768,
                "seam_fix_mode": "half tile",
            },
        },
        "26": {"class_type": "CLIPTextEncode", "inputs": {"text": ["37", 0]}},
        "27": {"class_type": "CLIPTextEncode", "inputs": {"text": ["38", 0]}},
        "36": {"class_type": "SmartMetadataReader", "inputs": {"image": reader_image}},
        "37": {"class_type": "ShowText|pysssss", "inputs": {"text": ["36", 2]}},
        "38": {"class_type": "ShowText|pysssss", "inputs": {"text": ["36", 3]}},
        "40": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "90": {
            "class_type": "GeminiChatNode",
            "inputs": {"prompt": "unused Gemini prompt template"},
        },
        "91": {"class_type": "ShowText|pysssss", "inputs": {"text": ["90", 0]}},
        "99": {"class_type": "PreviewImage", "inputs": {"images": ["22", 0]}},
    }


def _upscale_workflow(reader_image="003.png"):
    return {
        "nodes": [
            {
                "id": 36,
                "type": "SmartMetadataReader",
                "inputs": {"image": reader_image},
                "outputs": [
                    {"name": "image"},
                    {"name": "mask"},
                    {"name": "positive"},
                    {"name": "negative"},
                    {"name": "seed"},
                    {"name": "steps"},
                    {"name": "cfg"},
                    {"name": "width"},
                    {"name": "height"},
                    {"name": "model_name"},
                    {"name": "filename"},
                    {"name": "setting"},
                ],
            },
            {
                "id": 91,
                "type": "ShowText|pysssss",
                "inputs": {"text": ["90", 0]},
                "widgets_values": ["wrong unused prompt"],
            },
        ]
    }


def _llm_showtext_upscale_prompt(show_class="ShowText|pysssss", include_cache=True):
    return {
        "22": {
            "class_type": "FaceDetailer",
            "inputs": {"image": ["24", 0], "positive": ["26", 0], "negative": ["27", 0]},
        },
        "24": {
            "class_type": "UltimateSDUpscale",
            "inputs": {
                "image": ["40", 0],
                "positive": ["26", 0],
                "negative": ["27", 0],
                "seed": 222,
                "steps": 9,
                "cfg": 1.5,
                "sampler_name": "Euler",
                "scheduler": "normal",
            },
        },
        "26": {"class_type": "CLIPTextEncode", "inputs": {"text": ["37", 0]}},
        "27": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad hands, low quality"}},
        "37": {"class_type": show_class, "inputs": {"text": ["90", 0]}},
        "40": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "90": {
            "class_type": "GeminiChatNode",
            "inputs": {
                "system_instruction": "system instruction should not enter prompt",
                "prompt": "Gemini prompt template should not enter prompt",
                "api_key": "secret-key",
                "model": "gemini-test-model",
            },
        },
        "91": {"class_type": "ShowText|pysssss", "inputs": {"text": ["90", 0]}},
        "99": {"class_type": "PreviewImage", "inputs": {"images": ["22", 0]}},
    }


def _llm_showtext_workflow(show_class="ShowText|pysssss", cached_text="cached gemini positive prompt"):
    widgets_values = [cached_text] if cached_text else []
    return {
        "nodes": [
            {
                "id": 37,
                "type": show_class,
                "inputs": {"text": ["90", 0]},
                "widgets_values": widgets_values,
            },
            {
                "id": 90,
                "type": "GeminiChatNode",
                "inputs": {
                    "system_instruction": "system instruction should not enter prompt",
                    "prompt": "Gemini prompt template should not enter prompt",
                    "api_key": "secret-key",
                    "model": "gemini-test-model",
                },
                "widgets_values": [
                    "Gemini prompt template should not enter prompt",
                    "system instruction should not enter prompt",
                    "secret-key",
                    "gemini-test-model",
                ],
            },
            {
                "id": 91,
                "type": "ShowText|pysssss",
                "inputs": {"text": ["90", 0]},
                "widgets_values": ["wrong unused prompt"],
            },
        ]
    }


def _bundle_for_prompt(prompt, workflow=None, filename="sample.png", width=1200, height=1600):
    from smart_metadata_reader.models import MetadataBundle

    return MetadataBundle(
        filename=filename,
        width=width,
        height=height,
        prompt_raw=json.dumps(prompt),
        workflow_raw=json.dumps(workflow) if workflow is not None else None,
        parameters_raw=None,
        source_format="ComfyUI prompt/workflow",
    )


def test_read_metadata_extracts_png_text_chunks_and_dimensions(tmp_path):
    from smart_metadata_reader.metadata_reader import read_metadata

    image_path = tmp_path / "sample.png"
    prompt_graph = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "positive trigger"},
        }
    }
    workflow_graph = {
        "nodes": [
            {
                "id": 1,
                "type": "CLIPTextEncode",
                "inputs": {"text": "positive trigger"},
            }
        ]
    }
    pnginfo = PngInfo()
    pnginfo.add_text("prompt", json.dumps(prompt_graph))
    pnginfo.add_text("workflow", json.dumps(workflow_graph))
    pnginfo.add_text(
        "parameters",
        (
            "positive\n"
            "Negative prompt: negative\n"
            "Steps: 20, Sampler: Euler, CFG scale: 7, Seed: 123, Size: 512x768"
        ),
    )

    Image.new("RGBA", (64, 32), (255, 0, 0, 128)).save(
        image_path, pnginfo=pnginfo
    )

    bundle = read_metadata(image_path)

    assert bundle.filename == "sample.png"
    assert bundle.width == 64
    assert bundle.height == 32
    assert json.loads(bundle.prompt_raw) == prompt_graph
    assert json.loads(bundle.workflow_raw) == workflow_graph
    assert "Negative prompt: negative" in bundle.parameters_raw
    assert bundle.source_format == "ComfyUI prompt/workflow"


def test_load_image_and_mask_returns_batched_image_and_mask_shapes(tmp_path):
    from smart_metadata_reader.image_io import load_image_and_mask

    image_path = tmp_path / "transparent.png"
    Image.new("RGBA", (10, 12), (10, 20, 30, 128)).save(image_path)

    image, mask, width, height = load_image_and_mask(image_path)

    assert width == 10
    assert height == 12
    assert tuple(image.shape) == (1, 12, 10, 3)
    assert tuple(mask.shape) == (1, 12, 10)


def test_parse_standard_a1111_parameters():
    from smart_metadata_reader.a1111_parser import parse_a1111_parameters

    parameters = (
        "masterpiece, character trigger\n"
        "Negative prompt: blurry, low quality\n"
        "Steps: 25, Sampler: Euler a, CFG scale: 7.5, Seed: 12345, "
        "Size: 768x1024, Model: demo_model"
    )

    result = parse_a1111_parameters(parameters)

    assert result.positive == "masterpiece, character trigger"
    assert result.negative == "blurry, low quality"
    assert result.steps == 25
    assert result.sampler_name == "Euler a"
    assert result.cfg == 7.5
    assert result.seed == 12345
    assert result.width == 768
    assert result.height == 1024
    assert result.model_name == "demo_model"
    assert result.status_message == "A1111 parameters fallback"
    assert result.partial_result["source_format"] == "A1111 parameters"


def test_parse_a1111_preserves_multiline_positive_and_negative_prompts():
    from smart_metadata_reader.a1111_parser import parse_a1111_parameters

    parameters = (
        "masterpiece, best quality\n"
        "character trigger, detailed eyes\n"
        "Negative prompt: low quality\n"
        "bad hands, extra fingers\n"
        "Steps: 30, Sampler: DPM++ 2M, CFG scale: 6, Seed: 999, Size: 512x768"
    )

    result = parse_a1111_parameters(parameters)

    assert result.positive == "masterpiece, best quality\ncharacter trigger, detailed eyes"
    assert result.negative == "low quality\nbad hands, extra fingers"
    assert result.steps == 30
    assert result.seed == 999


def test_parse_a1111_preserves_lora_triggers_nsfw_and_user_tags():
    from smart_metadata_reader.a1111_parser import parse_a1111_parameters

    parameters = (
        "<lora:role_lora:0.8>, role_lora_trigger, nsfw, nude, user handwritten tag\n"
        "Negative prompt: low quality\n"
        "Steps: 20, Sampler: Euler, CFG scale: 7, Seed: 101, Size: 640x960"
    )

    result = parse_a1111_parameters(parameters)

    assert result.positive == (
        "<lora:role_lora:0.8>, role_lora_trigger, nsfw, nude, user handwritten tag"
    )
    assert result.negative == "low quality"


def test_parse_a1111_without_negative_prompt_keeps_positive_and_empty_negative():
    from smart_metadata_reader.a1111_parser import parse_a1111_parameters

    parameters = (
        "solo, dramatic lighting\n"
        "Steps: 12, Sampler: Euler, CFG scale: 4.5, Seed: 42, Size: 320x448"
    )

    result = parse_a1111_parameters(parameters)

    assert result.positive == "solo, dramatic lighting"
    assert result.negative == ""
    assert result.steps == 12
    assert result.seed == 42


def test_parse_a1111_missing_partial_parameters_uses_defaults():
    from smart_metadata_reader.a1111_parser import parse_a1111_parameters

    result = parse_a1111_parameters("simple prompt only")

    assert result.positive == "simple prompt only"
    assert result.negative == ""
    assert result.seed == -1
    assert result.steps == 0
    assert result.cfg == 0.0
    assert result.width == 0
    assert result.height == 0
    assert result.model_name == ""


def test_parse_a1111_scheduler_and_model_hash_metadata():
    from smart_metadata_reader.a1111_parser import parse_a1111_parameters

    parameters = (
        "positive text\n"
        "Negative prompt: negative text\n"
        "Steps: 25, Sampler: DPM++ 2M, Scheduler: Karras, CFG scale: 7.25, "
        "Seed: 123, Size: 768x1024, Model hash: abc123, Model: forge_model"
    )

    result = parse_a1111_parameters(parameters)

    assert result.scheduler == "Karras"
    assert result.model_name == "forge_model"
    assert result.partial_result["model_hash"] == "abc123"


def test_parse_comfyui_ultimate_sd_upscale_sampler_like_node():
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle
    from smart_metadata_reader.models import MetadataBundle

    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base_model.safetensors"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": "upscale_detail.safetensors",
                "strength_model": 0.5,
                "strength_clip": 0.75,
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "main_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "role_lora_trigger, 1girl, nsfw"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality"}},
        "6": {
            "class_type": "UltimateSDUpscaleNoUpscale",
            "inputs": {
                "model": ["2", 0],
                "clip": ["2", 1],
                "vae": ["3", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "seed": 221,
                "steps": 9,
                "cfg_scale": 1.5,
                "sampler": "Euler",
                "scheduler_name": "normal",
                "tile_width": 768,
                "seam_fix_mode": "half tile",
                "upscale_model": ["9", 0],
            },
        },
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}},
        "9": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x.pth"}},
    }
    bundle = MetadataBundle(
        filename="usdu.png",
        width=4800,
        height=6400,
        prompt_raw=json.dumps(prompt),
        workflow_raw=None,
        parameters_raw=None,
        source_format="ComfyUI prompt/workflow",
    )

    result = parse_metadata_bundle(bundle)

    assert result.positive == "role_lora_trigger, 1girl, nsfw"
    assert result.negative == "low quality"
    assert result.seed == 221
    assert result.steps == 9
    assert result.cfg == 1.5
    assert result.sampler_name == "Euler"
    assert result.scheduler == "normal"
    assert result.model_name == "base_model.safetensors"
    assert result.vae_name == "main_vae.safetensors"
    assert [lora.lora_name for lora in result.loras] == ["upscale_detail.safetensors"]
    assert "UltimateSDUpscaleNoUpscale 6" in result.debug_trace
    assert "tile_width" not in result.positive
    assert result.status_message == "OK"


def test_failed_comfyui_parse_includes_output_chain_diagnostics_in_setting():
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle
    from smart_metadata_reader.models import MetadataBundle

    prompt = {
        "1": {"class_type": "UnsupportedGenerator", "inputs": {"image": ["3", 0]}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
    }
    bundle = MetadataBundle(
        filename="unsupported.png",
        width=512,
        height=768,
        prompt_raw=json.dumps(prompt),
        workflow_raw=json.dumps({"nodes": []}),
        parameters_raw=None,
        source_format="ComfyUI prompt/workflow",
    )

    result = parse_metadata_bundle(bundle)

    assert result.status_message == "FAILED"
    assert "no supported sampler candidate found" in result.debug_trace
    assert "SaveImage 2.images <- UnsupportedGenerator 1" in result.debug_trace
    assert "Failure Reason:" in result.setting
    assert "final output nodes found: SaveImage 2" in result.setting


def test_read_metadata_extracts_ascii_and_unicode_exif_user_comment(tmp_path):
    from smart_metadata_reader.metadata_reader import read_metadata

    ascii_path = tmp_path / "ascii.jpeg"
    unicode_path = tmp_path / "unicode.jpeg"
    _save_jpeg_with_user_comment(ascii_path, "ASCII prompt\nSteps: 9, Seed: 1")
    _save_jpeg_with_user_comment(
        unicode_path,
        "UNICODE prompt\nSteps: 10, Seed: 2",
        prefix=b"UNICODE\x00",
    )

    ascii_bundle = read_metadata(ascii_path)
    unicode_bundle = read_metadata(unicode_path)

    assert ascii_bundle.user_comment_raw == "ASCII prompt\nSteps: 9, Seed: 1"
    assert unicode_bundle.user_comment_raw == "UNICODE prompt\nSteps: 10, Seed: 2"
    assert ascii_bundle.source_format == "EXIF UserComment"
    assert unicode_bundle.source_format == "EXIF UserComment"


def test_exif_user_comment_civitai_metadata_fallback(tmp_path):
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata

    positive = (
        "A green tree frog wearing a tiny straw hat sitting on a lily pad with "
        "a fishing rod made from a twig and spider silk."
    )
    resources = [
        {
            "type": "checkpoint",
            "modelVersionId": 2442439,
            "modelName": "Z Image Turbo",
            "modelVersionName": "Turbo",
        },
        {
            "type": "lora",
            "weight": 0.5,
            "modelVersionId": 2460437,
            "modelName": "Midjourney Luneva Cinematic Lora",
            "modelVersionName": "LORA: R128-5000step",
        },
        {
            "type": "lora",
            "weight": 0.8,
            "modelVersionId": 2515203,
            "modelName": "[ZIT] Detail Slider",
            "modelVersionName": "v1.0",
        },
    ]
    civitai_metadata = {
        "workflow": "txt2img",
        "cfgScale": 1,
        "steps": 9,
        "seed": 2214950711,
        "prompt": positive,
        "resources": [
            {"modelVersionId": 2442439, "strength": 1, "type": "Checkpoint"},
            {"modelVersionId": 2460437, "strength": 0.5, "type": "LORA"},
            {"modelVersionId": 2515203, "strength": 0.8, "type": "LORA"},
        ],
    }
    user_comment = (
        f"{positive}\n"
        "Steps: 9, Sampler: Euler, CFG scale: 1, Seed: 2214950711, "
        "Size: 832x1216, Model type: Z-Image Turbo, "
        f"Civitai resources: {json.dumps(resources)}, "
        f"Civitai metadata: {json.dumps(civitai_metadata)}"
    )
    image_path = tmp_path / "civitai.jpeg"
    _save_jpeg_with_user_comment(image_path, user_comment, prefix=b"UNICODE\x00")

    bundle = read_metadata(image_path)
    result = parse_metadata_bundle(bundle)

    assert bundle.user_comment_raw.startswith("A green tree frog")
    assert result.positive == positive
    assert result.negative == ""
    assert result.seed == 2214950711
    assert result.steps == 9
    assert result.cfg == 1.0
    assert result.sampler_name == "Euler"
    assert result.width == 832
    assert result.height == 1216
    assert result.model_name == "Z Image Turbo"
    assert [(lora.lora_name, lora.strength_model) for lora in result.loras] == [
        ("Midjourney Luneva Cinematic Lora", 0.5),
        ("[ZIT] Detail Slider", 0.8),
    ]
    assert result.partial_result["source_format"] == "EXIF UserComment / Civitai metadata"
    assert result.status_message == "EXIF UserComment / Civitai metadata fallback"
    assert "NO_METADATA" not in result.setting


def test_nested_smart_metadata_reader_outputs_resolve_referenced_image_prompts(tmp_path):
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata

    original_path = tmp_path / "003.png"
    _save_png_with_prompt(original_path, _simple_prompt_graph())

    upscale_path = tmp_path / "upscaled.png"
    _save_png_with_prompt(
        upscale_path,
        _upscale_prompt_graph(),
        workflow=_upscale_workflow(),
    )

    result = parse_metadata_bundle(read_metadata(upscale_path))

    assert result.positive == "original positive prompt"
    assert result.negative == "original negative prompt"
    assert result.status_message != "FAILED"
    assert not (result.status_message == "OK" and result.positive == "" and result.negative == "")
    assert "Resolved SmartMetadataReader positive output from referenced image 003.png" in result.debug_trace
    assert "Resolved SmartMetadataReader negative output from referenced image 003.png" in result.debug_trace


def test_unconnected_showtext_workflow_cache_does_not_pollute_nested_reader_prompt(tmp_path):
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata

    original_path = tmp_path / "003.png"
    _save_png_with_prompt(original_path, _simple_prompt_graph())
    upscale_path = tmp_path / "upscaled.png"
    _save_png_with_prompt(
        upscale_path,
        _upscale_prompt_graph(),
        workflow=_upscale_workflow(),
    )

    result = parse_metadata_bundle(read_metadata(upscale_path))

    assert result.positive == "original positive prompt"
    assert "wrong unused prompt" not in result.positive
    assert "wrong unused prompt" not in result.negative


def test_nested_smart_metadata_reader_falls_back_to_current_output_contract_without_workflow_outputs(tmp_path):
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata

    original_path = tmp_path / "003.png"
    _save_png_with_prompt(original_path, _simple_prompt_graph())
    upscale_path = tmp_path / "upscaled.png"
    _save_png_with_prompt(upscale_path, _upscale_prompt_graph())

    result = parse_metadata_bundle(read_metadata(upscale_path))

    assert result.positive == "original positive prompt"
    assert result.negative == "original negative prompt"


def test_nested_smart_metadata_reader_prefers_workflow_output_names_over_indices(tmp_path):
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata

    original_path = tmp_path / "003.png"
    _save_png_with_prompt(original_path, _simple_prompt_graph())
    prompt = _upscale_prompt_graph()
    prompt["37"]["inputs"]["text"] = ["36", 3]
    prompt["38"]["inputs"]["text"] = ["36", 2]
    workflow = _upscale_workflow()
    workflow["nodes"][0]["outputs"][2] = {"name": "negative"}
    workflow["nodes"][0]["outputs"][3] = {"name": "positive"}
    upscale_path = tmp_path / "upscaled.png"
    _save_png_with_prompt(upscale_path, prompt, workflow=workflow)

    result = parse_metadata_bundle(read_metadata(upscale_path))

    assert result.positive == "original positive prompt"
    assert result.negative == "original negative prompt"


def test_nested_smart_metadata_reader_missing_image_returns_partial_unresolved(tmp_path):
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata

    upscale_path = tmp_path / "upscaled.png"
    _save_png_with_prompt(
        upscale_path,
        _upscale_prompt_graph(reader_image="missing.png"),
        workflow=_upscale_workflow(reader_image="missing.png"),
    )

    result = parse_metadata_bundle(read_metadata(upscale_path))

    assert result.positive == ""
    assert result.negative == ""
    assert result.status_message == "PARTIAL"
    assert result.confidence <= 0.6
    assert result.partial_result["unresolved"][0]["class_type"] == "SmartMetadataReader"
    assert result.partial_result["unresolved"][0]["output_index"] == 2
    assert result.partial_result["unresolved"][0]["resolved_output_role"] == "positive"
    assert result.partial_result["unresolved"][0]["referenced_image_filename"] == "missing.png"
    assert "referenced image not found" in result.partial_result["unresolved"][0]["reason"]
    assert "Unresolved:" in result.setting


def test_nested_smart_metadata_reader_rejects_unsafe_image_paths(tmp_path):
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata

    for unsafe_path in ("../secret.png", "C:\\Users\\xxx\\secret.png"):
        upscale_path = tmp_path / f"upscaled-{unsafe_path[0].encode().hex()}.png"
        _save_png_with_prompt(
            upscale_path,
            _upscale_prompt_graph(reader_image=unsafe_path),
            workflow=_upscale_workflow(reader_image=unsafe_path),
        )

        result = parse_metadata_bundle(read_metadata(upscale_path))

        assert result.positive == ""
        assert result.status_message == "PARTIAL"
        assert result.partial_result["unresolved"][0]["class_type"] == "SmartMetadataReader"
        assert "unsafe referenced image path" in result.partial_result["unresolved"][0]["reason"]


def test_nested_smart_metadata_reader_respects_actual_connected_output_role(tmp_path):
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata

    original_path = tmp_path / "003.png"
    _save_png_with_prompt(original_path, _simple_prompt_graph())
    prompt = _upscale_prompt_graph()
    prompt["37"]["inputs"]["text"] = ["36", 3]
    upscale_path = tmp_path / "upscaled.png"
    _save_png_with_prompt(upscale_path, prompt, workflow=_upscale_workflow())

    result = parse_metadata_bundle(read_metadata(upscale_path))

    assert result.positive == "original negative prompt"
    assert "SmartMetadataReader output negative used in positive chain" in result.debug_trace


def test_showtext_cache_from_llm_chain_is_used_without_reading_llm_template():
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle

    result = parse_metadata_bundle(
        _bundle_for_prompt(
            _llm_showtext_upscale_prompt(),
            workflow=_llm_showtext_workflow(),
        )
    )

    assert result.positive == "cached gemini positive prompt"
    assert result.negative == "bad hands, low quality"
    assert "Gemini prompt template" not in result.positive
    assert "system instruction" not in result.positive
    assert "secret-key" not in result.positive
    assert result.status_message != "FAILED"
    assert "using cached ShowText text" in result.debug_trace
    assert "wrong unused prompt" not in result.positive


def test_showtext_cache_compatible_class_from_llm_chain_is_used():
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle

    result = parse_metadata_bundle(
        _bundle_for_prompt(
            _llm_showtext_upscale_prompt(show_class="ShowText"),
            workflow=_llm_showtext_workflow(show_class="ShowText"),
        )
    )

    assert result.positive == "cached gemini positive prompt"
    assert "Gemini prompt template" not in result.positive


def test_showtext_cache_is_single_segment_inside_string_function_chain():
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle

    prompt = _llm_showtext_upscale_prompt()
    prompt["26"]["inputs"]["text"] = ["39", 0]
    prompt["39"] = {
        "class_type": "StringFunction|pysssss",
        "inputs": {
            "action": "append",
            "text_a": "manual prompt",
            "text_b": ["37", 0],
            "text_c": "preset prompt",
        },
    }

    result = parse_metadata_bundle(
        _bundle_for_prompt(prompt, workflow=_llm_showtext_workflow())
    )

    assert "manual prompt" in result.positive
    assert "cached gemini positive prompt" in result.positive
    assert "preset prompt" in result.positive
    assert "Gemini prompt template" not in result.positive
    assert "system instruction" not in result.positive
    assert result.positive.index("manual prompt") < result.positive.index("cached gemini positive prompt")
    assert result.positive.index("cached gemini positive prompt") < result.positive.index("preset prompt")


def test_conditioning_combine_showtext_cache_does_not_override_other_branches():
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle

    prompt = _llm_showtext_upscale_prompt()
    prompt["24"]["inputs"]["positive"] = ["41", 0]
    prompt["41"] = {
        "class_type": "ConditioningCombine",
        "inputs": {"conditioning_1": ["42", 0], "conditioning_2": ["43", 0]},
    }
    prompt["42"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "branch A manual prompt"}}
    prompt["43"] = {"class_type": "CLIPTextEncode", "inputs": {"text": ["37", 0]}}

    result = parse_metadata_bundle(
        _bundle_for_prompt(prompt, workflow=_llm_showtext_workflow())
    )

    assert "branch A manual prompt" in result.positive
    assert "cached gemini positive prompt" in result.positive
    assert "Gemini prompt template" not in result.positive


def test_showtext_missing_cache_with_llm_upstream_returns_partial_unresolved():
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle

    result = parse_metadata_bundle(
        _bundle_for_prompt(
            _llm_showtext_upscale_prompt(include_cache=False),
            workflow=_llm_showtext_workflow(cached_text=""),
        )
    )

    assert result.positive == ""
    assert "Gemini prompt template" not in result.positive
    assert "system instruction" not in result.positive
    assert result.status_message == "PARTIAL"
    assert result.confidence <= 0.6
    assert result.partial_result["unresolved"][0]["class_type"] == "ShowText|pysssss"
    assert (
        result.partial_result["unresolved"][0]["reason"]
        == "ShowText cache missing and upstream is LLM runtime output not embedded"
    )
    assert "Unresolved:" in result.setting
