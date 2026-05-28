import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo


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
