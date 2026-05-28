import json
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from smart_metadata_reader.models import LoraInfo, MetadataBundle, ParseResult


def test_format_setting_outputs_complete_human_readable_text():
    from smart_metadata_reader.settings_formatter import format_setting

    result = ParseResult(
        filename="001.png",
        model_name="model.safetensors",
        vae_name="vae.vae.safetensors",
        loras=[
            LoraInfo(
                node_id="10",
                class_type="LoraLoader",
                lora_name="lora_a.safetensors",
                strength_model=0.8,
                strength_clip=1.0,
                path=["KSampler 9.model", "LoraLoader 10"],
            ),
            LoraInfo(
                node_id="11",
                class_type="LoraLoader",
                lora_name="lora_b.safetensors",
                strength_model=0.6,
                strength_clip=0.75,
                path=["KSampler 9.model", "LoraLoader 11"],
            ),
        ],
        seed=123456789,
        steps=30,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        width=1200,
        height=1600,
        status_message="OK",
        confidence=0.94,
        partial_result={"source_format": "ComfyUI prompt/workflow"},
    )

    setting = format_setting(result)

    assert "Filename: 001.png" in setting
    assert "Source: ComfyUI prompt/workflow" in setting
    assert "Model: model.safetensors" in setting
    assert "VAE: vae.vae.safetensors" in setting
    assert "LoRA:" in setting
    assert "* lora_a.safetensors (model 0.80, clip 1.00)" in setting
    assert "* lora_b.safetensors (model 0.60, clip 0.75)" in setting
    assert "Seed: 123456789" in setting
    assert "Steps: 30" in setting
    assert "CFG: 7.0" in setting
    assert "Sampler: euler" in setting
    assert "Scheduler: normal" in setting
    assert "Size: 1200 x 1600" in setting
    assert "Status: OK" in setting
    assert "Confidence: 0.94" in setting


def test_format_setting_handles_missing_fields_without_error():
    from smart_metadata_reader.settings_formatter import format_setting

    setting = format_setting(ParseResult())

    assert "Filename: unknown" in setting
    assert "Source: unknown" in setting
    assert "Model: unknown" in setting
    assert "VAE: unknown" in setting
    assert "LoRA: none" in setting
    assert "Seed: -1" in setting
    assert "Steps: 0" in setting
    assert "CFG: 0.0" in setting
    assert "Sampler: unknown" in setting
    assert "Scheduler: unknown" in setting
    assert "Size: unknown" in setting


def test_format_setting_displays_unresolved_partial_result_entries():
    from smart_metadata_reader.settings_formatter import format_setting

    result = ParseResult(
        filename="partial.png",
        status_message="PARTIAL",
        confidence=0.52,
        partial_result={
            "source_format": "ComfyUI prompt/workflow",
            "unresolved": [
                {
                    "node_id": "88",
                    "class_type": "UnknownConditioningRouter",
                    "field": "positive_conditioning",
                    "role": "positive",
                    "reason": "no conditioning adapter matched",
                }
            ],
        },
    )

    setting = format_setting(result)

    assert "Status: PARTIAL" in setting
    assert "Confidence: 0.52" in setting
    assert "Unresolved:" in setting
    assert (
        "* Node 88 UnknownConditioningRouter input positive_conditioning "
        "role=positive reason=no conditioning adapter matched"
    ) in setting


def test_format_setting_displays_a1111_source_and_status():
    from smart_metadata_reader.settings_formatter import format_setting

    result = ParseResult(
        filename="",
        positive="<lora:role:0.8>, nsfw",
        negative="low quality",
        seed=123,
        steps=25,
        cfg=7.5,
        sampler_name="Euler a",
        width=768,
        height=1024,
        model_name="demo_model",
        status_message="A1111 parameters fallback",
        confidence=0.75,
        partial_result={"source_format": "A1111 parameters"},
    )

    setting = format_setting(result)

    assert "Source: A1111 parameters" in setting
    assert "Status: A1111 parameters fallback" in setting
    assert "Model: demo_model" in setting


def test_format_setting_is_plain_text_not_json():
    from smart_metadata_reader.settings_formatter import format_setting

    setting = format_setting(
        ParseResult(
            filename="plain.png",
            partial_result={"source_format": "ComfyUI prompt/workflow"},
        )
    )

    assert not setting.strip().startswith("{")
    with pytest.raises(json.JSONDecodeError):
        json.loads(setting)


def test_node_mappings_expose_smart_metadata_reader():
    import nodes

    assert nodes.NODE_CLASS_MAPPINGS["SmartMetadataReader"] is nodes.SmartMetadataReader
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["SmartMetadataReader"] == "Smart Metadata Reader"


def test_plugin_root_init_exports_comfyui_mappings():
    root_dir = pytest.importorskip("pathlib").Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "comfyui_smart_metadata_reader_plugin",
        root_dir / "__init__.py",
        submodule_search_locations=[str(root_dir)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        sys.modules.pop(f"{spec.name}.nodes", None)

    assert "SmartMetadataReader" in module.NODE_CLASS_MAPPINGS
    assert module.NODE_DISPLAY_NAME_MAPPINGS["SmartMetadataReader"] == "Smart Metadata Reader"


def test_input_types_prefers_comfyui_filename_list(monkeypatch):
    import nodes

    monkeypatch.setattr(
        nodes,
        "folder_paths",
        SimpleNamespace(get_filename_list=lambda category: ["z.png", "nested/a.webp"]),
    )

    input_types = nodes.SmartMetadataReader.INPUT_TYPES()

    assert input_types["required"]["image"][0] == ["z.png", "nested/a.webp"]


def test_smart_metadata_reader_input_types_include_uploadable_image(tmp_path, monkeypatch):
    import nodes

    (tmp_path / "a.png").write_bytes(b"fake")
    (tmp_path / "b.webp").write_bytes(b"fake")
    (tmp_path / "note.txt").write_text("ignore me")

    monkeypatch.setattr(
        nodes,
        "folder_paths",
        SimpleNamespace(get_input_directory=lambda: str(tmp_path)),
    )

    input_types = nodes.SmartMetadataReader.INPUT_TYPES()
    required = input_types["required"]

    assert required["image"][0] == ["a.png", "b.webp"]
    assert required["image"][1]["image_upload"] is True
    assert required["parameter_index"][0] == "INT"
    assert required["parameter_index"][1]["default"] == 0
    assert required["parameter_index"][1]["min"] == 0
    assert required["prefer_cached_text"] == ("BOOLEAN", {"default": True})
    assert required["include_raw_json"] == ("BOOLEAN", {"default": True})
    assert required["max_depth"][0] == "INT"
    assert required["max_depth"][1]["default"] == 40
    assert required["max_depth"][1]["min"] == 1
    assert required["max_depth"][1]["max"] == 200


def test_smart_metadata_reader_return_contract():
    import nodes

    node_cls = nodes.SmartMetadataReader

    assert node_cls.RETURN_TYPES == (
        "IMAGE",
        "MASK",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
    )
    assert node_cls.RETURN_NAMES == (
        "image",
        "mask",
        "positive",
        "negative",
        "seed",
        "steps",
        "cfg",
        "width",
        "height",
        "model_name",
        "filename",
        "setting",
    )
    assert len(node_cls.RETURN_TYPES) == len(node_cls.RETURN_NAMES)
    assert hasattr(node_cls, node_cls.FUNCTION)
    assert node_cls.CATEGORY


def test_smart_metadata_reader_method_returns_expected_tuple_order(tmp_path, monkeypatch):
    import nodes

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake image bytes")

    calls = {}

    monkeypatch.setattr(
        nodes,
        "folder_paths",
        SimpleNamespace(get_annotated_filepath=lambda image: str(image_path)),
    )

    def fake_read_metadata(path):
        calls["read_metadata"] = path
        return MetadataBundle(
            filename="sample.png",
            width=1200,
            height=1600,
            prompt_raw="{}",
            workflow_raw=None,
            parameters_raw=None,
            source_format="ComfyUI prompt/workflow",
        )

    def fake_parse_metadata_bundle(
        bundle,
        parameter_index,
        prefer_cached_text,
        include_raw_json,
        max_depth,
    ):
        calls["parse"] = {
            "bundle": bundle,
            "parameter_index": parameter_index,
            "prefer_cached_text": prefer_cached_text,
            "include_raw_json": include_raw_json,
            "max_depth": max_depth,
        }
        return ParseResult(
            positive="real positive",
            negative="real negative",
            seed=123,
            steps=30,
            cfg=7.5,
            width=1200,
            height=1600,
            model_name="model.safetensors",
            filename="sample.png",
            setting="Filename: sample.png\nStatus: OK",
        )

    monkeypatch.setattr(nodes, "read_metadata", fake_read_metadata)
    monkeypatch.setattr(nodes, "parse_metadata_bundle", fake_parse_metadata_bundle)
    monkeypatch.setattr(nodes, "load_image_and_mask", lambda path: ("IMAGE_TENSOR", "MASK_TENSOR", 1200, 1600))

    output = nodes.SmartMetadataReader().read_metadata(
        "sample.png",
        parameter_index=2,
        prefer_cached_text=False,
        include_raw_json=False,
        max_depth=12,
    )

    assert output == (
        "IMAGE_TENSOR",
        "MASK_TENSOR",
        "real positive",
        "real negative",
        123,
        30,
        7.5,
        1200,
        1600,
        "model.safetensors",
        "sample.png",
        "Filename: sample.png\nStatus: OK",
    )
    assert calls["read_metadata"] == str(image_path)
    assert calls["parse"]["parameter_index"] == 2
    assert calls["parse"]["prefer_cached_text"] is False
    assert calls["parse"]["include_raw_json"] is False
    assert calls["parse"]["max_depth"] == 12


def test_smart_metadata_reader_parse_failure_returns_failed_setting(tmp_path, monkeypatch):
    import nodes

    image_path = tmp_path / "broken-metadata.png"
    image_path.write_bytes(b"fake image bytes")

    monkeypatch.setattr(
        nodes,
        "folder_paths",
        SimpleNamespace(get_annotated_filepath=lambda image: str(image_path)),
    )
    monkeypatch.setattr(
        nodes,
        "read_metadata",
        lambda path: MetadataBundle(
            filename="broken-metadata.png",
            width=640,
            height=480,
            prompt_raw=None,
            workflow_raw=None,
            parameters_raw=None,
            source_format="none",
        ),
    )
    monkeypatch.setattr(nodes, "parse_metadata_bundle", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")))
    monkeypatch.setattr(nodes, "load_image_and_mask", lambda path: ("IMAGE_TENSOR", "MASK_TENSOR", 640, 480))

    output = nodes.SmartMetadataReader().read_metadata("broken-metadata.png", 0, True, True, 40)

    assert output[:2] == ("IMAGE_TENSOR", "MASK_TENSOR")
    assert output[2] == ""
    assert output[3] == ""
    assert output[4] == -1
    assert output[5] == 0
    assert output[6] == 0.0
    assert output[7] == 640
    assert output[8] == 480
    assert output[9] == ""
    assert output[10] == "broken-metadata.png"
    assert "Status: FAILED" in output[11]
    assert "boom" in output[11]


def test_smart_metadata_reader_image_load_failure_raises_clear_error(monkeypatch):
    import nodes

    monkeypatch.setattr(
        nodes,
        "folder_paths",
        SimpleNamespace(get_annotated_filepath=lambda image: "missing.png"),
    )
    monkeypatch.setattr(
        nodes,
        "load_image_and_mask",
        lambda path: (_ for _ in ()).throw(OSError("cannot identify image file")),
    )

    with pytest.raises(RuntimeError, match="Failed to load image 'missing.png'"):
        nodes.SmartMetadataReader().read_metadata("missing.png", 0, True, True, 40)


def test_smart_metadata_reader_is_changed_tracks_file_content(tmp_path, monkeypatch):
    import nodes

    image_path = tmp_path / "changed.png"
    image_path.write_bytes(b"first")
    monkeypatch.setattr(
        nodes,
        "folder_paths",
        SimpleNamespace(get_annotated_filepath=lambda image: str(image_path)),
    )

    first = nodes.SmartMetadataReader.IS_CHANGED("changed.png")
    image_path.write_bytes(b"second")
    second = nodes.SmartMetadataReader.IS_CHANGED("changed.png")

    assert first
    assert second
    assert first != second


def test_smart_metadata_reader_validate_inputs_checks_image_exists(tmp_path, monkeypatch):
    import nodes

    existing = tmp_path / "exists.png"
    existing.write_bytes(b"fake")
    missing = tmp_path / "missing.png"
    monkeypatch.setattr(
        nodes,
        "folder_paths",
        SimpleNamespace(get_annotated_filepath=lambda image: str(tmp_path / image)),
    )

    assert nodes.SmartMetadataReader.VALIDATE_INPUTS("exists.png") is True
    invalid = nodes.SmartMetadataReader.VALIDATE_INPUTS("missing.png")
    assert isinstance(invalid, str)
    assert "Invalid image file" in invalid
    assert str(missing) in invalid


def test_smart_metadata_reader_validate_inputs_uses_comfyui_exists_helper(monkeypatch):
    import nodes

    monkeypatch.setattr(
        nodes,
        "folder_paths",
        SimpleNamespace(exists_annotated_filepath=lambda image: image == "ok.png"),
    )

    assert nodes.SmartMetadataReader.VALIDATE_INPUTS("ok.png") is True
    assert "Invalid image file" in nodes.SmartMetadataReader.VALIDATE_INPUTS("bad.png")
