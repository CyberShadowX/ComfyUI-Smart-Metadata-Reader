from smart_metadata_reader.graph import GraphIndex


def select_sampler(prompt):
    from smart_metadata_reader.sampler_selector import select_final_sampler

    return select_final_sampler(GraphIndex(prompt=prompt, workflow=None))


def test_selects_ksampler_upstream_of_save_image_via_vae_decode():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["9", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
    }

    selection = select_sampler(prompt)

    assert selection.sampler.node_id == "1"
    assert selection.confidence == 1.0
    assert "SaveImage 3.images <- VAEDecode 2.samples <- KSampler 1" in selection.debug_trace


def test_selects_ksampler_upstream_of_preview_image_via_vae_decode():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["9", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
    }

    selection = select_sampler(prompt)

    assert selection.sampler.node_id == "1"
    assert "PreviewImage 3.images" in selection.debug_trace


def test_selects_sampler_through_image_passthrough_before_save_image():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["9", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
        "3": {"class_type": "ImagePassThrough", "inputs": {"image": ["2", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
    }

    selection = select_sampler(prompt)

    assert selection.sampler.node_id == "1"
    assert "ImagePassThrough 3.image" in selection.debug_trace


def test_multiple_samplers_selects_one_connected_to_final_save_image():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["9", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "8": {"class_type": "KSampler", "inputs": {"positive": ["9", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
    }

    selection = select_sampler(prompt)

    assert selection.sampler.node_id == "1"
    assert "Selected KSampler 1 because it is upstream of final SaveImage" in selection.debug_trace


def test_fallback_selects_sampler_with_conditioning_inputs_when_no_output_node_exists():
    prompt = {
        "5": {"class_type": "PrimitiveString", "inputs": {"text": "not a sampler"}},
        "6": {
            "class_type": "KSampler",
            "inputs": {"positive": ["9", 0], "negative": ["10", 0]},
        },
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative"}},
    }

    selection = select_sampler(prompt)

    assert selection.sampler.node_id == "6"
    assert selection.confidence < 1.0
    assert "sampler selected by fallback heuristic" in selection.debug_trace


def test_extracts_ksampler_settings():
    from smart_metadata_reader.adapters.samplers import extract_sampler_settings
    from smart_metadata_reader.models import NodeRecord

    sampler = NodeRecord(
        node_id="1",
        class_type="KSampler",
        inputs={
            "seed": 123,
            "steps": 30,
            "cfg": 7.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.85,
        },
    )

    assert extract_sampler_settings(sampler) == {
        "seed": 123,
        "steps": 30,
        "cfg": 7.5,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 0.85,
    }


def test_extracts_ksampler_advanced_and_custom_field_variants():
    from smart_metadata_reader.adapters.samplers import extract_sampler_settings
    from smart_metadata_reader.models import NodeRecord

    advanced = NodeRecord(
        node_id="2",
        class_type="KSamplerAdvanced",
        inputs={
            "noise_seed": 456,
            "steps": 22,
            "cfg_scale": 6.0,
            "sampler": "dpmpp_2m",
            "schedule": "karras",
        },
    )
    custom = NodeRecord(
        node_id="3",
        class_type="SamplerCustom",
        inputs={
            "noise_seed": 789,
            "steps": 18,
            "cfg": 5.5,
            "sampler_name": "custom_sampler",
            "scheduler": "sgm_uniform",
        },
    )

    assert extract_sampler_settings(advanced) == {
        "seed": 456,
        "steps": 22,
        "cfg": 6.0,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
    }
    assert extract_sampler_settings(custom) == {
        "seed": 789,
        "steps": 18,
        "cfg": 5.5,
        "sampler_name": "custom_sampler",
        "scheduler": "sgm_uniform",
    }


def test_selects_ultimate_sd_upscale_sampler_like_node_upstream_of_save_image():
    prompt = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative"}},
        "7": {
            "class_type": "UltimateSDUpscale",
            "inputs": {
                "positive": ["1", 0],
                "negative": ["2", 0],
                "model": ["9", 0],
                "vae": ["10", 0],
                "seed": 123,
                "steps": 12,
                "cfg": 2.5,
                "sampler_name": "euler",
                "scheduler_name": "normal",
                "tile_width": 768,
                "seam_fix_mode": "band pass",
            },
        },
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0]}},
        "9": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
    }

    selection = select_sampler(prompt)

    assert selection.sampler.node_id == "7"
    assert selection.sampler.class_type == "UltimateSDUpscale"
    assert selection.selected_by == "output_chain"
    assert "SaveImage 8.images <- UltimateSDUpscale 7" in selection.debug_trace


def test_unsupported_output_chain_error_includes_diagnostics():
    from smart_metadata_reader.sampler_selector import select_final_sampler

    prompt = {
        "1": {
            "class_type": "UnsupportedGenerator",
            "inputs": {"image": ["3", 0], "tile_width": 768},
        },
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
    }

    try:
        select_final_sampler(GraphIndex(prompt=prompt, workflow=None))
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected unsupported output chain to fail")

    assert "no supported sampler candidate found" in message
    assert "final output nodes found: SaveImage 2" in message
    assert "SaveImage 2.images <- UnsupportedGenerator 1" in message
    assert "input keys: image, tile_width" in message
