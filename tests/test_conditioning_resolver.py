import json

from smart_metadata_reader.graph import GraphIndex
from smart_metadata_reader.metadata_reader import parse_metadata_bundle
from smart_metadata_reader.models import MetadataBundle
from smart_metadata_reader.prompt_merge import merge_prompt_segments


def resolver_for(prompt, max_depth=40):
    from smart_metadata_reader.resolver import ConditioningResolver

    graph = GraphIndex(prompt=prompt, workflow=None)
    sampler = graph.get_node("1")
    resolver = ConditioningResolver(
        graph=graph,
        max_depth=max_depth,
        prefer_cached_text=True,
    )
    return resolver, sampler


def segment_texts(segments):
    return [segment.text for segment in segments]


def test_resolves_simple_ksampler_positive_to_clip_text():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["2", 0]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "simple positive"}},
    }

    resolver, sampler = resolver_for(prompt)

    assert segment_texts(resolver.resolve_positive(sampler)) == ["simple positive"]


def test_resolves_controlnet_apply_passthrough():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["3", 0]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "controlnet positive"}},
        "3": {
            "class_type": "ControlNetApply",
            "inputs": {"conditioning": ["2", 0]},
        },
    }

    resolver, sampler = resolver_for(prompt)

    assert segment_texts(resolver.resolve_positive(sampler)) == ["controlnet positive"]


def test_resolves_controlnet_apply_advanced_passthrough():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["3", 0]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "advanced positive"}},
        "3": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {"positive": ["2", 0]},
        },
    }

    resolver, sampler = resolver_for(prompt)

    assert segment_texts(resolver.resolve_positive(sampler)) == ["advanced positive"]


def test_resolves_inpaint_model_conditioning_passthrough():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["3", 0]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "inpaint positive"}},
        "3": {
            "class_type": "InpaintModelConditioning",
            "inputs": {"positive": ["2", 0]},
        },
    }

    resolver, sampler = resolver_for(prompt)

    assert segment_texts(resolver.resolve_positive(sampler)) == ["inpaint positive"]


def test_conditioning_combine_merges_both_branches():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["4", 0]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "first branch"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "second branch"}},
        "4": {
            "class_type": "ConditioningCombine",
            "inputs": {"conditioning_1": ["2", 0], "conditioning_2": ["3", 0]},
        },
    }

    resolver, sampler = resolver_for(prompt)
    segments = resolver.resolve_positive(sampler)

    assert segment_texts(segments) == ["first branch", "second branch"]
    assert merge_prompt_segments(segments) == "first branch, second branch"


def test_conditioning_concat_merges_multiple_clip_text_nodes_into_positive():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["4", 0]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "subject tags"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "style tags"}},
        "4": {
            "class_type": "ConditioningConcat",
            "inputs": {"conditioning_to": ["2", 0], "conditioning_from": ["3", 0]},
        },
    }

    resolver, sampler = resolver_for(prompt)

    assert segment_texts(resolver.resolve_positive(sampler)) == [
        "subject tags",
        "style tags",
    ]


def test_resolves_negative_chain_separately():
    prompt = {
        "1": {
            "class_type": "KSampler",
            "inputs": {"positive": ["2", 0], "negative": ["3", 0]},
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive text"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative text"}},
    }

    resolver, sampler = resolver_for(prompt)

    assert segment_texts(resolver.resolve_negative(sampler)) == ["negative text"]


def test_unknown_conditioning_node_returns_unresolved_without_prompt_text():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["9", 0]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "should not guess"}},
        "9": {
            "class_type": "UnknownConditioningRouter",
            "inputs": {"mystery": ["2", 0]},
        },
    }

    resolver, sampler = resolver_for(prompt)

    assert resolver.resolve_positive(sampler) == []
    assert resolver.unresolved[0]["node_id"] == "9"
    assert resolver.unresolved[0]["class_type"] == "UnknownConditioningRouter"
    assert resolver.unresolved[0]["field"] == "mystery"
    assert resolver.unresolved[0]["role"] == "positive"
    assert "UnknownConditioningRouter 9" in resolver.debug_trace


def test_cycle_guard_prevents_infinite_recursion():
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"positive": ["4", 0]}},
        "4": {"class_type": "ControlNetApply", "inputs": {"conditioning": ["5", 0]}},
        "5": {"class_type": "ControlNetApply", "inputs": {"conditioning": ["4", 0]}},
    }

    resolver, sampler = resolver_for(prompt, max_depth=10)

    assert resolver.resolve_positive(sampler) == []
    assert resolver.unresolved[0]["reason"] == "cycle"
    assert resolver.unresolved[0]["role"] == "positive"


def bundle_for_prompt(prompt, workflow=None, filename="sample.png", width=1200, height=1600):
    return MetadataBundle(
        filename=filename,
        width=width,
        height=height,
        prompt_raw=json.dumps(prompt),
        workflow_raw=json.dumps(workflow) if workflow is not None else None,
        parameters_raw=None,
        source_format="ComfyUI prompt/workflow",
    )


def test_parse_metadata_bundle_resolves_full_comfyui_graph_end_to_end():
    prompt = {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 0]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "clip": ["6", 1],
                "positive": ["10", 0],
                "negative": ["20", 0],
                "seed": 123456789,
                "steps": 30,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "final.vae.safetensors"}},
        "5": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "anime_model.safetensors"},
        },
        "6": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["5", 0],
                "clip": ["5", 1],
                "lora_name": "role_lora.safetensors",
                "strength_model": 0.8,
                "strength_clip": 1.0,
            },
        },
        "10": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["11", 0],
                "image": "control image filename should not enter prompt",
                "strength": 0.7,
                "control_net": "control_v11p_sd15",
            },
        },
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": ["12", 0]}},
        "12": {
            "class_type": "StringFunction|pysssss",
            "inputs": {
                "action": "append",
                "text_a": "role_lora_trigger, character_name, NSFW",
                "text_b": ["13", 0],
                "text_c": ["14", 0],
            },
        },
        "13": {
            "class_type": "ShowText|pysssss",
            "inputs": {
                "text": ["15", 0],
                "text_0": "1girl, solo, squatting, stage lights, anime style",
            },
        },
        "14": {
            "class_type": "DeepTranslatorTextNode",
            "inputs": {"text": "translated supplement, dramatic lighting"},
        },
        "15": {
            "class_type": "GeminiChatNode",
            "inputs": {
                "system_instruction": "You are an anime image prompt extraction assistant",
                "prompt": "Analyze this image and convert it into a prompt template",
            },
        },
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": ["21", 0]}},
        "21": {
            "class_type": "StringFunction|pysssss",
            "inputs": {"action": "append", "text_a": "bad hands, low quality"},
        },
        "50": {
            "class_type": "GeminiChatNode",
            "inputs": {"prompt": "unconnected prompt template must be ignored"},
        },
        "51": {
            "class_type": "PrimitiveString",
            "inputs": {"value": "unconnected primitive text must be ignored"},
        },
    }

    result = parse_metadata_bundle(bundle_for_prompt(prompt))

    assert "role_lora_trigger" in result.positive
    assert "character_name" in result.positive
    assert "NSFW" in result.positive
    assert "1girl, solo, squatting, stage lights, anime style" in result.positive
    assert "translated supplement, dramatic lighting" in result.positive
    assert result.negative == "bad hands, low quality"
    assert "Analyze this image" not in result.positive
    assert "control_v11p_sd15" not in result.positive
    assert "control image filename" not in result.positive
    assert "unconnected prompt template" not in result.positive
    assert "unconnected primitive text" not in result.positive
    assert result.seed == 123456789
    assert result.steps == 30
    assert result.cfg == 7.0
    assert result.sampler_name == "euler"
    assert result.scheduler == "normal"
    assert result.model_name == "anime_model.safetensors"
    assert result.vae_name == "final.vae.safetensors"
    assert [lora.lora_name for lora in result.loras] == ["role_lora.safetensors"]
    assert result.status_message == "OK"
    assert result.confidence >= 0.9
    assert "Seed: 123456789" in result.setting
    assert "Model: anime_model.safetensors" in result.setting
    assert "SAMPLER_SELECTION" in result.debug_trace
    assert "POSITIVE_TRACE" in result.debug_trace
    assert "NEGATIVE_TRACE" in result.debug_trace


def test_parse_metadata_bundle_uses_a1111_parameters_when_comfyui_graph_cannot_start():
    bundle = MetadataBundle(
        filename="a1111.png",
        width=0,
        height=0,
        prompt_raw=None,
        workflow_raw=None,
        parameters_raw=(
            "role_lora_trigger, NSFW, cinematic lighting\n"
            "Negative prompt: low quality, bad hands\n"
            "Steps: 25, Sampler: Euler a, CFG scale: 7.5, "
            "Seed: 12345, Size: 768x1024, Model: demo_model"
        ),
        source_format="A1111 parameters",
    )

    result = parse_metadata_bundle(bundle)

    assert result.positive == "role_lora_trigger, NSFW, cinematic lighting"
    assert result.negative == "low quality, bad hands"
    assert result.seed == 12345
    assert result.steps == 25
    assert result.cfg == 7.5
    assert result.width == 768
    assert result.height == 1024
    assert result.model_name == "demo_model"
    assert result.status_message == "A1111 parameters fallback"
    assert "Source: A1111 parameters" in result.setting


def test_parse_metadata_bundle_omits_raw_json_when_requested_but_still_parses():
    prompt = {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["4", 0], "negative": ["5", 0]},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "visible positive"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "visible negative"}},
    }

    result = parse_metadata_bundle(bundle_for_prompt(prompt), include_raw_json=False)

    assert result.positive == "visible positive"
    assert result.negative == "visible negative"
    assert result.raw_prompt_json == ""
    assert result.raw_workflow_json == ""


def test_parse_metadata_bundle_keeps_partial_result_for_unknown_conditioning_node():
    prompt = {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["9", 0], "negative": ["5", 0]},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "do not guess"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative ok"}},
        "9": {
            "class_type": "UnknownConditioningRouter",
            "inputs": {"positive_conditioning": ["4", 0]},
        },
        "10": {
            "class_type": "PrimitiveString",
            "inputs": {"value": "nearby text must not be scanned"},
        },
    }

    result = parse_metadata_bundle(bundle_for_prompt(prompt))

    assert result.positive == ""
    assert result.negative == "negative ok"
    assert "do not guess" not in result.positive
    assert "nearby text" not in result.positive
    assert result.status_message == "PARTIAL"
    assert result.confidence <= 0.6
    assert result.partial_result["unresolved"][0]["node_id"] == "9"
    assert "Unresolved:" in result.setting


def test_parse_metadata_bundle_ignores_unconnected_text_and_llm_nodes():
    prompt = {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["4", 0], "negative": ["5", 0]},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "real prompt"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "6": {
            "class_type": "GeminiChatNode",
            "inputs": {
                "system_instruction": "unconnected system prompt",
                "prompt": "unconnected user prompt template",
            },
        },
        "7": {"class_type": "PrimitiveString", "inputs": {"value": "unconnected text"}},
    }

    result = parse_metadata_bundle(bundle_for_prompt(prompt))

    assert result.positive == "real prompt"
    assert "unconnected system prompt" not in result.positive
    assert "unconnected user prompt template" not in result.positive
    assert "unconnected text" not in result.positive


def test_parse_metadata_bundle_controlnet_parameters_do_not_enter_prompt():
    prompt = {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["6", 0], "negative": ["5", 0]},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "real control prompt"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "6": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["4", 0],
                "image": "pose_control_image.png",
                "strength": 0.55,
                "control_net": "controlnet-depth.safetensors",
            },
        },
    }

    result = parse_metadata_bundle(bundle_for_prompt(prompt))

    assert result.positive == "real control prompt"
    assert "pose_control_image" not in result.positive
    assert "0.55" not in result.positive
    assert "controlnet-depth" not in result.positive


def test_parse_metadata_bundle_conditioning_combine_merges_both_branches():
    prompt = {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["8", 0], "negative": ["7", 0]},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "branch alpha"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "branch beta"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "8": {
            "class_type": "ConditioningCombine",
            "inputs": {"conditioning_1": ["4", 0], "conditioning_2": ["5", 0]},
        },
    }

    result = parse_metadata_bundle(bundle_for_prompt(prompt))

    assert result.positive == "branch alpha, branch beta"


def test_llm_template_upstream_of_showtext_without_cache_is_not_used():
    prompt = {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["4", 0], "negative": ["7", 0]},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ["5", 0]}},
        "5": {"class_type": "ShowText|pysssss", "inputs": {"text": ["6", 0]}},
        "6": {
            "class_type": "GeminiChatNode",
            "inputs": {
                "system_instruction": "You are an anime image prompt extraction assistant",
                "prompt": "Analyze this image and convert it into a final prompt",
            },
        },
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    }

    result = parse_metadata_bundle(bundle_for_prompt(prompt))

    assert "You are an anime image prompt extraction assistant" not in result.positive
    assert "Analyze this image" not in result.positive
    assert result.positive == ""
    assert "LLM template input skipped, no cached generated output found" in result.debug_trace


def test_showtext_text_0_cache_beats_upstream_llm_template():
    prompt = {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["4", 0], "negative": ["7", 0]},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ["5", 0]}},
        "5": {
            "class_type": "ShowText|pysssss",
            "inputs": {
                "text": ["6", 0],
                "text_0": "1girl, solo, squatting, stage lights, anime style",
            },
        },
        "6": {
            "class_type": "GeminiChatNode",
            "inputs": {
                "system_instruction": "system prompt template should stay hidden",
                "prompt": "user prompt template should stay hidden",
            },
        },
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    }

    result = parse_metadata_bundle(bundle_for_prompt(prompt))

    assert result.positive == "1girl, solo, squatting, stage lights, anime style"
    assert "system prompt template" not in result.positive
    assert "user prompt template" not in result.positive
