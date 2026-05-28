import json

from smart_metadata_reader.graph import GraphIndex
from smart_metadata_reader.models import NodeRecord, PromptSegment
from smart_metadata_reader.prompt_merge import merge_prompt_segments


def graph_and_sampler(prompt, sampler_id="9"):
    graph = GraphIndex(prompt=prompt, workflow=None)
    return graph, graph.get_node(sampler_id)


def test_extract_model_name_from_checkpoint_loader_simple_ckpt_name():
    from smart_metadata_reader.lora_extractor import extract_model_name

    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "dream_model.safetensors"},
        },
        "9": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    graph, sampler = graph_and_sampler(prompt)

    assert extract_model_name(graph, sampler) == "dream_model.safetensors"


def test_extract_model_name_from_checkpoint_loader_field_variants():
    from smart_metadata_reader.lora_extractor import extract_model_name

    for field_name in ("checkpoint", "model_name", "name"):
        prompt = {
            "1": {
                "class_type": "CheckpointLoader",
                "inputs": {field_name: f"{field_name}_model.safetensors"},
            },
            "9": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        }
        graph, sampler = graph_and_sampler(prompt)

        assert extract_model_name(graph, sampler) == f"{field_name}_model.safetensors"


def test_sampler_model_lora_loader_chain_extracts_model_and_lora():
    from smart_metadata_reader.lora_extractor import extract_loras, extract_model_name

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
                "lora_name": "character_lora.safetensors",
                "strength_model": 0.8,
                "strength_clip": 1.0,
            },
        },
        "9": {"class_type": "KSampler", "inputs": {"model": ["2", 0]}},
    }
    graph, sampler = graph_and_sampler(prompt)
    loras = extract_loras(graph, sampler)

    assert extract_model_name(graph, sampler) == "base_model.safetensors"
    assert [(lora.lora_name, lora.strength_model, lora.strength_clip) for lora in loras] == [
        ("character_lora.safetensors", 0.8, 1.0)
    ]
    assert loras[0].path == [
        "KSampler 9.model",
        "LoraLoader 2",
        "CheckpointLoaderSimple 1",
    ]


def test_multiple_chained_lora_loaders_are_returned_checkpoint_to_sampler_order():
    from smart_metadata_reader.lora_extractor import extract_loras

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
                "lora_name": "first_lora.safetensors",
                "strength_model": 0.5,
                "strength_clip": 0.6,
            },
        },
        "3": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["2", 0],
                "clip": ["2", 1],
                "lora_name": "second_lora.safetensors",
                "strength_model": 0.7,
                "strength_clip": 0.8,
            },
        },
        "9": {"class_type": "KSampler", "inputs": {"model": ["3", 0]}},
    }
    graph, sampler = graph_and_sampler(prompt)

    assert [lora.lora_name for lora in extract_loras(graph, sampler)] == [
        "first_lora.safetensors",
        "second_lora.safetensors",
    ]


def test_lora_loader_model_only_extracts_lora_name_and_strength_model():
    from smart_metadata_reader.lora_extractor import extract_loras

    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base_model.safetensors"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "model_only.safetensors",
                "strength_model": 0.35,
            },
        },
        "9": {"class_type": "KSampler", "inputs": {"model": ["2", 0]}},
    }
    graph, sampler = graph_and_sampler(prompt)
    [lora] = extract_loras(graph, sampler)

    assert lora.node_id == "2"
    assert lora.class_type == "LoraLoaderModelOnly"
    assert lora.lora_name == "model_only.safetensors"
    assert lora.strength_model == 0.35
    assert lora.strength_clip is None


def test_sampler_clip_chain_lora_is_discovered():
    from smart_metadata_reader.lora_extractor import extract_loras

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
                "lora_name": "clip_chain_lora.safetensors",
                "strength_model": 0.1,
                "strength_clip": 0.9,
            },
        },
        "9": {"class_type": "KSampler", "inputs": {"clip": ["2", 1]}},
    }
    graph, sampler = graph_and_sampler(prompt)

    assert [lora.lora_name for lora in extract_loras(graph, sampler)] == [
        "clip_chain_lora.safetensors"
    ]


def test_extract_vae_name_from_vae_decode_chain_and_vae_loader_fallback():
    from smart_metadata_reader.lora_extractor import extract_vae_name

    prompt_with_decode_chain = {
        "1": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "main_vae.safetensors"},
        },
        "2": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["1", 0]},
        },
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "9": {"class_type": "KSampler", "inputs": {}},
    }
    assert extract_vae_name(GraphIndex(prompt_with_decode_chain, None)) == "main_vae.safetensors"

    fallback_prompt = {
        "4": {"class_type": "VAELoader", "inputs": {"name": "fallback_vae.safetensors"}},
    }
    assert extract_vae_name(GraphIndex(fallback_prompt, None)) == "fallback_vae.safetensors"


def test_extract_vae_name_prefers_final_save_image_output_chain_with_multiple_vae_branches():
    from smart_metadata_reader.lora_extractor import extract_vae_name

    prompt = {
        "1": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "unused_branch_vae.safetensors"},
        },
        "2": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["1", 0]},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "final_output_vae.safetensors"},
        },
        "4": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["3", 0]},
        },
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
        "8": {"class_type": "KSampler", "inputs": {}},
        "9": {"class_type": "KSampler", "inputs": {}},
    }

    assert extract_vae_name(GraphIndex(prompt, None)) == "final_output_vae.safetensors"


def test_lora_trigger_words_remain_prompt_text_and_are_not_filtered_by_extractor():
    from smart_metadata_reader.lora_extractor import extract_loras

    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "base_model.safetensors"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "role_lora.safetensors",
                "strength_model": 0.8,
            },
        },
        "9": {"class_type": "KSampler", "inputs": {"model": ["2", 0]}},
    }
    graph, sampler = graph_and_sampler(prompt)
    prompt_text = merge_prompt_segments(
        [
            PromptSegment(
                text="role_lora_trigger, character_name, style tag",
                node_id="22",
                class_type="PrimitiveString",
                field="text",
                path=["PrimitiveString 22.text"],
            )
        ]
    )

    loras_json = json.dumps([lora.__dict__ for lora in extract_loras(graph, sampler)])

    assert "role_lora.safetensors" in loras_json
    assert prompt_text == "role_lora_trigger, character_name, style tag"
