import json
import random
from copy import deepcopy

from smart_metadata_reader.metadata_reader import parse_metadata_bundle
from smart_metadata_reader.models import MetadataBundle


def bundle_for_prompt(prompt):
    return MetadataBundle(
        filename="randomized.png",
        width=768,
        height=1024,
        prompt_raw=json.dumps(prompt),
        workflow_raw=None,
        parameters_raw=None,
        source_format="ComfyUI prompt/workflow",
    )


def base_prompt():
    return {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        "2": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["6", 0]},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["5", 0],
                "clip": ["5", 1],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "seed": 98765,
                "steps": 18,
                "cfg": 6.25,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "stable_model.safetensors"},
        },
        "5": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["4", 0],
                "clip": ["4", 1],
                "lora_name": "style_lora.safetensors",
                "strength_model": 0.6,
                "strength_clip": 0.75,
            },
        },
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "main.vae.safetensors"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "subject trigger, style token"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality"}},
        "9": {"class_type": "PrimitiveString", "inputs": {"value": "unconnected text"}},
    }


def test_parser_results_do_not_depend_on_specific_node_ids():
    prompt = base_prompt()
    remapped_prompt = remap_node_ids(prompt)

    original = parse_metadata_bundle(bundle_for_prompt(prompt))
    remapped = parse_metadata_bundle(bundle_for_prompt(remapped_prompt))

    assert remapped.positive == original.positive
    assert remapped.negative == original.negative
    assert remapped.seed == original.seed
    assert remapped.steps == original.steps
    assert remapped.cfg == original.cfg
    assert remapped.sampler_name == original.sampler_name
    assert remapped.scheduler == original.scheduler
    assert remapped.model_name == original.model_name
    assert remapped.vae_name == original.vae_name
    assert lora_values(remapped) == lora_values(original)


def remap_node_ids(prompt):
    ids = list(prompt.keys())
    rng = random.Random(1234)
    new_ids = [str(value) for value in rng.sample(range(100, 999), len(ids))]
    mapping = dict(zip(ids, new_ids))

    remapped = {}
    for old_id, node in prompt.items():
        new_node = deepcopy(node)
        new_node["inputs"] = remap_value(new_node.get("inputs", {}), mapping)
        remapped[mapping[old_id]] = new_node
    return remapped


def remap_value(value, mapping):
    if isinstance(value, list):
        if len(value) == 2 and str(value[0]) in mapping and isinstance(value[1], int):
            return [mapping[str(value[0])], value[1]]
        return [remap_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: remap_value(item, mapping) for key, item in value.items()}
    return value


def lora_values(result):
    return [
        (lora.lora_name, lora.strength_model, lora.strength_clip)
        for lora in result.loras
    ]
