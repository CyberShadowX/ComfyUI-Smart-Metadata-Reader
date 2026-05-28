from smart_metadata_reader.models import PromptSegment


def segment(text: str) -> PromptSegment:
    return PromptSegment(
        text=text,
        node_id="1",
        class_type="TestTextNode",
        field="text",
        path=["TestTextNode 1.text"],
    )


def test_prompt_merge_removes_empty_segments_and_extra_commas_only():
    from smart_metadata_reader.prompt_merge import merge_prompt_segments

    merged = merge_prompt_segments(
        [
            segment("  masterpiece,,  "),
            segment(""),
            segment(" , , best quality "),
            segment("\n\ncinematic lighting\n\n"),
        ]
    )

    assert merged == "masterpiece, best quality, cinematic lighting"


def test_prompt_merge_preserves_lora_trigger_words_and_tags():
    from smart_metadata_reader.prompt_merge import merge_prompt_segments

    merged = merge_prompt_segments(
        [
            segment("<lora:character_lora:0.8>"),
            segment("raiden_shogun, purple hair"),
            segment("style trigger phrase"),
        ]
    )

    assert merged == (
        "<lora:character_lora:0.8>, "
        "raiden_shogun, purple hair, "
        "style trigger phrase"
    )


def test_prompt_merge_preserves_nsfw_terms():
    from smart_metadata_reader.prompt_merge import merge_prompt_segments

    merged = merge_prompt_segments(
        [
            segment("nsfw"),
            segment("nude, explicit"),
            segment("user handwritten tag"),
        ]
    )

    assert merged == "nsfw, nude, explicit, user handwritten tag"


def test_prompt_merge_preserves_repeated_meaningful_tokens():
    from smart_metadata_reader.prompt_merge import merge_prompt_segments

    merged = merge_prompt_segments(
        [
            segment("masterpiece"),
            segment("masterpiece"),
            segment("score_9, score_9"),
        ]
    )

    assert merged == "masterpiece, masterpiece, score_9, score_9"


def test_trace_collector_groups_lines_by_section():
    from smart_metadata_reader.trace import TraceCollector

    trace = TraceCollector()
    trace.add("POSITIVE_TRACE", "KSampler 3.positive")
    trace.add("POSITIVE_TRACE", " -> CLIPTextEncode 6.text")
    trace.add("NEGATIVE_TRACE", "KSampler 3.negative")

    assert trace.render() == (
        "POSITIVE_TRACE:\n"
        "KSampler 3.positive\n"
        " -> CLIPTextEncode 6.text\n\n"
        "NEGATIVE_TRACE:\n"
        "KSampler 3.negative"
    )


def resolve_text(prompt, workflow=None, node_id="1", prefer_cached_text=True):
    from smart_metadata_reader.adapters.base import TextResolutionContext
    from smart_metadata_reader.adapters.clip_text import (
        CLIPTextEncodeAdapter,
        CLIPTextEncodeSDXLAdapter,
    )
    from smart_metadata_reader.adapters.pysssss import (
        ShowTextAdapter,
        StringFunctionAdapter,
    )
    from smart_metadata_reader.adapters.text_sources import (
        DeepTranslatorTextAdapter,
        GenericTextAdapter,
    )
    from smart_metadata_reader.graph import GraphIndex
    from smart_metadata_reader.trace import TraceCollector

    trace = TraceCollector()
    context = TextResolutionContext(
        graph=GraphIndex(prompt=prompt, workflow=workflow),
        prefer_cached_text=prefer_cached_text,
        trace=trace,
    )
    context.registry.register(CLIPTextEncodeAdapter())
    context.registry.register(CLIPTextEncodeSDXLAdapter())
    context.registry.register(StringFunctionAdapter())
    context.registry.register(ShowTextAdapter())
    context.registry.register(DeepTranslatorTextAdapter())
    context.registry.register(GenericTextAdapter())
    segments = context.resolve_node(node_id=node_id, role="positive", path=[])
    return segments, trace.render()


def test_clip_text_encode_direct_string_returns_prompt_segment():
    prompt = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "masterpiece, character trigger"},
        }
    }

    segments, _trace = resolve_text(prompt)

    assert [segment.text for segment in segments] == ["masterpiece, character trigger"]
    assert segments[0].node_id == "1"
    assert segments[0].class_type == "CLIPTextEncode"
    assert segments[0].field == "text"


def test_clip_text_encode_resolves_upstream_text_node():
    prompt = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ["2", 0]},
        },
        "2": {
            "class_type": "PrimitiveString",
            "inputs": {"text": "upstream user text"},
        },
    }

    segments, _trace = resolve_text(prompt)

    assert [segment.text for segment in segments] == ["upstream user text"]
    assert segments[0].node_id == "2"
    assert segments[0].class_type == "PrimitiveString"


def test_clip_text_encode_sdxl_resolves_multiple_prompt_fields():
    prompt = {
        "1": {
            "class_type": "CLIPTextEncodeSDXL",
            "inputs": {
                "text_g": "global style text",
                "text_l": "local subject text",
            },
        }
    }

    segments, _trace = resolve_text(prompt)

    assert [segment.text for segment in segments] == [
        "global style text",
        "local subject text",
    ]


def test_string_function_append_resolves_three_parts_in_order():
    prompt = {
        "1": {
            "class_type": "StringFunction|pysssss",
            "inputs": {
                "action": "append",
                "text_a": "role trigger",
                "text_b": ["2", 0],
                "text_c": ["3", 0],
            },
        },
        "2": {
            "class_type": "PrimitiveString",
            "inputs": {"text": "cached model text"},
        },
        "3": {
            "class_type": "DeepTranslatorTextNode",
            "inputs": {"text": "translated user supplement"},
        },
    }

    segments, _trace = resolve_text(prompt)

    assert [segment.text for segment in segments] == [
        "role trigger",
        "cached model text",
        "translated user supplement",
    ]


def test_show_text_prefers_text_0_cache_over_upstream_llm_template():
    prompt = {
        "1": {
            "class_type": "ShowText|pysssss",
            "inputs": {
                "text": ["2", 0],
                "text_0": "final shown prompt",
            },
        },
        "2": {
            "class_type": "GeminiChatNode",
            "inputs": {
                "prompt": "describe this image as danbooru tags",
                "system_instruction": "you are a prompt generator",
            },
        },
    }

    segments, trace = resolve_text(prompt)

    assert [segment.text for segment in segments] == ["final shown prompt"]
    assert "GeminiChatNode" not in trace


def test_show_text_without_cache_does_not_treat_gemini_template_as_prompt():
    prompt = {
        "1": {
            "class_type": "ShowText|pysssss",
            "inputs": {"text": ["2", 0]},
        },
        "2": {
            "class_type": "GeminiChatNode",
            "inputs": {
                "prompt": "describe this image as danbooru tags",
                "system_instruction": "you are a prompt generator",
            },
        },
    }

    segments, trace = resolve_text(prompt)

    assert segments == []
    assert "GeminiChatNode" in trace
    assert "low confidence" in trace


def test_show_text_uses_workflow_cache_when_prompt_lacks_text_0():
    prompt = {
        "1": {
            "class_type": "ShowText|pysssss",
            "inputs": {"text": ["2", 0]},
        },
        "2": {
            "class_type": "GeminiChatNode",
            "inputs": {"prompt": "template, not final prompt"},
        },
    }
    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "ShowText|pysssss",
                "inputs": {"text_0": "workflow cached shown result"},
            }
        ]
    }

    segments, _trace = resolve_text(prompt, workflow=workflow)

    assert [segment.text for segment in segments] == ["workflow cached shown result"]


def test_show_text_widget_cache_fallback_lowers_confidence_and_traces_source():
    prompt = {
        "1": {
            "class_type": "ShowText|pysssss",
            "inputs": {"text": ["2", 0]},
        },
        "2": {
            "class_type": "GeminiChatNode",
            "inputs": {"prompt": "template, not final prompt"},
        },
    }
    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "ShowText|pysssss",
                "widgets_values": ["widget cached shown result"],
            },
            {
                "id": 2,
                "type": "GeminiChatNode",
                "widgets_values": ["do not use this llm template"],
            },
        ]
    }

    segments, trace = resolve_text(prompt, workflow=workflow)

    assert [segment.text for segment in segments] == ["widget cached shown result"]
    assert segments[0].confidence < 1.0
    assert "workflow widget cache fallback" in trace
    assert "do not use this llm template" not in [segment.text for segment in segments]


def test_deep_translator_text_node_uses_text_input():
    prompt = {
        "1": {
            "class_type": "DeepTranslatorTextNode",
            "inputs": {"text": "translated prompt text"},
        }
    }

    segments, _trace = resolve_text(prompt)

    assert [segment.text for segment in segments] == ["translated prompt text"]
