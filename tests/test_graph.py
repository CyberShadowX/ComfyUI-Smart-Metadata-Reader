def test_model_dataclasses_are_importable():
    from smart_metadata_reader.models import (
        LoraInfo,
        MetadataBundle,
        NodeRecord,
        ParseResult,
        PromptSegment,
    )

    bundle = MetadataBundle(
        filename="sample.png",
        width=512,
        height=768,
        prompt_raw=None,
        workflow_raw=None,
        parameters_raw=None,
        source_format="none",
    )
    node = NodeRecord(node_id="1", class_type="CLIPTextEncode", inputs={})
    segment = PromptSegment(
        text="character trigger",
        node_id="1",
        class_type="CLIPTextEncode",
        field="text",
        path=["CLIPTextEncode 1.text"],
    )
    lora = LoraInfo(
        node_id="2",
        class_type="LoraLoader",
        lora_name="character.safetensors",
        strength_model=0.8,
        strength_clip=1.0,
        path=["KSampler 3.model", "LoraLoader 2"],
    )
    result = ParseResult(
        positive="character trigger",
        negative="",
        filename=bundle.filename,
        loras=[lora],
    )

    assert node.node_id == "1"
    assert segment.text == "character trigger"
    assert result.seed == -1
    assert result.steps == 0
    assert result.cfg == 0.0
    assert result.width == 0
    assert result.height == 0


def test_graph_index_reads_api_prompt_nodes_as_primary_records():
    from smart_metadata_reader.graph import GraphIndex

    prompt = {
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ["22", 0]},
        },
        "22": {
            "class_type": "StringFunction|pysssss",
            "inputs": {"text_a": "character trigger"},
        },
    }
    workflow = {
        "nodes": [
            {
                "id": 6,
                "type": "CLIPTextEncode",
                "inputs": {"text": "stale widget text"},
            }
        ]
    }

    graph = GraphIndex(prompt=prompt, workflow=workflow)

    node = graph.get_node("6")
    assert node.class_type == "CLIPTextEncode"
    assert node.inputs["text"] == ["22", 0]
    assert graph.get_node(22).class_type == "StringFunction|pysssss"


def test_graph_index_reads_ui_workflow_nodes_when_prompt_is_absent():
    from smart_metadata_reader.graph import GraphIndex

    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "PrimitiveString",
                "inputs": {"text": "workflow text"},
                "widgets_values": ["workflow widget"],
            }
        ]
    }

    graph = GraphIndex(prompt=None, workflow=workflow)

    node = graph.get_node("1")
    assert node.class_type == "PrimitiveString"
    assert node.inputs["text"] == "workflow text"
    assert node.widgets_values == ["workflow widget"]


def test_graph_index_detects_links_and_downstream_nodes():
    from smart_metadata_reader.graph import GraphIndex

    prompt = {
        "1": {"class_type": "PrimitiveString", "inputs": {"text": "hello"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": [2, 0]}},
    }

    graph = GraphIndex(prompt=prompt, workflow=None)

    assert graph.is_link(["1", 0])
    assert graph.is_link([1, 0])
    assert graph.link_target([1, 0]) == ("1", 0)
    assert graph.upstream_link_inputs("3") == [("positive", [2, 0])]
    assert [node.node_id for node in graph.downstream_nodes("1")] == ["2"]


def test_workflow_cache_prefers_named_inputs_then_widget_values():
    from smart_metadata_reader.graph import GraphIndex

    prompt = {
        "10": {
            "class_type": "ShowText|pysssss",
            "inputs": {"text": ["9", 0]},
        },
        "11": {
            "class_type": "ShowText|pysssss",
            "inputs": {"text": ["9", 0]},
        },
    }
    workflow = {
        "nodes": [
            {
                "id": 10,
                "type": "ShowText|pysssss",
                "inputs": {"text_0": "cached shown text"},
            },
            {
                "id": 11,
                "type": "ShowText|pysssss",
                "widgets_values": ["widget cached text"],
            },
        ]
    }

    graph = GraphIndex(prompt=prompt, workflow=workflow)

    assert graph.workflow_cache_for("10", ["text_0", "text"]) == "cached shown text"
    assert graph.workflow_cache_for("11", ["text_0"]) == "widget cached text"
