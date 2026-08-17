from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

from grounding.boxes import parse_bbox
from grounding.data import load_query_groups
from grounding.prompts import (
    PromptCache,
    build_prompt,
    load_provider,
    prepare_prompt_results,
)
from grounding.postprocess import validate_submission
from grounding.query_router import classify_query
from grounding.backends.internvl import _unwrap_image_features
from grounding.runner import _infer_resilient, _units
from grounding.runner import RunConfig, run
from grounding.types import ImageGroup, PromptRequest, PromptResult


def build_rewritten_prompt(request: PromptRequest) -> str:
    return f"rewritten: {request.record.query}"


class BatchPromptProvider:
    name = "batch-test"
    version = "1"
    prefix = ""

    def __init__(self):
        self.calls = []

    def build_all(self, requests):
        self.calls.append(tuple(request.record.query for request in requests))
        return {
            request.record.query_id: PromptResult(
                f"batch prompt: {request.record.query}", self.name, self.version
            )
            for request in requests
        }


def _dataset(root: Path) -> Path:
    image = root / "Images" / "visible"
    image.mkdir(parents=True)
    Image.new("RGB", (16, 8), "white").save(image / "000001.png")
    queries = root / "queries"
    queries.mkdir()
    path = queries / "queries.json"
    path.write_text(json.dumps({
        "a": {"visible": "Images/visible/000001.png", "query": "first"},
        "b": {"visible": "Images/visible/000001.png", "query": "second"},
    }), encoding="utf-8")
    return path


def test_grouping_and_prompt_cache(tmp_path: Path):
    query_file = _dataset(tmp_path)
    groups, _ = load_query_groups(tmp_path, query_file)
    provider = load_provider(__name__ + ":build_rewritten_prompt")
    cache = PromptCache(tmp_path / "prompts.jsonl")
    result = prepare_prompt_results(groups, provider, cache)
    assert len(groups) == 1 and len(groups[0].records) == 2
    assert result["a"].text == "rewritten: first"


def test_parse_boxes():
    assert parse_bbox("<box><10><20><900><800></box>") == [0.01, 0.02, 0.9, 0.8]
    assert parse_bbox('{"bbox_2d":[10,20,900,800]}') == [0.01, 0.02, 0.9, 0.8]
    assert parse_bbox("<box><900><800><10><20></box>") is None


def test_same_image_units_are_grouped(tmp_path: Path):
    groups, _ = load_query_groups(tmp_path, _dataset(tmp_path))
    prompts = {record.query_id: type("P", (), {"text": record.query}) for record in groups[0].records}
    batches = list(_units(groups, prompts, 2))
    assert len(batches) == 1 and len(batches[0][0].records) == 2


class FakeBackend:
    def __init__(self):
        self.calls = []

    def infer(self, units):
        from grounding.types import BackendResponse
        self.calls.append([len(unit.records) for unit in units])
        count = sum(len(unit.records) for unit in units)
        if count > 1:
            raise RuntimeError("CUDA out of memory")
        return BackendResponse(tuple("<box><1><2><3><4></box>" for _ in range(count)), {"fake": True})

    @staticmethod
    def is_oom(error):
        return "out of memory" in str(error).lower()

    @staticmethod
    def clear_cuda_cache():
        pass


def test_oom_retry_splits():
    from grounding.types import InferenceUnit
    unit = InferenceUnit("a", Path("a"), (object(), object(), object()), ("a", "b", "c"))
    backend = FakeBackend()
    rows = _infer_resilient(backend, [unit])
    assert sum(len(texts) for _, texts, _ in rows) == 3


def test_submission_schema_preserves_source_fields():
    queries = {"q": {"visible": "Images/visible/a.png", "infrared": "Images/infrared/a.png", "depth": "Images/depth/a.png", "query": "object"}}
    prediction = {"q": {**queries["q"], "bbox": [0.1, 0.2, 0.8, 0.9]}}
    assert validate_submission(prediction, queries) == []
    changed = {"q": {**prediction["q"], "query": "changed"}}
    assert validate_submission(changed, queries)


def test_internvl_grounding_prompt_is_strict():
    prompt = build_prompt("Two white umbrellas above the outdoor dining area", "internvl_json_union")
    assert "bbox_2d" in prompt
    assert "0-1000" in prompt
    assert "multiple instances" in prompt
    assert "all matching instances" in prompt


def test_global_prompt_profiles_cover_model_formats():
    query = "Two white umbrellas above the outdoor dining area"
    json_prompt = build_prompt(query, "internvl_json_union")
    native_prompt = build_prompt(query, "internvl_native_box")
    minimal_prompt = build_prompt(query, "internvl_minimal")
    assert '{"bbox_2d": [x1, y1, x2, y2]}' in json_prompt
    assert "<box><x1><y1><x2><y2></box>" in native_prompt
    assert '{"bbox_2d": [x1, y1, x2, y2]}' in minimal_prompt
    assert all("multiple instances" in prompt for prompt in (json_prompt, native_prompt, minimal_prompt))


def test_query_router_prioritizes_explicit_workflow_cues():
    assert classify_query("Two white umbrellas above the dining area").workflow == "multi_instance"
    assert classify_query("From left to right, the second stone pier").workflow == "ordinal"
    assert classify_query("The red letter L closest to the camera").workflow == "text_sign"
    assert classify_query("The man beside the white lamp").workflow == "relation"
    assert classify_query("A red car").workflow == "default"


def test_locateanything_special_prompt_profiles():
    multi = build_prompt("people wearing red shirts", "locateanything_multi")
    text = build_prompt("the word EXIT", "locateanything_text")
    assert "Locate all the instances that match the following description" in multi
    assert "one box per instance" in multi
    assert "Locate the text that matches the following description" in text


def test_global_prompt_profile_is_query_to_prompt():
    prompt = build_prompt("red vehicle", "qwen_json")
    assert "red vehicle" in prompt
    assert "bbox_2d" in prompt


def test_custom_prompt_provider_receives_full_query_set(tmp_path: Path):
    query_file = _dataset(tmp_path)
    groups, _ = load_query_groups(tmp_path, query_file)
    provider = BatchPromptProvider()
    prepare_prompt_results(groups, provider, PromptCache(tmp_path / "prompts.jsonl"))
    assert provider.calls == [("first", "second")]


def test_prompt_rebuild_keeps_full_query_set_when_cache_is_partial(tmp_path: Path):
    query_file = _dataset(tmp_path)
    groups, _ = load_query_groups(tmp_path, query_file)
    provider = BatchPromptProvider()
    cache = PromptCache(tmp_path / "prompts.jsonl")
    cache.put(groups[0].records[0], PromptResult("cached", provider.name, provider.version))
    prepare_prompt_results(groups, provider, cache)
    assert provider.calls == [("first", "second")]


def test_full_set_change_invalidates_all_cached_prompts(tmp_path: Path):
    class ContextProvider(BatchPromptProvider):
        def build_all(self, requests):
            context = ",".join(request.record.query for request in requests)
            self.calls.append(context)
            return {
                request.record.query_id: PromptResult(context, self.name, self.version)
                for request in requests
            }

    query_file = _dataset(tmp_path)
    groups, _ = load_query_groups(tmp_path, query_file)
    provider = ContextProvider()
    cache = PromptCache(tmp_path / "prompts.jsonl")
    prepare_prompt_results(groups, provider, cache)
    changed = replace(groups[0].records[1], query="changed")
    changed_group = ImageGroup(groups[0].image_key, groups[0].image_path, (groups[0].records[0], changed))
    results = prepare_prompt_results([changed_group], provider, PromptCache(cache.path))
    assert results["a"].text == "first,changed"


def test_prompt_cache_namespace_isolated_between_prompt_experiments(tmp_path: Path):
    query_file = _dataset(tmp_path)
    groups, _ = load_query_groups(tmp_path, query_file)
    record = groups[0].records[0]
    provider = load_provider(None)
    path = tmp_path / "prompts.jsonl"
    json_cache = PromptCache(path, namespace="internvl:json_union")
    json_cache.put(record, PromptResult("json prompt", provider.name, provider.version))
    assert PromptCache(path, namespace="internvl:json_union").get(record, provider) is not None
    native_cache = PromptCache(path, namespace="internvl:native_box")
    assert native_cache.get(record, provider) is None
    native_cache.put(record, PromptResult("native prompt", provider.name, provider.version))
    assert PromptCache(path, namespace="internvl:json_union").get(record, provider) is not None


def test_internvl_feature_return_can_be_tensor():
    import torch
    feature = torch.zeros((1, 4, 8))
    assert _unwrap_image_features(feature, torch) is feature


def test_runner_writes_submission_and_resume_files(tmp_path: Path):
    query_file = _dataset(tmp_path)
    output = tmp_path / "predictions.json"
    config = RunConfig(tmp_path, query_file, output, tmp_path / "partial.json", tmp_path / "raw.jsonl", tmp_path / "prompts.jsonl", batch_size=2)
    provider = load_provider(None)
    result = run(config, provider, FakeBackend())
    assert result["count"] == 2 and output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload) == {"a", "b"}
    assert validate_submission(payload, json.loads(query_file.read_text(encoding="utf-8"))) == []
