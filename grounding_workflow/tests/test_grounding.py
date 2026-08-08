from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from grounding.boxes import parse_bbox
from grounding.data import load_query_groups
from grounding.prompts import PromptCache, load_provider, prepare_prompt_results
from grounding.postprocess import validate_submission
from grounding.backends.internvl import build_prompt as build_internvl_prompt
from grounding.runner import _infer_resilient, _units
from grounding.runner import RunConfig, run
from grounding.types import PromptRequest


def build_prompt(request: PromptRequest) -> str:
    return f"rewritten: {request.record.query}"


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
    provider = load_provider(__name__ + ":build_prompt")
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
    prompt = build_internvl_prompt("Two white umbrellas above the outdoor dining area")
    assert "bbox_2d" in prompt
    assert "0-1000" in prompt
    assert "multiple instances" in prompt
    assert "all matching instances" in prompt


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
