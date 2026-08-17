from __future__ import annotations

import json

from filter_workflow_queries import main as filter_main
from grounding.prompts import build_prompt
from grounding.query_router import route_query


def test_query_router_uses_high_precision_priority():
    assert route_query("the traffic sign beside two cars").workflow == "text_sign"
    assert route_query("the second vehicle from left to right").workflow == "ordinal"
    assert route_query("two white umbrellas").workflow == "multi_instance"
    assert route_query("red cars in the parking lot").workflow == "multi_instance"
    assert route_query("the number of cars").workflow == "multi_instance"
    assert route_query("the words on the screen").workflow == "text_sign"
    assert route_query("red dogs near the gate").workflow == "multi_instance"
    assert route_query("the bicycle beside the tree").workflow == "relation"
    assert route_query("the red bicycle").workflow == "default"
    assert route_query("the camera lens").workflow == "default"


def test_filter_cli_outputs_records_and_full_counts(tmp_path, capsys):
    queries = {
        "q1": {"visible": "Images/visible/1.png", "query": "the rightmost car"},
        "q2": {"visible": "Images/visible/2.png", "query": "a blue logo"},
        "q3": {"visible": "Images/visible/3.png", "query": "the bicycle"},
    }
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(queries), encoding="utf-8")

    assert filter_main(["--queries", str(path), "--workflow", "ordinal", "--limit", "1"]) == 0
    captured = capsys.readouterr()
    rows = [json.loads(line) for line in captured.out.splitlines()]
    assert rows == [{
        "query_id": "q1",
        "visible": "Images/visible/1.png",
        "query": "the rightmost car",
        "workflow": "ordinal",
        "tags": ["rightmost"],
        "reason": "matched ordinal trigger: rightmost",
    }]
    assert "default=1" in captured.err
    assert "ordinal=1" in captured.err
    assert "text_sign=1" in captured.err


def test_locateanything_workflow_prompt_profiles():
    multi = build_prompt("white umbrellas", "locateanything_multi")
    text = build_prompt("EXIT", "locateanything_text")
    assert "Locate all the instances that match the following description" in multi
    assert "Please locate the text referred as" in text
