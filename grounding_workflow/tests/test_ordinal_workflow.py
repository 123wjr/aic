from __future__ import annotations

from grounding.ordinal_workflow import (
    build_plan_prompt,
    build_selector_prompt,
    choose_candidate_from_text,
    parse_ordinal_plan,
    parse_candidate_boxes,
    plan_ordinal_query,
)


def test_plan_ordinal_query_removes_order_words():
    plan = plan_ordinal_query("From left to right, the second stone pier")
    assert plan.target == "stone pier"
    assert plan.order == "left_to_right"
    assert plan.rank == 1


def test_plan_ordinal_query_handles_edge_orders():
    leftmost = plan_ordinal_query("The leftmost security camera")
    rightmost = plan_ordinal_query("The rightmost car")
    assert leftmost.target == "security camera"
    assert leftmost.order == "left_to_right"
    assert leftmost.rank == 0
    assert rightmost.target == "car"
    assert rightmost.order == "right_to_left"
    assert rightmost.rank == 0


def test_parse_candidate_boxes_returns_all_valid_boxes():
    text = "<box><10><20><100><200></box> noise <box><300><50><500><250></box>"
    assert parse_candidate_boxes(text) == [[0.01, 0.02, 0.1, 0.2], [0.3, 0.05, 0.5, 0.25]]


def test_choose_candidate_from_text_prefers_candidate_index():
    boxes = [[0.01, 0.02, 0.1, 0.2], [0.3, 0.05, 0.5, 0.25]]
    assert choose_candidate_from_text('{"candidate": 2}', boxes) == boxes[1]
    assert choose_candidate_from_text("candidate: 1", boxes) == boxes[0]


def test_selector_prompt_contains_original_query_and_candidates():
    prompt = build_selector_prompt("From left to right, the second stone pier", [[0.1, 0.2, 0.3, 0.4]])
    assert "From left to right, the second stone pier" in prompt
    assert "candidate" in prompt
    assert "[100, 200, 300, 400]" in prompt


def test_parse_ordinal_plan_from_qwen_json():
    plan = parse_ordinal_plan(
        '{"target":"stone pier","order":"left_to_right","rank":1,"confidence":0.98,"is_simple_ordinal":true}'
    )
    assert plan.target == "stone pier"
    assert plan.order == "left_to_right"
    assert plan.rank == 1
    assert plan.confidence == 0.98
    assert plan.is_simple_ordinal is True


def test_plan_prompt_requires_structured_query_analysis():
    prompt = build_plan_prompt("The pier behind the fourth pier from the left")
    assert "target" in prompt
    assert "is_simple_ordinal" in prompt
    assert "Do not treat an ordinal inside a reference phrase as the main order" in prompt
