import pytest

from src.evaluation.stage39b_v2_runner import (
    build_plain_english_prompt,
    parse_plain_english_response,
    pending_v2,
    v2_backend_specification,
)


def request():
    return {
        "request_id": "l2_test",
        "source_gloss": "you good_at physics you",
        "training_candidates": [
            {"gloss": "you good_at physics you", "english": "are you good at physics"},
            {"gloss": "hello", "english": "hello"},
            {"gloss": "good afternoon", "english": "good afternoon"},
        ],
    }


def test_v2_prompt_has_no_json_placeholder():
    prompt = build_plain_english_prompt(request())
    assert "Return only the English sentence" in prompt
    assert '"sentence":"..."' not in prompt
    assert "ASL gloss: you good_at physics you" in prompt


def test_v2_backend_is_deterministic_and_reference_free():
    value = v2_backend_specification()
    assert value["prompt_version"] == 2
    assert value["training"] is False
    assert value["decoding"]["do_sample"] is False
    assert value["execution"]["outer_reference_available_to_generator"] is False


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Good afternoon", "Good afternoon"),
        ("English: Are you ready for lunch?", "Are you ready for lunch?"),
        ('"I take out the trash."', "I take out the trash."),
    ],
)
def test_plain_output_is_accepted(raw, expected):
    assert parse_plain_english_response(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        'sentence":"...", added_content_tokens:[]',
        'added_content_tokens:[], constraint_passed:true',
        "first line\nsecond line",
    ],
)
def test_invalid_output_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_plain_english_response(raw)


def test_resume_skips_only_completed_ids():
    queue = [
        {"request_id": "a"},
        {"request_id": "b"},
    ]
    assert [row["request_id"] for row in pending_v2(queue, {"a": {}})] == ["b"]


def test_unknown_completed_id_is_rejected():
    with pytest.raises(RuntimeError):
        pending_v2([{"request_id": "a"}], {"outside": {}})
