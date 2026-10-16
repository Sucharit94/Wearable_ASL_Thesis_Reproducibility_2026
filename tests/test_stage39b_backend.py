import json

import pytest

from src.evaluation.stage39b_runner import (
    MODEL_REVISION,
    append_response_rows,
    backend_specification,
    parse_generator_response,
    pending_requests,
    read_completed_responses,
)


def test_backend_is_pinned_and_reference_free():
    value = backend_specification()
    assert len(MODEL_REVISION) == 40
    assert value["training"] is False
    assert value["decoding"]["do_sample"] is False
    assert value["execution"]["outer_reference_available_to_generator"] is False
    assert value["execution"]["api_key_required"] is False


def test_parse_json_response():
    value = parse_generator_response(
        '{"sentence":"I need help.","added_content_tokens":[],"constraint_passed":true}'
    )
    assert value["sentence"] == "I need help."
    assert value["parse_mode"] == "json"
    assert value["reported_constraint_passed"] is True


def test_parse_fenced_json_response():
    value = parse_generator_response(
        '```json\n{"sentence":"Hello.","added_content_tokens":[]}\n```'
    )
    assert value["sentence"] == "Hello."
    assert value["parse_mode"] == "json"


def test_parse_raw_text_fallback():
    value = parse_generator_response("English: Please sit down.")
    assert value["sentence"] == "Please sit down."
    assert value["parse_mode"] == "raw_text_fallback"


def test_empty_response_is_rejected():
    with pytest.raises(ValueError):
        parse_generator_response("   ")


def test_pending_requests_resumes_by_id():
    queue = [
        {"request_id": "a", "prompt": "one"},
        {"request_id": "b", "prompt": "two"},
    ]
    assert [row["request_id"] for row in pending_requests(queue, {"a": {}})] == ["b"]


def test_pending_rejects_unknown_completed_id():
    with pytest.raises(RuntimeError):
        pending_requests([{"request_id": "a"}], {"outside": {}})


def test_append_and_read_completed_responses(tmp_path):
    path = tmp_path / "responses.jsonl"
    rows = [
        {"request_id": "a", "status": "COMPLETE", "sentence": "hello"},
        {"request_id": "b", "status": "COMPLETE", "sentence": "goodbye"},
    ]
    append_response_rows(path, rows)
    assert set(read_completed_responses(path)) == {"a", "b"}
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_duplicate_saved_response_is_rejected(tmp_path):
    path = tmp_path / "responses.jsonl"
    row = {"request_id": "a", "status": "COMPLETE", "sentence": "hello"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        read_completed_responses(path)
