"""scripts/fake_ollama.py is a teaching/offline-testing tool. Keep it correct, and keep it clearly NOT a model."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from conftest import GOOD_FEATURES

SPEC = importlib.util.spec_from_file_location("fake_ollama", Path(__file__).resolve().parent.parent / "scripts" / "fake_ollama.py")
fake = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fake)

SENTENCE = ("Predict the exam score for a student who studies 12.5 hours per week, has 88% attendance, scored 72, 75 and 70 in the "
            "three mock tests, belongs to the Medium income bracket, lives in Pune, enrolled on 2025-06-01, has shoe size 9 and lucky number 42.")


def test_it_extracts_all_ten_fields_from_the_sample_sentence():
    assert fake.extract_features(SENTENCE) == GOOD_FEATURES


def test_it_refuses_to_extract_from_an_incomplete_sentence():
    assert fake.extract_features("Predict for a student who studied 6 hours daily and attended 85% of classes.") is None


def test_it_asks_for_a_tool_call_then_repeats_the_tool_result():
    first = fake.reply_for([{"role": "user", "content": SENTENCE}])
    assert first["tool_calls"][0]["function"]["name"] == "predict_exam_score"
    assert json.loads(first["tool_calls"][0]["function"]["arguments"]) == GOOD_FEATURES
    second = fake.reply_for([{"role": "user", "content": SENTENCE}, {"role": "tool", "content": json.dumps({"predicted_final_score": 91.5})}])
    assert "91.5" in second["content"]


def test_it_is_openly_a_stand_in_for_everything_else():
    assert "stand-in" in fake.reply_for([{"role": "user", "content": "What is the capital of France?"}])["content"]
