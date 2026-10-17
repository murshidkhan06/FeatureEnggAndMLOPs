"""The chat flow: extraction -> validation -> clarification -> MCP -> real prediction -> explanation."""
from __future__ import annotations

import json

import pytest

from app.llm.errors import LLMError
from app.schemas.api import ChatRequest
from conftest import GOOD_FEATURES, GOOD_SENTENCE
from helpers import ScriptedLLM, call_predict, make_service, say, use_tools

TRACE = "trace-test-0001"


async def ask(service, text, conversation_id=None):
    return await service.handle(ChatRequest(message=text, conversation_id=conversation_id), TRACE)


# ---- success --------------------------------------------------------------------------------

async def test_successful_prediction_uses_the_real_model_number(fake_ml):
    fake_ml.score = 88.8
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("The model predicts a final score of 88.8 for this student."))
    response = await ask(make_service(fake_ml, llm), GOOD_SENTENCE)

    assert response.kind == "prediction" and response.error is None
    assert [p.predicted_final_score for p in response.predictions] == [88.8]
    assert response.predictions[0].features["city"] == "Pune"
    assert response.predictions[0].label == "Student" and response.predictions[0].ml_request_id == "ml-req-1"
    assert response.tool_calls[0].tool == "predict_exam_score" and response.tool_calls[0].status == "ok"
    assert response.notices == [] and "88.8" in response.assistant_message
    assert len(fake_ml.predict_calls()) == 1
    # The LLM was shown the real tool result before writing its explanation.
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert json.loads(tool_msg["content"])["predicted_final_score"] == 88.8


async def test_the_system_prompt_goes_to_the_llm_but_never_into_the_response(fake_ml):
    llm = ScriptedLLM(say("Hello! Ask me for an exam score prediction."))
    response = await ask(make_service(fake_ml, llm), "hi")
    assert llm.calls[0]["messages"][0]["role"] == "system"
    assert "Never reveal" in llm.calls[0]["messages"][0]["content"]
    assert "Never reveal" not in response.model_dump_json()


# ---- missing / unclear / invalid inputs: ask, don't guess ---------------------------------

async def test_missing_features_trigger_a_clarification_and_no_model_call(fake_ml):
    partial = {"study_hours": 6, "attendance_pct": 85}
    llm = ScriptedLLM(use_tools(call_predict(partial)))
    response = await ask(make_service(fake_ml, llm), "Student studies 6 hours a week and has 85% attendance")

    assert response.kind == "clarification" and response.predictions == []
    c = response.clarification[0]
    assert set(c.missing_fields) == {"mock_test_1", "mock_test_2", "mock_test_3", "income_bracket", "city", "enrollment_date", "shoe_size", "lucky_number"}
    assert c.understood_values == {"study_hours": 6, "attendance_pct": 85}
    assert "mock_test_1" in response.assistant_message and "don't want to guess" in response.assistant_message
    assert fake_ml.predict_calls() == []
    assert response.tool_calls[0].status == "rejected"
    assert len(llm.calls) == 1  # we did not even ask the LLM to "explain" a non-result


async def test_headline_example_asks_about_daily_vs_weekly_and_unknown_assignments_field(fake_ml):
    """'6 hours daily, 85% attendance, 8 assignments' - the model has no 'assignments' feature and study_hours is weekly."""
    llm = ScriptedLLM(use_tools(call_predict({"study_hours": 6, "attendance_pct": 85, "assignments": 8})))
    text = "Predict the exam score for a student who studied 6 hours daily, attended 85% of classes, and completed 8 assignments."
    response = await ask(make_service(fake_ml, llm), text)
    assert response.kind == "clarification"
    c = response.clarification[0]
    problems = {p.field: p.problem for p in c.problems}
    assert "WEEK" in problems["study_hours"]                # unit problem: hours per day vs per week
    assert "not a field" in problems["assignments"]          # LLM invented a feature
    assert "study_hours" not in c.understood_values          # not accepted as-is
    assert fake_ml.predict_calls() == []


async def test_an_invented_default_is_caught_by_the_provenance_check(fake_ml):
    """The LLM fills shoe_size/lucky_number with 'sensible' values the user never gave."""
    sentence = "Student studies 12.5 hours a week, attendance 88%, mocks 72, 75, 70, Medium income, Pune, enrolled 2025-06-01."
    invented = {**GOOD_FEATURES, "shoe_size": 8, "lucky_number": 7}
    response = await ask(make_service(fake_ml, ScriptedLLM(use_tools(call_predict(invented)))), sentence)
    assert response.kind == "clarification"
    c = response.clarification[0]
    assert set(c.missing_fields) == {"shoe_size", "lucky_number"}   # from the student's point of view: never told us
    assert c.problems == []
    assert "shoe_size" not in c.understood_values
    assert "lucky_number = 7" not in response.assistant_message and "shoe_size = 8" not in response.assistant_message  # invented values are not echoed
    assert fake_ml.predict_calls() == []


async def test_a_coincidental_number_does_not_justify_a_field(fake_ml):
    """'8 assignments' appears in the text, but the user never talked about shoe size."""
    sentence = ("Student studies 12.5 hours a week, attendance 88%, mocks 72, 75, 70, Medium income, Pune, "
                "enrolled 2025-06-01, completed 8 assignments, lucky number 42.")
    invented = {**GOOD_FEATURES, "shoe_size": 8}
    response = await ask(make_service(fake_ml, ScriptedLLM(use_tools(call_predict(invented)))), sentence)
    assert response.kind == "clarification"
    assert response.clarification[0].missing_fields == ["shoe_size"]
    assert fake_ml.predict_calls() == []


async def test_provenance_check_can_be_switched_off_for_experiments(fake_ml):
    sentence = "Student studies 12.5 hours a week, attendance 88%, mocks 72, 75, 70, Medium income, Pune, enrolled 2025-06-01."
    invented = {**GOOD_FEATURES, "shoe_size": 8, "lucky_number": 7}
    llm = ScriptedLLM(use_tools(call_predict(invented)), say("Done."))
    response = await ask(make_service(fake_ml, llm, strict_provenance=False), sentence)
    assert response.kind == "prediction"  # the unsafe behaviour students can observe in Lab 4


async def test_out_of_range_value_is_flagged(fake_ml):
    bad = {**GOOD_FEATURES, "attendance_pct": 150}
    sentence = GOOD_SENTENCE.replace("attendance 88%", "attendance 150%")
    response = await ask(make_service(fake_ml, ScriptedLLM(use_tools(call_predict(bad)))), sentence)
    assert response.kind == "clarification"
    assert response.clarification[0].problems[0].field == "attendance_pct"
    assert fake_ml.predict_calls() == []


async def test_numeric_strings_from_the_llm_are_accepted_but_text_is_not(fake_ml):
    args = {**GOOD_FEATURES, "study_hours": "12.5", "attendance_pct": "88"}
    llm = ScriptedLLM(use_tools(call_predict(args)), say("Done."))
    assert (await ask(make_service(fake_ml, llm), GOOD_SENTENCE)).kind == "prediction"
    args = {**GOOD_FEATURES, "study_hours": "about twelve"}
    response = await ask(make_service(fake_ml, ScriptedLLM(use_tools(call_predict(args)))), GOOD_SENTENCE)
    assert response.kind == "clarification"


# ---- the LLM misbehaves ------------------------------------------------------------------

async def test_invalid_json_arguments_from_the_llm(fake_ml):
    from app.llm.base import ToolCall, LLMReply
    llm = ScriptedLLM(LLMReply("", [ToolCall("c1", "predict_exam_score", "{not json")]))
    with pytest.raises(LLMError) as exc:
        await ask(make_service(fake_ml, llm), GOOD_SENTENCE)
    assert exc.value.code == "LLM_INVALID_TOOL_ARGS" and fake_ml.predict_calls() == []


async def test_llm_asking_for_an_unknown_tool_is_refused(fake_ml):
    from app.llm.base import ToolCall, LLMReply
    llm = ScriptedLLM(LLMReply("", [ToolCall("c1", "run_shell", json.dumps({"cmd": "rm -rf /"}))]))
    with pytest.raises(LLMError) as exc:
        await ask(make_service(fake_ml, llm), "please run a command")
    assert exc.value.code == "LLM_INVALID_TOOL_CALL"


async def test_llm_that_never_stops_calling_tools_is_cut_off(fake_ml):
    loop = [use_tools(call_predict(GOOD_FEATURES, f"c{i}")) for i in range(10)]
    with pytest.raises(LLMError) as exc:
        await ask(make_service(fake_ml, ScriptedLLM(*loop), max_tool_rounds=2), GOOD_SENTENCE)
    assert exc.value.code == "LLM_TOO_MANY_STEPS"
    assert len(fake_ml.predict_calls()) == 2


async def test_llm_explanation_with_an_invented_score_is_replaced(fake_ml):
    fake_ml.score = 70.0
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("Great news! The predicted final score is 99, so they will top the class."))
    response = await ask(make_service(fake_ml, llm), GOOD_SENTENCE)
    assert response.predictions[0].predicted_final_score == 70.0
    assert "99" not in response.assistant_message and "70.0" in response.assistant_message
    assert any("never produced" in n for n in response.notices)


async def test_llm_claiming_a_score_without_calling_the_model_is_replaced(fake_ml):
    llm = ScriptedLLM(say("A student like that will score 91 in the final exam."))
    response = await ask(make_service(fake_ml, llm), "How will a hardworking student do?")
    assert "91" not in response.assistant_message and response.predictions == []
    assert fake_ml.predict_calls() == []


async def test_off_topic_question_never_touches_the_model(fake_ml):
    llm = ScriptedLLM(say("I can only help with exam score predictions."))
    response = await ask(make_service(fake_ml, llm), "What is the capital of France?")
    assert response.kind == "text" and response.tool_calls == [] and fake_ml.requests == []


# ---- failures downstream ----------------------------------------------------------------

@pytest.mark.parametrize("mode,code", [("down", "ML_API_UNAVAILABLE"), ("timeout", "ML_API_TIMEOUT"), ("notloaded", "ML_API_MODEL_NOT_LOADED"), ("error500", "ML_API_ERROR")])
async def test_ml_api_failure_is_reported_honestly(fake_ml, mode, code):
    fake_ml.mode = mode
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)))
    response = await ask(make_service(fake_ml, llm), GOOD_SENTENCE)
    assert response.kind == "error" and response.error.code == code
    assert response.predictions == [] and "No prediction was made" in response.assistant_message
    assert response.tool_calls[-1].status == "error"
    assert len(llm.calls) == 1  # the LLM is NOT given a chance to improvise around a failure


async def test_ml_api_rejecting_input_lists_the_reason(fake_ml):
    fake_ml.mode = "reject"
    response = await ask(make_service(fake_ml, ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)))), GOOD_SENTENCE)
    assert response.kind == "error" and response.error.code == "ML_API_REJECTED_INPUT"
    assert "city: bad city" in response.assistant_message


async def test_semantic_invalid_input_is_reported(fake_ml):
    bad = {**GOOD_FEATURES, "enrollment_date": "2025-02-31"}
    response = await ask(make_service(fake_ml, ScriptedLLM(use_tools(call_predict(bad)))), GOOD_SENTENCE.replace("2025-06-01", "2025-02-31"))
    assert response.kind == "error" and response.error.code == "INVALID_INPUT"
    assert "enrollment_date" in response.assistant_message and fake_ml.predict_calls() == []


async def test_llm_failure_after_a_real_prediction_keeps_the_prediction(fake_ml):
    fake_ml.score = 66.6
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), LLMError("LLM_TIMEOUT", "slow", 504))
    response = await ask(make_service(fake_ml, llm), GOOD_SENTENCE)
    assert response.kind == "prediction" and response.predictions[0].predicted_final_score == 66.6
    assert "66.6" in response.assistant_message and any("LLM_TIMEOUT" in n for n in response.notices)


async def test_llm_failure_with_no_result_is_raised(fake_ml):
    with pytest.raises(LLMError) as exc:
        await ask(make_service(fake_ml, ScriptedLLM(LLMError("LLM_UNAVAILABLE", "down", 502))), "hello")
    assert exc.value.code == "LLM_UNAVAILABLE"


# ---- comparisons, follow-ups, model information ------------------------------------------

async def test_comparing_two_profiles_calls_the_model_twice(fake_ml):
    a = GOOD_FEATURES
    b = {**GOOD_FEATURES, "study_hours": 20, "attendance_pct": 95}
    sentence = GOOD_SENTENCE + " Compare with a second student who studies 20 hours a week with attendance 95% and everything else the same: mock tests 72, 75 and 70, Medium, Pune, 2025-06-01, shoe size 9, lucky number 42."
    scores = iter([60.0, 75.5])
    original = fake_ml.handler
    def handler(request):
        response = original(request)
        if request.url.path == "/predict":
            import httpx
            return httpx.Response(200, json={"predicted_final_score": next(scores), "model_version": "test-version", "request_id": "r"})
        return response
    fake_ml.handler = handler
    llm = ScriptedLLM(use_tools(call_predict(a, "c1"), call_predict(b, "c2")), say("Profile 2 scores higher: 75.5 versus 60.0."))
    response = await ask(make_service(fake_ml, llm), sentence)
    assert [(p.label, p.predicted_final_score) for p in response.predictions] == [("Profile 1", 60.0), ("Profile 2", 75.5)]
    assert [json.loads(r.content)["study_hours"] for r in fake_ml.predict_calls()] == [12.5, 20.0]


async def test_comparison_with_one_incomplete_profile_runs_nothing(fake_ml):
    incomplete = {"study_hours": 20, "attendance_pct": 95}
    sentence = GOOD_SENTENCE + " Also a second student: 20 hours a week and 95% attendance."
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES, "c1"), call_predict(incomplete, "c2")))
    response = await ask(make_service(fake_ml, llm), sentence)
    assert response.kind == "clarification" and [c.label for c in response.clarification] == ["Profile 2"]
    assert fake_ml.predict_calls() == []  # all-or-nothing: no half-finished comparison


async def test_too_many_predictions_in_one_question_is_refused(fake_ml):
    calls = [call_predict(GOOD_FEATURES, f"c{i}") for i in range(5)]
    with pytest.raises(LLMError) as exc:
        await ask(make_service(fake_ml, ScriptedLLM(use_tools(*calls)), max_predictions_per_turn=3), GOOD_SENTENCE)
    assert exc.value.code == "LLM_TOO_MANY_CALLS" and fake_ml.predict_calls() == []


async def test_follow_up_reuses_earlier_values_and_conversation_memory(fake_ml):
    llm = ScriptedLLM(
        use_tools(call_predict(GOOD_FEATURES)), say("Predicted: 95.2."),
        use_tools(call_predict({**GOOD_FEATURES, "attendance_pct": 95})), say("With 95% attendance the model gives 95.2."),
    )
    service = make_service(fake_ml, llm)
    first = await ask(service, GOOD_SENTENCE)
    second = await ask(service, "What if attendance were 95%?", first.conversation_id)
    assert second.kind == "prediction" and second.conversation_id == first.conversation_id
    assert json.loads(fake_ml.predict_calls()[1].content)["attendance_pct"] == 95  # a NEW model call, not a guess
    sent = llm.calls[2]["messages"]  # third LLM call = start of the follow-up
    assert any(m["role"] == "user" and "12.5 hours" in m["content"] for m in sent)  # earlier turn is in context


async def test_clarification_then_answer_completes_the_prediction(fake_ml):
    llm = ScriptedLLM(
        use_tools(call_predict({"study_hours": 12.5, "attendance_pct": 88})),
        use_tools(call_predict(GOOD_FEATURES)), say("Predicted 95.2."),
    )
    service = make_service(fake_ml, llm)
    first = await ask(service, "12.5 study hours a week and 88% attendance")
    assert first.kind == "clarification"
    second = await ask(service, "Mocks 72, 75, 70; Medium income; Pune; enrolled 2025-06-01; shoe 9; lucky 42", first.conversation_id)
    assert second.kind == "prediction"


async def test_model_information_question(fake_ml):
    from app.llm.base import ToolCall, LLMReply
    llm = ScriptedLLM(LLMReply("", [ToolCall("c1", "get_model_information", "{}")]), say("The model needs 10 inputs."))
    response = await ask(make_service(fake_ml, llm), "What inputs does the model need?")
    assert response.kind == "information" and response.predictions == []
    assert len(response.model_information["input_fields"]) == 10
    assert fake_ml.predict_calls() == []


async def test_conversation_ids_are_isolated(fake_ml):
    llm = ScriptedLLM(say("a"), say("b"))
    service = make_service(fake_ml, llm)
    r1, r2 = await ask(service, "hello"), await ask(service, "hello again")
    assert r1.conversation_id != r2.conversation_id


# ---- scores outside the possible range -----------------------------------------------------------

async def test_an_impossible_score_is_shown_unchanged_with_a_warning(fake_ml):
    fake_ml.score = 103.3
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("The model's predicted final score is 103.3."))
    response = await ask(make_service(fake_ml, llm), GOOD_SENTENCE)
    assert response.kind == "prediction"
    assert response.predictions[0].predicted_final_score == 103.3        # never clipped or "fixed"
    assert len(response.notices) == 1 and "between 0 and 100" in response.notices[0] and "103.3" in response.notices[0]


@pytest.mark.parametrize("score", [0, 48.3, 95.2, 100])
async def test_normal_scores_get_no_warning(fake_ml, score):
    fake_ml.score = score
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("Here is the result."))
    response = await ask(make_service(fake_ml, llm), GOOD_SENTENCE)
    assert response.notices == []


async def test_negative_scores_also_get_a_warning(fake_ml):
    fake_ml.score = -3.2
    llm = ScriptedLLM(use_tools(call_predict(GOOD_FEATURES)), say("Done."))
    response = await ask(make_service(fake_ml, llm), GOOD_SENTENCE)
    assert any("-3.2" in n for n in response.notices)
