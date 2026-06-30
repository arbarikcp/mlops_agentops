"""Unit tests for platform/llm/llm_eval.py (Day 103)."""

import pytest
from llm.llm_eval import (
    EvalDataset,
    EvalExample,
    EvalMethod,
    EvalReport,
    EvalResult,
    JudgeConfig,
)


class TestEvalExample:
    def test_basic(self):
        e = EvalExample(input="What is 2+2?", expected_output="4")
        assert e.expected_output == "4"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="input"):
            EvalExample(input="")

    def test_to_dict(self):
        e = EvalExample(input="x")
        assert e.to_dict()["input"] == "x"


class TestEvalDataset:
    def test_basic(self):
        ds = EvalDataset(name="golden", examples=[EvalExample(input="x")])
        assert ds.size() == 1

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            EvalDataset(name="", examples=[EvalExample(input="x")])

    def test_empty_examples_raises(self):
        with pytest.raises(ValueError, match="examples"):
            EvalDataset(name="x", examples=[])

    def test_to_dict(self):
        ds = EvalDataset(name="golden", examples=[EvalExample(input="x")])
        assert len(ds.to_dict()["examples"]) == 1


class TestJudgeConfig:
    def test_basic(self):
        jc = JudgeConfig(judge_model="gpt-4", rubric_template="Score 1-5")
        assert jc.score_range == (1, 5)

    def test_empty_judge_model_raises(self):
        with pytest.raises(ValueError, match="judge_model"):
            JudgeConfig(judge_model="", rubric_template="x")

    def test_empty_rubric_raises(self):
        with pytest.raises(ValueError, match="rubric_template"):
            JudgeConfig(judge_model="x", rubric_template="")

    def test_invalid_score_range_raises(self):
        with pytest.raises(ValueError, match="score_range"):
            JudgeConfig(judge_model="x", rubric_template="y", score_range=(5, 1))

    def test_to_dict(self):
        jc = JudgeConfig(judge_model="x", rubric_template="y")
        assert jc.to_dict()["score_range"] == [1, 5]


class TestEvalResult:
    def test_basic(self):
        r = EvalResult(example_input="x", method=EvalMethod.EXACT_MATCH, score=1.0)
        assert r.normalized_score() == 1.0

    def test_invalid_max_score_raises(self):
        with pytest.raises(ValueError, match="max_score"):
            EvalResult(example_input="x", method=EvalMethod.EXACT_MATCH, score=0.5, max_score=0)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="score"):
            EvalResult(example_input="x", method=EvalMethod.EXACT_MATCH, score=2.0, max_score=1.0)

    def test_passed_threshold(self):
        r = EvalResult(example_input="x", method=EvalMethod.ROUGE_L, score=0.8, max_score=1.0)
        assert r.passed() is True
        assert r.passed(threshold=0.9) is False

    def test_to_dict(self):
        r = EvalResult(example_input="x", method=EvalMethod.LLM_JUDGE, score=3, max_score=5)
        d = r.to_dict()
        assert d["normalized_score"] == 0.6


class TestEvalReport:
    def test_mean_score(self):
        results = [
            EvalResult(example_input="a", method=EvalMethod.EXACT_MATCH, score=1.0),
            EvalResult(example_input="b", method=EvalMethod.EXACT_MATCH, score=0.5),
        ]
        report = EvalReport(dataset_name="golden", results=results)
        assert report.mean_score() == pytest.approx(0.75)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="dataset_name"):
            EvalReport(dataset_name="", results=[])

    def test_pass_rate(self):
        results = [
            EvalResult(example_input="a", method=EvalMethod.EXACT_MATCH, score=1.0),
            EvalResult(example_input="b", method=EvalMethod.EXACT_MATCH, score=0.2),
        ]
        report = EvalReport(dataset_name="golden", results=results)
        assert report.pass_rate(threshold=0.5) == 0.5

    def test_worst_examples(self):
        results = [
            EvalResult(example_input="a", method=EvalMethod.EXACT_MATCH, score=1.0),
            EvalResult(example_input="b", method=EvalMethod.EXACT_MATCH, score=0.2),
            EvalResult(example_input="c", method=EvalMethod.EXACT_MATCH, score=0.5),
        ]
        report = EvalReport(dataset_name="golden", results=results)
        worst = report.worst_examples(n=2)
        assert worst[0].example_input == "b"
        assert worst[1].example_input == "c"

    def test_empty_results_mean_score(self):
        report = EvalReport(dataset_name="x", results=[])
        assert report.mean_score() == 0.0

    def test_to_dict(self):
        results = [EvalResult(example_input="a", method=EvalMethod.EXACT_MATCH, score=1.0)]
        report = EvalReport(dataset_name="golden", results=results)
        d = report.to_dict()
        assert "mean_score" in d
