"""Tests for the evaluator module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.evaluator import (
    AVAILABLE_HF_MODELS,
    ConfigurationError,
    EvaluationError,
    MAX_RAW_RESPONSE_LENGTH,
    _build_messages,
    _build_prompt,
    get_config,
    parse_evaluation_response,
    validate_and_build_evaluation,
    validate_config,
)
from app.models import IssueCategory, IssueSeverity


class TestGetConfig:
    """Tests for get_config function."""

    def test_returns_dict(self):
        """Config should return a dictionary."""
        config = get_config()
        assert isinstance(config, dict)

    def test_has_required_keys(self):
        """Config should have all required keys."""
        config = get_config()
        required_keys = [
            "llm_provider",
            "hf_api_token",
            "hf_model_id",
            "perplexity_api_key",
            "perplexity_model",
        ]
        for key in required_keys:
            assert key in config

    def test_default_provider_is_huggingface(self):
        """Default provider should be huggingface."""
        with patch.dict("os.environ", {}, clear=True):
            config = get_config()
            assert config["llm_provider"] == "huggingface"


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_huggingface_config(self):
        """Valid HuggingFace config should pass validation."""
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "huggingface", "HF_API_TOKEN": "test_token"},
        ):
            config = validate_config()
            assert config["llm_provider"] == "huggingface"

    def test_valid_perplexity_config(self):
        """Valid Perplexity config should pass validation."""
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "perplexity", "PERPLEXITY_API_KEY": "test_key"},
        ):
            config = validate_config()
            assert config["llm_provider"] == "perplexity"

    def test_invalid_provider_raises_error(self):
        """Invalid provider should raise ConfigurationError."""
        with patch.dict("os.environ", {"LLM_PROVIDER": "invalid_provider"}):
            with pytest.raises(ConfigurationError) as exc_info:
                validate_config()
            assert "Invalid LLM_PROVIDER" in str(exc_info.value)


class TestBuildMessages:
    """Tests for _build_messages function."""

    def test_returns_list_of_dicts(self):
        """Should return a list of message dictionaries."""
        messages = _build_messages("test prompt")
        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_has_system_and_user_roles(self):
        """Should have system and user messages."""
        messages = _build_messages("test prompt")
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_user_message_contains_prompt(self):
        """User message should contain the prompt."""
        prompt = "analyze this code"
        messages = _build_messages(prompt)
        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["content"] == prompt


class TestBuildPrompt:
    """Tests for _build_prompt function."""

    def test_basic_prompt(self):
        """Should build prompt with code."""
        code = "def foo(): pass"
        prompt = _build_prompt(code)
        assert code in prompt

    def test_prompt_with_filename(self):
        """Should include filename when provided."""
        code = "def foo(): pass"
        filename = "test.py"
        prompt = _build_prompt(code, filename)
        assert filename in prompt
        assert code in prompt

    def test_prompt_without_filename(self):
        """Should not have filename prefix when not provided."""
        code = "def foo(): pass"
        prompt = _build_prompt(code)
        assert "Filename:" not in prompt


class TestParseEvaluationResponse:
    """Tests for parse_evaluation_response function."""

    def test_valid_json_response(self):
        """Should parse valid JSON response."""
        response = json.dumps({
            "overall_score": 85,
            "summary": "Good code",
            "issues": [],
            "strengths": ["Clear"],
            "improvement_areas": ["Add tests"],
            "time_complexity": "O(1)",
            "memory_complexity": "O(1)",
        })
        result = parse_evaluation_response(response)
        assert result["overall_score"] == 85
        assert result["summary"] == "Good code"

    def test_json_with_surrounding_text(self):
        """Should extract JSON from response with surrounding text."""
        response = 'Here is the evaluation: {"overall_score": 90, "summary": "Great"} End.'
        result = parse_evaluation_response(response)
        assert result["overall_score"] == 90

    def test_invalid_json_returns_fallback(self):
        """Should return fallback dict for invalid JSON."""
        response = "This is not valid JSON at all"
        result = parse_evaluation_response(response)
        assert result["overall_score"] == 50
        assert "Could not parse" in result["summary"]

    def test_truncates_long_raw_response(self):
        """Should truncate long raw responses in error message."""
        long_response = "x" * (MAX_RAW_RESPONSE_LENGTH + 100)
        result = parse_evaluation_response(long_response)
        # Check that the raw response preview is truncated
        raw_preview = result["improvement_areas"][1]
        assert "[truncated]" in raw_preview


class TestValidateAndBuildEvaluation:
    """Tests for validate_and_build_evaluation function."""

    def test_valid_data(self):
        """Should build CodeEvaluation from valid data."""
        data = {
            "overall_score": 80,
            "summary": "Good code",
            "issues": [
                {
                    "category": "performance",
                    "severity": "medium",
                    "line_number": 5,
                    "description": "Slow loop",
                    "suggestion": "Use list comprehension",
                }
            ],
            "strengths": ["Clear naming"],
            "improvement_areas": ["Add docstrings"],
            "time_complexity": "O(n)",
            "memory_complexity": "O(1)",
        }
        evaluation = validate_and_build_evaluation(data)
        assert evaluation.overall_score == 80
        assert evaluation.summary == "Good code"
        assert len(evaluation.issues) == 1
        assert evaluation.issues[0].category == IssueCategory.PERFORMANCE

    def test_clamps_score_to_valid_range(self):
        """Should clamp score to 0-100 range."""
        data = {"overall_score": 150, "summary": "Test"}
        evaluation = validate_and_build_evaluation(data)
        assert evaluation.overall_score == 100

        data = {"overall_score": -10, "summary": "Test"}
        evaluation = validate_and_build_evaluation(data)
        assert evaluation.overall_score == 0

    def test_handles_invalid_category(self):
        """Should use default category for invalid values."""
        data = {
            "overall_score": 70,
            "summary": "Test",
            "issues": [
                {
                    "category": "invalid_category",
                    "severity": "high",
                    "description": "Test issue",
                    "suggestion": "Fix it",
                }
            ],
        }
        evaluation = validate_and_build_evaluation(data)
        assert evaluation.issues[0].category == IssueCategory.READABILITY

    def test_handles_invalid_severity(self):
        """Should use default severity for invalid values."""
        data = {
            "overall_score": 70,
            "summary": "Test",
            "issues": [
                {
                    "category": "bug",
                    "severity": "super_critical",
                    "description": "Test issue",
                    "suggestion": "Fix it",
                }
            ],
        }
        evaluation = validate_and_build_evaluation(data)
        assert evaluation.issues[0].severity == IssueSeverity.MEDIUM

    def test_handles_missing_fields(self):
        """Should use defaults for missing fields."""
        data = {}
        evaluation = validate_and_build_evaluation(data)
        assert evaluation.overall_score == 50
        assert evaluation.summary == "Evaluation completed."
        assert evaluation.issues == []
        assert evaluation.strengths == []

    def test_handles_non_numeric_score(self):
        """Should use default for non-numeric score."""
        data = {"overall_score": "not a number", "summary": "Test"}
        evaluation = validate_and_build_evaluation(data)
        assert evaluation.overall_score == 50


class TestAvailableModels:
    """Tests for available models constant."""

    def test_models_is_list(self):
        """Available models should be a list."""
        assert isinstance(AVAILABLE_HF_MODELS, list)

    def test_models_not_empty(self):
        """Should have at least one model available."""
        assert len(AVAILABLE_HF_MODELS) > 0

    def test_models_are_strings(self):
        """All models should be strings."""
        for model in AVAILABLE_HF_MODELS:
            assert isinstance(model, str)
