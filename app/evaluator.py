"""LLM API integration for code evaluation (HuggingFace & Perplexity)."""

import asyncio
import json
import logging
import os
import time
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from app.models import CodeEvaluation, CodeIssue, IssueCategory, IssueSeverity

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
MAX_TOKENS = 2048
TEMPERATURE = 0.1
MAX_RAW_RESPONSE_LENGTH = 500  # Truncate raw responses in error messages


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    HUGGINGFACE = "huggingface"
    PERPLEXITY = "perplexity"


# Available HuggingFace models for multi-model comparison
AVAILABLE_HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "openai/gpt-oss-20b",
    "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
]


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


class EvaluationError(Exception):
    """Raised when code evaluation fails."""

    pass


def get_config() -> dict[str, Any]:
    """Get configuration from environment variables.

    Returns:
        Configuration dictionary with LLM provider settings.
    """
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "huggingface").lower(),
        "hf_api_token": os.getenv("HF_API_TOKEN"),
        "hf_model_id": os.getenv("HF_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        "perplexity_api_key": os.getenv("PERPLEXITY_API_KEY"),
        "perplexity_model": os.getenv(
            "PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online"
        ),
    }


def validate_config() -> dict[str, Any]:
    """Validate configuration and return it if valid.

    Returns:
        Validated configuration dictionary.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    config = get_config()

    # Validate provider
    valid_providers = [p.value for p in LLMProvider]
    if config["llm_provider"] not in valid_providers:
        raise ConfigurationError(
            f"Invalid LLM_PROVIDER: {config['llm_provider']}. "
            f"Valid options: {valid_providers}"
        )

    # Check that at least one provider is configured
    if not config["hf_api_token"] and not config["perplexity_api_key"]:
        logger.warning(
            "No LLM API keys configured. Set HF_API_TOKEN or PERPLEXITY_API_KEY."
        )

    return config


SYSTEM_PROMPT = """You are an expert Python code reviewer. Analyze the given code and respond with a JSON object containing your evaluation.

You MUST respond with ONLY valid JSON in this exact format, no other text:
{
    "overall_score": <number 0-100>,
    "summary": "<brief summary>",
    "issues": [
        {
            "category": "<performance|readability|security|maintainability|bug>",
            "severity": "<critical|high|medium|low>",
            "line_number": <number or null>,
            "description": "<issue description>",
            "suggestion": "<how to fix>",
            "example_fix": "<code example or null>"
        }
    ],
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvement_areas": ["<area 1>", "<area 2>"],
    "time_complexity": "<O(n) or null>",
    "memory_complexity": "<O(n) or null>"
}"""

EVALUATION_PROMPT = """Analyze this Python code and provide your evaluation as JSON:

```python
{code}
```

Remember: Respond with ONLY the JSON object, no markdown, no explanation."""


def _build_messages(prompt: str) -> list[dict[str, str]]:
    """Build the message list for LLM API calls.

    Args:
        prompt: The user prompt to include.

    Returns:
        List of message dictionaries.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _build_prompt(code: str, filename: str | None = None) -> str:
    """Build the evaluation prompt with optional filename context.

    Args:
        code: The code to evaluate.
        filename: Optional filename for context.

    Returns:
        Formatted prompt string.
    """
    context = f"Filename: {filename}\n\n" if filename else ""
    return context + EVALUATION_PROMPT.format(code=code)


def get_hf_client() -> InferenceClient:
    """Create and return a HuggingFace Inference client.

    Returns:
        Configured InferenceClient.

    Raises:
        ConfigurationError: If HF_API_TOKEN is not set.
    """
    config = get_config()
    if not config["hf_api_token"]:
        raise ConfigurationError("HF_API_TOKEN environment variable is not set")
    return InferenceClient(token=config["hf_api_token"])


def get_perplexity_client() -> OpenAI:
    """Create and return a Perplexity client (OpenAI-compatible).

    Returns:
        Configured OpenAI client for Perplexity.

    Raises:
        ConfigurationError: If PERPLEXITY_API_KEY is not set.
    """
    config = get_config()
    if not config["perplexity_api_key"]:
        raise ConfigurationError("PERPLEXITY_API_KEY environment variable is not set")
    return OpenAI(
        api_key=config["perplexity_api_key"],
        base_url="https://api.perplexity.ai",
    )


def parse_evaluation_response(response_text: str) -> dict:
    """Parse the LLM response into an evaluation dictionary.

    Args:
        response_text: Raw response from the LLM.

    Returns:
        Parsed evaluation dictionary.
    """
    text = response_text.strip()

    # Try to extract JSON from the response - find the outermost braces
    brace_count = 0
    start_idx = None
    end_idx = None

    for i, char in enumerate(text):
        if char == "{":
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                end_idx = i + 1
                break

    if start_idx is not None and end_idx is not None:
        text = text[start_idx:end_idx]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Could not parse LLM response as JSON: %s", str(e))
        # Truncate raw response for safety and readability
        truncated_response = response_text[:MAX_RAW_RESPONSE_LENGTH]
        if len(response_text) > MAX_RAW_RESPONSE_LENGTH:
            truncated_response += "... [truncated]"

        return {
            "overall_score": 50,
            "summary": "Could not parse response as JSON. Try a different model.",
            "issues": [],
            "strengths": [],
            "improvement_areas": [
                "The model did not return valid JSON",
                f"Raw response preview: {truncated_response}",
            ],
            "time_complexity": None,
            "memory_complexity": None,
        }


def validate_and_build_evaluation(data: dict) -> CodeEvaluation:
    """Validate parsed data and build a CodeEvaluation object.

    Args:
        data: Parsed evaluation dictionary.

    Returns:
        Validated CodeEvaluation object.
    """
    issues = []
    for issue_data in data.get("issues", []):
        try:
            category = IssueCategory(issue_data.get("category", "readability").lower())
        except ValueError:
            category = IssueCategory.READABILITY

        try:
            severity = IssueSeverity(issue_data.get("severity", "medium").lower())
        except ValueError:
            severity = IssueSeverity.MEDIUM

        issues.append(
            CodeIssue(
                category=category,
                severity=severity,
                line_number=issue_data.get("line_number"),
                description=issue_data.get("description", "No description provided"),
                suggestion=issue_data.get("suggestion", "No suggestion provided"),
                example_fix=issue_data.get("example_fix"),
            )
        )

    score = data.get("overall_score", 50)
    if not isinstance(score, (int, float)):
        score = 50
    score = max(0, min(100, score))

    return CodeEvaluation(
        overall_score=score,
        summary=data.get("summary", "Evaluation completed."),
        issues=issues,
        strengths=data.get("strengths", []),
        improvement_areas=data.get("improvement_areas", []),
        time_complexity=data.get("time_complexity"),
        memory_complexity=data.get("memory_complexity"),
    )


def _call_huggingface(messages: list[dict], model_id: str) -> str:
    """Call HuggingFace API with the given messages.

    Args:
        messages: List of message dictionaries.
        model_id: The model ID to use.

    Returns:
        Response text from the model.

    Raises:
        EvaluationError: If the API call fails.
    """
    try:
        client = get_hf_client()
        response = client.chat_completion(
            model=model_id,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content
    except HfHubHTTPError as e:
        logger.error("HuggingFace API error: %s", str(e))
        raise EvaluationError(f"HuggingFace API error: {e}") from e
    except Exception as e:
        logger.error("Unexpected error calling HuggingFace: %s", str(e))
        raise EvaluationError(f"HuggingFace error: {e}") from e


def _call_perplexity(messages: list[dict], model_id: str) -> str:
    """Call Perplexity API with the given messages.

    Args:
        messages: List of message dictionaries.
        model_id: The model ID to use.

    Returns:
        Response text from the model.

    Raises:
        EvaluationError: If the API call fails.
    """
    try:
        client = get_perplexity_client()
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content
    except APITimeoutError as e:
        logger.error("Perplexity timeout: %s", str(e))
        raise EvaluationError(f"Perplexity timeout: {e}") from e
    except APIConnectionError as e:
        logger.error("Perplexity connection error: %s", str(e))
        raise EvaluationError(f"Perplexity connection error: {e}") from e
    except RateLimitError as e:
        logger.error("Perplexity rate limit: %s", str(e))
        raise EvaluationError(f"Perplexity rate limit exceeded: {e}") from e
    except Exception as e:
        logger.error("Unexpected error calling Perplexity: %s", str(e))
        raise EvaluationError(f"Perplexity error: {e}") from e


def evaluate_with_huggingface(prompt: str) -> str:
    """Evaluate code using HuggingFace Inference API.

    Args:
        prompt: The formatted evaluation prompt.

    Returns:
        Raw response text from the model.
    """
    config = get_config()
    messages = _build_messages(prompt)
    return _call_huggingface(messages, config["hf_model_id"])


def evaluate_with_model(prompt: str, model_id: str) -> str:
    """Evaluate code using a specific HuggingFace model.

    Args:
        prompt: The formatted evaluation prompt.
        model_id: The HuggingFace model ID to use.

    Returns:
        Raw response text from the model.
    """
    messages = _build_messages(prompt)
    return _call_huggingface(messages, model_id)


def evaluate_with_perplexity(prompt: str) -> str:
    """Evaluate code using Perplexity API.

    Args:
        prompt: The formatted evaluation prompt.

    Returns:
        Raw response text from the model.
    """
    config = get_config()
    messages = _build_messages(prompt)
    return _call_perplexity(messages, config["perplexity_model"])


def _timed_evaluation(
    evaluate_fn, prompt: str, *args
) -> tuple[CodeEvaluation | None, float | None, str | None]:
    """Execute an evaluation function with timing and error handling.

    Args:
        evaluate_fn: The evaluation function to call.
        prompt: The prompt to pass to the function.
        *args: Additional arguments for the function.

    Returns:
        Tuple of (evaluation, response_time, error).
    """
    try:
        start = time.time()
        response_text = evaluate_fn(prompt, *args) if args else evaluate_fn(prompt)
        response_time = round(time.time() - start, 2)
        evaluation_data = parse_evaluation_response(response_text)
        evaluation = validate_and_build_evaluation(evaluation_data)
        return evaluation, response_time, None
    except (ConfigurationError, EvaluationError) as e:
        logger.warning("Evaluation failed: %s", str(e))
        return None, None, str(e)
    except Exception as e:
        logger.error("Unexpected evaluation error: %s", str(e))
        return None, None, str(e)


def evaluate_code_sync(code: str, filename: str | None = None) -> CodeEvaluation:
    """Evaluate Python code using the configured LLM provider.

    Args:
        code: The Python code to evaluate.
        filename: Optional filename for additional context.

    Returns:
        CodeEvaluation object containing the analysis results.

    Raises:
        ConfigurationError: If the provider configuration is invalid.
        EvaluationError: If the evaluation fails.
    """
    config = get_config()
    prompt = _build_prompt(code, filename)

    if config["llm_provider"] == LLMProvider.PERPLEXITY.value:
        response_text = evaluate_with_perplexity(prompt)
    elif config["llm_provider"] == LLMProvider.HUGGINGFACE.value:
        response_text = evaluate_with_huggingface(prompt)
    else:
        raise ConfigurationError(
            f"Unknown LLM provider: {config['llm_provider']}. "
            "Use 'huggingface' or 'perplexity'."
        )

    evaluation_data = parse_evaluation_response(response_text)
    return validate_and_build_evaluation(evaluation_data)


async def evaluate_code(code: str, filename: str | None = None) -> CodeEvaluation:
    """Async wrapper for evaluate_code_sync.

    Runs the synchronous evaluation in a thread pool to avoid blocking.

    Args:
        code: The Python code to evaluate.
        filename: Optional filename for additional context.

    Returns:
        CodeEvaluation object containing the analysis results.
    """
    return await asyncio.to_thread(evaluate_code_sync, code, filename)


def compare_providers(code: str, filename: str | None = None) -> dict:
    """Evaluate code using both providers and compare results.

    Args:
        code: The Python code to evaluate.
        filename: Optional filename for additional context.

    Returns:
        Dictionary with results from both providers and timing info.
    """
    config = get_config()
    prompt = _build_prompt(code, filename)

    results = {
        "huggingface": None,
        "perplexity": None,
        "huggingface_time": None,
        "perplexity_time": None,
        "huggingface_error": None,
        "perplexity_error": None,
    }

    # Try HuggingFace
    if config["hf_api_token"]:
        eval_result, time_taken, error = _timed_evaluation(
            evaluate_with_huggingface, prompt
        )
        results["huggingface"] = eval_result
        results["huggingface_time"] = time_taken
        results["huggingface_error"] = error
    else:
        results["huggingface_error"] = "HF_API_TOKEN not configured"

    # Try Perplexity
    if config["perplexity_api_key"]:
        eval_result, time_taken, error = _timed_evaluation(
            evaluate_with_perplexity, prompt
        )
        results["perplexity"] = eval_result
        results["perplexity_time"] = time_taken
        results["perplexity_error"] = error
    else:
        results["perplexity_error"] = "PERPLEXITY_API_KEY not configured"

    return results


def compare_hf_models(
    code: str, models: list[str], filename: str | None = None
) -> list[dict]:
    """Evaluate code using multiple HuggingFace models.

    Args:
        code: The Python code to evaluate.
        models: List of HuggingFace model IDs to use.
        filename: Optional filename for additional context.

    Returns:
        List of result dictionaries with model_name, evaluation, response_time, and error.
    """
    config = get_config()
    if not config["hf_api_token"]:
        return [
            {
                "model_name": model,
                "evaluation": None,
                "response_time": None,
                "error": "HF_API_TOKEN not configured",
            }
            for model in models
        ]

    prompt = _build_prompt(code, filename)

    results = []
    for model_id in models:
        eval_result, time_taken, error = _timed_evaluation(
            evaluate_with_model, prompt, model_id
        )
        results.append(
            {
                "model_name": model_id,
                "evaluation": eval_result,
                "response_time": time_taken,
                "error": error,
            }
        )

    return results
