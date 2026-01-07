"""LLM API integration for code evaluation (HuggingFace & Perplexity)."""

import json
import os
import re
from enum import Enum

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from openai import OpenAI

from app.models import CodeEvaluation, CodeIssue, IssueCategory, IssueSeverity

# Load environment variables
load_dotenv()


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    HUGGINGFACE = "huggingface"
    PERPLEXITY = "perplexity"


def get_config():
    """Get configuration from environment variables."""
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "huggingface").lower(),
        "hf_api_token": os.getenv("HF_API_TOKEN"),
        "hf_model_id": os.getenv("HF_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        "perplexity_api_key": os.getenv("PERPLEXITY_API_KEY"),
        "perplexity_model": os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-online"),
    }

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


def get_hf_client() -> InferenceClient:
    """Create and return a HuggingFace Inference client."""
    config = get_config()
    if not config["hf_api_token"]:
        raise ValueError("HF_API_TOKEN environment variable is not set")
    return InferenceClient(token=config["hf_api_token"])


def get_perplexity_client() -> OpenAI:
    """Create and return a Perplexity client (OpenAI-compatible)."""
    config = get_config()
    if not config["perplexity_api_key"]:
        raise ValueError("PERPLEXITY_API_KEY environment variable is not set")
    return OpenAI(
        api_key=config["perplexity_api_key"],
        base_url="https://api.perplexity.ai",
    )


def parse_evaluation_response(response_text: str) -> dict:
    """
    Parse the LLM response into an evaluation dictionary.

    Args:
        response_text: Raw response from the LLM.

    Returns:
        Parsed evaluation dictionary.

    Raises:
        ValueError: If response cannot be parsed as JSON.
    """
    text = response_text.strip()

    # Try to extract JSON from the response - find the outermost braces
    brace_count = 0
    start_idx = None
    end_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                end_idx = i + 1
                break

    if start_idx is not None and end_idx is not None:
        text = text[start_idx:end_idx]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If JSON parsing fails, return a default evaluation with warning and raw response
        print(f"WARNING: Could not parse LLM response as JSON.")
        return {
            "overall_score": 50,
            "summary": "⚠️ WARNING: Could not parse response as JSON. Raw LLM response shown below.",
            "issues": [],
            "strengths": [],
            "improvement_areas": [
                "Try a different model like 'mistralai/Mistral-7B-Instruct-v0.3'",
                f"RAW RESPONSE: {response_text}"
            ],
            "time_complexity": None,
            "memory_complexity": None,
        }


def validate_and_build_evaluation(data: dict) -> CodeEvaluation:
    """
    Validate parsed data and build a CodeEvaluation object.

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

        issues.append(CodeIssue(
            category=category,
            severity=severity,
            line_number=issue_data.get("line_number"),
            description=issue_data.get("description", "No description provided"),
            suggestion=issue_data.get("suggestion", "No suggestion provided"),
            example_fix=issue_data.get("example_fix"),
        ))

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


def evaluate_with_huggingface(prompt: str) -> str:
    """
    Evaluate code using HuggingFace Inference API.

    Args:
        prompt: The formatted evaluation prompt.

    Returns:
        Raw response text from the model.
    """
    config = get_config()
    client = get_hf_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = client.chat_completion(
        model=config["hf_model_id"],
        messages=messages,
        max_tokens=2048,
        temperature=0.1,
    )

    return response.choices[0].message.content


def evaluate_with_perplexity(prompt: str) -> str:
    """
    Evaluate code using Perplexity API.

    Args:
        prompt: The formatted evaluation prompt.

    Returns:
        Raw response text from the model.
    """
    config = get_config()
    client = get_perplexity_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=config["perplexity_model"],
        messages=messages,
        max_tokens=2048,
        temperature=0.1,
    )

    return response.choices[0].message.content


def evaluate_code_sync(code: str, filename: str | None = None) -> CodeEvaluation:
    """
    Evaluate Python code using the configured LLM provider.

    Args:
        code: The Python code to evaluate.
        filename: Optional filename for additional context.

    Returns:
        CodeEvaluation object containing the analysis results.

    Raises:
        ValueError: If the response cannot be parsed or provider is invalid.
        Exception: If the API request fails.
    """
    config = get_config()
    context = ""
    if filename:
        context = f"Filename: {filename}\n\n"

    prompt = context + EVALUATION_PROMPT.format(code=code)

    if config["llm_provider"] == LLMProvider.PERPLEXITY:
        response_text = evaluate_with_perplexity(prompt)
    elif config["llm_provider"] == LLMProvider.HUGGINGFACE:
        response_text = evaluate_with_huggingface(prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {config['llm_provider']}. Use 'huggingface' or 'perplexity'.")

    evaluation_data = parse_evaluation_response(response_text)
    return validate_and_build_evaluation(evaluation_data)


async def evaluate_code(code: str, filename: str | None = None) -> CodeEvaluation:
    """
    Async wrapper for evaluate_code_sync.

    Args:
        code: The Python code to evaluate.
        filename: Optional filename for additional context.

    Returns:
        CodeEvaluation object containing the analysis results.
    """
    return evaluate_code_sync(code, filename)


def compare_providers(code: str, filename: str | None = None) -> dict:
    """
    Evaluate code using both providers and compare results.

    Args:
        code: The Python code to evaluate.
        filename: Optional filename for additional context.

    Returns:
        Dictionary with results from both providers and timing info.
    """
    import time

    config = get_config()
    context = ""
    if filename:
        context = f"Filename: {filename}\n\n"
    prompt = context + EVALUATION_PROMPT.format(code=code)

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
        try:
            start = time.time()
            response_text = evaluate_with_huggingface(prompt)
            results["huggingface_time"] = round(time.time() - start, 2)
            evaluation_data = parse_evaluation_response(response_text)
            results["huggingface"] = validate_and_build_evaluation(evaluation_data)
        except Exception as e:
            results["huggingface_error"] = str(e)
    else:
        results["huggingface_error"] = "HF_API_TOKEN not configured"

    # Try Perplexity
    if config["perplexity_api_key"]:
        try:
            start = time.time()
            response_text = evaluate_with_perplexity(prompt)
            results["perplexity_time"] = round(time.time() - start, 2)
            evaluation_data = parse_evaluation_response(response_text)
            results["perplexity"] = validate_and_build_evaluation(evaluation_data)
        except Exception as e:
            results["perplexity_error"] = str(e)
    else:
        results["perplexity_error"] = "PERPLEXITY_API_KEY not configured"

    return results
