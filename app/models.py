"""Pydantic models for the code evaluation system."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class IssueCategory(str, Enum):
    """Categories of code issues."""

    PERFORMANCE = "performance"
    READABILITY = "readability"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    BUG = "bug"


class IssueSeverity(str, Enum):
    """Severity levels for code issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CodeIssue(BaseModel):
    """Represents a single code issue found during evaluation."""

    category: IssueCategory = Field(description="Category of the issue")
    severity: IssueSeverity = Field(description="Severity level of the issue")
    line_number: int | None = Field(
        default=None, description="Line number where the issue occurs"
    )
    description: str = Field(description="Detailed description of the issue")
    suggestion: str = Field(description="Suggestion for fixing the issue")
    example_fix: str | None = Field(
        default=None, description="Example code fix if applicable"
    )


class CodeEvaluation(BaseModel):
    """Complete evaluation result for submitted code."""

    overall_score: float = Field(
        ge=0, le=100, description="Overall code quality score from 0-100"
    )
    summary: str = Field(description="Brief summary of the code evaluation")
    issues: list[CodeIssue] = Field(
        default_factory=list, description="List of identified issues"
    )
    strengths: list[str] = Field(
        default_factory=list, description="List of code strengths"
    )
    improvement_areas: list[str] = Field(
        default_factory=list, description="Areas that could be improved"
    )
    time_complexity: str | None = Field(
        default=None, description="Estimated time complexity"
    )
    memory_complexity: str | None = Field(
        default=None, description="Estimated memory complexity"
    )


class CodeSubmission(BaseModel):
    """Request model for code submission."""

    code: str = Field(min_length=1, description="Python code to evaluate")
    filename: str | None = Field(
        default=None, description="Optional filename for context"
    )


class EvaluationResponse(BaseModel):
    """Response model for evaluation submission."""

    evaluation_id: int = Field(description="Unique identifier for the evaluation")
    message: str = Field(description="Status message")


class EvaluationResult(BaseModel):
    """Complete evaluation result with metadata."""

    id: int = Field(description="Evaluation ID")
    code: str = Field(description="Submitted code")
    filename: str | None = Field(description="Optional filename")
    evaluation: CodeEvaluation = Field(description="Evaluation results")
    created_at: datetime = Field(description="Timestamp of evaluation")


class EvaluationSummary(BaseModel):
    """Summary of an evaluation for history listing."""

    id: int = Field(description="Evaluation ID")
    filename: str | None = Field(description="Optional filename")
    overall_score: float = Field(description="Overall score")
    summary: str = Field(description="Brief summary")
    created_at: datetime = Field(description="Timestamp of evaluation")


class HistoryResponse(BaseModel):
    """Response model for evaluation history."""

    evaluations: list[EvaluationSummary] = Field(description="List of evaluations")
    total: int = Field(description="Total number of evaluations")
