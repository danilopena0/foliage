"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from app.models import (
    CodeEvaluation,
    CodeIssue,
    CodeSubmission,
    EvaluationResponse,
    IssueCategory,
    IssueSeverity,
)


class TestCodeIssue:
    """Tests for the CodeIssue model."""

    def test_valid_code_issue(self):
        """Test creating a valid CodeIssue."""
        issue = CodeIssue(
            category=IssueCategory.SECURITY,
            severity=IssueSeverity.HIGH,
            line_number=10,
            description="SQL injection vulnerability",
            suggestion="Use parameterized queries",
            example_fix="cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        )

        assert issue.category == IssueCategory.SECURITY
        assert issue.severity == IssueSeverity.HIGH
        assert issue.line_number == 10
        assert "SQL injection" in issue.description

    def test_code_issue_without_optional_fields(self):
        """Test CodeIssue with only required fields."""
        issue = CodeIssue(
            category=IssueCategory.READABILITY,
            severity=IssueSeverity.LOW,
            description="Variable name could be more descriptive",
            suggestion="Rename 'x' to 'counter'",
        )

        assert issue.line_number is None
        assert issue.example_fix is None


class TestCodeEvaluation:
    """Tests for the CodeEvaluation model."""

    def test_valid_evaluation(self, sample_evaluation: CodeEvaluation):
        """Test creating a valid CodeEvaluation."""
        assert sample_evaluation.overall_score == 75.0
        assert len(sample_evaluation.issues) == 1
        assert len(sample_evaluation.strengths) == 2

    def test_score_boundaries(self):
        """Test that score must be between 0 and 100."""
        with pytest.raises(ValidationError):
            CodeEvaluation(
                overall_score=101,
                summary="Invalid score",
                issues=[],
                strengths=[],
                improvement_areas=[],
            )

        with pytest.raises(ValidationError):
            CodeEvaluation(
                overall_score=-1,
                summary="Invalid score",
                issues=[],
                strengths=[],
                improvement_areas=[],
            )

    def test_valid_score_at_boundaries(self):
        """Test valid scores at boundaries."""
        eval_0 = CodeEvaluation(
            overall_score=0,
            summary="Minimum score",
            issues=[],
            strengths=[],
            improvement_areas=[],
        )
        assert eval_0.overall_score == 0

        eval_100 = CodeEvaluation(
            overall_score=100,
            summary="Maximum score",
            issues=[],
            strengths=[],
            improvement_areas=[],
        )
        assert eval_100.overall_score == 100


class TestCodeSubmission:
    """Tests for the CodeSubmission model."""

    def test_valid_submission(self):
        """Test creating a valid submission."""
        submission = CodeSubmission(
            code="print('hello')",
            filename="test.py",
        )

        assert submission.code == "print('hello')"
        assert submission.filename == "test.py"

    def test_submission_without_filename(self):
        """Test submission without optional filename."""
        submission = CodeSubmission(code="x = 1")
        assert submission.filename is None

    def test_empty_code_rejected(self):
        """Test that empty code is rejected."""
        with pytest.raises(ValidationError):
            CodeSubmission(code="")


class TestEvaluationResponse:
    """Tests for the EvaluationResponse model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        response = EvaluationResponse(
            evaluation_id=1,
            message="Evaluation completed",
        )

        assert response.evaluation_id == 1
        assert response.message == "Evaluation completed"
