"""Tests for database operations."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.database import (
    DatabaseError,
    EvaluationRecord,
    _utc_now,
    create_evaluation,
    get_evaluation_by_id,
    get_evaluation_count,
    get_evaluation_history,
)
from app.models import CodeEvaluation


class TestEvaluationRecord:
    """Tests for the EvaluationRecord model."""

    def test_create_evaluation_record(
        self, test_db: Session, sample_evaluation: CodeEvaluation
    ):
        """Test creating an evaluation record."""
        record = create_evaluation(
            db=test_db,
            code="print('hello')",
            filename="test.py",
            evaluation_dict=sample_evaluation.model_dump(),
        )

        assert record.id is not None
        assert record.code == "print('hello')"
        assert record.filename == "test.py"
        assert record.overall_score == 75.0

    def test_evaluation_json_serialization(
        self, test_db: Session, sample_evaluation: CodeEvaluation
    ):
        """Test that evaluation data is correctly serialized/deserialized."""
        eval_dict = sample_evaluation.model_dump()
        record = create_evaluation(
            db=test_db,
            code="x = 1",
            filename=None,
            evaluation_dict=eval_dict,
        )

        retrieved_eval = record.get_evaluation()
        assert retrieved_eval["overall_score"] == eval_dict["overall_score"]
        assert retrieved_eval["summary"] == eval_dict["summary"]
        assert len(retrieved_eval["issues"]) == len(eval_dict["issues"])


class TestDatabaseQueries:
    """Tests for database query functions."""

    def test_get_evaluation_by_id(
        self, test_db: Session, sample_evaluation: CodeEvaluation
    ):
        """Test retrieving an evaluation by ID."""
        record = create_evaluation(
            db=test_db,
            code="test code",
            filename="test.py",
            evaluation_dict=sample_evaluation.model_dump(),
        )

        retrieved = get_evaluation_by_id(test_db, record.id)
        assert retrieved is not None
        assert retrieved.id == record.id
        assert retrieved.code == "test code"

    def test_get_evaluation_by_id_not_found(self, test_db: Session):
        """Test retrieving non-existent evaluation returns None."""
        retrieved = get_evaluation_by_id(test_db, 99999)
        assert retrieved is None

    def test_get_evaluation_history(
        self, test_db: Session, sample_evaluation: CodeEvaluation
    ):
        """Test retrieving evaluation history."""
        for i in range(3):
            create_evaluation(
                db=test_db,
                code=f"code {i}",
                filename=f"file{i}.py",
                evaluation_dict=sample_evaluation.model_dump(),
            )

        history = get_evaluation_history(test_db)
        assert len(history) == 3

    def test_get_evaluation_history_ordering(
        self, test_db: Session, sample_evaluation: CodeEvaluation
    ):
        """Test that history is ordered by created_at descending."""
        for i in range(3):
            create_evaluation(
                db=test_db,
                code=f"code {i}",
                filename=f"file{i}.py",
                evaluation_dict=sample_evaluation.model_dump(),
            )

        history = get_evaluation_history(test_db)
        assert history[0].filename == "file2.py"
        assert history[2].filename == "file0.py"

    def test_get_evaluation_history_pagination(
        self, test_db: Session, sample_evaluation: CodeEvaluation
    ):
        """Test history pagination."""
        for i in range(5):
            create_evaluation(
                db=test_db,
                code=f"code {i}",
                filename=f"file{i}.py",
                evaluation_dict=sample_evaluation.model_dump(),
            )

        page1 = get_evaluation_history(test_db, skip=0, limit=2)
        page2 = get_evaluation_history(test_db, skip=2, limit=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    def test_get_evaluation_count(
        self, test_db: Session, sample_evaluation: CodeEvaluation
    ):
        """Test counting evaluations."""
        assert get_evaluation_count(test_db) == 0

        for i in range(3):
            create_evaluation(
                db=test_db,
                code=f"code {i}",
                filename=None,
                evaluation_dict=sample_evaluation.model_dump(),
            )

        assert get_evaluation_count(test_db) == 3


class TestUtcNow:
    """Tests for the _utc_now function."""

    def test_returns_datetime(self):
        """Should return a datetime object."""
        result = _utc_now()
        assert isinstance(result, datetime)

    def test_is_timezone_aware(self):
        """Should return a timezone-aware datetime."""
        result = _utc_now()
        assert result.tzinfo is not None

    def test_uses_utc(self):
        """Should use UTC timezone."""
        result = _utc_now()
        assert result.tzinfo == UTC


class TestDatabaseErrorHandling:
    """Tests for database error handling."""

    def test_create_evaluation_with_missing_required_field(self, test_db: Session):
        """Test that missing required fields raise appropriate errors."""
        # Missing overall_score should raise an error
        with pytest.raises((DatabaseError, KeyError)):
            create_evaluation(
                db=test_db,
                code="test",
                filename=None,
                evaluation_dict={"summary": "test"},  # missing overall_score
            )

    def test_database_error_exception_exists(self):
        """DatabaseError exception should be defined."""
        assert DatabaseError is not None
        assert issubclass(DatabaseError, Exception)
