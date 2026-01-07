"""Tests for database operations."""

import pytest
from sqlalchemy.orm import Session

from app.database import (
    EvaluationRecord,
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
