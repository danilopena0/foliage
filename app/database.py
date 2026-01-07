"""Database setup and models using SQLAlchemy with SQLite."""

import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from sqlalchemy import DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./evaluations.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class EvaluationRecord(Base):
    """Database model for storing code evaluations."""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    def set_evaluation(self, evaluation_dict: dict) -> None:
        """Store evaluation data as JSON."""
        self.evaluation_json = json.dumps(evaluation_dict)

    def get_evaluation(self) -> dict:
        """Retrieve evaluation data from JSON."""
        return json.loads(self.evaluation_json)


def init_db() -> None:
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.

    Yields:
        Database session that auto-closes after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Yields:
        Database session that auto-closes after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_evaluation(
    db: Session,
    code: str,
    filename: str | None,
    evaluation_dict: dict,
) -> EvaluationRecord:
    """
    Create a new evaluation record in the database.

    Args:
        db: Database session.
        code: The submitted Python code.
        filename: Optional filename.
        evaluation_dict: The evaluation results as a dictionary.

    Returns:
        The created evaluation record.
    """
    record = EvaluationRecord(
        code=code,
        filename=filename,
        overall_score=evaluation_dict["overall_score"],
        summary=evaluation_dict["summary"],
    )
    record.set_evaluation(evaluation_dict)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_evaluation_by_id(db: Session, evaluation_id: int) -> EvaluationRecord | None:
    """
    Retrieve an evaluation by its ID.

    Args:
        db: Database session.
        evaluation_id: The evaluation ID to look up.

    Returns:
        The evaluation record if found, None otherwise.
    """
    return db.query(EvaluationRecord).filter(EvaluationRecord.id == evaluation_id).first()


def get_evaluation_history(
    db: Session, skip: int = 0, limit: int = 100
) -> list[EvaluationRecord]:
    """
    Retrieve evaluation history with pagination.

    Args:
        db: Database session.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        List of evaluation records.
    """
    return (
        db.query(EvaluationRecord)
        .order_by(EvaluationRecord.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_evaluation_count(db: Session) -> int:
    """
    Get the total count of evaluations.

    Args:
        db: Database session.

    Returns:
        Total number of evaluations.
    """
    return db.query(EvaluationRecord).count()
