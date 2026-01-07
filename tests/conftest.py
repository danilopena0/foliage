"""Pytest configuration and fixtures."""

import os
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import CodeEvaluation, CodeIssue, IssueCategory, IssueSeverity


TEST_DATABASE_URL = "sqlite:///./test_evaluations.db"


@pytest.fixture(scope="function")
def test_db() -> Generator[Session, None, None]:
    """Create a fresh test database for each test."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db: Session) -> Generator[TestClient, None, None]:
    """Create a test client with database override."""

    def override_get_db() -> Generator[Session, None, None]:
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_code() -> str:
    """Sample Python code for testing."""
    return '''
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
'''


@pytest.fixture
def sample_evaluation() -> CodeEvaluation:
    """Sample evaluation result for testing."""
    return CodeEvaluation(
        overall_score=75.0,
        summary="The code is functional but could be improved.",
        issues=[
            CodeIssue(
                category=IssueCategory.PERFORMANCE,
                severity=IssueSeverity.LOW,
                line_number=3,
                description="Loop could be replaced with built-in sum()",
                suggestion="Use sum(numbers) instead of manual loop",
                example_fix="return sum(numbers)",
            )
        ],
        strengths=["Clear function name", "Simple logic"],
        improvement_areas=["Use built-in functions", "Add type hints"],
        time_complexity="O(n)",
        memory_complexity="O(1)",
    )


@pytest.fixture
def mock_evaluator(sample_evaluation: CodeEvaluation):
    """Mock the code evaluator to avoid API calls during tests."""
    with patch("app.routes.evaluate_code_sync") as mock:
        mock.return_value = sample_evaluation
        yield mock
