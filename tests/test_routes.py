"""Tests for API routes."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client: TestClient):
        """Test health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "llm-code-evaluator"


class TestEvaluateEndpoint:
    """Tests for the /api/evaluate endpoint."""

    def test_evaluate_code_success(
        self, client: TestClient, sample_code: str, mock_evaluator
    ):
        """Test successful code evaluation."""
        response = client.post(
            "/api/evaluate",
            json={"code": sample_code, "filename": "test.py"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "evaluation_id" in data
        assert data["message"] == "Code evaluation completed successfully"

    def test_evaluate_code_without_filename(
        self, client: TestClient, sample_code: str, mock_evaluator
    ):
        """Test evaluation without filename."""
        response = client.post(
            "/api/evaluate",
            json={"code": sample_code},
        )

        assert response.status_code == 201
        data = response.json()
        assert "evaluation_id" in data

    def test_evaluate_empty_code(self, client: TestClient):
        """Test that empty code returns validation error."""
        response = client.post(
            "/api/evaluate",
            json={"code": ""},
        )

        assert response.status_code == 422

    def test_evaluate_missing_code(self, client: TestClient):
        """Test that missing code returns validation error."""
        response = client.post(
            "/api/evaluate",
            json={},
        )

        assert response.status_code == 422


class TestGetEvaluationEndpoint:
    """Tests for the /api/evaluation/{id} endpoint."""

    def test_get_evaluation_success(
        self, client: TestClient, sample_code: str, mock_evaluator
    ):
        """Test retrieving an evaluation by ID."""
        submit_response = client.post(
            "/api/evaluate",
            json={"code": sample_code, "filename": "test.py"},
        )
        evaluation_id = submit_response.json()["evaluation_id"]

        response = client.get(f"/api/evaluation/{evaluation_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == evaluation_id
        assert data["code"] == sample_code
        assert data["filename"] == "test.py"
        assert "evaluation" in data
        assert data["evaluation"]["overall_score"] == 75.0

    def test_get_evaluation_not_found(self, client: TestClient):
        """Test retrieving non-existent evaluation returns 404."""
        response = client.get("/api/evaluation/99999")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestHistoryEndpoint:
    """Tests for the /api/history endpoint."""

    def test_get_history_empty(self, client: TestClient):
        """Test history endpoint with no evaluations."""
        response = client.get("/api/history")

        assert response.status_code == 200
        data = response.json()
        assert data["evaluations"] == []
        assert data["total"] == 0

    def test_get_history_with_evaluations(
        self, client: TestClient, sample_code: str, mock_evaluator
    ):
        """Test history endpoint after submitting evaluations."""
        client.post("/api/evaluate", json={"code": sample_code, "filename": "test1.py"})
        client.post("/api/evaluate", json={"code": sample_code, "filename": "test2.py"})

        response = client.get("/api/history")

        assert response.status_code == 200
        data = response.json()
        assert len(data["evaluations"]) == 2
        assert data["total"] == 2

    def test_get_history_pagination(
        self, client: TestClient, sample_code: str, mock_evaluator
    ):
        """Test history endpoint with pagination."""
        for i in range(5):
            client.post("/api/evaluate", json={"code": sample_code})

        response = client.get("/api/history?skip=2&limit=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data["evaluations"]) == 2
        assert data["total"] == 5


class TestHomeEndpoint:
    """Tests for the home page."""

    def test_home_page_loads(self, client: TestClient):
        """Test that the home page loads successfully."""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "LLM Code Evaluator" in response.text
