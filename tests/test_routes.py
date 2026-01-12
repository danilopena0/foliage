"""Tests for API routes."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.evaluator import ConfigurationError, EvaluationError


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

    def test_home_page_has_form_elements(self, client: TestClient):
        """Test that the home page contains required form elements."""
        response = client.get("/")
        html = response.text

        # Check for form
        assert '<form id="evaluationForm">' in html

        # Check for code textarea
        assert 'id="code"' in html
        assert 'name="code"' in html

        # Check for filename input
        assert 'id="filename"' in html
        assert 'name="filename"' in html

        # Check for submit buttons
        assert 'id="submitBtn"' in html
        assert 'id="compareBtn"' in html
        assert 'id="compareModelsBtn"' in html

    def test_home_page_has_results_section(self, client: TestClient):
        """Test that the home page contains results display elements."""
        response = client.get("/")
        html = response.text

        # Check for results containers
        assert 'id="results"' in html
        assert 'id="singleResult"' in html
        assert 'id="comparisonResult"' in html
        assert 'id="multiModelResult"' in html

        # Check for loading indicator
        assert 'id="loading"' in html
        assert 'class="spinner"' in html

        # Check for error display
        assert 'id="error"' in html

    def test_home_page_has_model_selector(self, client: TestClient):
        """Test that the home page contains model selection UI."""
        response = client.get("/")
        html = response.text

        assert 'id="modelCheckboxes"' in html
        assert 'id="modelCount"' in html
        assert "Select 2-4 models" in html

    def test_home_page_has_required_scripts(self, client: TestClient):
        """Test that the home page contains necessary JavaScript."""
        response = client.get("/")
        html = response.text

        # Check for key JavaScript functions
        assert "loadModels" in html
        assert "runEvaluation" in html
        assert "renderEvaluation" in html
        assert "escapeHtml" in html

        # Check for API endpoints in JS
        assert "/api/evaluate" in html
        assert "/api/compare" in html
        assert "/api/models" in html
        assert "/api/compare-models" in html

    def test_home_page_has_valid_html_structure(self, client: TestClient):
        """Test that the home page has valid HTML structure."""
        response = client.get("/")
        html = response.text

        # Check for proper HTML5 structure
        assert "<!DOCTYPE html>" in html
        assert '<html lang="en">' in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "</html>" in html

        # Check for required meta tags
        assert 'charset="UTF-8"' in html
        assert 'name="viewport"' in html


class TestModelsEndpoint:
    """Tests for the /api/models endpoint."""

    def test_list_models(self, client: TestClient):
        """Test listing available models."""
        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0


class TestCompareModelsEndpoint:
    """Tests for the /api/compare-models endpoint."""

    def test_compare_models_validation_min_models(self, client: TestClient):
        """Test that at least 2 models are required."""
        response = client.post(
            "/api/compare-models",
            json={
                "code": "print('hello')",
                "models": ["model1"],  # Only 1 model
            },
        )
        assert response.status_code == 422

    def test_compare_models_validation_max_models(self, client: TestClient):
        """Test that at most 4 models are allowed."""
        response = client.post(
            "/api/compare-models",
            json={
                "code": "print('hello')",
                "models": ["m1", "m2", "m3", "m4", "m5"],  # 5 models
            },
        )
        assert response.status_code == 422


class TestGetEvaluationValidation:
    """Tests for input validation on get evaluation endpoint."""

    def test_negative_evaluation_id(self, client: TestClient):
        """Test that negative evaluation ID returns 400."""
        response = client.get("/api/evaluation/-1")
        assert response.status_code == 400
        assert "positive integer" in response.json()["detail"]

    def test_zero_evaluation_id(self, client: TestClient):
        """Test that zero evaluation ID returns 400."""
        response = client.get("/api/evaluation/0")
        assert response.status_code == 400


class TestHistoryValidation:
    """Tests for input validation on history endpoint."""

    def test_negative_skip(self, client: TestClient):
        """Test that negative skip returns validation error."""
        response = client.get("/api/history?skip=-1")
        assert response.status_code == 422

    def test_negative_limit(self, client: TestClient):
        """Test that negative limit returns validation error."""
        response = client.get("/api/history?limit=-1")
        assert response.status_code == 422

    def test_limit_too_large(self, client: TestClient):
        """Test that limit > 1000 returns validation error."""
        response = client.get("/api/history?limit=1001")
        assert response.status_code == 422


class TestErrorHandling:
    """Tests for error handling in routes."""

    def test_configuration_error_returns_503(self, client: TestClient, sample_code: str):
        """Test that ConfigurationError returns 503."""
        with patch("app.routes.evaluate_code_sync") as mock:
            mock.side_effect = ConfigurationError("No API key configured")
            response = client.post(
                "/api/evaluate",
                json={"code": sample_code},
            )
            assert response.status_code == 503
            assert "Configuration error" in response.json()["detail"]

    def test_evaluation_error_returns_502(self, client: TestClient, sample_code: str):
        """Test that EvaluationError returns 502."""
        with patch("app.routes.evaluate_code_sync") as mock:
            mock.side_effect = EvaluationError("API timeout")
            response = client.post(
                "/api/evaluate",
                json={"code": sample_code},
            )
            assert response.status_code == 502
            assert "LLM evaluation failed" in response.json()["detail"]
