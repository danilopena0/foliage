# LLM Code Evaluator

A Python code quality evaluation system powered by LLMs. Supports **HuggingFace** (free tier) and **Perplexity** APIs. Submit Python code and receive detailed analysis including quality scores, issue detection, and improvement suggestions.

## Features

- **Code Quality Scoring**: Get an overall score (0-100) for your Python code
- **Issue Detection**: Identifies bugs, security vulnerabilities, performance issues, and more
- **Actionable Suggestions**: Specific recommendations with example fixes
- **Complexity Analysis**: Time and memory complexity estimates
- **Evaluation History**: Store and retrieve past evaluations
- **Web Interface**: Simple HTML form for code submission

## Tech Stack

- **Backend**: FastAPI, Python 3.11+
- **AI**: HuggingFace Inference API or Perplexity API
- **Database**: SQLAlchemy + SQLite (auto-created, no setup needed)
- **Validation**: Pydantic v2
- **Testing**: Pytest

## Project Structure

```
foliage/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── models.py        # Pydantic models
│   ├── evaluator.py     # LLM API integration (HuggingFace/Perplexity)
│   ├── database.py      # SQLAlchemy setup
│   └── routes.py        # API endpoints
├── templates/
│   └── index.html       # Web form
├── tests/
│   ├── conftest.py      # Test fixtures
│   ├── test_models.py   # Model tests
│   ├── test_routes.py   # API tests
│   └── test_database.py # Database tests
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

### Prerequisites

- Python 3.11 or higher
- One of the following:
  - HuggingFace account (free tier available)
  - Perplexity API key

### Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd foliage
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Copy the example env file
   cp .env.example.example .env.example

   # Edit .env.example and fill in your credentials
   ```

   Or set them directly (choose one provider):

   **Option A: HuggingFace (free)**
   ```bash
   export LLM_PROVIDER=huggingface
   export HF_API_TOKEN=your_token  # Get from https://huggingface.co/settings/tokens
   ```

   **Option B: Perplexity**
   ```bash
   export LLM_PROVIDER=perplexity
   export PERPLEXITY_API_KEY=your_api_key  # Get from https://perplexity.ai/settings/api
   ```

5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

6. Open http://localhost:8000 in your browser

### Docker (Optional)

> **Note:** Docker is **not required** for local development. It's only useful if you want to:
> - Deploy to a server without installing Python/dependencies
> - Share the app with others as a single runnable container
> - Ensure consistent environment across machines
>
> For local use, just run `uvicorn app.main:app --reload` directly.

1. Build the image:
   ```bash
   docker build -t llm-code-evaluator .
   ```

2. Run the container:
   ```bash
   # With HuggingFace
   docker run -p 8000:8000 -e LLM_PROVIDER=huggingface -e HF_API_TOKEN=your_token llm-code-evaluator

   # With Perplexity
   docker run -p 8000:8000 -e LLM_PROVIDER=perplexity -e PERPLEXITY_API_KEY=your_key llm-code-evaluator
   ```

## API Endpoints

### POST /api/evaluate

Submit Python code for evaluation.

**Request:**
```json
{
  "code": "def hello():\n    print('Hello')",
  "filename": "hello.py"
}
```

**Response:**
```json
{
  "evaluation_id": 1,
  "message": "Code evaluation completed successfully"
}
```

### GET /api/evaluation/{id}

Retrieve evaluation results by ID.

**Response:**
```json
{
  "id": 1,
  "code": "def hello():\n    print('Hello')",
  "filename": "hello.py",
  "evaluation": {
    "overall_score": 85.0,
    "summary": "Clean, simple function with good practices.",
    "issues": [],
    "strengths": ["Clear function name", "Simple implementation"],
    "improvement_areas": ["Add type hints", "Add docstring"],
    "time_complexity": "O(1)",
    "memory_complexity": "O(1)"
  },
  "created_at": "2025-01-07T12:00:00"
}
```

### GET /api/history

List past evaluations with pagination.

**Query Parameters:**
- `skip` (default: 0): Number of records to skip
- `limit` (default: 100): Maximum records to return

**Response:**
```json
{
  "evaluations": [
    {
      "id": 1,
      "filename": "hello.py",
      "overall_score": 85.0,
      "summary": "Clean, simple function.",
      "created_at": "2025-01-07T12:00:00"
    }
  ],
  "total": 1
}
```

### GET /health

Health check endpoint.

## Running Tests

```bash
pytest
```

With coverage:
```bash
pytest --cov=app --cov-report=html
```

## Issue Categories

The evaluator detects issues in the following categories:

- **performance**: Inefficient algorithms, unnecessary operations
- **readability**: Poor naming, complex logic, missing documentation
- **security**: Injection vulnerabilities, unsafe operations
- **maintainability**: Code organization, SOLID violations
- **bug**: Logic errors, edge cases, potential runtime errors

## Severity Levels

- **critical**: Must fix immediately, potential security risk or crash
- **high**: Important issue that should be addressed soon
- **medium**: Should be fixed when time permits
- **low**: Minor improvement suggestion

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider to use (`huggingface` or `perplexity`) | `huggingface` |
| `HF_API_TOKEN` | HuggingFace API token | - |
| `HF_MODEL_ID` | HuggingFace model to use | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| `PERPLEXITY_API_KEY` | Perplexity API key | - |
| `PERPLEXITY_MODEL` | Perplexity model to use | `llama-3.1-sonar-small-128k-online` |
| `DATABASE_URL` | Database connection URL | `sqlite:///./evaluations.db` |

## Supported Models

### HuggingFace (free tier)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (default)
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-2-70b-chat-hf`
- `HuggingFaceH4/zephyr-7b-beta`

### Perplexity
- `llama-3.1-sonar-small-128k-online` (default, fast)
- `llama-3.1-sonar-large-128k-online` (better quality)
- `llama-3.1-sonar-huge-128k-online` (best quality)

## About SQLite

This app uses **SQLite** to store evaluation history. Key points:

- **No setup required** - SQLite creates a file (`evaluations.db`) automatically on first run
- **Purpose** - Saves evaluations so you can retrieve them later via `/api/evaluation/{id}` or `/api/history`
- **Local only** - The database file is stored in your project directory
- **Optional** - If you don't need history, the app still works (evaluations just won't persist)

For production with multiple users, you could swap SQLite for PostgreSQL by changing the `DATABASE_URL` environment variable.

## License

MIT
