"""API routes for the code evaluation system."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import (
    create_evaluation,
    get_db,
    get_evaluation_by_id,
    get_evaluation_count,
    get_evaluation_history,
)
from app.evaluator import evaluate_code_sync
from app.models import (
    CodeEvaluation,
    CodeSubmission,
    EvaluationResponse,
    EvaluationResult,
    EvaluationSummary,
    HistoryResponse,
)


router = APIRouter()


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit code for evaluation",
    description="Submit Python code for quality evaluation using Claude AI.",
)
async def submit_evaluation(
    submission: CodeSubmission,
    db: Session = Depends(get_db),
) -> EvaluationResponse:
    """
    Submit Python code for evaluation.

    Args:
        submission: The code submission containing the code and optional filename.
        db: Database session.

    Returns:
        EvaluationResponse with the evaluation ID.

    Raises:
        HTTPException: If evaluation fails.
    """
    try:
        evaluation = evaluate_code_sync(submission.code, submission.filename)
        evaluation_dict = evaluation.model_dump()

        record = create_evaluation(
            db=db,
            code=submission.code,
            filename=submission.filename,
            evaluation_dict=evaluation_dict,
        )

        return EvaluationResponse(
            evaluation_id=record.id,
            message="Code evaluation completed successfully",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.get(
    "/evaluation/{evaluation_id}",
    response_model=EvaluationResult,
    summary="Get evaluation results",
    description="Retrieve the structured evaluation results for a specific evaluation ID.",
)
async def get_evaluation(
    evaluation_id: int,
    db: Session = Depends(get_db),
) -> EvaluationResult:
    """
    Retrieve evaluation results by ID.

    Args:
        evaluation_id: The ID of the evaluation to retrieve.
        db: Database session.

    Returns:
        EvaluationResult with complete evaluation data.

    Raises:
        HTTPException: If evaluation not found.
    """
    record = get_evaluation_by_id(db, evaluation_id)

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation with ID {evaluation_id} not found",
        )

    evaluation_data = record.get_evaluation()
    evaluation = CodeEvaluation(**evaluation_data)

    return EvaluationResult(
        id=record.id,
        code=record.code,
        filename=record.filename,
        evaluation=evaluation,
        created_at=record.created_at,
    )


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Get evaluation history",
    description="Retrieve a list of past evaluations with pagination support.",
)
async def get_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """
    Retrieve evaluation history with pagination.

    Args:
        skip: Number of records to skip (for pagination).
        limit: Maximum number of records to return.
        db: Database session.

    Returns:
        HistoryResponse with list of evaluation summaries and total count.
    """
    records = get_evaluation_history(db, skip=skip, limit=limit)
    total = get_evaluation_count(db)

    summaries = [
        EvaluationSummary(
            id=record.id,
            filename=record.filename,
            overall_score=record.overall_score,
            summary=record.summary,
            created_at=record.created_at,
        )
        for record in records
    ]

    return HistoryResponse(evaluations=summaries, total=total)
