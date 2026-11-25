"""
NewsSentinel - News Analyzer API Routes
Stage 4: Production ML Prediction Endpoints
===========================================
Provides RESTful endpoints for headline and article text analysis,
retrieving platform statistics, model metadata, and health telemetry.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from app.services.analyzer_service import analyzer_service
from app.config import settings

router = APIRouter(prefix="/api", tags=["analyzer"])


class AnalyzeRequest(BaseModel):
    headline: Optional[str] = Field(
        None,
        max_length=1000,
        description="The headline or title of the news article (optional)."
    )
    text: Optional[str] = Field(
        None,
        max_length=50000,
        description="The article body, claim, or excerpt to evaluate."
    )


class APIResponse(BaseModel):
    success: bool
    timestamp: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/analyze", response_model=APIResponse)
def analyze_text(payload: AnalyzeRequest):
    """
    Process submitted headline and article text.
    Executes real-time inference using the Phase 3 trained Linear SVM champion model,
    computes confidence and class probability distribution, and extracts linguistic metrics.
    """
    headline = payload.headline or ""
    text = payload.text or ""

    result = analyzer_service.process_analysis_request(text=text, headline=headline)

    if not result["success"]:
        return APIResponse(
            success=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Validation failed",
            error=result["error"],
            data=None
        )

    pred_class = result["data"]["prediction"]
    conf = result["data"]["confidence_display"]
    if result["data"].get("is_uncertain"):
        msg = f"Assessment complete: UNCERTAIN — NEEDS VERIFICATION ({conf} confidence, below 65% threshold)."
    else:
        msg = f"Assessment complete: MODEL PREDICTION: {pred_class.upper()} with {conf} confidence."

    return APIResponse(
        success=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
        message=msg,
        data=result["data"],
        error=None
    )


@router.get("/history")
def get_analysis_history(limit: int = 50, offset: int = 0):
    """Retrieve persisted analysis history records from SQLite with pagination support."""
    records = analyzer_service.get_history(limit=limit, offset=offset)
    total = analyzer_service.get_history_count()
    return {
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": records
    }


@router.get("/stats")
def get_stats():
    """Retrieve platform analytics and Phase 3 model performance metrics."""
    return {
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": analyzer_service.get_system_stats()
    }


@router.get("/model/metrics")
def get_model_metrics():
    """Retrieve Phase 3 model benchmark comparisons and evaluation matrix."""
    return {
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": analyzer_service.get_model_metadata(),
        "comparison": analyzer_service.get_model_comparison()
    }


@router.get("/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "stage": settings.STAGE,
        "environment": settings.APP_ENV,
        "model_loaded": True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
