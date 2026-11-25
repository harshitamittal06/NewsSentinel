"""
NewsSentinel - Web Pages Route Handler
Stage 4: Connected Pages & Empirical Telemetry
=============================================
Renders modern futuristic dark UI templates with Jinja2,
passing live platform statistics, recent analyses, and model performance metrics.
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from app.config import settings
from app.services.analyzer_service import analyzer_service

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))


def get_base_context(active_page: str, title: str) -> dict:
    """Helper to generate common page context for templates."""
    stats = analyzer_service.get_system_stats()
    return {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "stage": 4,
        "active_page": active_page,
        "page_title": f"{title} | {settings.APP_NAME}",
        "stats": stats,
    }


@router.get("/", name="dashboard")
def dashboard_page(request: Request):
    """Render the main command dashboard."""
    context = get_base_context(active_page="dashboard", title="Command Dashboard")
    context["recent_records"] = analyzer_service.get_history()[:5]
    return templates.TemplateResponse(request=request, name="dashboard.html", context=context)


@router.get("/analyzer", name="analyzer")
def analyzer_page(request: Request):
    """Render the dedicated News Analyzer interface."""
    context = get_base_context(active_page="analyzer", title="News Intelligence Analyzer")
    return templates.TemplateResponse(request=request, name="analyzer.html", context=context)


@router.get("/history", name="history")
def history_page(request: Request):
    """Render the Analysis History log."""
    context = get_base_context(active_page="history", title="Analysis History")
    context["history_records"] = analyzer_service.get_history()
    return templates.TemplateResponse(request=request, name="history.html", context=context)


@router.get("/performance", name="performance")
def performance_page(request: Request):
    """Render the Model Performance & Architecture Blueprint."""
    context = get_base_context(active_page="performance", title="Model Performance & Metrics")
    context["metadata"] = analyzer_service.get_model_metadata()
    context["comparison"] = analyzer_service.get_model_comparison()
    return templates.TemplateResponse(request=request, name="performance.html", context=context)


@router.get("/about", name="about")
def about_page(request: Request):
    """Render the About & Academic Methodology page."""
    context = get_base_context(active_page="about", title="About NewsSentinel")
    return templates.TemplateResponse(request=request, name="about.html", context=context)
