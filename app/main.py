"""
NewsSentinel - Main Application Entrypoint
FastAPI Backend Application Configuration & Initialization
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.routes.pages import router as pages_router
from app.routes.analyzer import router as analyzer_router

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-Powered Fake News and Misinformation Pattern Detection System. "
        "A modular, research-oriented platform for natural language processing and news credibility analysis."
    ),
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Mount static asset directory
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Templates for error pages
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# Include application routers
app.include_router(pages_router)
app.include_router(analyzer_router)


# Custom Error Handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with responsive dark UI page or JSON for API calls."""
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "status_code": exc.status_code
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={
            "app_name": settings.APP_NAME,
            "status_code": exc.status_code,
            "message": exc.detail or "The requested resource could not be found.",
            "page_title": f"Error {exc.status_code} | {settings.APP_NAME}",
            "active_page": ""
        },
        status_code=exc.status_code
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unhandled server exceptions."""
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal Server Error",
                "details": str(exc) if settings.DEBUG else None
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={
            "app_name": settings.APP_NAME,
            "status_code": 500,
            "message": "An unexpected server error occurred. Please try again later.",
            "page_title": f"Server Error | {settings.APP_NAME}",
            "active_page": ""
        },
        status_code=500
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
