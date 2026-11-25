"""
API and Web Routes Package for NewsSentinel.
"""
from .pages import router as pages_router
from .analyzer import router as analyzer_router

__all__ = ["pages_router", "analyzer_router"]
