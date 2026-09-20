"""FastAPI Application Factory, Lifespan, Middleware, and Universal Exception Handlers.

Root entrypoint for the AI Incident Judgment & Scoring Service.
Coordinates application lifespan, CORS configuration, centralized error handling,
and REST routing under the /api namespace.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.api.v1.router import api_v1_router
from app.config import settings
from app.judgment.rubric import RUBRIC_VERSION


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan: log initialization and teardown."""
    # Log/verify initialization (provider, rubric version)
    print(
        f"Starting {settings.app_name} | Provider: {settings.llm_provider} | Rubric: {RUBRIC_VERSION}"
    )
    yield
    print("Shutting down application...")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Scores technical incident report quality and assesses severity using structured LLM perception.",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Universal Exception Handler
@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions with a structured JSON 500 response."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "INTERNAL_SERVER_ERROR",
                "message": str(exc),
            }
        },
    )


# Root redirect endpoint
@app.get("/", include_in_schema=False, summary="Redirect to interactive API documentation")
async def root_redirect() -> RedirectResponse:
    """Redirect GET / to /docs (307 Temporary Redirect)."""
    return RedirectResponse(url="/docs")


# Include API routers under /api
app.include_router(api_v1_router, prefix="/api")


def create_app() -> FastAPI:
    """Application factory returning the configured FastAPI application."""
    return app


__all__ = ["app", "create_app", "lifespan", "universal_exception_handler"]
