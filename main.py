"""Root application entrypoint delegating to app.main."""

from app.main import app, create_app

__all__ = ["app", "create_app"]
