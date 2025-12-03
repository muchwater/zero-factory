"""API Routes"""

from .health import router as health_router
# from .tumbler import router as tumbler_router  # TODO: Enable when embedding_generator is ready

__all__ = ["health_router"]  # "tumbler_router"
