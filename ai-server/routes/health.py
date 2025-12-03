"""Health check and status endpoints"""

from fastapi import APIRouter
import os

router = APIRouter(tags=["Health"])


@router.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "AI Model Server is running",
        "status": "healthy",
        "version": "0.2.0"
    }


@router.get("/health")
async def health_check():
    """헬스체크"""
    from main import container_detector, reusable_classifier, beverage_detector

    return {
        "status": "healthy",
        "device": os.getenv("DEVICE", "cpu"),
        "models_loaded": {
            "container_detector": container_detector is not None,
            "reusable_classifier": reusable_classifier is not None,
            "beverage_detector": beverage_detector is not None,
        }
    }
