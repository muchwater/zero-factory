"""
AI Model Server - FastAPI Application
Container verification API with YOLO detection, reusable classification, and beverage detection
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

# 환경 변수
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = Path('/app/models/weights')

# 글로벌 모델 변수
container_detector = None
reusable_classifier = None
beverage_detector = None

# FastAPI 앱 생성
app = FastAPI(
    title="Container Verification API",
    description="AI-powered container verification: detection → reusable classification → beverage detection",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_models():
    """서버 시작 시 모델 로드"""
    global container_detector, reusable_classifier, beverage_detector

    print("="*60)
    print("Loading AI Models...")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model directory: {MODEL_DIR}")

    try:
        # 1. Container Detector (YOLO)
        from models.container_detector import ContainerDetector
        container_detector = ContainerDetector(
            model_path='yolov8m.pt',  # medium 모델로 변경 (더 높은 정확도)
            confidence_threshold=0.10,
            device=DEVICE
        )

        # 2. Reusable Classifier
        reusable_model_path = MODEL_DIR / 'reusable_classifier_best.pth'
        if reusable_model_path.exists():
            from models.reusable_classifier_model import ReusableClassifierPredictor
            reusable_classifier = ReusableClassifierPredictor(
                model_path=str(reusable_model_path),
                device=DEVICE
            )
        else:
            print(f"⚠️  Reusable classifier not found: {reusable_model_path}")
            print("   Please train the model first using notebook 01")

        # 3. Beverage Detector
        beverage_model_path = MODEL_DIR / 'beverage_detector_best.pth'
        if beverage_model_path.exists():
            from models.beverage_detector_model import BeverageDetectorPredictor
            beverage_detector = BeverageDetectorPredictor(
                model_path=str(beverage_model_path),
                device=DEVICE
            )
        else:
            print(f"⚠️  Beverage detector not found: {beverage_model_path}")
            print("   Please train the model first using notebook 02")

        print("="*60)
        print("✓ Models loaded successfully!")
        print("="*60)

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()


# 라우터 등록
from routes.container import router as container_router
from routes.health import router as health_router

app.include_router(container_router)
app.include_router(health_router)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Container Verification API",
        "version": "1.0.0",
        "endpoints": {
            "verify": "/container/verify",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
