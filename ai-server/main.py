"""
AI Model Server for Reusable Container Verification
FastAPI 서버 - 다회용기 검증 AI 서비스
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import os
from dotenv import load_dotenv
from pathlib import Path

# 모델 import
from models.reusable_classifier import ReusableClassifierInference
from models.beverage_detector import BeverageDetectorInference
from models.embedding_generator import EmbeddingGenerator
from models.cup_detector import CupDetector

# 라우터 import
from routes import health_router, tumbler_router

# 환경 변수 로드
load_dotenv()

app = FastAPI(
    title="Reusable Container AI Service",
    description="AI 기반 다회용기 검증 서비스",
    version="0.2.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health_router)
app.include_router(tumbler_router)

# 전역 모델 인스턴스
classifier: Optional[ReusableClassifierInference] = None
beverage_detector: Optional[BeverageDetectorInference] = None
embedding_generator: Optional[EmbeddingGenerator] = None
cup_detector: Optional[CupDetector] = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로딩"""
    global classifier, embedding_generator, beverage_detector, cup_detector

    print("🚀 AI Model Server Starting...")

    # 디바이스 설정
    device = os.getenv('DEVICE', 'cpu')
    print(f"Device: {device}")

    # YOLO Cup Detector 로드
    try:
        cup_detector = CupDetector(model_name='yolov8n.pt', device=device)
        print("✅ YOLO cup detector loaded")
    except Exception as e:
        print(f"❌ Failed to load cup detector: {e}")

    # 모델 파일 경로
    models_dir = Path("models/weights")
    classifier_path = models_dir / "reusable_classifier.pth"
    beverage_path = models_dir / "beverage_detector.pth"
    siamese_path = models_dir / "siamese_network.pth"
    embeddings_db_path = models_dir / "cup_code_embeddings_siamese.json"

    # Reusable Classifier 로드
    try:
        if classifier_path.exists():
            classifier = ReusableClassifierInference(
                model_path=str(classifier_path),
                device=device
            )
            print("✅ Reusable classifier loaded")
        else:
            print(f"⚠️  Reusable classifier not found at {classifier_path}")
            print("   → Train model using notebooks/01_reusable_classifier.ipynb")
    except Exception as e:
        print(f"❌ Failed to load reusable classifier: {e}")

    # Beverage Detector 로드
    try:
        if beverage_path.exists():
            beverage_detector = BeverageDetectorInference(
                model_path=str(beverage_path),
                device=device,
                num_classes=3  # with_beverage, empty, unclear
            )
            print("✅ Beverage detector loaded")
        else:
            print(f"⚠️  Beverage detector not found at {beverage_path}")
            print("   → Train model using notebooks/03_beverage_detector.ipynb")
    except Exception as e:
        print(f"❌ Failed to load beverage detector: {e}")

    # Siamese Network Embedding Generator 로드
    try:
        if siamese_path.exists():
            embedding_generator = EmbeddingGenerator(
                model_path=str(siamese_path),
                embeddings_db_path=str(embeddings_db_path) if embeddings_db_path.exists() else None,
                device=device,
                embedding_dim=256
            )
            print("✅ Siamese Network embedding generator loaded")
        else:
            print(f"⚠️  Siamese Network not found at {siamese_path}")
            print("   → Train model using notebooks/04_siamese_network_training.ipynb")
    except Exception as e:
        print(f"❌ Failed to load embedding generator: {e}")

    print("\n" + "="*60)
    print("✅ Server ready!")
    print("="*60)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
