"""
AI Model Server for Reusable Container Verification
FastAPI 서버 - 다회용기 검증 AI 서비스
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import os
from dotenv import load_dotenv
from pathlib import Path

# 모델 import
from models.reusable_classifier import ReusableClassifierInference
from models.beverage_detector import BeverageDetectorInference

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

# 전역 모델 인스턴스
classifier: Optional[ReusableClassifierInference] = None
beverage_detector: Optional[BeverageDetectorInference] = None
embedding_generator = None  # TODO: 임베딩 모델은 추후 구현


# Response Models
class ClassificationResponse(BaseModel):
    """일회용/다회용 분류 응답"""
    is_reusable: bool
    confidence: float
    predicted_class: str
    probabilities: dict
    message: str


class EmbeddingResponse(BaseModel):
    """임베딩 벡터 응답"""
    embedding: List[float]
    dimension: int


class BeverageVerificationResponse(BaseModel):
    """음료 검증 응답"""
    has_beverage: bool
    confidence: float
    predicted_class: str
    is_valid: bool
    probabilities: dict
    message: str


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로딩"""
    global classifier, embedding_generator, beverage_detector

    print("🚀 AI Model Server Starting...")

    # 디바이스 설정
    device = os.getenv('DEVICE', 'cpu')
    print(f"Device: {device}")

    # 모델 파일 경로
    models_dir = Path("models/weights")
    classifier_path = models_dir / "reusable_classifier.pth"
    beverage_path = models_dir / "beverage_detector.pth"

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

    # TODO: Embedding Generator 로드
    print("⚠️  Embedding generator not implemented yet")

    print("\n" + "="*60)
    print("✅ Server ready!")
    print("="*60)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "AI Model Server is running",
        "status": "healthy",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """헬스체크"""
    return {
        "status": "healthy",
        "device": os.getenv("DEVICE", "cpu"),
        "models_loaded": {
            "classifier": classifier is not None,
            "embedding_generator": embedding_generator is not None,
            "beverage_detector": beverage_detector is not None,
        }
    }


@app.post("/classify-reusable", response_model=ClassificationResponse)
async def classify_reusable(file: UploadFile = File(...)):
    """
    다회용기 vs 일회용기 분류

    이미지를 업로드하면 다회용기인지 일회용기인지 분류합니다.
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier model not loaded. Please train the model first."
        )

    try:
        # 이미지 읽기
        image_bytes = await file.read()

        # 모델 추론
        result = classifier.predict(image_bytes)

        # 메시지 생성
        if result['is_reusable']:
            message = f"✅ Reusable container detected (confidence: {result['confidence']:.1%})"
        else:
            message = f"❌ Disposable container detected (confidence: {result['confidence']:.1%})"

        return ClassificationResponse(
            is_reusable=result['is_reusable'],
            confidence=result['confidence'],
            predicted_class=result['class'],
            probabilities=result['probabilities'],
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/generate-embedding", response_model=EmbeddingResponse)
async def generate_embedding(file: UploadFile = File(...)):
    """
    이미지 임베딩 벡터 생성 (512차원)
    TODO: 실제 구현
    """
    try:
        # TODO: 실제 모델 추론
        dummy_embedding = [0.0] * 512
        return EmbeddingResponse(
            embedding=dummy_embedding,
            dimension=512
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify-beverage", response_model=BeverageVerificationResponse)
async def verify_beverage(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.7
):
    """
    음료 포함 여부 검증

    다회용기에 음료가 담겨있는지 확인합니다.
    사용 인증 시 활용할 수 있습니다.

    Args:
        file: 이미지 파일
        confidence_threshold: 신뢰도 임계값 (기본 0.7)
    """
    if beverage_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Beverage detector model not loaded. Please train the model first."
        )

    try:
        # 이미지 읽기
        image_bytes = await file.read()

        # 모델 추론
        result = beverage_detector.predict(image_bytes, confidence_threshold)

        return BeverageVerificationResponse(
            has_beverage=result['has_beverage'],
            confidence=result['confidence'],
            predicted_class=result['class'],
            is_valid=result['is_valid'],
            probabilities=result['probabilities'],
            message=result['message']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
