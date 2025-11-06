"""
AI Model Server for Reusable Container Verification
FastAPI 서버 - 다회용기 검증 AI 서비스
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = FastAPI(
    title="Reusable Container AI Service",
    description="AI 기반 다회용기 검증 서비스",
    version="0.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: 전역 모델 인스턴스
classifier = None
embedding_generator = None
beverage_detector = None


# Response Models
class ClassificationResponse(BaseModel):
    """일회용/다회용 분류 응답"""
    is_reusable: bool
    confidence: float
    message: str


class EmbeddingResponse(BaseModel):
    """임베딩 벡터 응답"""
    embedding: List[float]
    dimension: int


class BeverageVerificationResponse(BaseModel):
    """음료 검증 응답"""
    has_beverage: bool
    confidence: float
    message: str


@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 모델 로딩
    TODO: 실제 모델 로딩 구현
    """
    global classifier, embedding_generator, beverage_detector

    print("🚀 AI Model Server Starting...")
    print(f"Device: {os.getenv('DEVICE', 'cpu')}")

    # TODO: 모델 로딩 구현
    # from models.classifier import ReusableClassifier
    # from models.embedding import EmbeddingGenerator
    # from models.beverage_detector import BeverageDetector

    # classifier = ReusableClassifier(...)
    # embedding_generator = EmbeddingGenerator(...)
    # beverage_detector = BeverageDetector(...)

    print("✅ Server ready (models not loaded yet - TODO)")


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
    TODO: 실제 구현
    """
    try:
        # TODO: 실제 모델 추론
        return ClassificationResponse(
            is_reusable=True,
            confidence=0.85,
            message="TODO: 실제 모델 구현 필요"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
async def verify_beverage(file: UploadFile = File(...)):
    """
    음료 포함 여부 검증
    TODO: 실제 구현
    """
    try:
        # TODO: 실제 모델 추론
        return BeverageVerificationResponse(
            has_beverage=True,
            confidence=0.90,
            message="TODO: 실제 모델 구현 필요"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
