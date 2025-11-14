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
from models.embedding_generator import EmbeddingGenerator
from models.cup_detector import CupDetector

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
embedding_generator: Optional[EmbeddingGenerator] = None
cup_detector: Optional[CupDetector] = None


# Response Models
class TumblerRegistrationResponse(BaseModel):
    """텀블러 등록 응답"""
    success: bool
    is_reusable: bool
    embedding: List[float]
    message: str
    confidence: Optional[float] = None
    error: Optional[str] = None


class UsageVerificationResponse(BaseModel):
    """사용 검증 응답"""
    success: bool
    has_beverage: bool
    embedding: List[float]
    message: str
    confidence: Optional[float] = None
    error: Optional[str] = None


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
            "cup_detector": cup_detector is not None,
            "classifier": classifier is not None,
            "embedding_generator": embedding_generator is not None,
            "beverage_detector": beverage_detector is not None,
        }
    }


@app.post("/register-tumbler", response_model=TumblerRegistrationResponse)
async def register_tumbler(file: UploadFile = File(...)):
    """
    텀블러 등록 API

    1) YOLO로 텀블러/컵 영역 자르기 (텀블러/컵이 없거나 2개 이상이면 실패)
    2) 고성능 ResNet으로 다회용기 검증
    3) Siamese로 임베딩 추출

    Args:
        file: 이미지 파일

    Returns:
        성공여부, 다회용기여부, 임베딩 벡터
    """
    # 모델 체크
    if cup_detector is None:
        raise HTTPException(status_code=503, detail="Cup detector not loaded")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Reusable classifier not loaded")
    if embedding_generator is None:
        raise HTTPException(status_code=503, detail="Embedding generator not loaded")

    try:
        # 이미지 읽기
        image_bytes = await file.read()

        # Step 1: YOLO로 텀블러/컵 영역 감지 및 자르기
        detection_result = cup_detector.detect(image_bytes)

        if not detection_result['success']:
            return TumblerRegistrationResponse(
                success=False,
                is_reusable=False,
                embedding=[],
                message=f"Detection failed: {detection_result['error']}",
                error=detection_result['error']
            )

        # Cropped 이미지를 bytes로 변환
        from io import BytesIO
        cropped_image = detection_result['cropped_image']
        buffer = BytesIO()
        cropped_image.save(buffer, format='JPEG')
        cropped_bytes = buffer.getvalue()

        # Step 2: ResNet으로 다회용기 검증
        classification_result = classifier.predict(cropped_bytes)

        if not classification_result['is_reusable']:
            return TumblerRegistrationResponse(
                success=True,
                is_reusable=False,
                embedding=[],
                message=f"Not a reusable container (confidence: {classification_result['confidence']:.1%})",
                confidence=classification_result['confidence'],
                error="Disposable container detected"
            )

        # Step 3: Siamese Network로 임베딩 추출
        embedding = embedding_generator.generate_embedding(cropped_bytes)

        return TumblerRegistrationResponse(
            success=True,
            is_reusable=True,
            embedding=embedding.tolist(),
            message=f"Reusable tumbler registered successfully (confidence: {classification_result['confidence']:.1%})",
            confidence=classification_result['confidence']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/verify-usage", response_model=UsageVerificationResponse)
async def verify_usage(file: UploadFile = File(...)):
    """
    사용 검증 API

    1) YOLO로 텀블러/컵 영역 자르기 (텀블러/컵이 없거나 2개 이상이면 실패)
    2) 속도빠른 MobileNet으로 음료 검증
    3) Siamese로 임베딩 추출

    Args:
        file: 이미지 파일

    Returns:
        성공여부, 음료여부, 임베딩 벡터
    """
    # 모델 체크
    if cup_detector is None:
        raise HTTPException(status_code=503, detail="Cup detector not loaded")
    if beverage_detector is None:
        raise HTTPException(status_code=503, detail="Beverage detector not loaded")
    if embedding_generator is None:
        raise HTTPException(status_code=503, detail="Embedding generator not loaded")

    try:
        # 이미지 읽기
        image_bytes = await file.read()

        # Step 1: YOLO로 텀블러/컵 영역 감지 및 자르기
        detection_result = cup_detector.detect(image_bytes)

        if not detection_result['success']:
            return UsageVerificationResponse(
                success=False,
                has_beverage=False,
                embedding=[],
                message=f"Detection failed: {detection_result['error']}",
                error=detection_result['error']
            )

        # Cropped 이미지를 bytes로 변환
        from io import BytesIO
        cropped_image = detection_result['cropped_image']
        buffer = BytesIO()
        cropped_image.save(buffer, format='JPEG')
        cropped_bytes = buffer.getvalue()

        # Step 2: MobileNet으로 음료 검증
        beverage_result = beverage_detector.predict(cropped_bytes, confidence_threshold=0.6)

        if not beverage_result['has_beverage']:
            return UsageVerificationResponse(
                success=True,
                has_beverage=False,
                embedding=[],
                message=f"No beverage detected: {beverage_result['message']}",
                confidence=beverage_result['confidence'],
                error="No beverage in container"
            )

        # Step 3: Siamese Network로 임베딩 추출
        embedding = embedding_generator.generate_embedding(cropped_bytes)

        return UsageVerificationResponse(
            success=True,
            has_beverage=True,
            embedding=embedding.tolist(),
            message=f"Usage verified with beverage (confidence: {beverage_result['confidence']:.1%})",
            confidence=beverage_result['confidence']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
