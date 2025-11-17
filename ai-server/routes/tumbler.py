"""Tumbler registration and verification endpoints"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from io import BytesIO

from schemas import TumblerRegistrationResponse, UsageVerificationResponse

router = APIRouter(prefix="/tumbler", tags=["Tumbler"])


@router.post("/register", response_model=TumblerRegistrationResponse)
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
    from main import cup_detector, classifier, embedding_generator

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


@router.post("/verify", response_model=UsageVerificationResponse)
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
    from main import cup_detector, beverage_detector, embedding_generator

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
