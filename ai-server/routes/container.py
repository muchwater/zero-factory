"""
Container Verification API
통합 용기 검증 엔드포인트: 컨테이너 감지 → 다회용기 검증 → 음료 검증
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from io import BytesIO

from schemas.container_verification import ContainerVerificationResponse

router = APIRouter(prefix="/container", tags=["Container"])


@router.post("/verify", response_model=ContainerVerificationResponse)
async def verify_container(file: UploadFile = File(...)):
    """
    통합 용기 검증 API

    **처리 순서:**
    1. **컨테이너 감지**: YOLO로 bottle/cup 감지 및 크롭
       - 1개만 감지되면 통과, 아니면 실패 반환
    2. **다회용기 검증**: EfficientNet-B0로 다회용기 여부 확인
       - 다회용기가 아니면 실패 반환
    3. **음료 검증**: EfficientNet-B0로 음료 유무 확인
       - Yes, No, Unclear 중 하나 반환

    **Args:**
        file: 이미지 파일 (JPEG, PNG 등)

    **Returns:**
        ContainerVerificationResponse:
        - container_detected: 컨테이너 1개 감지 여부
        - is_reusable: 다회용기 여부 (container_detected=True일 때)
        - beverage_status: 음료 상태 (is_reusable=True일 때)

    **Example Success Response:**
    ```json
    {
      "container_detected": true,
      "num_containers": 1,
      "is_reusable": true,
      "reusable_confidence": 0.95,
      "beverage_status": "Yes",
      "has_beverage": true,
      "beverage_confidence": 0.88,
      "message": "Container verified successfully",
      "container_class": "cup",
      "container_confidence": 0.92
    }
    ```

    **Example Failure Responses:**
    ```json
    // 컨테이너 미감지
    {
      "container_detected": false,
      "num_containers": 0,
      "message": "Container detection failed",
      "error": "No container detected"
    }

    // 일회용기 감지
    {
      "container_detected": true,
      "num_containers": 1,
      "is_reusable": false,
      "reusable_confidence": 0.92,
      "message": "Not a reusable container",
      "error": "Disposable container detected"
    }
    ```
    """
    from main import container_detector, reusable_classifier, beverage_detector

    # 모델 체크
    if container_detector is None:
        raise HTTPException(status_code=503, detail="Container detector not loaded")
    if reusable_classifier is None:
        raise HTTPException(status_code=503, detail="Reusable classifier not loaded")
    if beverage_detector is None:
        raise HTTPException(status_code=503, detail="Beverage detector not loaded")

    try:
        # 이미지 읽기
        image_bytes = await file.read()

        # ===== Step 1: 컨테이너 감지 및 크롭 =====
        detection_result = container_detector.detect_and_crop(image_bytes)

        # 검출 실패 시 원본 이미지를 그대로 사용 (false negative 방지)
        if not detection_result['container_detected']:
            print(f"⚠️  Detection failed, using original image as fallback")
            cropped_bytes = image_bytes
            container_class = None
            container_confidence = None
        else:
            # Cropped 이미지를 bytes로 변환
            cropped_image = detection_result['cropped_image']
            buffer = BytesIO()
            cropped_image.save(buffer, format='JPEG', quality=95)
            cropped_bytes = buffer.getvalue()
            container_class = detection_result['class_name']
            container_confidence = detection_result['confidence']

        # ===== Step 2: 다회용기 검증 =====
        reusable_result = reusable_classifier.predict(cropped_bytes)

        if not reusable_result['is_reusable']:
            return ContainerVerificationResponse(
                container_detected=detection_result['container_detected'],
                num_containers=detection_result['num_containers'],
                is_reusable=False,
                reusable_confidence=reusable_result['confidence'],
                message=f"Not a reusable container (confidence: {reusable_result['confidence']:.1%})",
                error="Disposable container detected",
                container_class=container_class,
                container_confidence=container_confidence
            )

        # ===== Step 3: 음료 검증 =====
        beverage_result = beverage_detector.predict(cropped_bytes, unclear_threshold=0.7)

        return ContainerVerificationResponse(
            container_detected=detection_result['container_detected'],
            num_containers=detection_result['num_containers'],
            is_reusable=True,
            reusable_confidence=reusable_result['confidence'],
            beverage_status=beverage_result['beverage_status'],
            has_beverage=beverage_result['has_beverage'],
            beverage_confidence=beverage_result['confidence'],
            message=f"Container verified: {beverage_result['beverage_status']} beverage (confidence: {beverage_result['confidence']:.1%})",
            error=None,
            container_class=container_class,
            container_confidence=container_confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    헬스 체크: 모델 로드 상태 확인
    """
    from main import container_detector, reusable_classifier, beverage_detector

    return {
        "status": "healthy",
        "models": {
            "container_detector": container_detector is not None,
            "reusable_classifier": reusable_classifier is not None,
            "beverage_detector": beverage_detector is not None
        }
    }
