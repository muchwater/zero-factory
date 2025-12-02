"""
Container Verification Response Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ContainerVerificationResponse(BaseModel):
    """통합 용기 검증 응답"""

    # 컨테이너 감지
    container_detected: bool = Field(..., description="컨테이너가 1개 감지되었는지 여부")
    num_containers: int = Field(..., description="감지된 컨테이너 수")

    # 다회용기 검증 (container_detected=True일 때만 유효)
    is_reusable: Optional[bool] = Field(None, description="다회용기 여부 (True: 다회용, False: 일회용)")
    reusable_confidence: Optional[float] = Field(None, description="다회용기 검증 신뢰도", ge=0, le=1)

    # 음료 검증 (is_reusable=True일 때만 유효)
    beverage_status: Optional[str] = Field(None, description="음료 상태 (Yes, No, Unclear)")
    has_beverage: Optional[bool] = Field(None, description="음료 있음 여부")
    beverage_confidence: Optional[float] = Field(None, description="음료 검증 신뢰도", ge=0, le=1)

    # 메타 정보
    message: str = Field(..., description="처리 결과 메시지")
    error: Optional[str] = Field(None, description="오류 메시지")

    # 상세 정보
    container_class: Optional[str] = Field(None, description="감지된 컨테이너 클래스 (bottle, cup)")
    container_confidence: Optional[float] = Field(None, description="컨테이너 감지 신뢰도", ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "container_detected": True,
                "num_containers": 1,
                "is_reusable": True,
                "reusable_confidence": 0.95,
                "beverage_status": "Yes",
                "has_beverage": True,
                "beverage_confidence": 0.88,
                "message": "Container verified successfully",
                "error": None,
                "container_class": "cup",
                "container_confidence": 0.92
            }
        }
