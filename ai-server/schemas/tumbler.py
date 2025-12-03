"""Tumbler API Response Schemas"""

from typing import List, Optional
from pydantic import BaseModel


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
