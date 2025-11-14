"""
Image utilities for handling various image formats including HEIC
"""

from PIL import Image
import io
from typing import Tuple, Optional


def load_image_from_bytes(image_bytes: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    다양한 포맷의 이미지를 로드합니다 (HEIC, HEIF, PNG, JPEG 등)

    Args:
        image_bytes: 이미지 바이트 데이터

    Returns:
        Tuple[PIL.Image, error]: 성공 시 (Image, None), 실패 시 (None, error_message)
    """
    try:
        # 먼저 PIL로 시도 (JPEG, PNG, BMP, GIF 등 지원)
        image = Image.open(io.BytesIO(image_bytes))

        # RGB로 변환 (RGBA, Grayscale 등 처리)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('RGB')

        return image, None

    except Exception as e:
        # PIL이 실패하면 HEIC/HEIF 시도
        try:
            import pillow_heif

            # HEIC/HEIF 파일 읽기
            heif_file = pillow_heif.read_heif(image_bytes)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw"
            )

            # RGB로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image, None

        except ImportError:
            return None, "HEIC format not supported. Please install pillow-heif: pip install pillow-heif"

        except Exception as heif_error:
            return None, f"Failed to load image: {str(e)}. HEIC error: {str(heif_error)}"


def validate_image_format(image_bytes: bytes) -> Tuple[bool, str, Optional[str]]:
    """
    이미지 포맷 검증

    Returns:
        Tuple[is_valid, format, error]: (True, "JPEG", None) 또는 (False, None, error_message)
    """
    try:
        # 매직 넘버로 포맷 확인
        if image_bytes[:2] == b'\xff\xd8':
            return True, "JPEG", None
        elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return True, "PNG", None
        elif image_bytes[:4] in (b'ftypheic', b'ftypheix', b'ftyphevc', b'ftyphevx'):
            return True, "HEIC", None
        elif image_bytes[:4] == b'ftypmif1':
            return True, "HEIF", None
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return True, "GIF", None
        elif image_bytes[:2] in (b'BM', b'BA'):
            return True, "BMP", None
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return True, "WEBP", None
        else:
            # PIL로 확인 시도
            try:
                img = Image.open(io.BytesIO(image_bytes))
                return True, img.format or "UNKNOWN", None
            except:
                return False, None, "Unknown or unsupported image format"

    except Exception as e:
        return False, None, f"Format validation error: {str(e)}"


def convert_to_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    """
    PIL Image를 JPEG 바이트로 변환

    Args:
        image: PIL Image
        quality: JPEG 품질 (1-100)

    Returns:
        JPEG 이미지 바이트
    """
    buffer = io.BytesIO()

    # RGB 변환 (JPEG는 RGB만 지원)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(buffer, format='JPEG', quality=quality)
    return buffer.getvalue()
