"""
Cup Detector using YOLOv8
텀블러/컵 영역 감지 및 cropping
"""

import torch
import numpy as np
from PIL import Image
import io
from pathlib import Path
from typing import List, Optional, Tuple
import sys
sys.path.append('..')
from utils.image_utils import load_image_from_bytes


class CupDetector:
    """YOLOv8을 사용한 컵/텀블러 감지기"""

    def __init__(self, model_name: str = 'yolov8n.pt', device: str = 'cpu'):
        """
        Args:
            model_name: YOLO 모델 이름 (yolov8n.pt, yolov8s.pt 등)
            device: 'cpu' 또는 'cuda'
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.device = device
        self.model = YOLO(model_name)

        # COCO 데이터셋의 cup/bottle 관련 클래스
        # 41: cup, 39: bottle, 42: wine glass, 44: bowl
        self.target_classes = [41, 39, 42, 44]

        print(f"✅ YOLO model loaded: {model_name}")

    def detect(
        self,
        image_bytes: bytes,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        padding: float = 0.1,
        min_size: int = 50
    ) -> dict:
        """
        이미지에서 컵/텀블러 감지 및 cropping

        Args:
            image_bytes: 이미지 바이트 데이터
            conf_thresh: 감지 신뢰도 임계값
            iou_thresh: NMS IoU 임계값
            padding: bbox 주변 padding 비율
            min_size: 최소 crop 크기 (픽셀)

        Returns:
            dict: {
                'success': bool,
                'num_detected': int,
                'cropped_image': PIL.Image (단일 객체인 경우),
                'error': str (실패 시)
            }
        """
        # 이미지 로드 (다양한 포맷 지원: HEIC, PNG, JPEG 등)
        image, error = load_image_from_bytes(image_bytes)
        if error:
            return {
                'success': False,
                'num_detected': 0,
                'cropped_image': None,
                'error': f'Image loading failed: {error}'
            }

        img_width, img_height = image.size

        # YOLO 감지
        results = self.model.predict(
            source=image,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=self.target_classes,
            verbose=False
        )

        # 결과 처리
        num_detected = 0
        if len(results) > 0 and len(results[0].boxes) > 0:
            num_detected = len(results[0].boxes)

        # 검증: 정확히 1개의 컵/텀블러만 허용
        if num_detected == 0:
            return {
                'success': False,
                'num_detected': 0,
                'cropped_image': None,
                'error': 'No cup or tumbler detected in image'
            }
        elif num_detected > 1:
            return {
                'success': False,
                'num_detected': num_detected,
                'cropped_image': None,
                'error': f'Multiple objects detected ({num_detected}). Please ensure only one cup/tumbler is in the image'
            }

        # 정확히 1개 감지된 경우 - cropping
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        # Padding 적용
        width = x2 - x1
        height = y2 - y1

        pad_x = width * padding
        pad_y = height * padding

        x1_padded = max(0, int(x1 - pad_x))
        y1_padded = max(0, int(y1 - pad_y))
        x2_padded = min(img_width, int(x2 + pad_x))
        y2_padded = min(img_height, int(y2 + pad_y))

        # 최소 크기 체크
        crop_width = x2_padded - x1_padded
        crop_height = y2_padded - y1_padded

        if crop_width < min_size or crop_height < min_size:
            return {
                'success': False,
                'num_detected': 1,
                'cropped_image': None,
                'error': f'Detected object too small ({crop_width}x{crop_height})'
            }

        # Crop
        cropped = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))

        return {
            'success': True,
            'num_detected': 1,
            'cropped_image': cropped,
            'confidence': confidence,
            'class_id': class_id,
            'class_name': self.model.names[class_id],
            'bbox': (x1_padded, y1_padded, x2_padded, y2_padded),
            'error': None
        }

    def detect_from_file(self, image_path: str, **kwargs) -> dict:
        """파일 경로에서 감지"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return self.detect(image_bytes, **kwargs)
