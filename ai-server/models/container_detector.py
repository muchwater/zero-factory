"""
Container Detection using YOLO
Detects and crops bottle/cup containers from images
"""

import torch
from PIL import Image
import io
from ultralytics import YOLO
import numpy as np


class ContainerDetector:
    """YOLO 기반 용기(bottle, cup) 감지 및 크롭"""

    # COCO 데이터셋의 bottle, cup 클래스 ID
    CONTAINER_CLASSES = {
        39: 'bottle',
        41: 'cup',
    }

    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.25, device: str = 'cuda'):
        """
        Args:
            model_path: YOLO 모델 경로
            confidence_threshold: 감지 신뢰도 임계값
            device: 'cuda' 또는 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold

        print(f"Loading YOLO model: {model_path} on {self.device}")
        self.model = YOLO(model_path)
        print("✓ Container detector loaded")

    def detect_and_crop(self, image_bytes: bytes, padding_ratio: float = 0.1) -> dict:
        """
        이미지에서 용기를 감지하고 크롭합니다.

        Args:
            image_bytes: 입력 이미지 바이트
            padding_ratio: bbox 패딩 비율

        Returns:
            dict: {
                'success': bool,
                'container_detected': bool,
                'num_containers': int,
                'cropped_image': PIL.Image or None,
                'bbox': [x1, y1, x2, y2] or None,
                'class_name': str or None,
                'confidence': float or None,
                'error': str or None
            }
        """
        try:
            # 이미지 로드
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_array = np.array(image)

            # YOLO 추론
            results = self.model.predict(
                img_array,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )

            # 용기 클래스만 필터링
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in self.CONTAINER_CLASSES:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])

                        detections.append({
                            'class_id': cls_id,
                            'class_name': self.CONTAINER_CLASSES[cls_id],
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence
                        })

            # 감지된 용기가 없는 경우
            if len(detections) == 0:
                return {
                    'success': False,
                    'container_detected': False,
                    'num_containers': 0,
                    'cropped_image': None,
                    'bbox': None,
                    'class_name': None,
                    'confidence': None,
                    'error': 'No container detected'
                }

            # 2개 이상 감지된 경우
            if len(detections) > 1:
                return {
                    'success': False,
                    'container_detected': False,
                    'num_containers': len(detections),
                    'cropped_image': None,
                    'bbox': None,
                    'class_name': None,
                    'confidence': None,
                    'error': f'Multiple containers detected: {len(detections)}'
                }

            # 정확히 1개 감지된 경우
            detection = detections[0]
            bbox = detection['bbox']

            # Crop with padding
            cropped_image = self._crop_with_padding(image, bbox, padding_ratio)

            return {
                'success': True,
                'container_detected': True,
                'num_containers': 1,
                'cropped_image': cropped_image,
                'bbox': bbox,
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'container_detected': False,
                'num_containers': 0,
                'cropped_image': None,
                'bbox': None,
                'class_name': None,
                'confidence': None,
                'error': f'Detection error: {str(e)}'
            }

    def _crop_with_padding(self, image: Image.Image, bbox: list, padding_ratio: float) -> Image.Image:
        """
        bbox로 이미지를 크롭합니다 (패딩 추가).

        Args:
            image: 원본 이미지
            bbox: [x1, y1, x2, y2]
            padding_ratio: bbox 크기 대비 패딩 비율

        Returns:
            크롭된 이미지
        """
        w, h = image.size
        x1, y1, x2, y2 = bbox

        # bbox 크기 계산
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # 패딩 추가
        pad_x = int(bbox_w * padding_ratio)
        pad_y = int(bbox_h * padding_ratio)

        # 이미지 범위 내로 제한
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # 크롭
        cropped = image.crop((x1, y1, x2, y2))
        return cropped
