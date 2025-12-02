"""
용기(bottle, cup) bbox 기반 이미지 전처리 스크립트

YOLO를 사용하여 이미지에서 용기(bottle, cup)를 감지하고,
한 이미지에 하나의 용기만 있는 경우에만 bbox로 크롭하여 저장합니다.
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContainerPreprocessor:
    """용기 이미지 전처리 클래스"""

    # COCO 데이터셋의 bottle, cup 클래스 ID
    CONTAINER_CLASSES = {
        39: 'bottle',  # bottle
        41: 'cup',     # cup
        # 추가로 필요한 클래스가 있으면 여기에 추가
    }

    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 confidence_threshold: float = 0.25,
                 device: str = 'cuda'):
        """
        Args:
            model_name: YOLO 모델 이름 (yolov8n.pt, yolov8s.pt 등)
            confidence_threshold: 감지 신뢰도 임계값
            device: 'cuda' 또는 'cpu'
        """
        logger.info(f"Loading YOLO model: {model_name} on {device}")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.device = device

        # 통계
        self.stats = {
            'processed': 0,
            'success': 0,
            'no_container': 0,
            'multiple_containers': 0,
            'low_confidence': 0,
            'errors': 0
        }

    def detect_containers(self, image_path: str) -> Tuple[List[dict], np.ndarray]:
        """
        이미지에서 용기를 감지합니다.

        Returns:
            Tuple[detections, image]: (감지된 용기 리스트, 원본 이미지)
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # YOLO 추론
        results = self.model.predict(
            image,
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

        return detections, image

    def crop_container(self, image: np.ndarray, bbox: List[int],
                       padding_ratio: float = 0.1) -> np.ndarray:
        """
        bbox로 이미지를 크롭합니다 (패딩 추가).

        Args:
            image: 원본 이미지
            bbox: [x1, y1, x2, y2]
            padding_ratio: bbox 크기 대비 패딩 비율

        Returns:
            크롭된 이미지
        """
        h, w = image.shape[:2]
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
        cropped = image[y1:y2, x1:x2]
        return cropped

    def process_image(self,
                     input_path: str,
                     output_path: str,
                     padding_ratio: float = 0.1) -> Tuple[bool, str]:
        """
        단일 이미지를 처리합니다.

        Returns:
            Tuple[success, message]: (성공 여부, 메시지)
        """
        try:
            # 용기 감지
            detections, image = self.detect_containers(input_path)

            # 감지된 용기가 없는 경우
            if len(detections) == 0:
                self.stats['no_container'] += 1
                return False, "No container detected"

            # 2개 이상 감지된 경우
            if len(detections) > 1:
                self.stats['multiple_containers'] += 1
                return False, f"Multiple containers detected: {len(detections)}"

            # 정확히 1개 감지된 경우
            detection = detections[0]

            # 크롭
            cropped = self.crop_container(image, detection['bbox'], padding_ratio)

            # 저장
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            # BGR -> RGB 변환 후 저장
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            Image.fromarray(cropped_rgb).save(output_path, quality=95)

            self.stats['success'] += 1
            return True, f"Success: {detection['class_name']} (conf: {detection['confidence']:.2f})"

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error processing {input_path}: {str(e)}")
            return False, f"Error: {str(e)}"

    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         recursive: bool = True,
                         padding_ratio: float = 0.1) -> None:
        """
        디렉토리의 모든 이미지를 처리합니다.

        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리
            recursive: 하위 디렉토리 포함 여부
            padding_ratio: bbox 패딩 비율
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # 이미지 파일 찾기
        if recursive:
            image_files = list(input_path.rglob('*.png')) + \
                         list(input_path.rglob('*.jpg')) + \
                         list(input_path.rglob('*.jpeg'))
        else:
            image_files = list(input_path.glob('*.png')) + \
                         list(input_path.glob('*.jpg')) + \
                         list(input_path.glob('*.jpeg'))

        logger.info(f"Found {len(image_files)} images in {input_dir}")

        # 경고 로그 파일
        warning_log_path = output_path / 'preprocessing_warnings.txt'
        warning_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(warning_log_path, 'w', encoding='utf-8') as warning_log:
            # 각 이미지 처리
            for img_path in tqdm(image_files, desc="Processing images"):
                self.stats['processed'] += 1

                # 상대 경로 유지
                rel_path = img_path.relative_to(input_path)
                out_path = output_path / rel_path

                # 처리
                success, message = self.process_image(
                    str(img_path),
                    str(out_path),
                    padding_ratio
                )

                # 실패 시 경고 로그 작성
                if not success:
                    warning_msg = f"⚠️  {img_path.name}: {message}\n"
                    warning_log.write(warning_msg)
                    logger.warning(warning_msg.strip())

        # 통계 출력
        self.print_stats()
        logger.info(f"Warnings saved to: {warning_log_path}")

    def print_stats(self):
        """통계 출력"""
        logger.info("\n" + "="*60)
        logger.info("Preprocessing Statistics")
        logger.info("="*60)
        logger.info(f"Total processed: {self.stats['processed']}")
        logger.info(f"✅ Success: {self.stats['success']} ({self.stats['success']/max(1,self.stats['processed'])*100:.1f}%)")
        logger.info(f"⚠️  No container: {self.stats['no_container']}")
        logger.info(f"⚠️  Multiple containers: {self.stats['multiple_containers']}")
        logger.info(f"❌ Errors: {self.stats['errors']}")
        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="용기(bottle, cup) 이미지 전처리 - YOLO 기반 bbox 크롭"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/app/data',
        help='입력 이미지 디렉토리'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/app/data_preprocessed',
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLO 모델 (yolov8n.pt, yolov8s.pt, yolov8m.pt 등)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='감지 신뢰도 임계값 (0-1)'
    )
    parser.add_argument(
        '--padding',
        type=float,
        default=0.1,
        help='bbox 패딩 비율 (0-1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='연산 장치'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='하위 디렉토리 탐색 안 함'
    )

    args = parser.parse_args()

    # 전처리 실행
    preprocessor = ContainerPreprocessor(
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )

    preprocessor.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
        padding_ratio=args.padding
    )


if __name__ == '__main__':
    main()
