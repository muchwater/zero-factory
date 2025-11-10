"""
Beverage Detector
음료 포함 여부 검증 모델
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from pathlib import Path


class BeverageDetector(nn.Module):
    """MobileNetV3-Small 기반 음료 검증 모델"""

    def __init__(self, num_classes=3, pretrained=False):
        super(BeverageDetector, self).__init__()

        # MobileNetV3-Small 백본 (경량화)
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)

        # 분류 헤드 교체
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class BeverageDetectorInference:
    """추론용 Beverage Detector 래퍼"""

    def __init__(self, model_path: str, device: str = 'cpu', num_classes: int = 3):
        """
        Args:
            model_path: 모델 가중치 파일 경로 (.pth)
            device: 'cpu' 또는 'cuda'
            num_classes: 클래스 수 (2 or 3)
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model = BeverageDetector(num_classes=num_classes, pretrained=False)

        # 모델 가중치 로드
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Beverage detector loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 전처리 Transform (Notebook과 동일)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 클래스 이름 (3-class 기준)
        if num_classes == 3:
            self.class_names = ['empty', 'with_beverage', 'unclear']
        else:
            self.class_names = ['empty', 'with_beverage']

    def predict(self, image_bytes: bytes, confidence_threshold: float = 0.7) -> dict:
        """
        이미지에서 음료 포함 여부 예측

        Args:
            image_bytes: 이미지 바이트 데이터
            confidence_threshold: 신뢰도 임계값

        Returns:
            dict: {
                'has_beverage': bool,
                'confidence': float,
                'class': str,
                'is_valid': bool,
                'probabilities': dict,
                'message': str
            }
        """
        # 이미지 로딩
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 전처리
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 추론
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        predicted_class_name = self.class_names[predicted_class]

        # 음료 포함 여부 판단
        has_beverage = (predicted_class_name == 'with_beverage')

        # 검증 로직 (신뢰도 기반)
        is_valid = has_beverage and (confidence.item() >= confidence_threshold)

        # 메시지 생성
        if is_valid:
            message = "Beverage detected - Valid usage"
        elif has_beverage and confidence.item() < confidence_threshold:
            message = f"Beverage detected but low confidence ({confidence.item():.2%})"
        else:
            message = f"No beverage detected - {predicted_class_name}"

        # 확률 딕셔너리 생성
        prob_dict = {name: float(probabilities[0][i].item())
                     for i, name in enumerate(self.class_names)}

        return {
            'has_beverage': has_beverage,
            'confidence': float(confidence.item()),
            'class': predicted_class_name,
            'is_valid': is_valid,
            'probabilities': prob_dict,
            'message': message
        }

    def predict_from_file(self, image_path: str, confidence_threshold: float = 0.7) -> dict:
        """파일 경로에서 예측"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return self.predict(image_bytes, confidence_threshold)
