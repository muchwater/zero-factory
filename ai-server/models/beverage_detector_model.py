"""
Beverage Detector
EfficientNet-B0 based beverage detection classifier
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io


class BeverageDetector(nn.Module):
    """음료 여부 검증 모델 (EfficientNet-B0 기반)"""

    def __init__(self, num_classes=2):
        super(BeverageDetector, self).__init__()

        # EfficientNet-B0 백본
        self.backbone = models.efficientnet_b0(pretrained=False)

        # 마지막 FC layer 교체
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class BeverageDetectorPredictor:
    """음료 검증 모델 예측기"""

    CLASSES = ['empty', 'has_beverage']

    def __init__(self, model_path: str, device: str = 'cuda', img_size: int = 224):
        """
        Args:
            model_path: 모델 체크포인트 경로
            device: 'cuda' 또는 'cpu'
            img_size: 입력 이미지 크기
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.img_size = img_size

        # Transform 정의
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 모델 로드
        print(f"Loading beverage detector from: {model_path}")
        self.model = BeverageDetector(num_classes=2)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✓ Beverage detector loaded on {self.device}")

    def predict(self, image_bytes: bytes, unclear_threshold: float = 0.7) -> dict:
        """
        이미지에 음료가 있는지 예측합니다.

        Args:
            image_bytes: 이미지 바이트
            unclear_threshold: Unclear 판정을 위한 신뢰도 임계값

        Returns:
            dict: {
                'beverage_status': str,  # 'Yes', 'No', 'Unclear'
                'has_beverage': bool,
                'confidence': float,
                'probabilities': {'empty': float, 'has_beverage': float}
            }
        """
        try:
            # 이미지 로드
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # Transform 적용
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 예측
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

                pred_class = predicted.item()
                confidence_value = confidence.item()

            has_beverage = (pred_class == 1)  # 1 = has_beverage

            # Unclear 판정: 신뢰도가 낮으면 Unclear
            if confidence_value < unclear_threshold:
                beverage_status = 'Unclear'
            else:
                beverage_status = 'Yes' if has_beverage else 'No'

            return {
                'beverage_status': beverage_status,
                'has_beverage': has_beverage,
                'confidence': confidence_value,
                'probabilities': {
                    'empty': probs[0][0].item(),
                    'has_beverage': probs[0][1].item()
                }
            }

        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")
