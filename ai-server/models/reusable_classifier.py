"""
Reusable Container Classifier
다회용기/일회용기 분류 모델
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from pathlib import Path


class ReusableClassifier(nn.Module):
    """ResNet50 기반 다회용기 분류기"""

    def __init__(self, num_classes=2, pretrained=False):
        super(ReusableClassifier, self).__init__()

        # ResNet50 백본
        self.backbone = models.resnet50(pretrained=pretrained)

        # 분류 헤드 교체
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class ReusableClassifierInference:
    """추론용 Reusable Classifier 래퍼"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: 모델 가중치 파일 경로 (.pth)
            device: 'cpu' 또는 'cuda'
        """
        self.device = torch.device(device)
        self.model = ReusableClassifier(num_classes=2, pretrained=False)

        # 모델 가중치 로드
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Reusable classifier loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # 전처리 Transform (Notebook과 동일)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.class_names = ['disposable', 'reusable']

    def predict(self, image_bytes: bytes) -> dict:
        """
        이미지에서 다회용기 여부 예측

        Args:
            image_bytes: 이미지 바이트 데이터

        Returns:
            dict: {
                'is_reusable': bool,
                'confidence': float,
                'class': str,
                'probabilities': dict
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
        is_reusable = (predicted_class == 1)

        return {
            'is_reusable': is_reusable,
            'confidence': float(confidence.item()),
            'class': self.class_names[predicted_class],
            'probabilities': {
                'disposable': float(probabilities[0][0].item()),
                'reusable': float(probabilities[0][1].item())
            }
        }

    def predict_from_file(self, image_path: str) -> dict:
        """파일 경로에서 예측"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return self.predict(image_bytes)
