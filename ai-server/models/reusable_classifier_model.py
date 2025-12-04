"""
Reusable Container Classifier
EfficientNet-B0 based binary classifier
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import timm


class ReusableClassifier(nn.Module):
    """다회용기 분류 모델 (ResNetV2-101 BiT 기반)"""

    def __init__(self, num_classes=2, pretrained=False):
        super(ReusableClassifier, self).__init__()

        # ResNetV2-101 BiT (Big Transfer) 백본
        # ImageNet-21K (14M images, 21K classes) → ImageNet-1K fine-tuned
        self.backbone = timm.create_model(
            'resnetv2_101x1_bit.goog_in21k_ft_in1k',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3
        )

    def forward(self, x):
        return self.backbone(x)


class ReusableClassifierPredictor:
    """다회용기 분류 모델 예측기"""

    CLASSES = ['disposable', 'reusable']

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
        print(f"Loading reusable classifier from: {model_path}")
        self.model = ReusableClassifier(num_classes=2)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✓ Reusable classifier loaded on {self.device}")

    def predict(self, image_bytes: bytes) -> dict:
        """
        이미지가 다회용기인지 예측합니다.

        Args:
            image_bytes: 이미지 바이트

        Returns:
            dict: {
                'is_reusable': bool,
                'class_name': str,
                'confidence': float,
                'probabilities': {'disposable': float, 'reusable': float}
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

            is_reusable = (pred_class == 1)  # 1 = reusable
            class_name = self.CLASSES[pred_class]

            return {
                'is_reusable': is_reusable,
                'class_name': class_name,
                'confidence': confidence_value,
                'probabilities': {
                    'disposable': probs[0][0].item(),
                    'reusable': probs[0][1].item()
                }
            }

        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")
