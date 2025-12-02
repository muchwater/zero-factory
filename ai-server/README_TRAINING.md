# 모델 학습 가이드

두 가지 분류 모델을 학습하는 가이드입니다.

## 모델 개요

### 1. 다회용기 검증 모델 (Reusable Container Classifier)
- **목적**: 용기가 일회용인지 다회용인지 분류
- **클래스**: disposable (일회용) vs reusable (다회용)
- **데이터**: 232개 이미지 (disposable: 58, reusable: 174)

### 2. 음료 여부 검증 모델 (Beverage Detector)
- **목적**: 용기에 음료가 있는지 여부 판단
- **클래스**: empty (비어있음) vs has_beverage (음료 있음)
- **데이터**: 170개 이미지 (empty: 43, has_beverage: 127)
  - **참고**: unclear 클래스는 학습에서 제외됨

## 모델 아키텍처

두 모델 모두 동일한 아키텍처를 사용합니다:
- **백본**: EfficientNet-B0 (ImageNet pretrained)
- **분류 헤드**: Dropout(0.3) + Linear(num_classes)
- **입력 크기**: 224x224 RGB 이미지

## 데이터 증강

### Training Transform
- Resize to 224x224
- Random Horizontal Flip (50%)
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation)
- Random Affine (translate)
- Normalization (ImageNet mean/std)

### Validation Transform
- Resize to 224x224
- Normalization (ImageNet mean/std)

## 학습 설정

| 하이퍼파라미터 | 값 |
|---------------|-----|
| Batch Size | 32 |
| Image Size | 224x224 |
| Epochs | 50 (early stopping) |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| Train/Val Split | 80/20 |
| Patience | 10 epochs |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |
| Loss | CrossEntropyLoss (weighted) |

## 사용 방법

### 1. Jupyter Lab 실행

```bash
# GPU 서버에서
docker compose up jupyter

# 브라우저에서 접속
http://localhost:8888
```

### 2. 노트북 실행

#### 다회용기 검증 모델
1. [notebooks/01_reusable_classifier_training.ipynb](notebooks/01_reusable_classifier_training.ipynb) 열기
2. 모든 셀 실행 (Runtime > Run All)
3. 학습 완료 대기 (~30-50 epochs, GPU 기준 약 10-15분)

#### 음료 여부 검증 모델
1. [notebooks/02_beverage_detector_training.ipynb](notebooks/02_beverage_detector_training.ipynb) 열기
2. 모든 셀 실행 (Runtime > Run All)
3. 학습 완료 대기 (~30-50 epochs, GPU 기준 약 10-15분)

## 학습 결과

### 저장되는 파일

각 모델별로 다음 파일들이 [models/weights/](models/weights/) 에 저장됩니다:

#### 다회용기 검증 모델
- `reusable_classifier_best.pth` - 최고 성능 모델 체크포인트
- `reusable_classifier_metadata.json` - 모델 메타데이터
- `reusable_classifier_training_curves.png` - Loss/Accuracy 곡선
- `reusable_classifier_confusion_matrix.png` - Confusion matrix
- `reusable_classifier_confusion_matrix_normalized.png` - Normalized confusion matrix
- `reusable_classifier_sample_predictions.png` - 샘플 예측 결과

#### 음료 여부 검증 모델
- `beverage_detector_best.pth` - 최고 성능 모델 체크포인트
- `beverage_detector_metadata.json` - 모델 메타데이터
- `beverage_detector_training_curves.png` - Loss/Accuracy 곡선
- `beverage_detector_confusion_matrix.png` - Confusion matrix
- `beverage_detector_confusion_matrix_normalized.png` - Normalized confusion matrix
- `beverage_detector_sample_predictions.png` - 샘플 예측 결과

### 모델 체크포인트 구조

```python
checkpoint = {
    'epoch': int,                    # 학습된 epoch 수
    'model_state_dict': OrderedDict, # 모델 가중치
    'optimizer_state_dict': dict,    # Optimizer 상태
    'train_loss': float,            # 학습 loss
    'val_loss': float,              # 검증 loss
    'val_acc': float,               # 검증 accuracy
}
```

### 메타데이터 예시

```json
{
  "model_name": "reusable_classifier",
  "architecture": "EfficientNet-B0",
  "num_classes": 2,
  "classes": ["disposable", "reusable"],
  "img_size": 224,
  "batch_size": 32,
  "epochs_trained": 35,
  "learning_rate": 0.0001,
  "best_val_loss": 0.3245,
  "best_val_acc": 89.5,
  "train_samples": 185,
  "val_samples": 47,
  "class_counts": [46, 139]
}
```

## 학습 모니터링

### Jupyter Notebook에서
- 실시간 progress bar로 학습 진행 상황 확인
- Epoch마다 train/val loss, accuracy 출력
- Early stopping 상태 모니터링

### TensorBoard (선택사항)
추후 TensorBoard 통합 가능:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_name')
```

## 클래스 불균형 처리

두 모델 모두 클래스 불균형 문제를 해결하기 위해:
1. **Weighted Loss**: 클래스 개수에 반비례하는 가중치 적용
2. **데이터 증강**: Minority class에 더 많은 variation 생성

```python
# 클래스 가중치 계산
class_weights = [len(train_dataset) / count for count in class_counts]
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
```

## 학습 기법

### Early Stopping
- Validation loss가 10 epoch 동안 개선되지 않으면 학습 중단
- 최고 성능 모델만 저장

### Learning Rate Scheduling
- ReduceLROnPlateau: val loss가 5 epoch 개선 안되면 LR을 절반으로 감소
- Factor: 0.5
- Patience: 5

### Regularization
- Dropout: 0.3 (classifier head)
- Weight Decay: 1e-5
- 데이터 증강

## 모델 로드 및 사용

### 모델 로드
```python
import torch
from torchvision import models
import torch.nn as nn

# 모델 정의
class ReusableClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# 체크포인트 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ReusableClassifier(num_classes=2)
checkpoint = torch.load('models/weights/reusable_classifier_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 추론
```python
from PIL import Image
import torchvision.transforms as transforms

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 로드 및 예측
img = Image.open('test_image.png').convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

print(f"Predicted: {classes[pred_class]} ({confidence*100:.2f}%)")
```

## 트러블슈팅

### GPU 메모리 부족
```python
# Batch size 줄이기
BATCH_SIZE = 16  # 또는 8
```

### 과적합 (Overfitting)
- Dropout 비율 증가: `p=0.5`
- Weight decay 증가: `1e-4`
- 데이터 증강 강화
- Early stopping patience 감소

### 학습이 느린 경우
```python
# DataLoader num_workers 증가
train_loader = DataLoader(..., num_workers=8)
```

### Class imbalance가 심한 경우
```python
# 더 강한 가중치 적용
class_weights = [len(train_dataset) / count ** 0.5 for count in class_counts]
```

## 성능 평가 지표

### Classification Report
- Precision: 예측한 positive 중 실제 positive 비율
- Recall: 실제 positive 중 예측한 positive 비율
- F1-score: Precision과 Recall의 조화평균
- Support: 각 클래스의 샘플 수

### Confusion Matrix
- True Positive (TP), False Positive (FP)
- True Negative (TN), False Negative (FN)
- Normalized version으로 비율 확인

## 다음 단계

1. **모델 통합**: 두 모델을 FastAPI 서버에 통합
2. **앙상블**: 여러 모델의 예측을 결합하여 성능 향상
3. **하이퍼파라미터 튜닝**: Optuna 등을 사용한 자동 튜닝
4. **추가 데이터 수집**: 성능 개선을 위한 데이터 확보
5. **모델 경량화**: 모바일/엣지 디바이스 배포를 위한 최적화
