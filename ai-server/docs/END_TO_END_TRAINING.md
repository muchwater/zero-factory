# End-to-End 모델 학습 가이드

## 🎯 개요

Zero Factory 다회용기 검증 시스템의 모든 AI 모델을 한번에 학습하는 완전 자동화 가이드입니다.

## 📦 학습할 모델 (4개)

| # | 모델 | 용도 | 학습 시간 | 모델 크기 |
|---|------|------|----------|----------|
| 1 | **YOLO v8n** | 컵/뚜껑 위치 검출 | ~30분 | ~6MB |
| 2 | **Siamese Network** | 임베딩 (256dim) | ~20분 | ~4MB |
| 3 | **ResNet18** | 등록 API 분류기 (고정확도) | ~15분 | ~45MB |
| 4 | **MobileNetV3** | 검증 API 분류기 (고속도) | ~10분 | ~10MB |

**총 학습 시간**: 약 **1.5시간** (GPU 기준)

## 🚀 빠른 시작

### 1. 환경 확인

```bash
cd ai-server

# GPU 확인
nvidia-smi

# Python 패키지 확인
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "from ultralytics import YOLO; print('YOLO: OK')"
```

### 2. 데이터 준비 확인

```bash
# 필수 파일/디렉토리 확인
ls dataset/project-1-at-2025-11-12-05-32-de5d2a99.json
ls data/raw_images/ | head -5

# YOLO 데이터셋이 없으면 자동 생성됨
# (노트북 실행 시 자동으로 처리)
```

### 3. Jupyter Lab 실행

```bash
# Docker 사용 시
docker-compose up -d jupyter
# 브라우저에서 http://localhost:8888 접속

# 로컬 사용 시
source venv/bin/activate
jupyter lab notebooks/
```

### 4. End-to-End 학습 실행

1. Jupyter에서 `00_end_to_end_training.ipynb` 열기
2. **Runtime → Run All Cells** 실행 (또는 Shift+Enter로 순차 실행)
3. ☕ 커피 타임 (~1.5시간)

## 📊 상세 학습 프로세스

### Phase 1: YOLO 학습 (30분)

```
입력: Label Studio bbox 어노테이션
  ↓
데이터셋 변환 (Label Studio → YOLO format)
  ↓
YOLOv8n 학습 (100 epochs)
  ↓
검증 (mAP50, mAP50-95)
  ↓
출력: runs/detect/cup_detection/weights/best.pt
```

**성능 목표**:
- mAP50: > 0.90
- mAP50-95: > 0.70

### Phase 2: 분류 데이터셋 생성 (5분)

```
원본 이미지 + Label Studio JSON
  ↓
Container bbox로 크롭
  ↓
├─ reusable/ (다회용기 분류)
│  ├─ reusable/
│  └─ disposable/
└─ types/ (임베딩용 cup_code별)
   ├─ CUP001/
   └─ CUP002/...
```

### Phase 3: Siamese Network 학습 (20분)

```
입력: types/ (cup_code별 크롭 이미지)
  ↓
MobileNetV3-Small 백본
  ↓
Triplet Loss (margin=0.3)
  ↓
출력: 256차원 L2-normalized 임베딩
```

**성능 목표**:
- Intra-class distance: < 0.5
- Inter-class distance: > 1.0
- Distance gap: > 0.5

### Phase 4: ResNet 분류기 학습 (15분)

```
입력: reusable/ (다회용기/일회용기)
  ↓
ResNet18 (ImageNet pretrained)
  ↓
Fine-tuning (50 epochs)
  ↓
출력: 이진 분류 모델
```

**성능 목표**:
- Accuracy: > 95%
- F1 Score: > 0.95

### Phase 5: MobileNet 분류기 학습 (10분)

```
입력: reusable/ (동일 데이터)
  ↓
MobileNetV3-Small (ImageNet pretrained)
  ↓
Fine-tuning (50 epochs)
  ↓
출력: 경량 이진 분류 모델
```

**성능 목표**:
- Accuracy: > 92%
- Inference time: < 20ms (CPU)

## 🎓 학습 중 모니터링

### TensorBoard (선택사항)

```bash
# 새 터미널에서
tensorboard --logdir runs/detect

# 브라우저에서 http://localhost:6006 접속
```

### 진행 상황 확인

```bash
# YOLO 학습 진행
watch -n 5 "ls -lh runs/detect/cup_detection/weights/"

# 모델 파일 확인
ls -lh models/weights/
```

### GPU 모니터링

```bash
# GPU 사용량 실시간 모니터링
watch -n 1 nvidia-smi

# 특정 프로세스 확인
ps aux | grep python | grep train
```

## 🐛 트러블슈팅

### GPU 메모리 부족

**증상**: `CUDA out of memory`

**해결책**:
```python
# 노트북 내에서 배치 사이즈 조정
YOLO_CONFIG['batch'] = 8  # 기본값: -1 (auto)

# 또는 더 작은 모델 사용
YOLO_CONFIG['model'] = 'yolov8n.pt'  # 이미 nano 사용 중
```

### YOLO 데이터셋 변환 실패

**증상**: `No valid data found`

**해결책**:
```bash
# Label Studio JSON 확인
python3 -c "
import json
with open('dataset/project-1-at-2025-11-12-05-32-de5d2a99.json') as f:
    data = json.load(f)
print(f'Total images: {len(data)}')
print(f'First item keys: {data[0].keys()}')
"

# 이미지 파일 확인
ls data/raw_images/*.png | wc -l
```

### Jupyter Kernel 죽음

**증상**: Kernel dies during training

**해결책**:
```bash
# 메모리 제한 늘리기 (Docker)
# docker-compose.yml 수정:
# services:
#   jupyter:
#     mem_limit: 8g  # 기본: 4g

# 또는 스크립트로 직접 실행
python3 scripts/train_yolo.py --data data/yolo_dataset/data.yaml
```

### 학습 중단 후 재개

**YOLO**:
```bash
python3 scripts/train_yolo.py \
  --resume runs/detect/cup_detection/weights/last.pt
```

**PyTorch 모델** (Siamese, ResNet, MobileNet):
```python
# 체크포인트에서 로드
checkpoint = torch.load('models/weights/siamese_network.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## 📈 성능 검증

### 1. YOLO 성능 확인

```python
from ultralytics import YOLO

model = YOLO('runs/detect/cup_detection/weights/best.pt')
results = model.val()

print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")
```

### 2. Siamese Network 확인

```python
# t-SNE 시각화로 클러스터링 확인
# 노트북 Section 12 참조
```

### 3. 분류기 성능 확인

```python
from sklearn.metrics import classification_report

# 테스트 세트로 평가
predictions = model.predict(test_loader)
print(classification_report(y_true, predictions))
```

## 🎁 학습 완료 후

### 학습된 모델 위치

```
models/weights/
├── siamese_network.pth           # Siamese 임베딩
├── resnet_classifier.pth          # ResNet 분류기
├── mobilenet_classifier.pth       # MobileNet 분류기
└── cup_code_embeddings_siamese.json  # 사전 계산된 임베딩

runs/detect/cup_detection/weights/
├── best.pt                        # YOLO 최고 성능 모델
└── last.pt                        # YOLO 마지막 체크포인트
```

### 모델 크기 확인

```bash
du -sh models/weights/*
du -sh runs/detect/cup_detection/weights/best.pt
```

### 다음 단계: FastAPI 통합

1. **models/cup_detection_pipeline.py** 생성
   ```python
   class CupDetectionPipeline:
       def __init__(self):
           self.yolo = YOLO('runs/detect/cup_detection/weights/best.pt')
           self.siamese = load_siamese_network()
           self.resnet = load_resnet_classifier()
           self.mobilenet = load_mobilenet_classifier()
   ```

2. **API 엔드포인트 구현**
   - `POST /api/register` - 등록 (YOLO + ResNet + Siamese)
   - `POST /api/verify` - 검증 (YOLO + MobileNet + Siamese)

3. **Docker 이미지 빌드**
   ```bash
   docker build -t zero-factory-ai:latest .
   ```

4. **E2E 테스트**
   ```bash
   pytest tests/test_api_integration.py
   ```

## 📝 추가 학습 옵션

### 학습 시간 단축 (테스트용)

```python
# 노트북 내에서 에포크 수 줄이기
YOLO_CONFIG['epochs'] = 10      # 기본: 100
SIAMESE_EPOCHS = 10              # 기본: 100
RESNET_EPOCHS = 10               # 기본: 50
MOBILENET_EPOCHS = 10            # 기본: 50

# 총 학습 시간: ~15분
```

### 더 높은 정확도 원할 때

```python
# 더 큰 YOLO 모델 사용
YOLO_CONFIG['model'] = 'yolov8s.pt'  # Small (더 정확)
YOLO_CONFIG['epochs'] = 200

# ResNet 대신 더 큰 모델
# ResNet50, ResNet101 등
```

### 데이터 증강 조정

```python
# 학습 데이터가 부족할 때
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),  # 20 → 30
    transforms.ColorJitter(brightness=0.4, contrast=0.4),  # 강화
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # 추가
    # ...
])
```

## 🎉 완료!

모든 모델이 성공적으로 학습되었습니다! 이제 FastAPI 서버에 통합하여 실전 배포를 진행하세요.

### 체크리스트

- [x] YOLO 학습 완료
- [x] Siamese Network 학습 완료
- [x] ResNet 분류기 학습 완료
- [x] MobileNet 분류기 학습 완료
- [ ] FastAPI 서버 통합
- [ ] API 테스트
- [ ] Docker 배포
- [ ] 프로덕션 배포

---

**참고 자료**:
- [YOLO 공식 문서](https://docs.ultralytics.com/)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [Siamese Network 논문](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
