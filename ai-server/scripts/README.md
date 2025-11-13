# AI Server Scripts

데이터셋 변환 및 모델 학습을 위한 스크립트 모음

## 📂 스크립트 목록

### 1. convert_labelstudio_to_yolo.py
Label Studio 어노테이션을 YOLO 학습 포맷으로 변환

**기능**:
- Label Studio JSON → YOLO format 변환
- Bbox 좌표를 YOLO 포맷으로 변환 (center x, center y, width, height)
- Train/Val/Test 자동 분할
- data.yaml 설정 파일 자동 생성

**사용법**:
```bash
# 기본 사용
python3 convert_labelstudio_to_yolo.py \
  dataset/project-1-at-2025-11-12-05-32-de5d2a99.json \
  --image-dir data/raw_images \
  --output-dir data/yolo_dataset

# 커스텀 split 비율 (train/val/test)
python3 convert_labelstudio_to_yolo.py \
  dataset/export.json \
  --image-dir data/raw_images \
  --output-dir data/yolo_dataset \
  --split 0.7 0.15 0.15

# container만 학습 (lid 제외)
python3 convert_labelstudio_to_yolo.py \
  dataset/export.json \
  --image-dir data/raw_images \
  --output-dir data/yolo_dataset \
  --classes container
```

### 2. train_yolo.py
YOLO 모델 학습 스크립트

**기능**:
- YOLOv8 모델 학습
- 다양한 모델 크기 지원 (n/s/m/l/x)
- 학습 하이퍼파라미터 설정
- 체크포인트에서 재개
- 모델 검증

**사용법**:
```bash
# 기본 학습 (YOLOv8n, 100 epochs)
python3 train_yolo.py --data data/yolo_dataset/data.yaml

# 큰 모델로 학습 (더 높은 정확도)
python3 train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --model yolov8s.pt \
  --epochs 200

# 자동 배치 사이즈 (권장)
python3 train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --batch -1

# 커스텀 설정
python3 train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --model yolov8m.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --project runs/detect \
  --name cup_v2

# 학습 재개
python3 train_yolo.py \
  --resume runs/detect/cup_detection/weights/last.pt

# 모델 검증
python3 train_yolo.py \
  --validate runs/detect/cup_detection/weights/best.pt \
  --data data/yolo_dataset/data.yaml
```

**모델 크기**:
| 모델 | 파라미터 | 속도 | 정확도 | 추천 용도 |
|------|---------|------|--------|----------|
| yolov8n | 3.2M | 가장 빠름 | 낮음 | 실시간, 모바일 |
| yolov8s | 11.2M | 빠름 | 중간 | 일반적 사용 |
| yolov8m | 25.9M | 보통 | 높음 | 균형잡힌 성능 |
| yolov8l | 43.7M | 느림 | 매우 높음 | 고정확도 필요 |
| yolov8x | 68.2M | 가장 느림 | 최고 | 최고 성능 |

### 3. convert_labelstudio_to_dataset.py
Label Studio 어노테이션을 분류 데이터셋으로 변환

**기능**:
- Container bbox로 이미지 크롭
- 다회용기/일회용기 분류 데이터셋 생성
- 음료 유무 분류 데이터셋 생성
- 임베딩용 cup_code별 데이터셋 생성
- ZIP 아카이브 자동 생성

**사용법**:
```bash
# 다회용기 분류 데이터셋
python3 convert_labelstudio_to_dataset.py \
  dataset/export.json \
  --image-dir data/raw_images \
  --output-dir dataset_output \
  --task reusable

# 음료 검증 데이터셋
python3 convert_labelstudio_to_dataset.py \
  dataset/export.json \
  --image-dir data/raw_images \
  --output-dir dataset_output \
  --task beverage

# 모든 데이터셋 + 임베딩용
python3 convert_labelstudio_to_dataset.py \
  dataset/export.json \
  --image-dir data/raw_images \
  --output-dir dataset_output \
  --task both \
  --include-types
```

## 🔄 워크플로우

### 전체 파이프라인

```bash
# 1. Label Studio에서 어노테이션 완료 후 JSON export

# 2. YOLO 데이터셋 변환
python3 scripts/convert_labelstudio_to_yolo.py \
  dataset/project-1-at-2025-11-12-05-32-de5d2a99.json \
  --image-dir data/raw_images \
  --output-dir data/yolo_dataset

# 3. YOLO 모델 학습
python3 scripts/train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch -1

# 4. 분류 데이터셋 생성
python3 scripts/convert_labelstudio_to_dataset.py \
  dataset/project-1-at-2025-11-12-05-32-de5d2a99.json \
  --image-dir data/raw_images \
  --output-dir dataset_output \
  --task both \
  --include-types

# 5. 분류 모델 학습 (Jupyter Notebook)
# - notebooks/01_reusable_classifier.ipynb
# - notebooks/02_embedding_generator.ipynb
# - notebooks/03_beverage_detector.ipynb
```

## 📊 데이터셋 구조

### YOLO 데이터셋
```
data/yolo_dataset/
├── data.yaml              # YOLO 설정 파일
├── dataset_info.json      # 데이터셋 통계
├── train/
│   ├── images/           # 학습 이미지
│   └── labels/           # YOLO 라벨 (.txt)
│       └── image.txt     # <class_id> <x_center> <y_center> <width> <height>
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 분류 데이터셋
```
dataset_output/dataset_YYYYMMDD_HHMMSS.zip
├── reusable/
│   ├── reusable/         # 다회용기 (cropped)
│   ├── disposable/       # 일회용기 (cropped)
│   └── unclear/
├── beverage/
│   ├── with_beverage/    # 음료 있음 (cropped)
│   ├── empty/            # 빈 용기 (cropped)
│   └── unclear/
└── types/                # 임베딩용
    ├── CUP001/
    ├── CUP002/
    └── ...
```

## 🎯 학습 팁

### YOLO 모델 선택
- **개발/테스트**: yolov8n (빠른 실험)
- **프로덕션**: yolov8s 또는 yolov8m (균형잡힌 성능)
- **최고 정확도**: yolov8l 또는 yolov8x (충분한 GPU 메모리 필요)

### 하이퍼파라미터 튜닝
```bash
# Learning rate 조정
python3 train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --lr0 0.001 \
  --lrf 0.01

# Data augmentation 강도 조정
python3 train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --hsv-h 0.015 \
  --hsv-s 0.7 \
  --hsv-v 0.4 \
  --degrees 10 \
  --translate 0.1 \
  --scale 0.5 \
  --fliplr 0.5

# Augmentation 비활성화
python3 train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --no-augment
```

### GPU 메모리 부족 시
```bash
# 배치 사이즈 줄이기
python3 train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --batch 8

# 이미지 크기 줄이기
python3 train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --imgsz 416

# 작은 모델 사용
python3 train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --model yolov8n.pt
```

## 🔍 모니터링

### TensorBoard (선택사항)
```bash
# 학습 중 metrics 확인
tensorboard --logdir runs/detect

# 브라우저에서 http://localhost:6006 접속
```

### 학습 결과 확인
```bash
# 학습 완료 후
ls runs/detect/cup_detection/
# - weights/best.pt: 최고 성능 모델
# - weights/last.pt: 마지막 체크포인트
# - results.csv: 학습 메트릭
# - results.png: 학습 그래프
# - confusion_matrix.png: Confusion matrix
# - val_batch*.jpg: 검증 이미지 샘플
```

## 🐛 트러블슈팅

### ultralytics 미설치
```bash
pip install ultralytics
```

### PIL 이미지 에러
```bash
pip install Pillow
```

### YAML 파싱 에러
```bash
pip install pyyaml
```

### CUDA out of memory
- 배치 사이즈 줄이기: `--batch 8` 또는 `--batch 4`
- 이미지 크기 줄이기: `--imgsz 416`
- 작은 모델 사용: `--model yolov8n.pt`
- CPU 모드: `--device cpu` (느림)
