# AI Container Verification Server

YOLO + EfficientNet 기반 용기 검증 시스템

## 주요 기능

1. **컨테이너 감지**: YOLO를 사용하여 bottle/cup 자동 감지 및 크롭
2. **다회용기 검증**: EfficientNet-B0 기반 일회용/다회용 분류
3. **음료 검증**: EfficientNet-B0 기반 음료 유무 판단 (Yes/No/Unclear)

## 시스템 구성

```
┌─────────────┐
│   이미지    │
└──────┬──────┘
       ↓
┌─────────────────────────────┐
│ 1. Container Detection      │
│    (YOLO)                   │
│    - 1개만 감지 → 통과      │
│    - 0개 or 2개+ → 실패     │
└──────┬──────────────────────┘
       ↓
┌─────────────────────────────┐
│ 2. Reusable Classification  │
│    (EfficientNet-B0)        │
│    - 다회용 → 통과          │
│    - 일회용 → 실패          │
└──────┬──────────────────────┘
       ↓
┌─────────────────────────────┐
│ 3. Beverage Detection       │
│    (EfficientNet-B0)        │
│    - Yes: 음료 있음         │
│    - No: 비어있음           │
│    - Unclear: 불명확        │
└──────┬──────────────────────┘
       ↓
   JSON 응답
```

## 빠른 시작

### 1. 데이터 전처리
```bash
docker compose -f docker-compose.preprocess.yml up
```
→ YOLO로 용기를 감지하고 크롭하여 `data_preprocessed/`에 저장

### 2. 모델 학습
```bash
docker compose up jupyter
# 브라우저: http://localhost:8888/lab

# 실행 순서:
# 1. notebooks/01_reusable_classifier_training.ipynb
# 2. notebooks/02_beverage_detector_training.ipynb
```
→ 학습된 모델이 `models/weights/`에 저장됨

### 3. API 서버 실행
```bash
docker compose up ai-server
```
→ API 서버가 http://localhost:8000 에서 실행됨

### 4. API 테스트
```bash
# 헬스 체크
python test_api.py --health

# 이미지 검증
python test_api.py --image path/to/image.jpg

# 또는 curl
curl -X POST http://localhost:8000/container/verify \
  -F "file=@image.jpg"
```

## 프로젝트 구조

```
ai-server/
├── main.py                    # FastAPI 서버
├── models/                    # 모델 정의
│   ├── container_detector.py
│   ├── reusable_classifier_model.py
│   ├── beverage_detector_model.py
│   └── weights/              # 학습된 모델 (.pth)
├── routes/                    # API 라우터
│   ├── container.py
│   └── health.py
├── schemas/                   # 응답 스키마
├── notebooks/                 # Jupyter 학습 노트북
│   ├── 01_reusable_classifier_training.ipynb
│   ├── 02_beverage_detector_training.ipynb
│   └── gpu_memory_utils.py
├── scripts/                   # 전처리 스크립트
│   └── preprocess_container_images.py
├── data/                      # 원본 학습 데이터
├── data_preprocessed/         # 전처리된 데이터
├── utils/                     # 유틸리티
└── test_api.py               # API 테스트 스크립트
```

## 문서

- **[API_USAGE.md](API_USAGE.md)** - API 사용 가이드 및 예제
- **[README_API.md](README_API.md)** - API 서버 실행 가이드
- **[README_TRAINING.md](README_TRAINING.md)** - 모델 학습 가이드
- **[README_PREPROCESS.md](README_PREPROCESS.md)** - 데이터 전처리 가이드
- **[GPU_MEMORY_GUIDE.md](GPU_MEMORY_GUIDE.md)** - GPU 메모리 관리

## API 엔드포인트

### POST /container/verify

**요청:**
```bash
curl -X POST http://localhost:8000/container/verify \
  -F "file=@image.jpg"
```

**응답:**
```json
{
  "container_detected": true,
  "num_containers": 1,
  "is_reusable": true,
  "reusable_confidence": 0.95,
  "beverage_status": "Yes",
  "has_beverage": true,
  "beverage_confidence": 0.88,
  "message": "Container verified successfully",
  "container_class": "cup",
  "container_confidence": 0.92
}
```

## 모델 성능

### 다회용기 분류 모델
- **아키텍처**: EfficientNet-B0
- **정확도**: ~100% (validation set)
- **클래스**: disposable (58), reusable (174)

### 음료 검증 모델
- **아키텍처**: EfficientNet-B0
- **정확도**: ~95%+ (validation set)
- **클래스**: empty (43), has_beverage (127)

### 추론 성능 (RTX 2060)
- **전체 처리**: 150-250ms
- **GPU 메모리**: 요청당 ~500MB

## 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU (RTX 2060 이상 권장)
- **VRAM**: 6GB 이상
- **RAM**: 8GB 이상

### 소프트웨어
- Docker & Docker Compose
- NVIDIA Docker Runtime
- Python 3.10+
- CUDA 11.8+

## 개발 환경

### Docker Compose 서비스

```bash
# AI 서버
docker compose up ai-server

# Jupyter Lab (학습용)
docker compose up jupyter

# 데이터 전처리
docker compose -f docker-compose.preprocess.yml up
```

### 로컬 개발

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python main.py
```

## 문제 해결

### GPU 메모리 부족
```bash
# CPU 모드로 전환
DEVICE=cpu docker compose up ai-server
```

### 모델 미로드
```bash
# 모델 파일 확인
ls -la models/weights/

# 학습 필요
docker compose up jupyter
```

### Jupyter 연결 실패
```bash
# 커널 재시작
docker compose restart jupyter

# 로그 확인
docker logs ai-jupyter
```

## 기술 스택

- **Backend**: FastAPI
- **Object Detection**: YOLOv8 (Ultralytics)
- **Classification**: PyTorch + EfficientNet-B0
- **Image Processing**: Pillow, OpenCV
- **Container**: Docker, NVIDIA Runtime

## 라이선스

MIT License

## 기여

이슈 및 PR 환영합니다!
