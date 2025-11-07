# AI Model Server - 다회용기 검증 시스템

자체 학습 모델을 사용한 다회용기 검증 AI 서버

## 📋 목차
- [개요](#개요)
- [빠른 시작](#빠른-시작)
- [Docker 사용법](#docker-사용법)
- [로컬 개발](#로컬-개발)
- [모델 학습](#모델-학습)
- [API 문서](#api-문서)

---

## 개요

### 제공 기능
1. **다회용기 분류**: 일회용기 vs 다회용기 구분
2. **임베딩 생성**: 이미지를 512차원 벡터로 변환
3. **음료 검증**: 다회용기에 음료가 담겨있는지 확인

### 기술 스택
- **프레임워크**: FastAPI
- **모델**: PyTorch + Transformers (CLIP)
- **배포**: Docker + GPU 지원

---

## 🚀 빠른 시작

### 1. Docker로 실행 (권장)

#### GPU가 있는 경우
```bash
cd ai-server

# 환경 변수 설정
cp .env.example .env

# Docker Compose로 실행
docker-compose up -d ai-server

# 로그 확인
docker-compose logs -f ai-server
```

#### GPU가 없는 경우
`docker-compose.yml`에서 GPU 설정 제거:
```yaml
# deploy 섹션 주석 처리 또는 제거
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
```

그 다음 실행:
```bash
docker-compose up -d ai-server
```

### 2. 서버 확인

```bash
# 헬스체크
curl http://localhost:8000/health

# API 문서 확인
# 브라우저에서 http://localhost:8000/docs 접속
```

---

## 🐳 Docker 사용법

### 서비스 구성

**ai-server/docker-compose.yml**에 3개 서비스:

1. **ai-server**: FastAPI 서버 (포트 8000)
2. **jupyter**: Jupyter Lab 서버 (포트 8888, 선택사항)
3. **label-studio**: 데이터셋 어노테이션 툴 (포트 8080, 선택사항)

### 서비스 관리

```bash
# 모든 서비스 시작
docker-compose up -d

# 특정 서비스만 시작
docker-compose up -d ai-server
docker-compose up -d jupyter

# 서비스 중지
docker-compose down

# 재시작
docker-compose restart ai-server

# 로그 확인
docker-compose logs -f ai-server
docker-compose logs -f jupyter

# 컨테이너 접속
docker-compose exec ai-server bash
```

### Jupyter Notebook 사용

```bash
# Jupyter 서버 시작
docker-compose up -d jupyter

# 브라우저에서 접속
# http://localhost:8888
# (토큰 없이 접속 가능하도록 설정됨)
```

### Label Studio 사용 (데이터셋 어노테이션)

```bash
# Label Studio 서버 시작
docker-compose up -d label-studio

# 브라우저에서 접속
# http://localhost:8080

# 기본 로그인 정보:
# Email: admin@example.com
# Password: admin123
```

**주요 기능**:
- 이미지 자르기, 회전, 확대/축소
- Bounding Box, Polygon, Segmentation 어노테이션
- COCO, YOLO, Pascal VOC 등 다양한 포맷으로 내보내기
- 프로젝트별 데이터 관리

**데이터 위치**:
- 어노테이션할 이미지: `data/` 디렉토리에 배치
- 프로젝트 데이터: `label-studio/data/`에 자동 저장
- 내보내기 결과: `label-studio/export/`

### 볼륨 관리

데이터는 다음 디렉토리에 저장됩니다:

```
ai-server/
├── models/          # 학습된 모델 가중치
├── uploads/         # 업로드된 이미지
├── data/           # 학습 데이터
├── notebooks/      # Jupyter notebooks
└── label-studio/   # Label Studio 데이터
    ├── data/       # 프로젝트 및 어노테이션
    └── export/     # 내보내기 결과
```

---

## 💻 로컬 개발

### 1. 환경 설정

```bash
cd ai-server

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. GPU 설정 (선택사항)

CUDA가 설치된 경우:
```bash
# PyTorch GPU 버전 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일 편집
```

### 4. 서버 실행

```bash
# 개발 모드 (자동 재로드)
python main.py

# 또는
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🎓 모델 학습

### 학습 노트북

`notebooks/` 디렉토리에 3개의 Jupyter Notebook:

1. **01_reusable_classifier.ipynb**: 다회용기 분류 모델
2. **02_embedding_generator.ipynb**: CLIP 임베딩
3. **03_beverage_detector.ipynb**: 음료 검증 모델

### Jupyter 실행

#### 로컬에서:
```bash
source venv/bin/activate
jupyter lab notebooks/
```

#### Docker에서:
```bash
docker-compose up -d jupyter
# http://localhost:8888 접속
```

### 데이터 준비

학습 데이터는 다음 구조로 준비:

```
data/
├── reusable_classification/
│   ├── train/
│   │   ├── reusable/     # 다회용기 이미지 (최소 500장)
│   │   └── disposable/   # 일회용기 이미지 (최소 500장)
│   └── val/
│       ├── reusable/
│       └── disposable/
└── beverage_detection/
    ├── train/
    │   ├── with_beverage/    # 음료 있음 (최소 300장)
    │   └── without_beverage/ # 음료 없음 (최소 300장)
    └── val/
        ├── with_beverage/
        └── without_beverage/
```

### 학습 순서

1. **02_embedding_generator.ipynb** 먼저 실행 (사전학습 모델, 학습 불필요)
2. **01_reusable_classifier.ipynb** 실행 (데이터 준비 후)
3. **03_beverage_detector.ipynb** 실행 (데이터 준비 후)

학습된 모델은 `models/weights/`에 저장됩니다.

---

## 📖 API 문서

### 엔드포인트

#### 1. 다회용기 분류
```http
POST /classify-reusable
Content-Type: multipart/form-data

file: <image_file>
```

**응답**:
```json
{
  "is_reusable": true,
  "confidence": 0.95,
  "message": "다회용기로 판단됨"
}
```

#### 2. 임베딩 생성
```http
POST /generate-embedding
Content-Type: multipart/form-data

file: <image_file>
```

**응답**:
```json
{
  "embedding": [0.123, 0.456, ..., 0.789],
  "dimension": 512
}
```

#### 3. 음료 검증
```http
POST /verify-beverage
Content-Type: multipart/form-data

file: <image_file>
```

**응답**:
```json
{
  "has_beverage": true,
  "confidence": 0.92,
  "message": "음료가 담겨있음"
}
```

#### 4. 헬스체크
```http
GET /health
```

**응답**:
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": {
    "classifier": true,
    "embedding_generator": true,
    "beverage_detector": true
  }
}
```

### Swagger UI

서버 실행 후 브라우저에서 접속:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🔧 트러블슈팅

### GPU 메모리 부족
```bash
# .env 파일에서 배치 크기 줄이기
BATCH_SIZE=4
```

### CUDA 오류
```bash
# GPU 사용 가능 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"

# CPU 모드로 전환
# .env 파일에서
DEVICE=cpu
```

### Docker GPU 지원 안됨
```bash
# NVIDIA Container Toolkit 설치 확인
nvidia-container-toolkit --version

# Docker에서 GPU 테스트
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 포트 충돌
```bash
# docker-compose.yml에서 포트 변경
ports:
  - "8001:8000"  # 8000 → 8001
```

---

## 📊 성능 최적화

### 모델 최적화
- **양자화 (INT8)**: 모델 크기 75% 감소
- **ONNX 변환**: 추론 속도 20-30% 향상
- **배치 처리**: 여러 이미지 동시 처리

학습 노트북에 최적화 코드 포함.

### 서버 최적화
```bash
# 프로덕션 모드 (workers 추가)
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

---

## 📝 개발 로드맵

- [ ] 모델 구현 (classifier, embedding, beverage detector)
- [ ] FastAPI 엔드포인트 완성
- [ ] 모델 로딩 및 추론 구현
- [ ] 에러 핸들링 강화
- [ ] 로깅 시스템 구축
- [ ] 성능 모니터링
- [ ] 캐싱 전략
- [ ] 배치 처리 최적화

---

## 📄 라이선스

학습용 프로젝트
