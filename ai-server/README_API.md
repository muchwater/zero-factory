# Container Verification API

다회용기 검증 + 음료 검증 통합 API 서버

## 빠른 시작

### 1. 서버 실행

```bash
# Docker Compose
docker compose up ai-server

# 서버 실행 확인
curl http://localhost:8000
```

### 2. 모델 학습 (필수)

서버를 실행하기 전에 두 모델을 학습해야 합니다:

```bash
# Jupyter Lab 실행
docker compose up jupyter

# 브라우저에서 접속: http://localhost:8888/lab

# 노트북 실행 (순서대로)
# 1. notebooks/01_reusable_classifier_training.ipynb
# 2. notebooks/02_beverage_detector_training.ipynb
```

학습 완료 후 다음 파일이 생성됩니다:
- `models/weights/reusable_classifier_best.pth`
- `models/weights/beverage_detector_best.pth`

### 3. API 테스트

```bash
# 헬스 체크
python test_api.py --health

# 이미지 테스트
python test_api.py --image path/to/image.jpg
```

## API 엔드포인트

### POST /container/verify

이미지를 업로드하여 용기를 검증합니다.

**처리 순서:**
1. YOLO로 용기 감지 및 크롭 (1개만 통과)
2. 다회용기 검증 (다회용기만 통과)
3. 음료 검증 (Yes/No/Unclear)

**요청:**
```bash
curl -X POST http://localhost:8000/container/verify \
  -F "file=@image.jpg"
```

**성공 응답:**
```json
{
  "container_detected": true,
  "num_containers": 1,
  "is_reusable": true,
  "reusable_confidence": 0.95,
  "beverage_status": "Yes",
  "has_beverage": true,
  "beverage_confidence": 0.88,
  "message": "Container verified successfully"
}
```

상세한 API 문서는 [API_USAGE.md](API_USAGE.md)를 참조하세요.

## 프로젝트 구조

```
ai-server/
├── main.py                          # FastAPI 서버
├── models/
│   ├── container_detector.py       # YOLO 기반 용기 감지
│   ├── reusable_classifier_model.py # 다회용기 분류
│   ├── beverage_detector_model.py  # 음료 검증
│   └── weights/                     # 학습된 모델 저장
│       ├── reusable_classifier_best.pth
│       └── beverage_detector_best.pth
├── routes/
│   ├── container.py                # 컨테이너 검증 API
│   └── health.py                   # 헬스 체크
├── schemas/
│   └── container_verification.py  # 응답 스키마
├── notebooks/
│   ├── 01_reusable_classifier_training.ipynb
│   └── 02_beverage_detector_training.ipynb
└── data_preprocessed/              # 전처리된 학습 데이터
```

## 학습 데이터 전처리

모델 학습 전에 데이터를 전처리해야 합니다:

```bash
# GPU 서버에서 실행
docker compose -f docker-compose.preprocess.yml up

# 결과: data_preprocessed/ 폴더에 크롭된 이미지 생성
```

상세한 전처리 가이드는 [README_PREPROCESS.md](README_PREPROCESS.md)를 참조하세요.

## 성능

### 응답 시간 (RTX 2060 기준)
- 전체 처리: ~150-250ms
- 컨테이너 감지: ~50-100ms
- 다회용기 검증: ~30-50ms
- 음료 검증: ~30-50ms

### GPU 메모리
- 모델 로드: ~2GB
- 추론 요청당: ~500MB

## 문제 해결

### 모델이 로드되지 않음

```bash
# 모델 파일 확인
ls -la models/weights/

# 없으면 학습 필요
# notebooks/01_reusable_classifier_training.ipynb
# notebooks/02_beverage_detector_training.ipynb
```

### GPU 메모리 부족

```bash
# CPU 모드로 전환
DEVICE=cpu docker compose up ai-server
```

또는 `docker-compose.yml`에서:
```yaml
environment:
  - DEVICE=cpu
```

### 느린 추론

1. GPU 사용 확인: `nvidia-smi`
2. 이미지 크기 조정 (800x600 이하 권장)
3. 배치 처리 고려

## 개발

### 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python main.py
```

### 코드 수정 후 재시작

```bash
# Docker Compose는 자동 재로드
docker compose up ai-server

# 수동 재시작
docker compose restart ai-server
```

## 문서

- [API 사용 가이드](API_USAGE.md)
- [학습 가이드](README_TRAINING.md)
- [전처리 가이드](README_PREPROCESS.md)
- [GPU 메모리 관리](GPU_MEMORY_GUIDE.md)

## 라이선스

MIT License
