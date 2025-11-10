# 모델 배포 가이드

Jupyter Notebook에서 학습한 모델을 FastAPI 서버에 배포하는 방법을 설명합니다.

## 📋 전체 워크플로우

```
1. Notebook에서 모델 학습
   └→ models/weights/*.pth 저장

2. 서버 자동 로드
   └→ startup 시 모델 메모리에 로드

3. API 엔드포인트로 서빙
   └→ HTTP POST로 이미지 전송 → 추론 결과 반환
```

---

## 1️⃣ Notebook에서 모델 학습

### Reusable Classifier 학습

```bash
# Jupyter Notebook 실행
jupyter notebook notebooks/01_reusable_classifier.ipynb
```

**중요**: Notebook을 끝까지 실행하면 자동으로 모델이 저장됩니다:
- 저장 위치: `models/weights/reusable_classifier.pth`
- 저장 포맷: PyTorch state_dict

### Beverage Detector 학습

```bash
jupyter notebook notebooks/03_beverage_detector.ipynb
```

- 저장 위치: `models/weights/beverage_detector.pth`

---

## 2️⃣ 모델 파일 구조

학습 완료 후 다음과 같은 구조가 됩니다:

```
ai-server/
├── models/
│   ├── __init__.py
│   ├── reusable_classifier.py      # 모델 클래스 정의
│   ├── beverage_detector.py        # 모델 클래스 정의
│   └── weights/
│       ├── reusable_classifier.pth  # 학습된 가중치 ✅
│       └── beverage_detector.pth    # 학습된 가중치 ✅
├── main.py                          # FastAPI 서버
└── notebooks/
    ├── 01_reusable_classifier.ipynb
    └── 03_beverage_detector.ipynb
```

---

## 3️⃣ 서버 실행

### 로컬 실행 (개발)

```bash
cd ai-server
python main.py
```

서버 시작 시 자동으로 모델을 로드합니다:

```
🚀 AI Model Server Starting...
Device: cpu
✅ Reusable classifier loaded from models/weights/reusable_classifier.pth
✅ Beverage detector loaded from models/weights/beverage_detector.pth
⚠️  Embedding generator not implemented yet

============================================================
✅ Server ready!
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Docker 실행 (프로덕션)

```bash
cd ai-server
docker-compose up ai-server
```

---

## 4️⃣ API 사용 방법

### API 문서 확인

서버 실행 후 브라우저에서:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 1. Reusable Container 분류

```bash
# cURL 예시
curl -X POST "http://localhost:8000/classify-reusable" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/container_image.jpg"
```

**응답 예시:**
```json
{
  "is_reusable": true,
  "confidence": 0.92,
  "predicted_class": "reusable",
  "probabilities": {
    "disposable": 0.08,
    "reusable": 0.92
  },
  "message": "✅ Reusable container detected (confidence: 92.0%)"
}
```

### 2. Beverage 검증

```bash
curl -X POST "http://localhost:8000/verify-beverage?confidence_threshold=0.7" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/container_with_beverage.jpg"
```

**응답 예시:**
```json
{
  "has_beverage": true,
  "confidence": 0.88,
  "predicted_class": "with_beverage",
  "is_valid": true,
  "probabilities": {
    "empty": 0.05,
    "with_beverage": 0.88,
    "unclear": 0.07
  },
  "message": "Beverage detected - Valid usage"
}
```

### 3. Python 클라이언트 예시

```python
import requests

# Reusable 분류
with open('container.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify-reusable',
        files={'file': f}
    )
    result = response.json()
    print(f"Is reusable: {result['is_reusable']}")
    print(f"Confidence: {result['confidence']:.1%}")

# Beverage 검증
with open('container_with_drink.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/verify-beverage',
        files={'file': f},
        params={'confidence_threshold': 0.7}
    )
    result = response.json()
    print(f"Has beverage: {result['has_beverage']}")
    print(f"Is valid: {result['is_valid']}")
```

---

## 5️⃣ 모델 업데이트 프로세스

### 새로운 데이터로 재학습

```bash
# 1. Label Studio에서 더 많은 데이터 라벨링
# 2. 데이터셋 변환
.venv/bin/python scripts/convert_labelstudio_to_dataset.py \
  label-studio/data/export/latest.json \
  --image-dir data/raw_images \
  --output-dir dataset_output \
  --zip

# 3. Notebook에서 재학습
jupyter notebook notebooks/01_reusable_classifier.ipynb

# 4. 서버 재시작 (자동으로 새 모델 로드)
docker-compose restart ai-server
```

### 모델 버전 관리

모델 파일에 날짜를 포함해서 저장:

```python
# Notebook에서
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f'../models/weights/reusable_classifier_{timestamp}.pth'
torch.save(model.state_dict(), model_path)

# 심볼릭 링크로 최신 모델 지정
os.symlink(model_path, '../models/weights/reusable_classifier.pth')
```

---

## 6️⃣ 모델 구조 설명

### Reusable Classifier

**모델 파일**: `models/reusable_classifier.py`

```python
class ReusableClassifier(nn.Module):
    """ResNet50 기반 2-class 분류기"""
    - 백본: ResNet50 (ImageNet pretrained)
    - 헤드: FC(2048 → 512 → 2)
    - 출력: [disposable, reusable]
```

**추론 클래스**: `ReusableClassifierInference`
- 전처리: Resize(224) + Normalize(ImageNet)
- 출력: is_reusable, confidence, probabilities

### Beverage Detector

**모델 파일**: `models/beverage_detector.py`

```python
class BeverageDetector(nn.Module):
    """MobileNetV3-Small 기반 3-class 분류기"""
    - 백본: MobileNetV3-Small (경량화)
    - 헤드: FC(576 → 256 → 3)
    - 출력: [empty, with_beverage, unclear]
```

**추론 클래스**: `BeverageDetectorInference`
- 전처리: Resize(224) + Normalize(ImageNet)
- 검증 로직: confidence_threshold 기반
- 출력: has_beverage, is_valid, probabilities

---

## 7️⃣ 성능 모니터링

### 추론 속도 측정

```python
import time

start = time.time()
result = classifier.predict(image_bytes)
inference_time = time.time() - start

print(f"Inference time: {inference_time*1000:.2f}ms")
```

### 로깅 추가

```python
# main.py에 추가
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/classify-reusable")
async def classify_reusable(file: UploadFile = File(...)):
    logger.info(f"Classifying image: {file.filename}")
    # ... 추론 로직
    logger.info(f"Result: {result['class']} ({result['confidence']:.2%})")
```

---

## 8️⃣ 트러블슈팅

### ❌ 모델이 로드되지 않음

**증상**: `Classifier model not loaded` 에러

**해결**:
1. 모델 파일이 존재하는지 확인:
   ```bash
   ls -lh models/weights/
   ```
2. Notebook에서 모델을 학습했는지 확인
3. 경로가 올바른지 확인 (`models/weights/*.pth`)

### ❌ CUDA out of memory

**증상**: GPU 메모리 부족

**해결**:
1. CPU로 전환:
   ```bash
   export DEVICE=cpu
   ```
2. 배치 크기 줄이기 (Notebook에서)
3. 더 작은 모델 사용 (MobileNet 등)

### ❌ 추론이 너무 느림

**해결**:
1. ONNX로 변환:
   ```python
   # Notebook에서
   torch.onnx.export(model, dummy_input, 'model.onnx')
   ```
2. 양자화 적용:
   ```python
   model_quantized = torch.quantization.quantize_dynamic(
       model, {nn.Linear}, dtype=torch.qint8
   )
   ```

---

## 9️⃣ 프로덕션 체크리스트

배포 전 확인 사항:

- [ ] 모델이 충분한 데이터로 학습되었는가? (최소 100개/클래스)
- [ ] Validation accuracy가 목표치 이상인가? (>90%)
- [ ] 추론 속도가 요구사항을 만족하는가? (<100ms)
- [ ] 오분류 케이스를 분석했는가?
- [ ] 모델 파일이 버전 관리되는가?
- [ ] API 엔드포인트가 테스트되었는가?
- [ ] 에러 핸들링이 적절한가?
- [ ] 로깅이 설정되었는가?

---

## 📚 추가 리소스

- [Notebook 가이드](notebooks/README.md)
- [데이터셋 변환 가이드](DATASET_WORKFLOW.md)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [PyTorch 배포 가이드](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
