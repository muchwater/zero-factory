# Container Verification API 사용 가이드

통합 용기 검증 API - 컨테이너 감지, 다회용기 검증, 음료 검증을 한 번에 처리

## API 엔드포인트

### POST /container/verify

이미지를 업로드하여 용기를 검증합니다.

**요청:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: 이미지 파일 (`file`)

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
  "error": null,
  "container_class": "cup",
  "container_confidence": 0.92
}
```

## 처리 흐름

```
이미지 입력
    ↓
[1] 컨테이너 감지 (YOLO)
    ├─ 0개 감지 → ❌ 실패 반환
    ├─ 2개 이상 → ❌ 실패 반환
    └─ 1개 감지 → ✅ 다음 단계
        ↓
[2] 다회용기 검증 (EfficientNet-B0)
    ├─ 일회용 → ❌ 실패 반환
    └─ 다회용 → ✅ 다음 단계
        ↓
[3] 음료 검증 (EfficientNet-B0)
    ├─ Yes (음료 있음)
    ├─ No (비어있음)
    └─ Unclear (불명확)
        ↓
    ✅ 최종 결과 반환
```

## 응답 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `container_detected` | boolean | 컨테이너 1개 감지 여부 |
| `num_containers` | integer | 감지된 컨테이너 수 (0, 1, 2+) |
| `is_reusable` | boolean? | 다회용기 여부 (null: 감지 실패) |
| `reusable_confidence` | float? | 다회용기 검증 신뢰도 (0-1) |
| `beverage_status` | string? | 음료 상태 ("Yes", "No", "Unclear") |
| `has_beverage` | boolean? | 음료 있음 여부 |
| `beverage_confidence` | float? | 음료 검증 신뢰도 (0-1) |
| `message` | string | 처리 결과 메시지 |
| `error` | string? | 오류 메시지 (실패 시) |
| `container_class` | string? | 컨테이너 종류 ("bottle", "cup") |
| `container_confidence` | float? | 컨테이너 감지 신뢰도 (0-1) |

## 사용 예시

### cURL

```bash
# 로컬 이미지 업로드
curl -X POST http://localhost:8000/container/verify \
  -F "file=@/path/to/image.jpg"

# 원격 서버
curl -X POST https://your-server.com/container/verify \
  -F "file=@image.jpg"
```

### Python (requests)

```python
import requests

url = "http://localhost:8000/container/verify"
files = {"file": open("image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Container detected: {result['container_detected']}")
print(f"Is reusable: {result.get('is_reusable')}")
print(f"Beverage status: {result.get('beverage_status')}")
```

### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/container/verify', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log('Container detected:', data.container_detected);
    console.log('Is reusable:', data.is_reusable);
    console.log('Beverage status:', data.beverage_status);
  });
```

## 응답 시나리오

### 1. 성공 (다회용기 + 음료 있음)

```json
{
  "container_detected": true,
  "num_containers": 1,
  "is_reusable": true,
  "reusable_confidence": 0.95,
  "beverage_status": "Yes",
  "has_beverage": true,
  "beverage_confidence": 0.88,
  "message": "Container verified: Yes beverage (confidence: 88.0%)",
  "error": null,
  "container_class": "cup",
  "container_confidence": 0.92
}
```

### 2. 실패 - 컨테이너 미감지

```json
{
  "container_detected": false,
  "num_containers": 0,
  "is_reusable": null,
  "reusable_confidence": null,
  "beverage_status": null,
  "has_beverage": null,
  "beverage_confidence": null,
  "message": "Container detection failed",
  "error": "No container detected",
  "container_class": null,
  "container_confidence": null
}
```

### 3. 실패 - 여러 컨테이너 감지

```json
{
  "container_detected": false,
  "num_containers": 3,
  "is_reusable": null,
  "reusable_confidence": null,
  "beverage_status": null,
  "has_beverage": null,
  "beverage_confidence": null,
  "message": "Container detection failed",
  "error": "Multiple containers detected: 3",
  "container_class": null,
  "container_confidence": null
}
```

### 4. 실패 - 일회용기 감지

```json
{
  "container_detected": true,
  "num_containers": 1,
  "is_reusable": false,
  "reusable_confidence": 0.92,
  "beverage_status": null,
  "has_beverage": null,
  "beverage_confidence": null,
  "message": "Not a reusable container (confidence: 92.0%)",
  "error": "Disposable container detected",
  "container_class": "bottle",
  "container_confidence": 0.85
}
```

### 5. 성공 - 음료 없음

```json
{
  "container_detected": true,
  "num_containers": 1,
  "is_reusable": true,
  "reusable_confidence": 0.96,
  "beverage_status": "No",
  "has_beverage": false,
  "beverage_confidence": 0.91,
  "message": "Container verified: No beverage (confidence: 91.0%)",
  "error": null,
  "container_class": "cup",
  "container_confidence": 0.88
}
```

### 6. 성공 - 음료 상태 불명확

```json
{
  "container_detected": true,
  "num_containers": 1,
  "is_reusable": true,
  "reusable_confidence": 0.94,
  "beverage_status": "Unclear",
  "has_beverage": false,
  "beverage_confidence": 0.62,
  "message": "Container verified: Unclear beverage (confidence: 62.0%)",
  "error": null,
  "container_class": "cup",
  "container_confidence": 0.89
}
```

## 헬스 체크

### GET /container/health

모델 로드 상태를 확인합니다.

```bash
curl http://localhost:8000/container/health
```

응답:
```json
{
  "status": "healthy",
  "models": {
    "container_detector": true,
    "reusable_classifier": true,
    "beverage_detector": true
  }
}
```

## API 문서

### Swagger UI

```
http://localhost:8000/docs
```

### ReDoc

```
http://localhost:8000/redoc
```

## 서버 실행

### Docker Compose

```bash
docker compose up ai-server
```

### 로컬 실행

```bash
# 가상환경 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 에러 코드

| HTTP 코드 | 설명 |
|-----------|------|
| 200 | 성공 |
| 422 | 잘못된 요청 (파일 누락 등) |
| 500 | 서버 오류 |
| 503 | 모델 미로드 |

## 성능

### 응답 시간 (RTX 2060 기준)

- **컨테이너 감지**: ~50-100ms
- **다회용기 검증**: ~30-50ms
- **음료 검증**: ~30-50ms
- **전체**: ~150-250ms

### 동시 요청

- 권장: 최대 4 동시 요청
- GPU 메모리: 요청당 ~500MB

## 문제 해결

### 모델 미로드 (503 오류)

```bash
# 모델 파일 확인
ls -la models/weights/

# 필요한 파일:
# - reusable_classifier_best.pth
# - beverage_detector_best.pth
```

해결: 노트북으로 모델 학습 필요
- `notebooks/01_reusable_classifier_training.ipynb`
- `notebooks/02_beverage_detector_training.ipynb`

### GPU 메모리 부족

```python
# main.py에서 DEVICE 변경
DEVICE = 'cpu'  # 또는 환경변수 DEVICE=cpu
```

### 느린 응답

1. GPU 사용 확인: `nvidia-smi`
2. 배치 처리 고려
3. 이미지 크기 조정 (권장: 800x600 이하)

## 보안 권장사항

1. **파일 크기 제한**: 최대 10MB
2. **파일 형식 검증**: JPEG, PNG만 허용
3. **Rate Limiting**: IP당 분당 60회
4. **인증**: API 키 또는 JWT 토큰

## 라이선스

MIT License
