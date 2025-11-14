# AI Server API 사용 가이드

## API 엔드포인트

### 1. 텀블러 등록 API (`/register-tumbler`)

텀블러를 시스템에 등록합니다.

**처리 과정:**
1. YOLO로 텀블러/컵 영역 감지 및 자르기
   - 텀블러/컵이 없거나 2개 이상이면 실패
2. 고성능 ResNet으로 다회용기 검증
3. Siamese Network로 임베딩 추출

**요청:**
```bash
curl -X POST "http://localhost:8000/register-tumbler" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tumbler_image.jpg"
```

**성공 응답 예시:**
```json
{
  "success": true,
  "is_reusable": true,
  "embedding": [0.123, 0.456, ...],  // 256차원 벡터
  "message": "Reusable tumbler registered successfully (confidence: 95.3%)",
  "confidence": 0.953
}
```

**실패 응답 예시:**
```json
{
  "success": false,
  "is_reusable": false,
  "embedding": [],
  "message": "Detection failed: No cup or tumbler detected in image",
  "error": "No cup or tumbler detected in image"
}
```

---

### 2. 사용 검증 API (`/verify-usage`)

텀블러 사용을 검증합니다 (음료 포함 여부 확인).

**처리 과정:**
1. YOLO로 텀블러/컵 영역 감지 및 자르기
   - 텀블러/컵이 없거나 2개 이상이면 실패
2. 속도 빠른 MobileNet으로 음료 검증
3. Siamese Network로 임베딩 추출

**요청:**
```bash
curl -X POST "http://localhost:8000/verify-usage" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tumbler_with_beverage.jpg"
```

**성공 응답 예시:**
```json
{
  "success": true,
  "has_beverage": true,
  "embedding": [0.123, 0.456, ...],  // 256차원 벡터
  "message": "Usage verified with beverage (confidence: 87.6%)",
  "confidence": 0.876
}
```

**실패 응답 예시:**
```json
{
  "success": false,
  "has_beverage": false,
  "embedding": [],
  "message": "No beverage detected: No beverage detected - empty",
  "confidence": 0.923,
  "error": "No beverage in container"
}
```

---

## 실패 케이스

### 1. 텀블러/컵 감지 실패
- **원인:** 이미지에 텀블러/컵이 없음
- **에러:** "No cup or tumbler detected in image"

### 2. 여러 객체 감지
- **원인:** 이미지에 2개 이상의 텀블러/컵 감지
- **에러:** "Multiple objects detected (2). Please ensure only one cup/tumbler is in the image"

### 3. 일회용기 감지
- **원인:** ResNet이 일회용기로 판단
- **에러:** "Disposable container detected"

### 4. 음료 없음
- **원인:** MobileNet이 음료를 감지하지 못함
- **에러:** "No beverage in container"

---

## 모델 정보

| 모델 | 용도 | 특징 |
|------|------|------|
| **YOLOv8n** | 객체 감지 | 빠르고 가벼움, 배경 제거 |
| **ResNet50** | 다회용기 검증 (등록) | 고성능, 정확도 높음 |
| **MobileNetV3** | 음료 검증 (사용) | 빠른 추론 속도 |
| **Siamese Network** | 임베딩 생성 | 256차원 벡터 |

---

## 헬스체크

```bash
curl http://localhost:8000/health
```

**응답:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": {
    "cup_detector": true,
    "classifier": true,
    "embedding_generator": true,
    "beverage_detector": true
  }
}
```

---

## API 문서

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 통합 플로우 예시

### 텀블러 등록 + 사용 검증
```python
import requests
import numpy as np

# 1. 텀블러 등록
with open('tumbler.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/register-tumbler',
        files={'file': f}
    )
    registration = response.json()

    if registration['success']:
        tumbler_embedding = np.array(registration['embedding'])
        print(f"등록 성공! 임베딩: {len(tumbler_embedding)}차원")
    else:
        print(f"등록 실패: {registration['error']}")

# 2. 사용 검증
with open('tumbler_with_coffee.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/verify-usage',
        files={'file': f}
    )
    usage = response.json()

    if usage['success']:
        usage_embedding = np.array(usage['embedding'])

        # 코사인 유사도로 동일 텀블러인지 확인
        similarity = np.dot(tumbler_embedding, usage_embedding)
        print(f"사용 검증 성공! 유사도: {similarity:.3f}")

        if similarity > 0.7:
            print("✅ 동일한 텀블러입니다")
        else:
            print("❌ 다른 텀블러입니다")
    else:
        print(f"검증 실패: {usage['error']}")
```

---

## 주의사항

1. **이미지 품질:** 텀블러/컵이 명확하게 보이는 이미지 사용
2. **단일 객체:** 한 이미지에 하나의 텀블러/컵만 포함
3. **GPU 사용:** CUDA 사용 시 추론 속도 대폭 향상
4. **임베딩 저장:** 반환된 임베딩 벡터를 DB에 저장하여 매칭에 활용
5. **L2 정규화:** 임베딩은 이미 L2 정규화되어 있어 코사인 유사도는 내적으로 계산 가능
