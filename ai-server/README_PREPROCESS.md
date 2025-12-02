# 데이터셋 전처리 가이드

## 개요
YOLO를 사용하여 이미지에서 용기(bottle, cup)를 자동으로 감지하고, bbox 기반으로 크롭하여 전처리합니다.

## 전처리 규칙
- ✅ **한 이미지에 용기가 정확히 1개**: 성공적으로 크롭
- ⚠️ **용기가 감지되지 않음**: 건너뛰고 warning 로그
- ⚠️ **용기가 2개 이상 감지됨**: 건너뛰고 warning 로그
- ❌ **처리 중 오류 발생**: 건너뛰고 error 로그

## 사용 방법

### 1. Docker Compose 사용 (권장)

```bash
# GPU 서버에서 실행
docker-compose -f docker-compose.preprocess.yml up

# 완료 후 컨테이너 정리
docker-compose -f docker-compose.preprocess.yml down
```

### 2. 직접 실행 (Python)

```bash
# 기본 설정
python scripts/preprocess_container_images.py \
  --input_dir ./data \
  --output_dir ./data_preprocessed \
  --device cuda

# 커스텀 설정
python scripts/preprocess_container_images.py \
  --input_dir ./data \
  --output_dir ./data_preprocessed \
  --model yolov8s.pt \
  --confidence 0.3 \
  --padding 0.15 \
  --device cuda
```

## 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input_dir` | `/app/data` | 입력 이미지 디렉토리 |
| `--output_dir` | `/app/data_preprocessed` | 출력 디렉토리 |
| `--model` | `yolov8n.pt` | YOLO 모델 (n/s/m/l/x) |
| `--confidence` | `0.25` | 감지 신뢰도 임계값 (0-1) |
| `--padding` | `0.1` | bbox 패딩 비율 (10%) |
| `--device` | `cuda` | 연산 장치 (cuda/cpu) |
| `--no-recursive` | - | 하위 디렉토리 탐색 안 함 |

## 출력 결과

### 디렉토리 구조
```
data_preprocessed/
├── container_type/
│   ├── disposable/
│   │   └── (크롭된 이미지들)
│   └── reusable/
│       └── (크롭된 이미지들)
├── has_beverage/
│   ├── has_beverage/
│   ├── empty/
│   └── unclear/
└── preprocessing_warnings.txt  # ⚠️ 경고 로그
```

### 통계 리포트
```
============================================================
Preprocessing Statistics
============================================================
Total processed: 840
✅ Success: 652 (77.6%)
⚠️  No container: 103
⚠️  Multiple containers: 78
❌ Errors: 7
============================================================
```

### 경고 로그 예시 (`preprocessing_warnings.txt`)
```
⚠️  zf_bottle_123.png: No container detected
⚠️  zf_bottle_456.png: Multiple containers detected: 2
⚠️  zf_cup_789.png: No container detected
```

## YOLO 모델 선택

| 모델 | 크기 | 속도 | 정확도 |
|------|------|------|--------|
| `yolov8n.pt` | 6.3MB | 매우 빠름 | 보통 |
| `yolov8s.pt` | 21.5MB | 빠름 | 좋음 |
| `yolov8m.pt` | 49.7MB | 보통 | 매우 좋음 |
| `yolov8l.pt` | 83.7MB | 느림 | 우수 |
| `yolov8x.pt` | 130.5MB | 매우 느림 | 최고 |

**권장:**
- 빠른 테스트: `yolov8n.pt`
- 프로덕션: `yolov8s.pt` 또는 `yolov8m.pt`

## 감지 클래스

COCO 데이터셋 기반:
- **Class 39**: bottle (병, 플라스틱 병, 유리병 등)
- **Class 41**: cup (컵, 머그잔, 텀블러 등)

추가 클래스가 필요한 경우 `CONTAINER_CLASSES` 딕셔너리에 추가하세요.

## GPU 메모리 최적화

```yaml
# docker-compose.preprocess.yml
shm_size: '2gb'  # 공유 메모리 크기
deploy:
  resources:
    reservations:
      devices:
        - count: 1  # GPU 1개만 사용
```

## 트러블슈팅

### GPU 메모리 부족
```bash
# 더 작은 모델 사용
--model yolov8n.pt

# 또는 CPU로 전환
--device cpu
```

### 너무 많은 경고 (용기 미감지)
```bash
# 신뢰도 임계값 낮추기
--confidence 0.15
```

### bbox가 너무 타이트
```bash
# 패딩 늘리기
--padding 0.2  # 20%
```

## 다음 단계

전처리 완료 후:
1. `data_preprocessed/` 디렉토리 확인
2. `preprocessing_warnings.txt` 검토
3. 샘플 이미지 육안 검증
4. 모델 학습 시작
