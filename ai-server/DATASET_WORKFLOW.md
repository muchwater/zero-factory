# 데이터셋 변환 워크플로우

Label Studio에서 라벨링한 데이터를 학습용 데이터셋으로 변환하는 방법을 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [데이터 준비](#데이터-준비)
3. [데이터셋 변환](#데이터셋-변환)
4. [모델 학습](#모델-학습)
5. [트러블슈팅](#트러블슈팅)

## 개요

### 워크플로우

```
Label Studio 라벨링
    ↓
Export JSON
    ↓
convert_labelstudio_to_dataset.py 실행
    ↓
크롭된 데이터셋 생성 (ZIP 포함)
    ↓
Jupyter Notebook으로 모델 학습
```

### 데이터셋 구조

변환 후 다음과 같은 구조로 데이터셋이 생성됩니다:

```
dataset_output/
├── reusable/
│   ├── reusable/       # 재사용 용기 (142개)
│   ├── disposable/     # 일회용 용기 (78개)
│   └── unclear/        # 불분명 (0개)
├── beverage/
│   ├── with_beverage/  # 음료 있음 (42개)
│   ├── empty/          # 빈 용기 (150개)
│   └── unclear/        # 불분명 (28개)
├── dataset_reusable_YYYYMMDD_HHMMSS.zip
└── dataset_beverage_YYYYMMDD_HHMMSS.zip
```

## 데이터 준비

### 1. Label Studio에서 라벨링

1. Label Studio 접속: `http://localhost:8080`
2. 이미지 업로드 및 라벨링 작업 수행
3. 각 이미지에 대해:
   - **Container bbox** 1개만 표시 (필수)
   - **Container type** 선택: reusable / disposable / unclear
   - **Beverage status** 선택: has_beverage / empty / unclear
   - **Lid status** 선택 (선택사항)

### 2. Export JSON

Label Studio에서 데이터 export:

```bash
# Label Studio UI에서:
# Project → Export → JSON → Download
```

Export된 파일 위치:
```
ai-server/label-studio/data/export/project-1-at-YYYY-MM-DD-HH-MM-*.json
```

## 데이터셋 변환

### 스크립트 실행

```bash
# 프로젝트 루트에서 실행
.venv/bin/python ai-server/scripts/convert_labelstudio_to_dataset.py \
  ai-server/label-studio/data/export/project-1-at-2025-11-10-01-59-baddde76.json \
  --image-dir ai-server/data/raw_images \
  --output-dir ai-server/dataset_output \
  --zip
```

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `json_file` | Label Studio export JSON 파일 경로 | (필수) |
| `--image-dir` | 원본 이미지가 있는 디렉토리 | (필수) |
| `--output-dir` | 출력 디렉토리 | `./dataset_output` |
| `--task` | 생성할 데이터셋 종류 (`reusable` / `beverage` / `both`) | `both` |
| `--zip` | ZIP 파일 생성 여부 | False |

### 변환 과정

스크립트는 다음 작업을 수행합니다:

1. **JSON 파싱**: Label Studio export 파일 읽기
2. **필터링**: Container bbox가 정확히 1개인 데이터만 선택
3. **이미지 크롭**: Container 영역만 추출
4. **분류별 저장**:
   - Reusable/Disposable/Unclear
   - With_beverage/Empty/Unclear
5. **ZIP 생성** (옵션): 타임스탬프가 포함된 압축 파일

### 출력 예시

```
Parsing ai-server/label-studio/data/export/project-1-at-2025-11-10-01-59-baddde76.json...
⚠️  Skipping zf_bottle_102.png: No container bbox
⚠️  Skipping zf_bottle_115.png: Multiple containers (2)

=== Parsing Statistics ===
Total: 235
No annotation: 0
No container: 14
Multiple containers: 1
Valid: 220

✅ Found 220 valid images with container bbox

============================================================
Creating Reusable/Disposable Classification Dataset
============================================================
Processed 10/220 images...
...
Processed 220/220 images...

=== REUSABLE Dataset Statistics ===
reusable: 142 images
disposable: 78 images
unclear: 0 images
Total: 220 images

============================================================
Creating Beverage Status Classification Dataset
============================================================
...

=== BEVERAGE Dataset Statistics ===
with_beverage: 42 images
empty: 150 images
unclear: 28 images
Total: 220 images

============================================================
Creating ZIP archives...
============================================================
✅ Created: dataset_reusable_20251110_020249.zip (3.84 MB)
✅ Created: dataset_beverage_20251110_020249.zip (3.84 MB)
```

## 모델 학습

### 1. Reusable Container Classifier

Jupyter Notebook: `notebooks/01_reusable_classifier.ipynb`

```python
# 데이터셋 경로 자동 설정됨
DATA_DIR = '../dataset_output/reusable'

# Notebook 실행:
# 1. 모든 셀 실행
# 2. 학습 진행 (약 20 epochs)
# 3. 모델 저장: models/weights/reusable_classifier.pth
```

### 2. Beverage Detector

Jupyter Notebook: `notebooks/03_beverage_detector.ipynb`

```python
# 데이터셋 경로 자동 설정됨
DATA_DIR = '../dataset_output/beverage'

# Unclear 클래스 포함 여부 선택
INCLUDE_UNCLEAR = True  # 3-class 분류
# or
INCLUDE_UNCLEAR = False  # 2-class 분류 (with_beverage, empty만)
```

### 학습 결과 확인

학습이 완료되면 다음 파일들이 생성됩니다:

```
models/weights/
├── reusable_classifier.pth      # Reusable 분류 모델
├── reusable_classifier.onnx     # ONNX 포맷
├── beverage_detector.pth        # Beverage 검증 모델
└── beverage_detector.onnx       # ONNX 포맷
```

## 트러블슈팅

### ❌ Container가 없거나 2개 이상

**증상**: `Skipping {filename}: No container bbox` 또는 `Multiple containers`

**해결**:
1. Label Studio에서 해당 이미지 다시 확인
2. Container bbox를 정확히 1개만 그려야 함
3. 재라벨링 후 다시 export

### ❌ 이미지 파일을 찾을 수 없음

**증상**: `FileNotFoundError: Image not found: ...`

**해결**:
1. `--image-dir` 경로 확인
2. Label Studio에서 사용한 이미지와 동일한 파일명인지 확인

### ❌ 학습 데이터가 너무 적음

**증상**: 데이터셋 크기가 너무 작아 학습이 어려움

**권장 최소 데이터 수**:
- Reusable 분류: 각 클래스별 최소 50개 이상
- Beverage 검증: 각 클래스별 최소 30개 이상

**해결**:
1. Label Studio에서 더 많은 이미지 라벨링
2. Data Augmentation 활용 (Notebook에 기본 포함)

### ❌ 클래스 불균형

**증상**: 한 클래스의 데이터가 다른 클래스보다 훨씬 많음

**해결**:
1. 소수 클래스 데이터 추가 라벨링
2. Class weighting 적용 (Notebook에서 구현 가능)
3. Oversampling/Undersampling 기법 사용

## 추가 정보

### 스크립트 도움말

```bash
.venv/bin/python ai-server/scripts/convert_labelstudio_to_dataset.py --help
```

### Label Studio 라벨링 가이드

상세한 라벨링 지침은 다음 문서 참고:
- `ai-server/label-studio/LABELING_GUIDE.md`
- `ai-server/label-studio/LABELING_INSTRUCTIONS.html`

### 데이터셋 버전 관리

ZIP 파일에는 타임스탬프가 포함되어 있어 여러 버전 관리 가능:

```bash
dataset_reusable_20251110_020249.zip
dataset_reusable_20251111_143052.zip  # 새로운 버전
dataset_reusable_20251112_091234.zip  # 더 새로운 버전
```

### 성능 최적화

대량의 이미지 처리 시:
- 병렬 처리: 스크립트는 자동으로 멀티프로세싱 사용
- 메모리 절약: 이미지는 스트리밍 방식으로 처리
- 진행 상황: 10개 단위로 진행률 출력

## 문의

문제가 지속되면 이슈 등록:
- GitHub Issues: `https://github.com/your-repo/issues`
