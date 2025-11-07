# 제로팩토리 라벨링 가이드

## 프로젝트 설정 템플릿

### 통합 어노테이션 템플릿 (권장)

```xml
<View style="display: flex; gap: 20px;">
  <View style="flex: 0 0 40%; max-width: 40%;">
    <Image name="image" value="$image" zoom="true" rotateControl="true" brightnessControl="true" contrastControl="true"/>
  </View>

  <View style="flex: 1; overflow-y: auto; max-height: 85vh; padding-right: 10px;">

  <!-- 1단계: 다회용기 분류 -->
  <Header value="1. 용기 분류"/>
  <Choices name="container_type" toName="image" choice="single-radio" required="true">
    <Choice value="reusable" background="#28a745" hint="다회용기 (텀블러, 유리컵 등)"/>
    <Choice value="disposable" background="#dc3545" hint="일회용기 (종이컵, 플라스틱컵 등)"/>
    <Choice value="unclear" background="#6c757d" hint="불분명"/>
  </Choices>

  <!-- 2단계: 음료 유무 확인 -->
  <Header value="2. 음료 유무"/>
  <Choices name="beverage_status" toName="image" choice="single-radio" required="true">
    <Choice value="has_beverage" background="#007bff" hint="음료가 담겨있음"/>
    <Choice value="empty" background="#ffc107" hint="빈 용기"/>
    <Choice value="unclear" background="#6c757d" hint="불분명"/>
  </Choices>

  <!-- 3단계: 뚜껑 유무 확인 -->
  <Header value="3. 뚜껑 유무"/>
  <Choices name="lid_status" toName="image" choice="single-radio" required="true">
    <Choice value="has_lid" background="#6f42c1" hint="뚜껑 있음"/>
    <Choice value="no_lid" background="#fd7e14" hint="뚜껑 없음"/>
    <Choice value="unclear" background="#6c757d" hint="불분명"/>
  </Choices>

  <!-- 4단계: 객체 탐지 (선택사항) -->
  <Header value="4. 객체 위치 표시 (선택사항)"/>
  <RectangleLabels name="objects" toName="image" strokeWidth="3">
    <Label value="container" background="#17a2b8" hint="용기 전체"/>
    <Label value="beverage_surface" background="#28a745" hint="음료 표면"/>
    <Label value="lid" background="#ffc107" hint="뚜껑"/>
  </RectangleLabels>

  <!-- 메모 -->
  <TextArea name="notes" toName="image" placeholder="특이사항이나 애매한 케이스 설명..." rows="2"/>
  </View>
</View>
```

## 라벨링 가이드라인

### 1. 다회용기 분류

**다회용기 (reusable)**:
- 텀블러, 머그컵, 유리컵
- 스테인리스 컵, 보온병
- 재사용 가능한 플라스틱 컵

**일회용기 (disposable)**:
- 종이컵, 플라스틱컵 (투명/불투명)
- 테이크아웃 컵
- 1회용 용기

**불분명 (unclear)**:
- 사진이 흐림
- 각도가 좋지 않아 판단 어려움
- 가려져서 잘 안보임

### 2. 음료 유무

**음료 있음 (has_beverage)**:
- 음료가 명확히 보임
- 컵에 음료가 담겨있음
- 음료 표면이 보임

**빈 용기 (empty)**:
- 빈 컵
- 음료가 전혀 없음

**불분명 (unclear)**:
- 뚜껑이 있어서 내부가 안보임
- 각도 때문에 확인 어려움

### 3. 뚜껑 유무

**뚜껑 있음 (has_lid)**:
- 뚜껑이 명확히 보임
- 플라스틱/금속/실리콘 뚜껑 모두 포함
- 일부만 보여도 뚜껑이 있다고 판단

**뚜껑 없음 (no_lid)**:
- 뚜껑이 전혀 없음
- 용기 입구가 완전히 열려있음

**불분명 (unclear)**:
- 각도 때문에 뚜껑 유무 확인 어려움
- 일부분이 가려져서 판단 불가

### 4. 객체 위치 표시 (선택)

- **container**: 용기 전체를 감싸는 박스
- **beverage_surface**: 음료 표면 (있는 경우만)
- **lid**: 뚜껑 (있는 경우만)

## 단축키

- **Space**: 다음 이미지
- **Ctrl + Enter**: 제출
- **숫자 1-9**: 빠른 라벨 선택
- **Ctrl + Z**: 실행 취소

## 데이터 내보내기 및 변환

### 1. Label Studio에서 내보내기

1. 프로젝트 페이지에서 **Export** 클릭
2. **JSON** 선택
3. `export.json` 저장

### 2. 학습 데이터셋으로 변환

```bash
cd /home/ubuntu/zero-factory/ai-server/scripts

# 모든 데이터셋 생성
python convert_labelstudio_to_dataset.py export.json --task both

# 다회용기 분류 데이터셋만
python convert_labelstudio_to_dataset.py export.json --task reusable

# 음료 검증 데이터셋만
python convert_labelstudio_to_dataset.py export.json --task beverage

# 메타데이터만
python convert_labelstudio_to_dataset.py export.json --task metadata
```

### 3. 생성되는 디렉토리 구조

```
data/
├── reusable_classification/
│   ├── train/
│   │   ├── reusable/
│   │   └── disposable/
│   └── val/
│       ├── reusable/
│       └── disposable/
├── beverage_detection/
│   ├── train/
│   │   ├── with_beverage/
│   │   └── without_beverage/
│   └── val/
│       ├── with_beverage/
│       └── without_beverage/
└── annotations_metadata.json
```

## 각 시스템별 활용

### 1️⃣ 다회용기 분류 (notebooks/01_reusable_classifier.ipynb)
- `container_type` 필드 사용
- reusable vs disposable 이진 분류

### 2️⃣ 임베딩 생성 (notebooks/02_embedding_generator.ipynb)
- 모든 이미지 사용 (라벨 무관)
- 메타데이터로 검색 성능 향상

### 3️⃣ 음료 검증 (notebooks/03_beverage_detector.ipynb)
- `beverage_status` 필드 사용
- has_beverage vs empty 이진 분류

## 팁

1. **일관성**: 동일한 기준으로 라벨링
2. **주기적 검토**: 10-20개마다 이전 라벨 확인
3. **품질 마킹**: quality 필드로 이미지 품질 표시하면 학습 시 필터링 가능

## 추천 워크플로우

1. 이미지 100장 라벨링
2. Export → JSON
3. 변환 스크립트 실행
4. 간단한 모델 학습 및 테스트
5. 문제점 파악 후 가이드라인 개선
6. 반복

## 문의

라벨링 중 애매한 케이스나 질문사항은 notes 필드에 기록해주세요!
