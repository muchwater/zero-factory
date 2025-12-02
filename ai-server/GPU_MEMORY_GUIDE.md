# GPU 메모리 관리 가이드

RTX 2060 (6GB VRAM)에서 모델 학습 시 메모리 부족 문제 해결 방법

## 현재 상황

- **GPU**: NVIDIA GeForce RTX 2060
- **VRAM**: 6GB (5.6GB 사용 가능)
- **문제**: CUDA out of memory 오류

## 즉시 해결 방법

### 1. Jupyter 커널 재시작
노트북에서:
- Kernel → Restart Kernel
- 또는 상단의 ⟳ 버튼 클릭

### 2. GPU 메모리 정리 셀 실행
노트북의 "GPU 메모리 관리" 섹션에서:
```python
print_gpu_memory()
clear_gpu_memory()
```

### 3. 다른 GPU 프로세스 종료
```bash
# GPU 사용 프로세스 확인
nvidia-smi

# 불필요한 프로세스 종료
kill -9 <PID>
```

## 노트북 설정 최적화

이미 다음과 같이 최적화되었습니다:

### 변경된 설정
```python
# 기존
BATCH_SIZE = 32
num_workers = 4

# 최적화
BATCH_SIZE = 16  # GPU 메모리 절약
num_workers = 2  # CPU 부하 감소
```

## 추가 최적화 방법

### 방법 1: Batch Size 더 줄이기
```python
BATCH_SIZE = 8  # 또는 4
```

**장점**: 메모리 사용량 대폭 감소
**단점**: 학습 시간 증가, 수렴 느릴 수 있음

### 방법 2: 이미지 크기 줄이기
```python
IMG_SIZE = 192  # 기본 224에서 감소
# 또는
IMG_SIZE = 160
```

**장점**: 메모리 사용량 감소
**단점**: 모델 정확도 약간 감소 가능

### 방법 3: Mixed Precision Training (FP16)
노트북의 모델 정의 후 추가:
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# 학습 루프에서
for images, labels in train_loader:
    optimizer.zero_grad()

    # Mixed precision
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**장점**: 메모리 50% 절약, 속도 향상
**단점**: 구현 복잡도 증가

### 방법 4: Gradient Accumulation
작은 배치로 여러 번 학습하여 큰 배치 효과:
```python
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2  # 실질적으로 batch_size=16

optimizer.zero_grad()
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / ACCUMULATION_STEPS
    loss.backward()

    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 방법 5: 더 작은 모델 사용
```python
# EfficientNet-B0 대신 더 작은 모델
from torchvision.models import mobilenet_v3_small

model = mobilenet_v3_small(pretrained=True)
# ... 분류 헤드 수정
```

## 메모리 모니터링

### 학습 시작 전
```python
from notebooks.gpu_memory_utils import print_gpu_memory

print_gpu_memory()
```

출력 예시:
```
============================================================
GPU 0: NVIDIA GeForce RTX 2060
============================================================
Total Memory:     6144.00 MB
Allocated:        500.00 MB (8.1%)
Reserved:         800.00 MB (13.0%)
Free:             5644.00 MB (91.9%)
============================================================
```

### 학습 중 모니터링
```python
from notebooks.gpu_memory_utils import monitor_gpu_memory

@monitor_gpu_memory
def train_epoch(model, dataloader, criterion, optimizer, device):
    # ... 학습 코드
    pass
```

## 문제 해결 체크리스트

### OOM 오류 발생 시

1. **[ ]** Jupyter 커널 재시작
2. **[ ]** `clear_gpu_memory()` 실행
3. **[ ]** 다른 GPU 프로세스 확인 (`nvidia-smi`)
4. **[ ]** BATCH_SIZE를 16 → 8로 줄이기
5. **[ ]** num_workers를 2 → 0으로 줄이기
6. **[ ]** IMG_SIZE를 224 → 192로 줄이기
7. **[ ]** Mixed Precision Training 적용
8. **[ ]** 더 작은 모델로 변경

## 권장 설정 (RTX 2060 6GB)

### 최소 설정 (안정적)
```python
BATCH_SIZE = 8
IMG_SIZE = 192
num_workers = 0
```

### 균형 설정 (권장)
```python
BATCH_SIZE = 16
IMG_SIZE = 224
num_workers = 2
```

### 최대 설정 (다른 프로세스 없을 때)
```python
BATCH_SIZE = 24
IMG_SIZE = 224
num_workers = 4
```

## GPU 메모리 사용량 예상

| 설정 | 예상 메모리 | 학습 가능 여부 |
|------|------------|--------------|
| Batch=8, Size=192 | ~2.5GB | ✅ 안전 |
| Batch=16, Size=192 | ~3.5GB | ✅ 안전 |
| Batch=16, Size=224 | ~4.5GB | ✅ 권장 |
| Batch=24, Size=224 | ~6.0GB | ⚠️ 위험 |
| Batch=32, Size=224 | ~7.5GB | ❌ 불가능 |

## 시스템 GPU 프로세스 관리

### 브라우저 GPU 가속 비활성화
Chrome/Edge에서:
1. `chrome://settings/system`
2. "하드웨어 가속 사용" 비활성화

### VSCode GPU 가속 비활성화
```json
// settings.json
{
  "terminal.integrated.gpuAcceleration": "off"
}
```

### 그래픽 세션 최소화
필요 시:
```bash
# 터미널에서만 작업
sudo systemctl stop gdm3  # GUI 중지
# 학습 완료 후
sudo systemctl start gdm3  # GUI 재시작
```

## 디버깅 명령어

### GPU 메모리 실시간 모니터링
```bash
watch -n 1 nvidia-smi
```

### 특정 프로세스 GPU 메모리 확인
```bash
nvidia-smi pmon -i 0 -s um
```

### 모든 Python 프로세스 GPU 메모리 해제
```bash
# 주의: Jupyter 커널도 종료됨!
pkill -9 python
```

## 학습 전략

### 단계적 학습
1. **Phase 1**: 작은 설정으로 테스트
   - BATCH_SIZE=8, EPOCHS=5
   - 오류 없는지 확인

2. **Phase 2**: 설정 증가
   - BATCH_SIZE=16, EPOCHS=10
   - 메모리 사용량 모니터링

3. **Phase 3**: 전체 학습
   - 최적 설정으로 EPOCHS=50

## 참고 자료

- [PyTorch GPU 메모리 관리](https://pytorch.org/docs/stable/notes/cuda.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA GPU 최적화 가이드](https://docs.nvidia.com/deeplearning/performance/index.html)
