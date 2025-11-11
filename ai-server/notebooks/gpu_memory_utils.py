"""
GPU 메모리 관리 유틸리티

사용법:
    from gpu_memory_utils import clear_gpu_memory, print_gpu_memory, monitor_gpu_memory

    # GPU 메모리 정리
    clear_gpu_memory()

    # GPU 메모리 상태 확인
    print_gpu_memory()

    # 자동 모니터링 (데코레이터)
    @monitor_gpu_memory
    def train_model():
        ...
"""

import torch
import gc
import os


def clear_gpu_memory(verbose=True):
    """
    GPU 메모리를 완전히 정리합니다.

    Args:
        verbose (bool): 정리 전후 메모리 상태 출력 여부

    Returns:
        dict: 정리 전후 메모리 정보
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available. No GPU memory to clear.")
        return None

    # 정리 전 메모리 상태
    before_allocated = torch.cuda.memory_allocated(0) / 1024**2
    before_reserved = torch.cuda.memory_reserved(0) / 1024**2

    if verbose:
        print("=" * 60)
        print("GPU Memory Cleanup")
        print("=" * 60)
        print(f"Before cleanup:")
        print(f"  Allocated: {before_allocated:.2f} MB")
        print(f"  Reserved:  {before_reserved:.2f} MB")

    # 1. Python garbage collection
    gc.collect()

    # 2. PyTorch 캐시 정리
    torch.cuda.empty_cache()

    # 3. 모든 CUDA 스트림 동기화
    torch.cuda.synchronize()

    # 4. 가능하면 메모리 통계 리셋
    try:
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.reset_accumulated_memory_stats(0)
    except:
        pass

    # 정리 후 메모리 상태
    after_allocated = torch.cuda.memory_allocated(0) / 1024**2
    after_reserved = torch.cuda.memory_reserved(0) / 1024**2

    freed_allocated = before_allocated - after_allocated
    freed_reserved = before_reserved - after_reserved

    if verbose:
        print(f"\nAfter cleanup:")
        print(f"  Allocated: {after_allocated:.2f} MB (freed: {freed_allocated:.2f} MB)")
        print(f"  Reserved:  {after_reserved:.2f} MB (freed: {freed_reserved:.2f} MB)")
        print("=" * 60)

    return {
        'before': {'allocated': before_allocated, 'reserved': before_reserved},
        'after': {'allocated': after_allocated, 'reserved': after_reserved},
        'freed': {'allocated': freed_allocated, 'reserved': freed_reserved}
    }


def print_gpu_memory(device_id=0):
    """
    현재 GPU 메모리 상태를 출력합니다.

    Args:
        device_id (int): GPU 디바이스 ID (기본값: 0)
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    allocated = torch.cuda.memory_allocated(device_id) / 1024**2
    reserved = torch.cuda.memory_reserved(device_id) / 1024**2
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
    free = total - allocated

    print("=" * 60)
    print(f"GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
    print("=" * 60)
    print(f"Total Memory:     {total:.2f} MB")
    print(f"Allocated:        {allocated:.2f} MB ({allocated/total*100:.1f}%)")
    print(f"Reserved:         {reserved:.2f} MB ({reserved/total*100:.1f}%)")
    print(f"Free:             {free:.2f} MB ({free/total*100:.1f}%)")
    print("=" * 60)

    # 경고 메시지
    if allocated / total > 0.9:
        print("⚠️  WARNING: GPU memory usage is very high (>90%)")
    elif allocated / total > 0.8:
        print("⚠️  CAUTION: GPU memory usage is high (>80%)")


def get_gpu_memory_info(device_id=0):
    """
    GPU 메모리 정보를 딕셔너리로 반환합니다.

    Args:
        device_id (int): GPU 디바이스 ID

    Returns:
        dict: 메모리 정보
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated(device_id) / 1024**2
    reserved = torch.cuda.memory_reserved(device_id) / 1024**2
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
    free = total - allocated

    return {
        'device_id': device_id,
        'device_name': torch.cuda.get_device_name(device_id),
        'total_mb': total,
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'free_mb': free,
        'allocated_percent': allocated / total * 100,
        'free_percent': free / total * 100
    }


def monitor_gpu_memory(func):
    """
    함수 실행 전후 GPU 메모리를 모니터링하는 데코레이터

    사용 예:
        @monitor_gpu_memory
        def train_epoch():
            ...
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            print(f"\n[Before {func.__name__}]")
            print_gpu_memory()

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            print(f"\n[After {func.__name__}]")
            print_gpu_memory()

        return result

    return wrapper


def kill_gpu_processes():
    """
    현재 사용자의 GPU 프로세스를 종료합니다.
    (Jupyter 노트북에서는 권장하지 않음 - 커널이 종료될 수 있음)
    """
    import subprocess

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("⚠️  WARNING: This will kill all GPU processes owned by you.")
    print("This may restart your Jupyter kernel!")

    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled")
        return

    try:
        # nvidia-smi로 현재 사용자의 프로세스 찾기
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )

        pids = result.stdout.strip().split('\n')

        for pid in pids:
            if pid:
                print(f"Killing process {pid}...")
                os.system(f"kill -9 {pid}")

        print("✓ GPU processes killed")

    except Exception as e:
        print(f"Error: {e}")


def optimize_for_inference(model):
    """
    추론을 위해 모델을 최적화합니다.

    Args:
        model: PyTorch 모델

    Returns:
        model: 최적화된 모델
    """
    model.eval()

    # Gradient 계산 비활성화
    for param in model.parameters():
        param.requires_grad = False

    # float16 변환 (GPU에서만)
    if next(model.parameters()).is_cuda:
        model = model.half()
        print("✓ Model converted to float16 for inference")

    return model


# 간편 사용을 위한 별칭
clear = clear_gpu_memory
status = print_gpu_memory
info = get_gpu_memory_info


if __name__ == "__main__":
    # 테스트
    print("Testing GPU memory utilities...")
    print_gpu_memory()
    print("\nClearing GPU memory...")
    clear_gpu_memory()
