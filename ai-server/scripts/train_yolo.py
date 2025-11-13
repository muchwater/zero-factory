#!/usr/bin/env python3
"""
YOLO 모델 학습 스크립트

사용법:
    # 기본 학습 (YOLOv8n, 100 epochs)
    python train_yolo.py --data ./data/yolo_dataset/data.yaml

    # 모델 크기 선택 (n/s/m/l/x)
    python train_yolo.py --data ./data/yolo_dataset/data.yaml --model yolov8s.pt

    # 에포크 및 이미지 사이즈 설정
    python train_yolo.py --data ./data/yolo_dataset/data.yaml --epochs 200 --imgsz 640

    # GPU 선택
    python train_yolo.py --data ./data/yolo_dataset/data.yaml --device 0

    # 배치 사이즈 설정 (자동 조정도 가능)
    python train_yolo.py --data ./data/yolo_dataset/data.yaml --batch 16
    python train_yolo.py --data ./data/yolo_dataset/data.yaml --batch -1  # auto

    # 프로젝트 이름 및 실험 이름 지정
    python train_yolo.py --data ./data/yolo_dataset/data.yaml --project runs/detect --name cup_detection_v1

    # Resume training from checkpoint
    python train_yolo.py --resume runs/detect/cup_detection_v1/weights/last.pt

모델 크기:
    - yolov8n.pt: Nano (가장 빠름, 정확도 낮음)
    - yolov8s.pt: Small (균형잡힌 성능)
    - yolov8m.pt: Medium (높은 정확도)
    - yolov8l.pt: Large (매우 높은 정확도)
    - yolov8x.pt: XLarge (최고 정확도, 가장 느림)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch


def print_system_info():
    """시스템 정보 출력"""
    print("\n=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()


def validate_data_yaml(data_path: str):
    """data.yaml 파일 검증"""
    data_file = Path(data_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_file, 'r') as f:
        data_config = yaml.safe_load(f)

    print("\n=== Dataset Configuration ===")
    print(f"Path: {data_config.get('path')}")
    print(f"Train: {data_config.get('train')}")
    print(f"Val: {data_config.get('val')}")
    print(f"Test: {data_config.get('test')}")
    print(f"Classes ({data_config.get('nc')}): {data_config.get('names')}")

    # 경로 검증
    dataset_path = Path(data_config['path'])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    train_path = dataset_path / data_config['train']
    val_path = dataset_path / data_config['val']

    if not train_path.exists():
        raise FileNotFoundError(f"Train path not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val path not found: {val_path}")

    train_images = list(train_path.glob('*.*'))
    val_images = list(val_path.glob('*.*'))

    print(f"\nTrain images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print()

    return data_config


def train_yolo(
    data: str,
    model: str = 'yolov8n.pt',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = '0',
    project: str = 'runs/detect',
    name: str = 'cup_detection',
    resume: bool = False,
    patience: int = 50,
    save_period: int = 10,
    pretrained: bool = True,
    optimizer: str = 'auto',
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    augment: bool = True,
    **kwargs
):
    """YOLO 모델 학습

    Args:
        data: data.yaml 파일 경로
        model: 사전학습 모델 (yolov8n.pt, yolov8s.pt, etc.)
        epochs: 학습 에포크 수
        imgsz: 입력 이미지 크기
        batch: 배치 사이즈 (-1이면 자동)
        device: GPU 디바이스 ('0', '0,1', 'cpu')
        project: 프로젝트 디렉토리
        name: 실험 이름
        resume: 체크포인트에서 재개
        patience: Early stopping patience
        save_period: 체크포인트 저장 주기
        pretrained: 사전학습 가중치 사용
        optimizer: 옵티마이저 ('SGD', 'Adam', 'AdamW', 'auto')
        lr0: 초기 learning rate
        lrf: 최종 learning rate (lr0 * lrf)
        momentum: SGD momentum / Adam beta1
        weight_decay: Optimizer weight decay
        augment: 데이터 증강 사용
        **kwargs: 추가 학습 파라미터
    """

    # 시스템 정보 출력
    print_system_info()

    # 데이터셋 검증
    data_config = validate_data_yaml(data)

    # 모델 로드
    print(f"=== Loading Model: {model} ===")
    yolo = YOLO(model)

    if pretrained:
        print(f"Using pretrained weights from {model}")
    else:
        print(f"Training from scratch")

    # 학습 시작
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    print(f"Model: {model}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print(f"Optimizer: {optimizer}")
    print(f"Learning rate: {lr0} -> {lr0 * lrf}")
    print(f"Augmentation: {augment}")
    print(f"Output: {project}/{name}")
    print("="*60 + "\n")

    # 학습
    results = yolo.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        patience=patience,
        save_period=save_period,
        pretrained=pretrained,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        augment=augment,
        verbose=True,
        **kwargs
    )

    print("\n" + "="*60)
    print("✅ Training Completed!")
    print("="*60)

    # 결과 출력
    output_dir = Path(project) / name
    print(f"\nOutput directory: {output_dir}")
    print(f"  Weights: {output_dir / 'weights'}")
    print(f"    - best.pt: Best model")
    print(f"    - last.pt: Last checkpoint")
    print(f"  Results: {output_dir / 'results.csv'}")
    print(f"  Plots: {output_dir}/*.png")

    return results


def validate_model(weights: str, data: str, imgsz: int = 640, batch: int = 16, device: str = '0'):
    """학습된 모델 검증

    Args:
        weights: 모델 가중치 경로
        data: data.yaml 파일 경로
        imgsz: 입력 이미지 크기
        batch: 배치 사이즈
        device: GPU 디바이스
    """
    print("\n" + "="*60)
    print("Validating Model...")
    print("="*60)

    model = YOLO(weights)
    results = model.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device
    )

    print("\n=== Validation Results ===")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO model for cup detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_yolo.py --data ./data/yolo_dataset/data.yaml

  # Use larger model and more epochs
  python train_yolo.py --data ./data/yolo_dataset/data.yaml --model yolov8s.pt --epochs 200

  # Custom batch size and image size
  python train_yolo.py --data ./data/yolo_dataset/data.yaml --batch 32 --imgsz 640

  # Auto batch size (recommended)
  python train_yolo.py --data ./data/yolo_dataset/data.yaml --batch -1

  # Use multiple GPUs
  python train_yolo.py --data ./data/yolo_dataset/data.yaml --device 0,1

  # Resume from checkpoint
  python train_yolo.py --resume runs/detect/cup_detection/weights/last.pt

After training:
  # Validate the model
  python train_yolo.py --validate runs/detect/cup_detection/weights/best.pt --data ./data/yolo_dataset/data.yaml
        """
    )

    # Training arguments
    parser.add_argument('--data', type=str, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='Model size (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size, -1 for auto (default: 16)')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (0, 0,1, cpu) (default: 0)')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory (default: runs/detect)')
    parser.add_argument('--name', type=str, default='cup_detection', help='Experiment name (default: cup_detection)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (default: 50)')
    parser.add_argument('--save-period', type=int, default=10, help='Save checkpoint every N epochs (default: 10)')

    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='auto',
                       choices=['SGD', 'Adam', 'AdamW', 'auto'],
                       help='Optimizer (default: auto)')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate (default: 0.01)')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum (default: 0.937)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay (default: 0.0005)')

    # Data augmentation
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--hsv-h', type=float, default=0.015, help='HSV-Hue augmentation')
    parser.add_argument('--hsv-s', type=float, default=0.7, help='HSV-Saturation augmentation')
    parser.add_argument('--hsv-v', type=float, default=0.4, help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotation degrees')
    parser.add_argument('--translate', type=float, default=0.1, help='Translation fraction')
    parser.add_argument('--scale', type=float, default=0.5, help='Scaling factor')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Horizontal flip probability')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic augmentation probability')

    # Validation
    parser.add_argument('--validate', type=str, help='Validate model weights')

    args = parser.parse_args()

    # Validation mode
    if args.validate:
        if not args.data:
            parser.error("--data is required for validation")
        validate_model(args.validate, args.data, args.imgsz, args.batch, args.device)
        return

    # Training mode
    if args.resume:
        # Resume from checkpoint
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        # Start new training
        if not args.data:
            parser.error("--data is required for training")

        train_yolo(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            save_period=args.save_period,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            augment=not args.no_augment,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            fliplr=args.fliplr,
            mosaic=args.mosaic
        )


if __name__ == '__main__':
    main()
