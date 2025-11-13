#!/usr/bin/env python3
"""
Label Studio 어노테이션을 YOLO 학습 데이터셋으로 변환하는 스크립트

YOLO 포맷:
- 각 이미지마다 .txt 파일 생성
- 포맷: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
- class_id: 0=container, 1=lid

사용법:
    # 기본 사용 (80/10/10 split)
    python convert_labelstudio_to_yolo.py export.json --image-dir ./raw_images --output-dir ./yolo_dataset

    # train/val/test 비율 지정
    python convert_labelstudio_to_yolo.py export.json --image-dir ./raw_images --output-dir ./yolo_dataset --split 0.7 0.15 0.15

    # container만 학습 (lid 제외)
    python convert_labelstudio_to_yolo.py export.json --image-dir ./raw_images --output-dir ./yolo_dataset --classes container

    # 우리 파일구조에서
    python3 ./scripts/convert_labelstudio_to_yolo.py ./dataset/project-1-at-2025-11-12-05-32-de5d2a99.json --image-dir ./data/raw_images --output-dir ./data/yolo_dataset

출력 구조:
    yolo_dataset/
    ├── data.yaml              # YOLO 설정 파일
    ├── train/
    │   ├── images/
    │   │   ├── image1.png
    │   │   └── ...
    │   └── labels/
    │       ├── image1.txt
    │       └── ...
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
"""

import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
import argparse
from PIL import Image
import random
import yaml


def parse_labelstudio_for_yolo(json_path: str, target_classes: list = None):
    """Label Studio JSON을 파싱하고 YOLO 포맷으로 변환

    Args:
        json_path: Label Studio export JSON 파일 경로
        target_classes: 추출할 클래스 리스트 (None이면 ['container', 'lid'] 모두)

    Returns:
        tuple: (parsed_data, class_names, stats)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 클래스 설정
    if target_classes is None:
        target_classes = ['container', 'lid']

    class_to_id = {cls: idx for idx, cls in enumerate(target_classes)}

    results = []
    stats = {
        'total': len(data),
        'no_annotation': 0,
        'no_objects': 0,
        'valid': 0,
        'total_objects': 0,
        'objects_by_class': {cls: 0 for cls in target_classes}
    }

    for item in data:
        if not item.get('annotations'):
            stats['no_annotation'] += 1
            continue

        # 첫 번째 어노테이션 사용
        annotation = item['annotations'][0]
        result = annotation.get('result', [])

        # 이미지 경로 추출
        image_path = item['data'].get('image', '')
        if image_path.startswith('http'):
            # URL에서 파일명 추출
            image_filename = Path(urlparse(image_path).path).name
        else:
            image_filename = Path(image_path).name

        # Rectangle bbox 추출
        bboxes = []

        for r in result:
            if r.get('type') == 'rectanglelabels':
                # Rectangle bbox 추출
                label_names = r['value'].get('rectanglelabels', [])

                for label_name in label_names:
                    if label_name in target_classes:
                        # Label Studio는 percentage (0-100)로 저장
                        # YOLO는 normalized (0-1) center format 필요
                        x_percent = r['value']['x']
                        y_percent = r['value']['y']
                        w_percent = r['value']['width']
                        h_percent = r['value']['height']

                        # Center coordinates로 변환 (percentage)
                        x_center = x_percent + w_percent / 2
                        y_center = y_percent + h_percent / 2

                        # 0-1로 normalize
                        x_center_norm = x_center / 100
                        y_center_norm = y_center / 100
                        w_norm = w_percent / 100
                        h_norm = h_percent / 100

                        bboxes.append({
                            'class': label_name,
                            'class_id': class_to_id[label_name],
                            'x_center': x_center_norm,
                            'y_center': y_center_norm,
                            'width': w_norm,
                            'height': h_norm
                        })

                        stats['objects_by_class'][label_name] += 1
                        stats['total_objects'] += 1

        # 객체가 없는 이미지는 제외
        if len(bboxes) == 0:
            stats['no_objects'] += 1
            print(f"⚠️  Skipping {image_filename}: No target objects ({target_classes})")
            continue

        stats['valid'] += 1
        results.append({
            'image': image_filename,
            'image_path': image_path,
            'bboxes': bboxes
        })

    print(f"\n=== Parsing Statistics ===")
    print(f"Total images: {stats['total']}")
    print(f"No annotation: {stats['no_annotation']}")
    print(f"No target objects: {stats['no_objects']}")
    print(f"Valid images: {stats['valid']}")
    print(f"\nTotal objects: {stats['total_objects']}")
    for cls in target_classes:
        print(f"  {cls}: {stats['objects_by_class'][cls]}")

    return results, target_classes, stats


def split_dataset(data: list, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """데이터를 train/val/test로 분할

    Args:
        data: 파싱된 데이터 리스트
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        seed: 랜덤 시드

    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)

    total = len(data_copy)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': data_copy[:train_end],
        'val': data_copy[train_end:val_end],
        'test': data_copy[val_end:]
    }

    print(f"\n=== Dataset Split ===")
    print(f"Train: {len(splits['train'])} images ({len(splits['train'])/total*100:.1f}%)")
    print(f"Val: {len(splits['val'])} images ({len(splits['val'])/total*100:.1f}%)")
    print(f"Test: {len(splits['test'])} images ({len(splits['test'])/total*100:.1f}%)")

    return splits


def create_yolo_dataset(splits: dict, class_names: list, output_dir: str, image_dir: str):
    """YOLO 데이터셋 생성

    Args:
        splits: split_dataset()의 출력
        class_names: 클래스 이름 리스트
        output_dir: 출력 디렉토리
        image_dir: 원본 이미지 디렉토리
    """
    output_path = Path(output_dir)
    image_dir_path = Path(image_dir)

    stats = {split: {'images': 0, 'labels': 0, 'objects': 0, 'failed': []}
             for split in ['train', 'val', 'test']}

    for split_name, split_data in splits.items():
        if len(split_data) == 0:
            print(f"\n⚠️  Skipping {split_name}: No data")
            continue

        # 디렉토리 생성
        images_dir = output_path / split_name / 'images'
        labels_dir = output_path / split_name / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Creating {split_name.upper()} dataset ===")

        for idx, item in enumerate(split_data):
            try:
                # 이미지 복사
                src_image = image_dir_path / item['image']
                if not src_image.exists():
                    raise FileNotFoundError(f"Image not found: {src_image}")

                dst_image = images_dir / item['image']
                shutil.copy2(src_image, dst_image)
                stats[split_name]['images'] += 1

                # 라벨 파일 생성 (YOLO 포맷)
                label_file = labels_dir / f"{src_image.stem}.txt"
                with open(label_file, 'w') as f:
                    for bbox in item['bboxes']:
                        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                        line = f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
                        f.write(line)
                        stats[split_name]['objects'] += 1

                stats[split_name]['labels'] += 1

                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(split_data)} images...")

            except Exception as e:
                stats[split_name]['failed'].append(item['image'])
                print(f"❌ Failed to process {item['image']}: {e}")

        print(f"✅ {split_name}: {stats[split_name]['images']} images, {stats[split_name]['labels']} labels, {stats[split_name]['objects']} objects")
        if stats[split_name]['failed']:
            print(f"⚠️  Failed: {len(stats[split_name]['failed'])} images")

    return stats


def create_yaml_config(class_names: list, output_dir: str, dataset_name: str = "Zero Factory Cup Detection"):
    """YOLO data.yaml 설정 파일 생성

    Args:
        class_names: 클래스 이름 리스트
        output_dir: 출력 디렉토리
        dataset_name: 데이터셋 이름
    """
    output_path = Path(output_dir)

    # 절대 경로로 변환
    abs_output_path = output_path.resolve()

    config = {
        'path': str(abs_output_path),  # 데이터셋 루트 경로
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),  # number of classes
        'names': class_names  # class names
    }

    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"\n=== YOLO Config ===")
    print(f"Created: {yaml_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"\nYou can train with:")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")
    print(f"  # or")
    print(f"  python -m ultralytics.yolo.detect.train data={yaml_path} model=yolov8n.pt")


def create_dataset_info(stats: dict, output_dir: str, class_names: list):
    """데이터셋 통계 정보를 JSON으로 저장

    Args:
        stats: create_yolo_dataset()의 출력
        output_dir: 출력 디렉토리
        class_names: 클래스 이름 리스트
    """
    output_path = Path(output_dir)

    info = {
        'dataset_name': 'Zero Factory Cup Detection Dataset',
        'classes': class_names,
        'num_classes': len(class_names),
        'splits': {
            split: {
                'num_images': stats[split]['images'],
                'num_labels': stats[split]['labels'],
                'num_objects': stats[split]['objects'],
                'failed': stats[split]['failed']
            }
            for split in ['train', 'val', 'test']
        },
        'total_images': sum(stats[s]['images'] for s in ['train', 'val', 'test']),
        'total_objects': sum(stats[s]['objects'] for s in ['train', 'val', 'test'])
    }

    info_path = output_path / 'dataset_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nDataset info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Label Studio annotations to YOLO format dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (80/10/10 split)
  python convert_labelstudio_to_yolo.py export.json --image-dir ./images --output-dir ./yolo_dataset

  # Custom split ratios
  python convert_labelstudio_to_yolo.py export.json --image-dir ./images --output-dir ./yolo_dataset --split 0.7 0.15 0.15

  # Only detect containers (exclude lids)
  python convert_labelstudio_to_yolo.py export.json --image-dir ./images --output-dir ./yolo_dataset --classes container

  # For our project structure
  python3 ./scripts/convert_labelstudio_to_yolo.py ./dataset/project-1-at-2025-11-12-05-32-de5d2a99.json --image-dir ./data/raw_images --output-dir ./data/yolo_dataset

Training:
  cd ./data/yolo_dataset
  yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
        """
    )

    parser.add_argument('json_file', help='Label Studio export JSON file')
    parser.add_argument('--image-dir', required=True, help='Directory containing original images')
    parser.add_argument('--output-dir', required=True, help='Output directory for YOLO dataset')
    parser.add_argument('--classes', nargs='+', default=['container', 'lid'],
                       help='Classes to include (default: container lid)')
    parser.add_argument('--split', nargs=3, type=float, metavar=('TRAIN', 'VAL', 'TEST'),
                       default=[0.8, 0.1, 0.1],
                       help='Train/Val/Test split ratios (default: 0.8 0.1 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    # 검증
    if abs(sum(args.split) - 1.0) > 1e-6:
        parser.error("Split ratios must sum to 1.0")

    print("="*60)
    print("Label Studio to YOLO Converter")
    print("="*60)

    # 1. JSON 파싱
    print(f"\nParsing {args.json_file}...")
    results, class_names, parse_stats = parse_labelstudio_for_yolo(args.json_file, args.classes)

    if not results:
        print("\n❌ No valid data found. Please check:")
        print("  1. Images have bbox annotations")
        print(f"  2. Target classes exist: {args.classes}")
        return

    print(f"\n✅ Found {len(results)} valid images")

    # 2. 데이터 분할
    print("\n" + "="*60)
    print("Splitting dataset...")
    print("="*60)
    splits = split_dataset(results, args.split[0], args.split[1], args.split[2], args.seed)

    # 3. YOLO 데이터셋 생성
    print("\n" + "="*60)
    print("Creating YOLO dataset...")
    print("="*60)
    dataset_stats = create_yolo_dataset(splits, class_names, args.output_dir, args.image_dir)

    # 4. data.yaml 생성
    print("\n" + "="*60)
    print("Creating YOLO configuration...")
    print("="*60)
    create_yaml_config(class_names, args.output_dir)

    # 5. 통계 정보 저장
    create_dataset_info(dataset_stats, args.output_dir, class_names)

    # 최종 요약
    print("\n" + "="*60)
    print("✅ YOLO Dataset Created Successfully!")
    print("="*60)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Total images: {sum(dataset_stats[s]['images'] for s in ['train', 'val', 'test'])}")
    print(f"Total objects: {sum(dataset_stats[s]['objects'] for s in ['train', 'val', 'test'])}")
    print(f"\nNext steps:")
    print(f"  1. Review the dataset: ls {args.output_dir}")
    print(f"  2. Train YOLO model:")
    print(f"     yolo detect train data={args.output_dir}/data.yaml model=yolov8n.pt epochs=100 imgsz=640")


if __name__ == '__main__':
    main()
