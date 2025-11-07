#!/usr/bin/env python3
"""
Label Studio 어노테이션을 학습 데이터셋으로 변환하는 스크립트

사용법:
    python convert_labelstudio_to_dataset.py export.json
"""

import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
import argparse


def parse_labelstudio_json(json_path: str, base_image_dir: str = None):
    """Label Studio JSON을 파싱"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for item in data:
        if not item.get('annotations'):
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

        # 라벨 추출
        labels = {}
        for r in result:
            choice_name = r.get('from_name')
            if r.get('type') == 'choices':
                labels[choice_name] = r['value']['choices'][0]

        results.append({
            'image': image_filename,
            'image_path': image_path,
            'labels': labels,
            'annotation_id': annotation.get('id')
        })

    return results


def create_classification_dataset(results, output_dir: str, task: str):
    """분류 데이터셋 생성

    Args:
        results: 파싱된 결과
        output_dir: 출력 디렉토리
        task: 'reusable' 또는 'beverage'
    """
    output_path = Path(output_dir)

    if task == 'reusable':
        label_field = 'container_type'
        train_dir = output_path / 'reusable_classification' / 'train'
        val_dir = output_path / 'reusable_classification' / 'val'
        classes = ['reusable', 'disposable']
    elif task == 'beverage':
        label_field = 'beverage_status'
        train_dir = output_path / 'beverage_detection' / 'train'
        val_dir = output_path / 'beverage_detection' / 'val'
        classes = ['with_beverage', 'without_beverage']
        # Label Studio의 'has_beverage' -> 'with_beverage', 'empty' -> 'without_beverage' 변환
        label_mapping = {
            'has_beverage': 'with_beverage',
            'empty': 'without_beverage'
        }
    else:
        raise ValueError(f"Unknown task: {task}")

    # 디렉토리 생성
    for class_name in classes:
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)

    # 데이터 분할 (80% train, 20% val)
    split_idx = int(len(results) * 0.8)
    train_results = results[:split_idx]
    val_results = results[split_idx:]

    stats = {'train': {c: 0 for c in classes}, 'val': {c: 0 for c in classes}}

    # Train 데이터 복사
    for item in train_results:
        label = item['labels'].get(label_field)
        if label and label != 'unclear':
            if task == 'beverage' and label in label_mapping:
                label = label_mapping[label]

            if label in classes:
                # 이미지 파일 복사 (실제 구현 시 source path 필요)
                dest_path = train_dir / label / item['image']
                # shutil.copy(source_path, dest_path)
                stats['train'][label] += 1
                print(f"Train: {item['image']} -> {label}")

    # Validation 데이터 복사
    for item in val_results:
        label = item['labels'].get(label_field)
        if label and label != 'unclear':
            if task == 'beverage' and label in label_mapping:
                label = label_mapping[label]

            if label in classes:
                dest_path = val_dir / label / item['image']
                # shutil.copy(source_path, dest_path)
                stats['val'][label] += 1
                print(f"Val: {item['image']} -> {label}")

    print(f"\n=== {task.upper()} Dataset Statistics ===")
    print(f"Train: {stats['train']}")
    print(f"Val: {stats['val']}")
    print(f"Total: {sum(stats['train'].values()) + sum(stats['val'].values())} images")


def create_metadata_json(results, output_file: str):
    """메타데이터 JSON 생성 (임베딩 시스템용)"""
    metadata = []

    for item in results:
        metadata.append({
            'image': item['image'],
            'container_type': item['labels'].get('container_type'),
            'beverage_status': item['labels'].get('beverage_status'),
            'quality': item['labels'].get('quality', []),
            'notes': item['labels'].get('notes', '')
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata saved to: {output_file}")
    print(f"Total images: {len(metadata)}")


def main():
    parser = argparse.ArgumentParser(description='Convert Label Studio annotations to training dataset')
    parser.add_argument('json_file', help='Label Studio export JSON file')
    parser.add_argument('--output-dir', default='../data', help='Output directory')
    parser.add_argument('--task', choices=['reusable', 'beverage', 'both', 'metadata'],
                       default='both', help='Which dataset to create')

    args = parser.parse_args()

    # JSON 파싱
    print(f"Parsing {args.json_file}...")
    results = parse_labelstudio_json(args.json_file)
    print(f"Found {len(results)} annotated images")

    # 데이터셋 생성
    if args.task in ['reusable', 'both']:
        print("\n=== Creating Reusable Container Classification Dataset ===")
        create_classification_dataset(results, args.output_dir, 'reusable')

    if args.task in ['beverage', 'both']:
        print("\n=== Creating Beverage Detection Dataset ===")
        create_classification_dataset(results, args.output_dir, 'beverage')

    if args.task in ['metadata', 'both']:
        print("\n=== Creating Metadata JSON ===")
        metadata_file = Path(args.output_dir) / 'annotations_metadata.json'
        create_metadata_json(results, str(metadata_file))

    print("\n✅ Conversion complete!")
    print("\nNext steps:")
    print("1. Review the generated dataset structure")
    print("2. Copy actual image files to the directories")
    print("3. Run training notebooks in ai-server/notebooks/")


if __name__ == '__main__':
    main()
