#!/usr/bin/env python3
"""
Label Studio 어노테이션을 학습 데이터셋으로 변환하는 스크립트

사용법:
    # 기본 사용 (현재 디렉토리에 ZIP 파일 생성)
    python convert_labelstudio_to_dataset.py export.json --image-dir ./images

    # 임베딩용 types/ 디렉토리 포함
    python convert_labelstudio_to_dataset.py export.json --image-dir ./images --include-types

    # 출력 디렉토리 지정
    python convert_labelstudio_to_dataset.py export.json --image-dir ./images --output-dir ./datasets

    # 출력 디렉토리와 파일명 지정
    python convert_labelstudio_to_dataset.py export.json --image-dir ./images --output-dir ./datasets --output-file my_dataset.zip

    # reusable 데이터셋만 생성
    python convert_labelstudio_to_dataset.py export.json --image-dir ./images --task reusable

    # beverage 데이터셋과 types 포함
    python convert_labelstudio_to_dataset.py export.json --image-dir ./images --task beverage --include-types

    # 우리 파일구조에서,,,
    python3 ./scripts/convert_labelstudio_to_dataset.py ./label-studio/export/project-1-at-2025-11-10-01-59-baddde76.json --image-dir ./data/raw_images/ --output-dir ./dataset_output --include-types

출력 구조:
    dataset_YYYYMMDD_HHMMSS.zip
    ├── reusable/
    │   ├── reusable/       # 다회용 용기 이미지
    │   ├── disposable/     # 일회용 용기 이미지
    │   └── unclear/        # 불분명한 이미지
    ├── beverage/
    │   ├── with_beverage/  # 음료가 있는 이미지
    │   ├── empty/          # 빈 용기 이미지
    │   └── unclear/        # 불분명한 이미지
    └── types/              # --include-types 사용 시
        ├── CUP001/         # 컵 코드별 분류 (임베딩용)
        ├── CUP002/
        └── ...
"""

import json
import shutil
from pathlib import Path
from urllib.parse import urlparse
import argparse
from PIL import Image
import requests
from io import BytesIO
import zipfile
from datetime import datetime
import tempfile


def parse_labelstudio_json(json_path: str, base_image_dir: str = None):
    """Label Studio JSON을 파싱하고 container bbox 추출

    Args:
        json_path: Label Studio export JSON 파일 경로
        base_image_dir: 이미지가 저장된 디렉토리 (optional)

    Returns:
        list: 파싱된 결과 (container bbox 포함)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    stats = {
        'total': len(data),
        'no_annotation': 0,
        'no_container': 0,
        'multiple_containers': 0,
        'valid': 0
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

        # Container bbox와 라벨 추출
        labels = {}
        containers = []

        for r in result:
            if r.get('type') == 'choices':
                choice_name = r.get('from_name')
                labels[choice_name] = r['value']['choices'][0]
            elif r.get('type') == 'rectanglelabels':
                # Rectangle bbox 추출
                label_names = r['value'].get('rectanglelabels', [])
                if 'container' in label_names:
                    containers.append({
                        'x': r['value']['x'],
                        'y': r['value']['y'],
                        'width': r['value']['width'],
                        'height': r['value']['height']
                    })

        # Container가 정확히 1개인 경우만 포함
        if len(containers) == 0:
            stats['no_container'] += 1
            print(f"⚠️  Skipping {image_filename}: No container bbox")
            continue
        elif len(containers) > 1:
            stats['multiple_containers'] += 1
            print(f"⚠️  Skipping {image_filename}: Multiple containers ({len(containers)})")
            continue

        stats['valid'] += 1
        results.append({
            'image': image_filename,
            'image_path': image_path,
            'labels': labels,
            'container_bbox': containers[0],
            'annotation_id': annotation.get('id')
        })

    print(f"\n=== Parsing Statistics ===")
    print(f"Total: {stats['total']}")
    print(f"No annotation: {stats['no_annotation']}")
    print(f"No container: {stats['no_container']}")
    print(f"Multiple containers: {stats['multiple_containers']}")
    print(f"Valid: {stats['valid']}")

    return results


def load_and_crop_image(image_path: str, bbox: dict, image_dir: str = None):
    """이미지를 로드하고 bbox로 크롭

    Args:
        image_path: 이미지 경로 (URL 또는 파일명)
        bbox: Container bbox (x, y, width, height - percentage)
        image_dir: 로컬 이미지 디렉토리

    Returns:
        PIL.Image: 크롭된 이미지
    """
    # 이미지 로드
    if image_path.startswith('http'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        if image_dir:
            full_path = Path(image_dir) / Path(image_path).name
        else:
            full_path = Path(image_path)

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        img = Image.open(full_path)

    # Label Studio bbox는 percentage로 저장됨
    img_width, img_height = img.size
    x = int(bbox['x'] * img_width / 100)
    y = int(bbox['y'] * img_height / 100)
    width = int(bbox['width'] * img_width / 100)
    height = int(bbox['height'] * img_height / 100)

    # 크롭
    cropped = img.crop((x, y, x + width, y + height))
    return cropped


def create_classification_dataset(results, output_dir: str, task: str, image_dir: str = None):
    """분류 데이터셋 생성 (container 영역으로 크롭된 이미지)

    Args:
        results: 파싱된 결과
        output_dir: 출력 디렉토리
        task: 'reusable' 또는 'beverage'
        image_dir: 원본 이미지 디렉토리
    """
    output_path = Path(output_dir)

    if task == 'reusable':
        label_field = 'container_type'
        classes = ['reusable', 'disposable', 'unclear']
    elif task == 'beverage':
        label_field = 'beverage_status'
        # Label Studio의 'has_beverage' -> 'with_beverage', 'empty' -> 'empty' 매핑
        label_mapping = {
            'has_beverage': 'with_beverage',
            'empty': 'empty',
            'unclear': 'unclear'
        }
        classes = ['with_beverage', 'empty', 'unclear']
    else:
        raise ValueError(f"Unknown task: {task}")

    # 디렉토리 생성
    for class_name in classes:
        (output_path / task / class_name).mkdir(parents=True, exist_ok=True)

    stats = {c: 0 for c in classes}
    failed = []

    # 각 이미지를 크롭하고 분류별로 저장
    for idx, item in enumerate(results):
        label = item['labels'].get(label_field)

        if not label:
            print(f"⚠️  Skipping {item['image']}: No {label_field} label")
            continue

        # Beverage task의 경우 라벨 매핑
        if task == 'beverage' and label in label_mapping:
            label = label_mapping[label]

        if label not in classes:
            print(f"⚠️  Skipping {item['image']}: Unknown label '{label}'")
            continue

        try:
            # Container 영역으로 크롭
            cropped_img = load_and_crop_image(
                item['image_path'],
                item['container_bbox'],
                image_dir
            )

            # 저장
            dest_path = output_path / task / label / item['image']
            cropped_img.save(dest_path)

            stats[label] += 1
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(results)} images...")

        except Exception as e:
            failed.append(item['image'])
            print(f"❌ Failed to process {item['image']}: {e}")

    print(f"\n=== {task.upper()} Dataset Statistics ===")
    for class_name in classes:
        print(f"{class_name}: {stats[class_name]} images")
    print(f"Total: {sum(stats.values())} images")
    if failed:
        print(f"\n⚠️  Failed: {len(failed)} images")
        print(f"Failed images: {', '.join(failed[:10])}{' ...' if len(failed) > 10 else ''}")


def create_embedding_dataset(results, output_dir: str, image_dir: str = None):
    """임베딩용 데이터셋 생성 (cup_code별로 크롭된 이미지 분류)

    Args:
        results: 파싱된 결과
        output_dir: 출력 디렉토리
        image_dir: 원본 이미지 디렉토리
    """
    output_path = Path(output_dir)

    stats = {}
    failed = []
    skipped_no_cup_code = 0

    # 각 이미지를 크롭하고 cup_code별로 저장
    for idx, item in enumerate(results):
        cup_code = item['labels'].get('cup_code')

        if not cup_code:
            skipped_no_cup_code += 1
            continue

        # cup_code 디렉토리 생성
        cup_dir = output_path / 'types' / cup_code
        cup_dir.mkdir(parents=True, exist_ok=True)

        if cup_code not in stats:
            stats[cup_code] = 0

        try:
            # Container 영역으로 크롭
            cropped_img = load_and_crop_image(
                item['image_path'],
                item['container_bbox'],
                image_dir
            )

            # 저장
            dest_path = cup_dir / item['image']
            cropped_img.save(dest_path)

            stats[cup_code] += 1
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(results)} images...")

        except Exception as e:
            failed.append(item['image'])
            print(f"❌ Failed to process {item['image']}: {e}")

    print(f"\n=== EMBEDDING Dataset Statistics ===")
    for cup_code in sorted(stats.keys()):
        print(f"{cup_code}: {stats[cup_code]} images")
    print(f"Total: {sum(stats.values())} images")
    if skipped_no_cup_code > 0:
        print(f"Skipped (no cup_code): {skipped_no_cup_code} images")
    if failed:
        print(f"\n⚠️  Failed: {len(failed)} images")
        print(f"Failed images: {', '.join(failed[:10])}{' ...' if len(failed) > 10 else ''}")


def create_metadata_json(results, output_file: str):
    """메타데이터 JSON 생성 (임베딩 시스템용)"""
    metadata = []

    for item in results:
        metadata.append({
            'image': item['image'],
            'cup_code': item['labels'].get('cup_code'),
            'container_type': item['labels'].get('container_type'),
            'beverage_status': item['labels'].get('beverage_status'),
            'lid_status': item['labels'].get('lid_status'),
            'quality': item['labels'].get('quality', []),
            'notes': item['labels'].get('notes', '')
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata saved to: {output_file}")
    print(f"Total images: {len(metadata)}")


def create_cup_code_statistics(results, output_file: str):
    """컵 코드별 통계 생성"""
    cup_code_stats = {}

    for item in results:
        cup_code = item['labels'].get('cup_code')
        if cup_code:
            if cup_code not in cup_code_stats:
                cup_code_stats[cup_code] = {
                    'count': 0,
                    'with_beverage': 0,
                    'empty': 0,
                    'unclear_beverage': 0,
                    'has_lid': 0,
                    'no_lid': 0,
                    'unclear_lid': 0
                }

            cup_code_stats[cup_code]['count'] += 1

            # 음료 유무 통계
            beverage_status = item['labels'].get('beverage_status')
            if beverage_status == 'has_beverage':
                cup_code_stats[cup_code]['with_beverage'] += 1
            elif beverage_status == 'empty':
                cup_code_stats[cup_code]['empty'] += 1
            elif beverage_status == 'unclear':
                cup_code_stats[cup_code]['unclear_beverage'] += 1

            # 뚜껑 유무 통계
            lid_status = item['labels'].get('lid_status')
            if lid_status == 'has_lid':
                cup_code_stats[cup_code]['has_lid'] += 1
            elif lid_status == 'no_lid':
                cup_code_stats[cup_code]['no_lid'] += 1
            elif lid_status == 'unclear':
                cup_code_stats[cup_code]['unclear_lid'] += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cup_code_stats, f, indent=2, ensure_ascii=False)

    print(f"\nCup code statistics saved to: {output_file}")
    print("\n=== Cup Code Statistics ===")
    for cup_code, stats in sorted(cup_code_stats.items()):
        print(f"\n{cup_code}: {stats['count']} images")
        print(f"  Beverage: {stats['with_beverage']} with / {stats['empty']} empty / {stats['unclear_beverage']} unclear")
        print(f"  Lid: {stats['has_lid']} has / {stats['no_lid']} no / {stats['unclear_lid']} unclear")


def create_zip_archive(temp_dir: str, output_dir: str, output_filename: str, task: str, include_types: bool = False):
    """임시 디렉토리의 데이터셋을 ZIP 파일로 압축

    Args:
        temp_dir: 임시 데이터셋 디렉토리
        output_dir: 최종 ZIP 파일이 저장될 디렉토리
        output_filename: ZIP 파일명 (None이면 자동 생성)
        task: 'reusable', 'beverage', 또는 'both'
        include_types: types 디렉토리 포함 여부

    Returns:
        str: 생성된 ZIP 파일 경로
    """
    temp_path = Path(temp_dir)
    output_path = Path(output_dir)

    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 압축할 task 목록 결정
    tasks_to_zip = []
    if task in ['reusable', 'both']:
        tasks_to_zip.append('reusable')
    if task in ['beverage', 'both']:
        tasks_to_zip.append('beverage')

    # types 디렉토리 추가
    if include_types and (temp_path / 'types').exists():
        tasks_to_zip.append('types')

    # 존재하는 디렉토리만 필터링
    existing_tasks = [t for t in tasks_to_zip if (temp_path / t).exists()]

    if not existing_tasks:
        return None

    # 하나의 ZIP 파일에 모든 데이터셋 포함
    zip_filename = output_filename if output_filename else f"dataset_{timestamp}.zip"
    zip_path = output_path / zip_filename

    print(f"\nCreating {zip_filename}...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for task_name in existing_tasks:
            task_dir = temp_path / task_name

            # task_dir 내의 모든 파일을 ZIP에 추가
            for file_path in task_dir.rglob('*'):
                if file_path.is_file():
                    # ZIP 내부 경로를 task_name/class/filename 형식으로
                    arcname = file_path.relative_to(temp_path)
                    zipf.write(file_path, arcname)

            print(f"  Added {task_name}/ directory")

    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Created: {zip_path.name} ({zip_size:.2f} MB)")

    return str(zip_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Label Studio annotations to training dataset (cropped by container bbox)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create both reusable and beverage datasets as ZIP
  python convert_labelstudio_to_dataset.py export.json --image-dir ./images

  # Include types/ directory for embeddings
  python convert_labelstudio_to_dataset.py export.json --image-dir ./images --include-types

  # Specify output directory
  python convert_labelstudio_to_dataset.py export.json --image-dir ./images --output-dir ./output

  # Specify output directory and file name
  python convert_labelstudio_to_dataset.py export.json --image-dir ./images --output-dir ./output --output-file my_dataset.zip

  # Create only reusable dataset
  python convert_labelstudio_to_dataset.py export.json --image-dir ./images --task reusable

  # Create only beverage dataset with types
  python convert_labelstudio_to_dataset.py export.json --image-dir ./images --task beverage --include-types

Output:
  dataset_YYYYMMDD_HHMMSS.zip (contains the following structure)
    reusable/
      reusable/       # Reusable container images (cropped)
      disposable/     # Disposable container images (cropped)
      unclear/        # Unclear container images (cropped)
    beverage/
      with_beverage/  # Container with beverage images (cropped)
      empty/          # Empty container images (cropped)
      unclear/        # Unclear beverage status images (cropped)
    types/            # Only if --include-types is used
      CUP001/         # Cup code based classification (for embeddings)
      CUP002/
      ...
        """
    )
    parser.add_argument('json_file', help='Label Studio export JSON file')
    parser.add_argument('--image-dir', required=True, help='Directory containing original images')
    parser.add_argument('--output-dir', default='.', help='Output directory for ZIP file (default: current directory)')
    parser.add_argument('--output-file', help='Output ZIP file name (default: dataset_YYYYMMDD_HHMMSS.zip)')
    parser.add_argument('--task', choices=['reusable', 'beverage', 'both'],
                       default='both', help='Which dataset to create (default: both)')
    parser.add_argument('--include-types', action='store_true',
                       help='Include types/ directory with cup_code-based classification for embeddings')

    args = parser.parse_args()

    # JSON 파싱
    print(f"Parsing {args.json_file}...")
    results = parse_labelstudio_json(args.json_file, args.image_dir)

    if not results:
        print("\n❌ No valid data found. Please check:")
        print("  1. All images have exactly one 'container' bbox annotation")
        print("  2. Annotations are properly labeled")
        return

    print(f"\n✅ Found {len(results)} valid images with container bbox")

    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 데이터셋 생성 (임시 디렉토리에)
        if args.task in ['reusable', 'both']:
            print("\n" + "="*60)
            print("Creating Reusable/Disposable Classification Dataset")
            print("="*60)
            create_classification_dataset(results, temp_dir, 'reusable', args.image_dir)

        if args.task in ['beverage', 'both']:
            print("\n" + "="*60)
            print("Creating Beverage Status Classification Dataset")
            print("="*60)
            create_classification_dataset(results, temp_dir, 'beverage', args.image_dir)

        # 임베딩용 데이터셋 생성
        if args.include_types:
            print("\n" + "="*60)
            print("Creating Embedding Dataset (Cup Code Classification)")
            print("="*60)
            create_embedding_dataset(results, temp_dir, args.image_dir)

        print("\n" + "="*60)
        print("Dataset creation complete!")
        print("="*60)
        print("\nDataset structure:")
        if args.task in ['reusable', 'both']:
            print(f"  reusable/")
            for class_name in ['reusable', 'disposable', 'unclear']:
                count = len(list((temp_path / 'reusable' / class_name).glob('*')))
                print(f"    {class_name}/  ({count} images)")
        if args.task in ['beverage', 'both']:
            print(f"  beverage/")
            for class_name in ['with_beverage', 'empty', 'unclear']:
                count = len(list((temp_path / 'beverage' / class_name).glob('*')))
                print(f"    {class_name}/  ({count} images)")
        if args.include_types and (temp_path / 'types').exists():
            print(f"  types/")
            cup_codes = sorted([d.name for d in (temp_path / 'types').iterdir() if d.is_dir()])
            for cup_code in cup_codes:
                count = len(list((temp_path / 'types' / cup_code).glob('*')))
                print(f"    {cup_code}/  ({count} images)")

        # ZIP 파일 생성
        print("\n" + "="*60)
        print("Creating ZIP archive...")
        print("="*60)

        zip_file = create_zip_archive(
            temp_dir,
            args.output_dir,
            args.output_file,
            args.task,
            args.include_types
        )

        if zip_file:
            print("\n" + "="*60)
            print("✅ ZIP archive created!")
            print("="*60)
            print(f"  📦 {zip_file}")
        else:
            print("\n⚠️  No ZIP file created (no data to compress)")


if __name__ == '__main__':
    main()
