"""
API 테스트 스크립트
"""

import requests
import sys
from pathlib import Path


def test_container_verification(image_path: str, api_url: str = "http://localhost:8000"):
    """
    컨테이너 검증 API 테스트

    Args:
        image_path: 테스트할 이미지 경로
        api_url: API 서버 URL
    """
    endpoint = f"{api_url}/container/verify"

    # 이미지 파일 확인
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"Testing API: {endpoint}")
    print(f"Image: {image_path}")
    print("="*60)

    try:
        # API 요청
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(endpoint, files=files)

        # 응답 확인
        if response.status_code == 200:
            result = response.json()

            print("✅ API Response:")
            print("="*60)
            print(f"Container Detected: {result['container_detected']}")
            print(f"Num Containers: {result['num_containers']}")

            if result.get('container_class'):
                print(f"Container Class: {result['container_class']} ({result['container_confidence']:.1%})")

            if result.get('is_reusable') is not None:
                print(f"Is Reusable: {result['is_reusable']} ({result['reusable_confidence']:.1%})")

            if result.get('beverage_status'):
                print(f"Beverage Status: {result['beverage_status']} ({result['beverage_confidence']:.1%})")

            print(f"\nMessage: {result['message']}")

            if result.get('error'):
                print(f"Error: {result['error']}")

            print("="*60)

        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed. Is the server running at {api_url}?")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_health_check(api_url: str = "http://localhost:8000"):
    """헬스 체크 테스트"""
    endpoint = f"{api_url}/container/health"

    print(f"Health Check: {endpoint}")
    print("="*60)

    try:
        response = requests.get(endpoint)

        if response.status_code == 200:
            result = response.json()
            print("✅ Health Check:")
            print(f"Status: {result['status']}")
            print("Models:")
            for model, loaded in result['models'].items():
                status = "✅" if loaded else "❌"
                print(f"  {status} {model}: {loaded}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed. Is the server running at {api_url}?")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Container Verification API")
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='API server URL')
    parser.add_argument('--health', action='store_true', help='Run health check only')

    args = parser.parse_args()

    if args.health:
        test_health_check(args.url)
    elif args.image:
        test_container_verification(args.image, args.url)
    else:
        # 기본: 헬스 체크만
        test_health_check(args.url)
        print("\nUsage:")
        print("  python test_api.py --image path/to/image.jpg")
        print("  python test_api.py --health")
