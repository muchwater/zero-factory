"""
Embedding Generator using Siamese Network
샴 네트워크 기반 임베딩 생성기
"""

import torch
from torchvision import transforms
from PIL import Image
import io
import json
from pathlib import Path
from typing import Union, Optional, Dict, List
import numpy as np

try:
    from .siamese_network import SiameseNetworkInference
except ImportError:
    from siamese_network import SiameseNetworkInference


class EmbeddingGenerator:
    """
    Siamese Network를 사용한 임베딩 생성기

    기존 CLIP 모델을 대체하여 더 정확한 용기 분류를 위한
    256차원 임베딩 벡터를 생성합니다.
    """

    def __init__(
        self,
        model_path: str,
        embeddings_db_path: Optional[str] = None,
        device: str = 'cpu',
        embedding_dim: int = 256
    ):
        """
        Args:
            model_path: Siamese Network 모델 가중치 파일 경로 (.pth)
            embeddings_db_path: 저장된 임베딩 데이터베이스 경로 (.json)
            device: 'cpu' 또는 'cuda'
            embedding_dim: 임베딩 차원 (기본값: 256)
        """
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim

        # Siamese Network 로드
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = SiameseNetworkInference(
            model_path=model_path,
            embedding_dim=embedding_dim,
            device=device
        )

        print(f"✅ Siamese Network embedding generator loaded from {model_path}")

        # 임베딩 데이터베이스 로드
        self.embeddings_db = {}
        if embeddings_db_path and Path(embeddings_db_path).exists():
            self.load_embeddings_db(embeddings_db_path)
            print(f"✅ Loaded {len(self.embeddings_db)} embeddings from database")

        # 전처리 Transform (Siamese Network 학습 시와 동일하게)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image: Union[bytes, str, Image.Image]) -> torch.Tensor:
        """
        이미지 전처리

        Args:
            image: 이미지 (bytes, 파일 경로, 또는 PIL.Image)

        Returns:
            전처리된 torch.Tensor
        """
        # 이미지 로드
        if isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError("Image must be bytes, file path, or PIL.Image")

        # 전처리
        return self.transform(pil_image)

    def generate_embedding(self, image: Union[bytes, str, Image.Image]) -> np.ndarray:
        """
        이미지에서 임베딩 벡터 생성

        Args:
            image: 이미지 (bytes, 파일 경로, 또는 PIL.Image)

        Returns:
            256차원 L2-normalized 임베딩 벡터 (numpy array)
        """
        # 전처리
        image_tensor = self.preprocess_image(image)

        # 임베딩 생성
        embedding = self.model.predict(image_tensor)

        return embedding

    def generate_embedding_batch(self, images: List[Union[bytes, str, Image.Image]]) -> np.ndarray:
        """
        여러 이미지의 임베딩 벡터 배치 생성

        Args:
            images: 이미지 리스트

        Returns:
            (N, 256) 형태의 임베딩 벡터 배열
        """
        # 전처리
        image_tensors = [self.preprocess_image(img) for img in images]
        batch_tensor = torch.stack(image_tensors)

        # 배치 임베딩 생성
        embeddings = self.model.predict_batch(batch_tensor)

        return embeddings

    def load_embeddings_db(self, db_path: str):
        """
        저장된 임베딩 데이터베이스 로드

        Args:
            db_path: JSON 파일 경로
        """
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # JSON에서 numpy array로 변환
        self.embeddings_db = {
            cup_code: np.array(embedding)
            for cup_code, embedding in data.items()
        }

    def save_embeddings_db(self, db_path: str):
        """
        임베딩 데이터베이스 저장

        Args:
            db_path: JSON 파일 경로
        """
        # numpy array를 list로 변환
        data = {
            cup_code: embedding.tolist()
            for cup_code, embedding in self.embeddings_db.items()
        }

        db_path_obj = Path(db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(self.embeddings_db)} embeddings to {db_path}")

    def add_to_database(self, cup_code: str, embedding: np.ndarray):
        """
        데이터베이스에 임베딩 추가

        Args:
            cup_code: 용기 코드 (예: 'paper_cup', 'steel_mug')
            embedding: 임베딩 벡터
        """
        self.embeddings_db[cup_code] = embedding

    def match_container(
        self,
        image: Union[bytes, str, Image.Image],
        threshold: float = 0.7,
        top_k: int = 3
    ) -> List[tuple]:
        """
        이미지를 데이터베이스의 용기들과 매칭

        Args:
            image: 쿼리 이미지
            threshold: 최소 유사도 임계값 (기본값: 0.7)
            top_k: 반환할 상위 매칭 개수 (기본값: 3)

        Returns:
            [(cup_code, similarity), ...] 리스트 (유사도 내림차순)
        """
        # 쿼리 임베딩 생성
        query_embedding = self.generate_embedding(image)

        # 데이터베이스와 매칭
        matches = self.model.match_container(
            query_embedding,
            self.embeddings_db,
            threshold=threshold,
            top_k=top_k
        )

        return matches

    def compute_similarity(
        self,
        image1: Union[bytes, str, Image.Image],
        image2: Union[bytes, str, Image.Image],
        metric: str = 'cosine'
    ) -> float:
        """
        두 이미지 간 유사도 계산

        Args:
            image1: 첫 번째 이미지
            image2: 두 번째 이미지
            metric: 유사도 메트릭 ('cosine' 또는 'euclidean')

        Returns:
            유사도 점수
        """
        emb1 = self.generate_embedding(image1)
        emb2 = self.generate_embedding(image2)

        return self.model.compute_similarity(emb1, emb2, metric=metric)

    def get_embedding_dim(self) -> int:
        """임베딩 차원 반환"""
        return self.embedding_dim


if __name__ == "__main__":
    # 테스트 코드
    print("Testing EmbeddingGenerator...")

    # 모델 경로 확인
    model_path = Path("models/weights/siamese_network.pth")
    embeddings_db_path = Path("models/weights/cup_code_embeddings_siamese.json")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model using notebooks/04_siamese_network_training.ipynb")
        exit(1)

    # 임베딩 생성기 초기화
    generator = EmbeddingGenerator(
        model_path=str(model_path),
        embeddings_db_path=str(embeddings_db_path) if embeddings_db_path.exists() else None,
        device='cpu',
        embedding_dim=256
    )

    # 더미 이미지로 테스트
    print("\nTesting with dummy image...")
    dummy_image = Image.new('RGB', (224, 224), color='red')

    # 임베딩 생성 테스트
    embedding = generator.generate_embedding(dummy_image)
    print(f"✓ Generated embedding shape: {embedding.shape}")
    print(f"✓ Embedding dimension: {generator.get_embedding_dim()}")

    # L2 정규화 확인
    norm = np.linalg.norm(embedding)
    print(f"✓ Embedding L2 norm: {norm:.6f} (should be ~1.0)")

    # 배치 테스트
    batch_images = [dummy_image, dummy_image]
    batch_embeddings = generator.generate_embedding_batch(batch_images)
    print(f"✓ Batch embeddings shape: {batch_embeddings.shape}")

    # 유사도 테스트
    similarity = generator.compute_similarity(dummy_image, dummy_image, metric='cosine')
    print(f"✓ Self-similarity (cosine): {similarity:.6f} (should be ~1.0)")

    print("\n✅ All tests passed!")
