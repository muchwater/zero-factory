"""
Siamese Network for Cup Container Similarity Learning

This module implements a Siamese Network based on MobileNetV3-Small
for learning embeddings of reusable cup containers. The network is trained
with triplet loss to ensure that embeddings of the same cup_code are close
together while different cup_codes are far apart.

Architecture:
    - Encoder: MobileNetV3-Small (pretrained on ImageNet)
    - Embedding: 256-dimensional L2-normalized vectors
    - Loss: Triplet Loss with semi-hard negative mining

Usage:
    # Training
    model = SiameseNetwork(embedding_dim=256)
    anchor, positive, negative = batch
    anchor_emb, pos_emb, neg_emb = model.forward_triplet(anchor, positive, negative)

    # Inference
    inference = SiameseNetworkInference(model_path="siamese_network.pth")
    embedding = inference.predict(image)
    similarity = inference.compute_similarity(emb1, emb2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Union
import numpy as np
from pathlib import Path


class SiameseNetwork(nn.Module):
    """
    Siamese Network using MobileNetV3-Small as the encoder backbone.

    The network shares weights between branches and produces L2-normalized
    embeddings suitable for metric learning with triplet loss.

    Args:
        embedding_dim: Dimension of the output embedding (default: 256)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate before the embedding layer (default: 0.3)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super(SiameseNetwork, self).__init__()

        self.embedding_dim = embedding_dim

        # Load MobileNetV3-Small as the encoder backbone
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)

        # Extract feature extractor (remove classifier)
        self.encoder = nn.Sequential(*list(mobilenet.children())[:-1])

        # Get the feature dimension from MobileNetV3-Small
        # MobileNetV3-Small final conv output: 576 channels
        feature_dim = 576

        # Embedding head with dropout for regularization
        self.embedding_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights of the embedding head."""
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single input.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            L2-normalized embedding of shape (B, embedding_dim)
        """
        # Extract features
        features = self.encoder(x)

        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        # Generate embedding
        embedding = self.embedding_head(features)

        # L2 normalization
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (alias for forward_once).

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            L2-normalized embedding of shape (B, embedding_dim)
        """
        return self.forward_once(x)

    def forward_triplet(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for triplet inputs.

        Args:
            anchor: Anchor images of shape (B, 3, H, W)
            positive: Positive images (same cup_code as anchor) of shape (B, 3, H, W)
            negative: Negative images (different cup_code) of shape (B, 3, H, W)

        Returns:
            Tuple of (anchor_emb, positive_emb, negative_emb), each of shape (B, embedding_dim)
        """
        anchor_emb = self.forward_once(anchor)
        positive_emb = self.forward_once(positive)
        negative_emb = self.forward_once(negative)

        return anchor_emb, positive_emb, negative_emb

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


class SiameseNetworkInference:
    """
    Inference wrapper for Siamese Network.

    This class provides a simple interface for loading a trained model
    and generating embeddings for images, computing similarities, and
    matching against a database of known cup containers.

    Args:
        model_path: Path to the saved model checkpoint (.pth file)
        embedding_dim: Embedding dimension (default: 256)
        device: Device to run inference on ('cuda' or 'cpu', auto-detected if None)
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        embedding_dim: int = 256,
        device: Optional[str] = None
    ):
        self.embedding_dim = embedding_dim

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = SiameseNetwork(embedding_dim=embedding_dim, pretrained=False)

        # Load weights if provided
        if model_path is not None:
            self.load_model(model_path)

        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: Union[str, Path]):
        """
        Load model weights from a checkpoint.

        Args:
            model_path: Path to the .pth checkpoint file
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> np.ndarray:
        """
        Generate embedding for a single image or batch of images.

        Args:
            image: Input tensor of shape (3, H, W) or (B, 3, H, W)

        Returns:
            Embedding as numpy array of shape (embedding_dim,) or (B, embedding_dim)
        """
        # Add batch dimension if single image
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Move to device
        image = image.to(self.device)

        # Generate embedding
        embedding = self.model(image)

        # Convert to numpy
        embedding = embedding.cpu().numpy()

        # Remove batch dimension if single image
        if squeeze_output:
            embedding = embedding[0]

        return embedding

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Generate embeddings for a batch of images.

        Args:
            images: Batch of images of shape (B, 3, H, W)

        Returns:
            Embeddings as numpy array of shape (B, embedding_dim)
        """
        images = images.to(self.device)
        embeddings = self.model(images)
        return embeddings.cpu().numpy()

    @staticmethod
    def compute_similarity(
        embedding1: Union[torch.Tensor, np.ndarray],
        embedding2: Union[torch.Tensor, np.ndarray],
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding (1D array or tensor)
            embedding2: Second embedding (1D array or tensor)
            metric: Similarity metric ('cosine' or 'euclidean')

        Returns:
            Similarity score (higher = more similar for cosine, lower = more similar for euclidean)
        """
        # Convert to numpy if tensor
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.cpu().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.cpu().numpy()

        # Ensure 1D
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        if metric == 'cosine':
            # Cosine similarity (assumes L2-normalized embeddings)
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)

        elif metric == 'euclidean':
            # Euclidean distance
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)

        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'.")

    def match_container(
        self,
        query_embedding: Union[torch.Tensor, np.ndarray],
        database_embeddings: dict,
        threshold: float = 0.7,
        top_k: int = 3
    ) -> list:
        """
        Match a query embedding against a database of known cup containers.

        Args:
            query_embedding: Query embedding (1D array or tensor)
            database_embeddings: Dict mapping cup_code to embeddings
                                 e.g., {'paper_cup': array([...]), ...}
            threshold: Minimum cosine similarity to consider a match (default: 0.7)
            top_k: Number of top matches to return (default: 3)

        Returns:
            List of (cup_code, similarity) tuples, sorted by similarity (highest first)
        """
        # Convert query to numpy
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        query_embedding = query_embedding.flatten()

        # Compute similarities
        matches = []
        for cup_code, db_embedding in database_embeddings.items():
            if isinstance(db_embedding, torch.Tensor):
                db_embedding = db_embedding.cpu().numpy()
            db_embedding = db_embedding.flatten()

            similarity = self.compute_similarity(query_embedding, db_embedding, metric='cosine')

            if similarity >= threshold:
                matches.append((cup_code, similarity))

        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return matches[:top_k]

    def save_model(self, save_path: Union[str, Path], metadata: Optional[dict] = None):
        """
        Save the model to a checkpoint file.

        Args:
            save_path: Path to save the checkpoint
            metadata: Optional metadata to include (e.g., epoch, loss)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim
        }

        if metadata is not None:
            checkpoint.update(metadata)

        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Unit tests
    print("Testing SiameseNetwork...")

    # Test model initialization
    model = SiameseNetwork(embedding_dim=256)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    output = model(dummy_input)
    print(f"✓ Forward pass: input shape {dummy_input.shape} -> output shape {output.shape}")

    # Verify L2 normalization
    norms = torch.norm(output, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), "Embeddings not L2-normalized!"
    print(f"✓ Embeddings are L2-normalized (norms: {norms})")

    # Test triplet forward
    anchor = torch.randn(batch_size, 3, 224, 224)
    positive = torch.randn(batch_size, 3, 224, 224)
    negative = torch.randn(batch_size, 3, 224, 224)
    anchor_emb, pos_emb, neg_emb = model.forward_triplet(anchor, positive, negative)
    print(f"✓ Triplet forward pass successful")

    # Test inference wrapper
    print("\nTesting SiameseNetworkInference...")
    inference = SiameseNetworkInference(embedding_dim=256, device='cpu')

    # Test single image prediction
    single_image = torch.randn(3, 224, 224)
    embedding = inference.predict(single_image)
    print(f"✓ Single image prediction: shape {embedding.shape}")

    # Test batch prediction
    batch_images = torch.randn(4, 3, 224, 224)
    batch_embeddings = inference.predict_batch(batch_images)
    print(f"✓ Batch prediction: shape {batch_embeddings.shape}")

    # Test similarity computation
    emb1 = embedding
    emb2 = batch_embeddings[0]
    cosine_sim = SiameseNetworkInference.compute_similarity(emb1, emb2, metric='cosine')
    euclidean_dist = SiameseNetworkInference.compute_similarity(emb1, emb2, metric='euclidean')
    print(f"✓ Cosine similarity: {cosine_sim:.4f}")
    print(f"✓ Euclidean distance: {euclidean_dist:.4f}")

    # Test matching
    database = {
        'paper_cup': batch_embeddings[0],
        'steel_cup': batch_embeddings[1],
        'mug_cup': batch_embeddings[2]
    }
    matches = inference.match_container(emb1, database, threshold=0.0, top_k=3)
    print(f"✓ Matching result: {matches}")

    print("\n✅ All tests passed!")
