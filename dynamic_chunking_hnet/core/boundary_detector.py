"""
Similarity-Based Boundary Detector

Implements the routing module from H-Net paper that uses cosine similarity
to detect semantic boundaries between adjacent text representations.
"""

import numpy as np
from typing import Optional, Union
import logging

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..utils.exceptions import InvalidTextError, EmbeddingError

logger = logging.getLogger(__name__)


class SimilarityBasedBoundaryDetector:
    """
    Implements the routing module from H-Net paper that uses cosine similarity
    to detect semantic boundaries between adjacent text representations.
    
    According to the paper, the boundary probability is calculated as:
    p_t = 0.5 * (1 - cos_similarity(q_t, k_{t-1}))
    
    Where consecutive vectors with different contexts yield high boundary probability.
    """
    
    def __init__(self, embedding_dim: int = 384, device: str = 'cpu'):
        """
        Initialize the boundary detector.
        
        Args:
            embedding_dim: Dimension of input embeddings
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.embedding_dim = embedding_dim
        self.device = device
        
        logger.info(f"Initializing boundary detector with embedding_dim={embedding_dim}")
        
        if HAS_TORCH:
            # Simple linear projections for query and key (as in the paper)
            self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
            
            # Initialize with small random weights
            nn.init.xavier_uniform_(self.W_q.weight, gain=0.1)
            nn.init.xavier_uniform_(self.W_k.weight, gain=0.1)
            
            logger.info("Using PyTorch implementation with learnable projections")
        else:
            # Fallback to numpy implementation
            self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.1
            self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.1
            
            logger.warning("PyTorch not available, using numpy fallback implementation")
    
    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """Validate input embeddings."""
        if embeddings is None or len(embeddings) == 0:
            raise InvalidTextError("Empty embeddings provided")
        
        if len(embeddings.shape) != 2:
            raise InvalidTextError(f"Expected 2D embeddings, got shape {embeddings.shape}")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise InvalidTextError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
    
    def calculate_boundary_probabilities(
        self, 
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate boundary probabilities using cosine similarity.
        
        Args:
            embeddings: Array of shape (sequence_length, embedding_dim)
            
        Returns:
            boundary_probs: Array of shape (sequence_length,) with boundary probabilities
            
        Raises:
            InvalidTextError: If embeddings are invalid
            EmbeddingError: If computation fails
        """
        try:
            self._validate_embeddings(embeddings)
            
            if len(embeddings) < 2:
                logger.info("Single token sequence, returning boundary probability 1.0")
                return np.array([1.0])
            
            logger.debug(f"Computing boundary probabilities for {len(embeddings)} embeddings")
            
            if HAS_TORCH and isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings).float()
            
            if HAS_TORCH:
                # Use PyTorch implementation
                q = self.W_q(embeddings)  # Query projections
                k = self.W_k(embeddings)  # Key projections
                
                # Calculate cosine similarities between adjacent positions
                similarities = []
                for t in range(1, len(embeddings)):
                    cos_sim = F.cosine_similarity(q[t:t+1], k[t-1:t], dim=1)
                    similarities.append(cos_sim.item())
                
                similarities = np.array(similarities)
            else:
                # Numpy fallback implementation
                q = embeddings @ self.W_q.T  # Query projections
                k = embeddings @ self.W_k.T  # Key projections
                
                similarities = []
                for t in range(1, len(embeddings)):
                    # Cosine similarity between q[t] and k[t-1]
                    cos_sim = np.dot(q[t], k[t-1]) / (
                        np.linalg.norm(q[t]) * np.linalg.norm(k[t-1]) + 1e-8
                    )
                    similarities.append(cos_sim)
                
                similarities = np.array(similarities)
            
            # Convert to boundary probabilities using H-Net formula
            boundary_probs = 0.5 * (1 - similarities)
            
            # Clamp to valid probability range
            boundary_probs = np.clip(boundary_probs, 0.0, 1.0)
            
            # First position is always a boundary (p_1 = 1.0 as in paper)
            boundary_probs = np.concatenate([[1.0], boundary_probs])
            
            logger.debug(f"Computed boundary probabilities: mean={np.mean(boundary_probs):.3f}")
            return boundary_probs
            
        except Exception as e:
            logger.error(f"Failed to calculate boundary probabilities: {e}")
            if isinstance(e, (InvalidTextError, EmbeddingError)):
                raise
            raise EmbeddingError(f"Boundary probability calculation failed: {e}")
    
    def get_discrete_boundaries(
        self, 
        boundary_probs: np.ndarray, 
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Convert boundary probabilities to discrete boundary indicators.
        
        Args:
            boundary_probs: Boundary probabilities
            threshold: Threshold for boundary decision (0.0 to 1.0)
            
        Returns:
            boundaries: Binary array indicating boundaries (1=boundary, 0=no boundary)
            
        Raises:
            InvalidTextError: If inputs are invalid
        """
        if boundary_probs is None or len(boundary_probs) == 0:
            raise InvalidTextError("Empty boundary probabilities")
        
        if not 0.0 <= threshold <= 1.0:
            raise InvalidTextError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        logger.debug(f"Converting to discrete boundaries with threshold={threshold}")
        
        boundaries = (boundary_probs >= threshold).astype(int)
        
        # Ensure first position is always a boundary
        if len(boundaries) > 0:
            boundaries[0] = 1
        
        logger.debug(f"Created {np.sum(boundaries)} boundaries from {len(boundary_probs)} positions")
        
        return boundaries
    
    def adaptive_threshold(
        self, 
        boundary_probs: np.ndarray, 
        target_compression: float = 6.0
    ) -> float:
        """
        Calculate adaptive threshold to achieve target compression ratio.
        
        Args:
            boundary_probs: Boundary probabilities
            target_compression: Desired compression ratio (tokens per chunk)
            
        Returns:
            threshold: Adaptive threshold value
        """
        if len(boundary_probs) < 2:
            return 0.5
        
        # Calculate desired number of boundaries
        target_boundaries = max(1, int(len(boundary_probs) / target_compression))
        
        # Sort probabilities to find threshold
        sorted_probs = np.sort(boundary_probs)[::-1]  # Descending order
        
        if target_boundaries >= len(sorted_probs):
            threshold = 0.0  # Keep all boundaries
        else:
            threshold = sorted_probs[target_boundaries - 1]
        
        # Apply minimum threshold to avoid too many boundaries
        threshold = max(0.1, threshold)
        
        logger.debug(f"Adaptive threshold: {threshold:.3f} for target compression {target_compression}")
        
        return threshold