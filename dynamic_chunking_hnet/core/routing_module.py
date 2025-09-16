"""
Routing Module

Implements the chunking layer routing mechanism from H-Net.
Uses boundary probabilities to create dynamic chunks with target compression ratio.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from .boundary_detector import SimilarityBasedBoundaryDetector
from ..utils.exceptions import InvalidTextError, ChunkingError

logger = logging.getLogger(__name__)


class RoutingModule:
    """
    Implements the chunking layer routing mechanism from H-Net.
    Uses boundary probabilities to create dynamic chunks with target compression ratio.
    """
    
    def __init__(
        self, 
        target_compression_ratio: float = 6.0,
        boundary_detector: Optional[SimilarityBasedBoundaryDetector] = None
    ):
        """
        Initialize the routing module.
        
        Args:
            target_compression_ratio: Desired compression ratio (tokens per chunk)
            boundary_detector: Pre-initialized boundary detector, creates new if None
        """
        if target_compression_ratio <= 1.0:
            raise InvalidTextError(f"Compression ratio must be > 1.0, got {target_compression_ratio}")
        
        self.target_compression_ratio = target_compression_ratio
        
        if boundary_detector is None:
            self.boundary_detector = SimilarityBasedBoundaryDetector()
        else:
            self.boundary_detector = boundary_detector
        
        logger.info(f"Initialized routing module with compression ratio {target_compression_ratio}")
    
    def calculate_ratio_loss(
        self, 
        boundary_probs: np.ndarray, 
        boundaries: np.ndarray
    ) -> float:
        """
        Calculate the ratio loss as described in the H-Net paper.
        
        Args:
            boundary_probs: Boundary probabilities (G in paper)
            boundaries: Discrete boundary indicators (F in paper)
            
        Returns:
            ratio_loss: Loss value encouraging target compression ratio
        """
        if len(boundary_probs) == 0 or len(boundaries) == 0:
            return 0.0
        
        N = self.target_compression_ratio
        
        F = np.mean(boundaries)  # Fraction of vectors actually selected
        G = np.mean(boundary_probs)  # Average boundary probability
        
        # Avoid division by zero
        if N <= 1:
            return 0.0
        
        # Ratio loss from equation (10) in paper
        ratio_loss = (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))
        
        logger.debug(f"Ratio loss: {ratio_loss:.4f} (F={F:.3f}, G={G:.3f})")
        
        return ratio_loss
    
    def adaptive_threshold_selection(self, boundary_probs: np.ndarray) -> float:
        """
        Adaptively select threshold to approximate target compression ratio.
        
        Args:
            boundary_probs: Array of boundary probabilities
            
        Returns:
            threshold: Selected threshold value
        """
        if len(boundary_probs) == 0:
            return 0.5
        
        # Sort probabilities to find threshold that gives desired compression
        sorted_probs = np.sort(boundary_probs)[::-1]  # Descending order
        target_boundaries = max(1, int(len(boundary_probs) / self.target_compression_ratio))
        
        if target_boundaries >= len(sorted_probs):
            threshold = 0.0  # Keep all boundaries
        else:
            threshold = sorted_probs[target_boundaries - 1]
        
        # Apply minimum threshold to avoid too many boundaries
        threshold = max(0.1, threshold)
        
        logger.debug(
            f"Selected threshold {threshold:.3f} for {target_boundaries} "
            f"target boundaries from {len(boundary_probs)} positions"
        )
        
        return threshold
    
    def _validate_inputs(
        self, 
        embeddings: np.ndarray, 
        text_tokens: Optional[List[str]] = None
    ) -> None:
        """Validate input parameters."""
        if embeddings is None or len(embeddings) == 0:
            raise InvalidTextError("Empty embeddings provided")
        
        if text_tokens is not None and len(text_tokens) != len(embeddings):
            raise InvalidTextError(
                f"Token count ({len(text_tokens)}) doesn't match "
                f"embedding count ({len(embeddings)})"
            )
    
    def create_chunks(
        self, 
        embeddings: np.ndarray, 
        text_tokens: Optional[List[str]] = None
    ) -> Dict:
        """
        Create dynamic chunks from embeddings using the routing mechanism.
        
        Args:
            embeddings: Input embeddings of shape (sequence_length, embedding_dim)
            text_tokens: Optional text tokens for chunk creation
            
        Returns:
            Dictionary containing chunking results with keys:
            - 'chunks': List of token chunks (if text_tokens provided)
            - 'chunk_embeddings': List of embedding chunks
            - 'boundary_probs': Boundary probabilities
            - 'boundaries': Discrete boundary indicators
            - 'threshold': Selected threshold
            - 'compression_ratio': Actual compression ratio achieved
            - 'target_compression_ratio': Target compression ratio
            - 'ratio_loss': Calculated ratio loss
            - 'num_chunks': Number of chunks created
            
        Raises:
            InvalidTextError: If inputs are invalid
            ChunkingError: If chunking fails
        """
        try:
            self._validate_inputs(embeddings, text_tokens)
            
            logger.info(f"Creating chunks for {len(embeddings)} embeddings")
            
            # Calculate boundary probabilities
            boundary_probs = self.boundary_detector.calculate_boundary_probabilities(embeddings)
            
            # Adaptive threshold selection
            threshold = self.adaptive_threshold_selection(boundary_probs)
            
            # Get discrete boundaries
            boundaries = self.boundary_detector.get_discrete_boundaries(boundary_probs, threshold)
            
            # Create chunks by grouping tokens between boundaries
            chunks = []
            chunk_embeddings = []
            current_chunk_tokens = []
            current_chunk_embeddings = []
            
            for i, (is_boundary, embedding) in enumerate(zip(boundaries, embeddings)):
                # Add current token to chunk
                if text_tokens:
                    current_chunk_tokens.append(text_tokens[i])
                current_chunk_embeddings.append(embedding)
                
                # If this is a boundary and we have accumulated content, finalize chunk
                if is_boundary and len(current_chunk_embeddings) > 0 and i > 0:
                    if text_tokens:
                        chunks.append(current_chunk_tokens.copy())
                    chunk_embeddings.append(np.array(current_chunk_embeddings.copy()))
                    
                    # Start new chunk with current token/embedding
                    current_chunk_tokens = [text_tokens[i]] if text_tokens else []
                    current_chunk_embeddings = [embedding]
            
            # Add final chunk if it exists
            if len(current_chunk_embeddings) > 0:
                if text_tokens:
                    chunks.append(current_chunk_tokens)
                chunk_embeddings.append(np.array(current_chunk_embeddings))
            
            # Calculate metrics
            actual_compression_ratio = len(embeddings) / len(chunks) if len(chunks) > 0 else 1.0
            ratio_loss = self.calculate_ratio_loss(boundary_probs, boundaries)
            
            logger.info(
                f"Created {len(chunks)} chunks with compression ratio "
                f"{actual_compression_ratio:.2f} (target: {self.target_compression_ratio})"
            )
            
            result = {
                'chunks': chunks,
                'chunk_embeddings': chunk_embeddings,
                'boundary_probs': boundary_probs,
                'boundaries': boundaries,
                'threshold': threshold,
                'compression_ratio': actual_compression_ratio,
                'target_compression_ratio': self.target_compression_ratio,
                'ratio_loss': ratio_loss,
                'num_chunks': len(chunks)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create chunks: {e}")
            if isinstance(e, (InvalidTextError, ChunkingError)):
                raise
            raise ChunkingError(f"Chunk creation failed: {e}")
    
    def get_chunk_statistics(self, result: Dict) -> Dict:
        """
        Calculate detailed statistics for chunking results.
        
        Args:
            result: Result dictionary from create_chunks()
            
        Returns:
            Dictionary with detailed statistics
        """
        chunks = result.get('chunks', [])
        chunk_embeddings = result.get('chunk_embeddings', [])
        
        if not chunks and not chunk_embeddings:
            return {}
        
        # Calculate chunk sizes
        if chunks:
            chunk_sizes = [len(chunk) for chunk in chunks]
        else:
            chunk_sizes = [len(chunk_emb) for chunk_emb in chunk_embeddings]
        
        stats = {
            'num_chunks': len(chunk_sizes),
            'total_tokens': sum(chunk_sizes),
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'mean_chunk_size': np.mean(chunk_sizes) if chunk_sizes else 0,
            'std_chunk_size': np.std(chunk_sizes) if chunk_sizes else 0,
            'median_chunk_size': np.median(chunk_sizes) if chunk_sizes else 0,
            'chunk_size_variance': np.var(chunk_sizes) if chunk_sizes else 0,
        }
        
        # Add boundary statistics
        boundary_probs = result.get('boundary_probs', np.array([]))
        if len(boundary_probs) > 0:
            stats.update({
                'mean_boundary_prob': np.mean(boundary_probs),
                'std_boundary_prob': np.std(boundary_probs),
                'max_boundary_prob': np.max(boundary_probs),
                'min_boundary_prob': np.min(boundary_probs),
            })
        
        return stats