"""
Smoothing Module

Implements the smoothing module from H-Net for gradient flow and error correction.
Uses exponential moving average as described in equation (5) of the paper.
"""

import numpy as np
from typing import Optional
import logging

from ..utils.exceptions import InvalidTextError, ChunkingError

logger = logging.getLogger(__name__)


class SmoothingModule:
    """
    Implements the smoothing module from H-Net for gradient flow and error correction.
    Uses exponential moving average as described in equation (5) of the paper.
    
    The smoothing applies: z̄_t = P_t * z^_t + (1 - P_t) * z̄_{t-1}
    where P_t is the boundary probability acting as confidence score.
    """
    
    def __init__(self, alpha: float = 0.7, use_confidence_weighting: bool = True):
        """
        Initialize the smoothing module.
        
        Args:
            alpha: Default smoothing factor for EMA (0 < alpha <= 1)
            use_confidence_weighting: Whether to use boundary probabilities as weights
        """
        if not 0 < alpha <= 1:
            raise InvalidTextError(f"Alpha must be between 0 and 1, got {alpha}")
        
        self.alpha = alpha
        self.use_confidence_weighting = use_confidence_weighting
        
        logger.info(f"Initialized smoothing module with alpha={alpha}")
    
    def apply_smoothing(
        self, 
        chunk_embeddings: np.ndarray, 
        boundary_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply exponential moving average smoothing to chunk embeddings.
        
        According to the paper: z̄_t = P_t * z^_t + (1 - P_t) * z̄_{t-1}
        
        Args:
            chunk_embeddings: Compressed chunk embeddings of shape (num_chunks, embedding_dim)
            boundary_probs: Boundary probabilities for confidence weighting
            
        Returns:
            smoothed_embeddings: Smoothed embeddings with error correction
            
        Raises:
            InvalidTextError: If inputs are invalid
            ChunkingError: If smoothing fails
        """
        try:
            if chunk_embeddings is None or len(chunk_embeddings) == 0:
                logger.warning("Empty chunk embeddings provided")
                return np.array([])
            
            if len(chunk_embeddings.shape) != 2:
                raise InvalidTextError(f"Expected 2D embeddings, got shape {chunk_embeddings.shape}")
            
            num_chunks, embedding_dim = chunk_embeddings.shape
            
            logger.debug(f"Applying smoothing to {num_chunks} chunks of dimension {embedding_dim}")
            
            smoothed = np.zeros_like(chunk_embeddings)
            smoothed[0] = chunk_embeddings[0]  # First embedding unchanged
            
            for t in range(1, len(chunk_embeddings)):
                if self.use_confidence_weighting and boundary_probs is not None and t < len(boundary_probs):
                    # Use boundary probability as confidence score
                    P_t = boundary_probs[t]
                    # Clamp to reasonable range
                    P_t = np.clip(P_t, 0.1, 1.0)
                else:
                    # Use default alpha
                    P_t = self.alpha
                
                # Apply EMA smoothing: z̄_t = P_t * z^_t + (1 - P_t) * z̄_{t-1}
                smoothed[t] = P_t * chunk_embeddings[t] + (1 - P_t) * smoothed[t-1]
            
            logger.debug(f"Smoothing completed with mean confidence {np.mean(boundary_probs) if boundary_probs is not None else self.alpha:.3f}")
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Smoothing failed: {e}")
            if isinstance(e, InvalidTextError):
                raise
            raise ChunkingError(f"Smoothing operation failed: {e}")
    
    def straight_through_estimator(self, confidence_scores: np.ndarray) -> np.ndarray:
        """
        Apply Straight-Through Estimator (STE) for gradient stabilization.
        
        In training, this allows gradients to flow through discrete operations
        by using the continuous input for backward pass while using rounded
        values for forward pass.
        
        Args:
            confidence_scores: Raw confidence scores
            
        Returns:
            rounded_scores: Rounded scores with gradient preservation
        """
        if confidence_scores is None or len(confidence_scores) == 0:
            return np.array([])
        
        # Round to discrete values in forward pass
        rounded = np.round(confidence_scores)
        
        # In actual PyTorch implementation, gradients would flow through 
        # the continuous scores. Here we simulate the effect.
        
        logger.debug(f"Applied STE to {len(confidence_scores)} confidence scores")
        
        return rounded
    
    def upsample_with_confidence(
        self, 
        smoothed_chunks: np.ndarray,
        boundaries: np.ndarray,
        original_length: int,
        boundary_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Upsample compressed chunks back to original resolution with confidence weighting.
        
        Args:
            smoothed_chunks: Smoothed chunk embeddings of shape (num_chunks, embedding_dim)
            boundaries: Boundary indicators of shape (original_length,)
            original_length: Target length for upsampling
            boundary_probs: Confidence scores for weighting
            
        Returns:
            upsampled: Upsampled embeddings at original resolution
            
        Raises:
            InvalidTextError: If inputs are invalid
            ChunkingError: If upsampling fails
        """
        try:
            if len(smoothed_chunks) == 0:
                logger.warning("Empty smoothed chunks provided")
                embedding_dim = 1 if len(smoothed_chunks.shape) == 1 else smoothed_chunks.shape[1]
                return np.zeros((original_length, embedding_dim))
            
            if original_length <= 0:
                raise InvalidTextError(f"Original length must be positive, got {original_length}")
            
            if len(boundaries) != original_length:
                raise InvalidTextError(
                    f"Boundaries length ({len(boundaries)}) doesn't match "
                    f"original length ({original_length})"
                )
            
            logger.debug(f"Upsampling {len(smoothed_chunks)} chunks to {original_length} positions")
            
            # Create mapping from original positions to chunk indices
            chunk_idx = 0
            upsampled = np.zeros((original_length, smoothed_chunks.shape[1]))
            
            for t in range(original_length):
                # Determine which chunk this position belongs to
                if t < len(boundaries) and boundaries[t] == 1 and t > 0:
                    chunk_idx = min(chunk_idx + 1, len(smoothed_chunks) - 1)
                
                # Get confidence score for weighting
                if boundary_probs is not None and t < len(boundary_probs):
                    confidence = boundary_probs[t]
                else:
                    confidence = self.alpha
                
                # Apply confidence weighting as in equation (9) of paper
                confidence_weighted = self.straight_through_estimator(np.array([confidence]))[0]
                
                # Assign chunk embedding with confidence weighting
                chunk_idx_safe = min(chunk_idx, len(smoothed_chunks) - 1)
                upsampled[t] = confidence_weighted * smoothed_chunks[chunk_idx_safe]
            
            logger.debug(f"Upsampling completed to shape {upsampled.shape}")
            
            return upsampled
            
        except Exception as e:
            logger.error(f"Upsampling failed: {e}")
            if isinstance(e, InvalidTextError):
                raise
            raise ChunkingError(f"Upsampling operation failed: {e}")
    
    def calculate_smoothing_metrics(
        self, 
        original: np.ndarray, 
        smoothed: np.ndarray
    ) -> dict:
        """
        Calculate metrics to evaluate smoothing quality.
        
        Args:
            original: Original embeddings before smoothing
            smoothed: Embeddings after smoothing
            
        Returns:
            Dictionary with smoothing metrics
        """
        if len(original) == 0 or len(smoothed) == 0:
            return {}
        
        if original.shape != smoothed.shape:
            logger.warning(f"Shape mismatch: original {original.shape}, smoothed {smoothed.shape}")
            return {}
        
        # Calculate various smoothing metrics
        mse = np.mean((original - smoothed) ** 2)
        mae = np.mean(np.abs(original - smoothed))
        
        # Cosine similarity between original and smoothed
        orig_flat = original.flatten()
        smooth_flat = smoothed.flatten()
        
        cos_sim = np.dot(orig_flat, smooth_flat) / (
            np.linalg.norm(orig_flat) * np.linalg.norm(smooth_flat) + 1e-8
        )
        
        # Signal-to-noise ratio
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - smoothed) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae), 
            'cosine_similarity': float(cos_sim),
            'snr_db': float(snr),
            'smoothing_strength': 1.0 - cos_sim  # Higher = more smoothing applied
        }
        
        logger.debug(f"Smoothing metrics: MSE={mse:.4f}, Cosine Sim={cos_sim:.4f}")
        
        return metrics
    
    def adaptive_smoothing(
        self, 
        chunk_embeddings: np.ndarray,
        boundary_probs: np.ndarray,
        noise_threshold: float = 0.3
    ) -> np.ndarray:
        """
        Apply adaptive smoothing based on confidence levels.
        
        Args:
            chunk_embeddings: Input chunk embeddings
            boundary_probs: Boundary confidence probabilities
            noise_threshold: Threshold below which to apply stronger smoothing
            
        Returns:
            Adaptively smoothed embeddings
        """
        if len(chunk_embeddings) == 0:
            return chunk_embeddings
        
        smoothed = np.zeros_like(chunk_embeddings)
        smoothed[0] = chunk_embeddings[0]
        
        for t in range(1, len(chunk_embeddings)):
            confidence = boundary_probs[t] if t < len(boundary_probs) else self.alpha
            
            # Adapt smoothing strength based on confidence
            if confidence < noise_threshold:
                # Low confidence -> stronger smoothing
                alpha = max(0.1, confidence * 0.5)
            else:
                # High confidence -> lighter smoothing
                alpha = min(0.9, confidence)
            
            smoothed[t] = alpha * chunk_embeddings[t] + (1 - alpha) * smoothed[t-1]
        
        logger.debug(f"Applied adaptive smoothing with noise threshold {noise_threshold}")
        
        return smoothed