"""
Tests for SimilarityBasedBoundaryDetector
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from dynamic_chunking_hnet.core.boundary_detector import SimilarityBasedBoundaryDetector
from dynamic_chunking_hnet.utils.exceptions import InvalidTextError, EmbeddingError


class TestSimilarityBasedBoundaryDetector:
    """Test suite for SimilarityBasedBoundaryDetector."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = SimilarityBasedBoundaryDetector(embedding_dim=384)
        
        # Create sample embeddings with clear semantic boundaries
        np.random.seed(42)
        self.sample_embeddings = self._create_sample_embeddings()
    
    def _create_sample_embeddings(self) -> np.ndarray:
        """Create sample embeddings with semantic structure."""
        embeddings = []
        
        # First semantic context (3 embeddings)
        base_emb1 = np.random.randn(384) * 0.5
        for _ in range(3):
            noise = np.random.randn(384) * 0.1
            embeddings.append(base_emb1 + noise)
        
        # Second semantic context (4 embeddings)
        base_emb2 = np.random.randn(384) * 0.5
        for _ in range(4):
            noise = np.random.randn(384) * 0.1
            embeddings.append(base_emb2 + noise)
        
        return np.array(embeddings)
    
    def test_initialization_default(self):
        """Test default initialization."""
        detector = SimilarityBasedBoundaryDetector()
        
        assert detector.embedding_dim == 384
        assert detector.device == 'cpu'
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        detector = SimilarityBasedBoundaryDetector(
            embedding_dim=512,
            device='cuda'
        )
        
        assert detector.embedding_dim == 512
        assert detector.device == 'cuda'
    
    def test_validate_embeddings_valid(self):
        """Test validation with valid embeddings."""
        # Should not raise any exception
        self.detector._validate_embeddings(self.sample_embeddings)
    
    def test_validate_embeddings_empty(self):
        """Test validation with empty embeddings."""
        with pytest.raises(InvalidTextError, match="Empty embeddings"):
            self.detector._validate_embeddings(np.array([]))
    
    def test_validate_embeddings_wrong_shape(self):
        """Test validation with wrong shape."""
        wrong_shape = np.random.randn(10)  # 1D instead of 2D
        
        with pytest.raises(InvalidTextError, match="Expected 2D embeddings"):
            self.detector._validate_embeddings(wrong_shape)
    
    def test_validate_embeddings_wrong_dimension(self):
        """Test validation with wrong embedding dimension."""
        wrong_dim = np.random.randn(5, 256)  # 256 instead of 384
        
        with pytest.raises(InvalidTextError, match="Expected embedding dimension"):
            self.detector._validate_embeddings(wrong_dim)
    
    def test_calculate_boundary_probabilities_basic(self):
        """Test basic boundary probability calculation."""
        probs = self.detector.calculate_boundary_probabilities(self.sample_embeddings)
        
        # Check output shape and properties
        assert len(probs) == len(self.sample_embeddings)
        assert all(0 <= p <= 1 for p in probs)
        assert probs[0] == 1.0  # First position always boundary
        
        # Check that boundary is detected between semantic contexts (around position 3)
        assert probs[3] > probs[1]  # Should have higher boundary prob at context switch
    
    def test_calculate_boundary_probabilities_single_token(self):
        """Test with single token (edge case)."""
        single_embedding = self.sample_embeddings[:1]
        probs = self.detector.calculate_boundary_probabilities(single_embedding)
        
        assert len(probs) == 1
        assert probs[0] == 1.0
    
    def test_calculate_boundary_probabilities_two_tokens(self):
        """Test with two tokens (minimal case)."""
        two_embeddings = self.sample_embeddings[:2]
        probs = self.detector.calculate_boundary_probabilities(two_embeddings)
        
        assert len(probs) == 2
        assert probs[0] == 1.0
        assert 0 <= probs[1] <= 1
    
    @patch('dynamic_chunking_hnet.core.boundary_detector.HAS_TORCH', False)
    def test_numpy_fallback(self):
        """Test numpy fallback when PyTorch is not available."""
        detector = SimilarityBasedBoundaryDetector(embedding_dim=384)
        probs = detector.calculate_boundary_probabilities(self.sample_embeddings)
        
        assert len(probs) == len(self.sample_embeddings)
        assert probs[0] == 1.0
        assert all(0 <= p <= 1 for p in probs)
    
    def test_get_discrete_boundaries_basic(self):
        """Test discrete boundary detection."""
        probs = np.array([1.0, 0.3, 0.2, 0.8, 0.1, 0.9, 0.4])
        boundaries = self.detector.get_discrete_boundaries(probs, threshold=0.5)
        
        expected = np.array([1, 0, 0, 1, 0, 1, 0])  # Only probs >= 0.5
        np.testing.assert_array_equal(boundaries, expected)
    
    def test_get_discrete_boundaries_threshold_validation(self):
        """Test threshold validation."""
        probs = np.array([1.0, 0.5, 0.3])
        
        # Invalid thresholds
        with pytest.raises(InvalidTextError):
            self.detector.get_discrete_boundaries(probs, threshold=-0.1)
        
        with pytest.raises(InvalidTextError):
            self.detector.get_discrete_boundaries(probs, threshold=1.1)
    
    def test_get_discrete_boundaries_empty_input(self):
        """Test with empty boundary probabilities."""
        with pytest.raises(InvalidTextError):
            self.detector.get_discrete_boundaries(np.array([]), threshold=0.5)
    
    def test_get_discrete_boundaries_first_always_boundary(self):
        """Test that first position is always marked as boundary."""
        probs = np.array([0.1, 0.2, 0.3])  # All below threshold
        boundaries = self.detector.get_discrete_boundaries(probs, threshold=0.5)
        
        assert boundaries[0] == 1  # First position forced to be boundary
    
    def test_adaptive_threshold_basic(self):
        """Test adaptive threshold calculation."""
        probs = np.array([1.0, 0.8, 0.3, 0.6, 0.2, 0.9, 0.1, 0.4])
        
        threshold = self.detector.adaptive_threshold(probs, target_compression=4.0)
        
        # Should select threshold to get approximately 2 boundaries (8 tokens / 4 compression)
        assert 0.1 <= threshold <= 1.0
    
    def test_adaptive_threshold_edge_cases(self):
        """Test adaptive threshold with edge cases."""
        # Single token
        single_prob = np.array([1.0])
        threshold = self.detector.adaptive_threshold(single_prob, target_compression=2.0)
        assert threshold == 0.5
        
        # Very high compression ratio
        probs = np.array([1.0, 0.5, 0.3, 0.8])
        threshold = self.detector.adaptive_threshold(probs, target_compression=10.0)
        assert threshold >= 0.1  # Minimum threshold
    
    def test_adaptive_threshold_minimum_enforcement(self):
        """Test that minimum threshold is enforced."""
        # All low probabilities
        low_probs = np.array([0.05, 0.02, 0.01, 0.03, 0.04])
        threshold = self.detector.adaptive_threshold(low_probs, target_compression=2.0)
        
        assert threshold >= 0.1  # Should enforce minimum
    
    @pytest.mark.parametrize("embedding_dim", [128, 256, 384, 512, 768])
    def test_different_embedding_dimensions(self, embedding_dim):
        """Test with different embedding dimensions."""
        detector = SimilarityBasedBoundaryDetector(embedding_dim=embedding_dim)
        
        # Create embeddings with the correct dimension
        embeddings = np.random.randn(5, embedding_dim)
        
        probs = detector.calculate_boundary_probabilities(embeddings)
        
        assert len(probs) == 5
        assert probs[0] == 1.0
        assert all(0 <= p <= 1 for p in probs)
    
    @pytest.mark.parametrize("sequence_length", [1, 2, 5, 10, 50, 100])
    def test_different_sequence_lengths(self, sequence_length):
        """Test with different sequence lengths."""
        embeddings = np.random.randn(sequence_length, 384)
        
        probs = self.detector.calculate_boundary_probabilities(embeddings)
        
        assert len(probs) == sequence_length
        if sequence_length > 0:
            assert probs[0] == 1.0
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic for same input."""
        # Run multiple times with same input
        probs1 = self.detector.calculate_boundary_probabilities(self.sample_embeddings)
        probs2 = self.detector.calculate_boundary_probabilities(self.sample_embeddings)
        
        np.testing.assert_array_almost_equal(probs1, probs2)
    
    def test_similarity_properties(self):
        """Test that similar embeddings have low boundary probabilities."""
        # Create very similar embeddings
        base = np.random.randn(384)
        similar_embeddings = np.array([
            base + np.random.randn(384) * 0.01,  # Very small noise
            base + np.random.randn(384) * 0.01,
            base + np.random.randn(384) * 0.01,
        ])
        
        probs = self.detector.calculate_boundary_probabilities(similar_embeddings)
        
        # Boundary probabilities should be low for similar embeddings
        assert probs[1] < 0.7  # Second position should have low boundary prob
        assert probs[2] < 0.7  # Third position should have low boundary prob
    
    def test_dissimilar_embeddings(self):
        """Test that dissimilar embeddings have high boundary probabilities."""
        # Create very different embeddings
        dissimilar_embeddings = np.array([
            np.random.randn(384) * 2,
            -np.random.randn(384) * 2,  # Opposite direction
            np.random.randn(384) * 2,
        ])
        
        probs = self.detector.calculate_boundary_probabilities(dissimilar_embeddings)
        
        # Should detect boundaries between dissimilar embeddings
        assert probs[1] > 0.3  # Should have reasonable boundary probability