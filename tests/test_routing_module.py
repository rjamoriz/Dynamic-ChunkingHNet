"""
Tests for RoutingModule
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from dynamic_chunking_hnet.core.routing_module import RoutingModule
from dynamic_chunking_hnet.core.boundary_detector import SimilarityBasedBoundaryDetector
from dynamic_chunking_hnet.utils.exceptions import InvalidTextError, ChunkingError


class TestRoutingModule:
    """Test suite for RoutingModule."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.router = RoutingModule(target_compression_ratio=6.0)
        
        # Create sample data
        np.random.seed(42)
        self.sample_embeddings = np.random.randn(12, 384)
        self.sample_tokens = [f"token{i}" for i in range(12)]
        
    def test_initialization_default(self):
        """Test default initialization."""
        router = RoutingModule()
        
        assert router.target_compression_ratio == 6.0
        assert isinstance(router.boundary_detector, SimilarityBasedBoundaryDetector)
    
    def test_initialization_custom_compression_ratio(self):
        """Test initialization with custom compression ratio."""
        router = RoutingModule(target_compression_ratio=8.0)
        
        assert router.target_compression_ratio == 8.0
    
    def test_initialization_invalid_compression_ratio(self):
        """Test initialization with invalid compression ratio."""
        with pytest.raises(InvalidTextError):
            RoutingModule(target_compression_ratio=0.5)
        
        with pytest.raises(InvalidTextError):
            RoutingModule(target_compression_ratio=-1.0)
    
    def test_initialization_custom_boundary_detector(self):
        """Test initialization with custom boundary detector."""
        custom_detector = SimilarityBasedBoundaryDetector(embedding_dim=512)
        router = RoutingModule(
            target_compression_ratio=4.0,
            boundary_detector=custom_detector
        )
        
        assert router.boundary_detector is custom_detector
        assert router.target_compression_ratio == 4.0
    
    def test_calculate_ratio_loss_basic(self):
        """Test ratio loss calculation."""
        boundary_probs = np.array([1.0, 0.3, 0.8, 0.2, 0.9])
        boundaries = np.array([1, 0, 1, 0, 1])
        
        loss = self.router.calculate_ratio_loss(boundary_probs, boundaries)
        
        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative
    
    def test_calculate_ratio_loss_empty_arrays(self):
        """Test ratio loss with empty arrays."""
        empty_array = np.array([])
        
        loss = self.router.calculate_ratio_loss(empty_array, empty_array)
        assert loss == 0.0
    
    def test_adaptive_threshold_selection_basic(self):
        """Test adaptive threshold selection."""
        boundary_probs = np.array([1.0, 0.8, 0.6, 0.4, 0.9, 0.3, 0.7, 0.1])
        
        threshold = self.router.adaptive_threshold_selection(boundary_probs)
        
        assert 0.1 <= threshold <= 1.0  # Should be in valid range
    
    def test_adaptive_threshold_selection_empty(self):
        """Test adaptive threshold with empty probabilities."""
        threshold = self.router.adaptive_threshold_selection(np.array([]))
        assert threshold == 0.5
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        # Should not raise any exception
        self.router._validate_inputs(self.sample_embeddings, self.sample_tokens)
    
    def test_validate_inputs_empty_embeddings(self):
        """Test validation with empty embeddings."""
        with pytest.raises(InvalidTextError, match="Empty embeddings"):
            self.router._validate_inputs(np.array([]), [])
    
    def test_validate_inputs_mismatched_lengths(self):
        """Test validation with mismatched token and embedding counts."""
        short_tokens = ["token1", "token2"]
        
        with pytest.raises(InvalidTextError, match="Token count.*doesn't match"):
            self.router._validate_inputs(self.sample_embeddings, short_tokens)
    
    def test_create_chunks_basic(self):
        """Test basic chunk creation."""
        result = self.router.create_chunks(self.sample_embeddings, self.sample_tokens)
        
        # Check result structure
        assert 'chunks' in result
        assert 'chunk_embeddings' in result
        assert 'boundary_probs' in result
        assert 'boundaries' in result
        assert 'compression_ratio' in result
        assert 'num_chunks' in result
        
        # Check data validity
        assert len(result['chunks']) == len(result['chunk_embeddings'])
        assert len(result['boundary_probs']) == len(self.sample_embeddings)
        assert result['num_chunks'] > 0
        assert result['compression_ratio'] > 0
    
    def test_create_chunks_without_tokens(self):
        """Test chunk creation without text tokens."""
        result = self.router.create_chunks(self.sample_embeddings, text_tokens=None)
        
        # Should still work but chunks will be empty
        assert 'chunks' in result
        assert result['chunks'] == []  # No text tokens provided
        assert 'chunk_embeddings' in result
        assert len(result['chunk_embeddings']) > 0  # Should still have embedding chunks
    
    def test_create_chunks_single_embedding(self):
        """Test with single embedding."""
        single_embedding = self.sample_embeddings[:1]
        single_token = self.sample_tokens[:1]
        
        result = self.router.create_chunks(single_embedding, single_token)
        
        assert result['num_chunks'] == 1
        assert len(result['chunks']) == 1
        assert result['chunks'][0] == single_token
    
    def test_create_chunks_compression_ratio_effect(self):
        """Test that different compression ratios produce different numbers of chunks."""
        # Low compression (more chunks)
        router_low = RoutingModule(target_compression_ratio=3.0)
        result_low = router_low.create_chunks(self.sample_embeddings, self.sample_tokens)
        
        # High compression (fewer chunks)
        router_high = RoutingModule(target_compression_ratio=12.0)
        result_high = router_high.create_chunks(self.sample_embeddings, self.sample_tokens)
        
        # Low compression should produce more chunks
        assert result_low['num_chunks'] >= result_high['num_chunks']
    
    @pytest.mark.parametrize("compression_ratio", [2.0, 4.0, 6.0, 8.0, 10.0])
    def test_different_compression_ratios(self, compression_ratio):
        """Test with different compression ratios."""
        router = RoutingModule(target_compression_ratio=compression_ratio)
        result = router.create_chunks(self.sample_embeddings, self.sample_tokens)
        
        assert result['target_compression_ratio'] == compression_ratio
        assert result['num_chunks'] > 0
        assert result['compression_ratio'] > 0
        
        # Achieved compression should be roughly in the ballpark of target
        achieved = result['compression_ratio']
        assert 0.5 * compression_ratio <= achieved <= 2.0 * compression_ratio
    
    def test_chunk_boundary_consistency(self):
        """Test that chunks are consistent with detected boundaries."""
        result = self.router.create_chunks(self.sample_embeddings, self.sample_tokens)
        
        boundaries = result['boundaries']
        chunks = result['chunks']
        
        # Count boundaries and compare with chunks
        num_boundaries = np.sum(boundaries)
        num_chunks = len(chunks)
        
        # Should have roughly similar numbers (boundaries create chunks)
        assert abs(num_boundaries - num_chunks) <= 2  # Allow small discrepancy
    
    def test_chunk_content_validity(self):
        """Test that chunk contents are valid."""
        result = self.router.create_chunks(self.sample_embeddings, self.sample_tokens)
        
        chunks = result['chunks']
        
        # All chunks should be non-empty
        assert all(len(chunk) > 0 for chunk in chunks)
        
        # All tokens should be preserved
        all_chunk_tokens = []
        for chunk in chunks:
            all_chunk_tokens.extend(chunk)
        
        assert set(all_chunk_tokens) == set(self.sample_tokens)
    
    def test_embedding_chunk_consistency(self):
        """Test that embedding chunks match token chunks."""
        result = self.router.create_chunks(self.sample_embeddings, self.sample_tokens)
        
        chunks = result['chunks']
        chunk_embeddings = result['chunk_embeddings']
        
        assert len(chunks) == len(chunk_embeddings)
        
        # Each chunk should have matching number of embeddings and tokens
        for tokens, embeddings in zip(chunks, chunk_embeddings):
            assert len(tokens) == len(embeddings)
    
    def test_get_chunk_statistics_basic(self):
        """Test chunk statistics calculation."""
        result = self.router.create_chunks(self.sample_embeddings, self.sample_tokens)
        stats = self.router.get_chunk_statistics(result)
        
        # Check required statistics
        required_keys = [
            'num_chunks', 'total_tokens', 'min_chunk_size', 'max_chunk_size',
            'mean_chunk_size', 'std_chunk_size', 'median_chunk_size',
            'chunk_size_variance'
        ]
        
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float, np.number))
        
        # Sanity checks
        assert stats['num_chunks'] > 0
        assert stats['total_tokens'] == len(self.sample_tokens)
        assert stats['min_chunk_size'] <= stats['max_chunk_size']
    
    def test_get_chunk_statistics_with_boundary_probs(self):
        """Test statistics with boundary probabilities."""
        result = self.router.create_chunks(self.sample_embeddings, self.sample_tokens)
        stats = self.router.get_chunk_statistics(result)
        
        # Should include boundary statistics
        boundary_keys = [
            'mean_boundary_prob', 'std_boundary_prob',
            'max_boundary_prob', 'min_boundary_prob'
        ]
        
        for key in boundary_keys:
            assert key in stats
            assert 0 <= stats[key] <= 1  # Probabilities should be in [0,1]
    
    def test_get_chunk_statistics_empty_result(self):
        """Test statistics with empty result."""
        empty_result = {'chunks': [], 'chunk_embeddings': []}
        stats = self.router.get_chunk_statistics(empty_result)
        
        assert stats == {}
    
    def test_error_handling_invalid_embeddings(self):
        """Test error handling with invalid embeddings."""
        invalid_embeddings = "not an array"
        
        with pytest.raises(ChunkingError):
            self.router.create_chunks(invalid_embeddings, self.sample_tokens)
    
    def test_memory_efficiency_large_input(self):
        """Test memory efficiency with larger input."""
        # Create larger input
        large_embeddings = np.random.randn(1000, 384)
        large_tokens = [f"token{i}" for i in range(1000)]
        
        result = self.router.create_chunks(large_embeddings, large_tokens)
        
        # Should still work and produce reasonable results
        assert result['num_chunks'] > 0
        assert len(result['chunks']) > 0
        assert result['compression_ratio'] > 1.0
    
    @patch.object(SimilarityBasedBoundaryDetector, 'calculate_boundary_probabilities')
    def test_boundary_detector_integration(self, mock_boundary_calc):
        """Test integration with boundary detector."""
        # Mock boundary detector response
        mock_probs = np.array([1.0, 0.3, 0.7, 0.2, 0.8])
        mock_boundary_calc.return_value = mock_probs
        
        small_embeddings = self.sample_embeddings[:5]
        small_tokens = self.sample_tokens[:5]
        
        result = self.router.create_chunks(small_embeddings, small_tokens)
        
        # Verify boundary detector was called
        mock_boundary_calc.assert_called_once_with(small_embeddings)
        
        # Verify mock probabilities were used
        np.testing.assert_array_equal(result['boundary_probs'], mock_probs)
    
    def test_threshold_effect_on_chunks(self):
        """Test how threshold affects number of chunks."""
        # Create embeddings with known boundary probabilities
        embeddings = self.sample_embeddings[:8]
        tokens = self.sample_tokens[:8]
        
        # Get boundary probabilities
        result = self.router.create_chunks(embeddings, tokens)
        boundary_probs = result['boundary_probs']
        
        # Test with different manual thresholds
        low_threshold = 0.3
        high_threshold = 0.7
        
        boundaries_low = self.router.boundary_detector.get_discrete_boundaries(
            boundary_probs, low_threshold
        )
        boundaries_high = self.router.boundary_detector.get_discrete_boundaries(
            boundary_probs, high_threshold
        )
        
        # Lower threshold should produce more boundaries
        assert np.sum(boundaries_low) >= np.sum(boundaries_high)