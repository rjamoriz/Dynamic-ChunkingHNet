"""
Tests for DynamicChunkingPipeline
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from dynamic_chunking_hnet.core.pipeline import DynamicChunkingPipeline
from dynamic_chunking_hnet.utils.exceptions import InvalidTextError, EmbeddingError, ModelNotFoundError


class TestDynamicChunkingPipeline:
    """Test suite for DynamicChunkingPipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.pipeline = DynamicChunkingPipeline(compression_ratio=6.0)
        
        # Sample texts for testing
        self.simple_text = "This is a simple test. It contains multiple sentences. Each sentence should be processed correctly."
        self.long_text = " ".join([f"Sentence {i} with some content." for i in range(20)])
        self.short_text = "Short text."
    
    def test_initialization_default(self):
        """Test default initialization."""
        pipeline = DynamicChunkingPipeline()
        
        assert pipeline.compression_ratio == 6.0
        assert pipeline.embedding_model is None
        assert pipeline.embedding_dim == 384
        assert pipeline.cache_embeddings is True
        assert pipeline.device == 'cpu'
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        pipeline = DynamicChunkingPipeline(
            compression_ratio=8.0,
            embedding_model="test-model",
            embedding_dim=512,
            cache_embeddings=False,
            device='cuda'
        )
        
        assert pipeline.compression_ratio == 8.0
        assert pipeline.embedding_model == "test-model"
        assert pipeline.embedding_dim == 512
        assert pipeline.cache_embeddings is False
        assert pipeline.device == 'cuda'
    
    def test_initialization_invalid_compression_ratio(self):
        """Test initialization with invalid compression ratio."""
        with pytest.raises(InvalidTextError):
            DynamicChunkingPipeline(compression_ratio=0.5)
    
    def test_validate_text_input_valid(self):
        """Test text validation with valid input."""
        # Should not raise any exception
        self.pipeline._validate_text_input(self.simple_text)
    
    def test_validate_text_input_empty(self):
        """Test validation with empty text."""
        with pytest.raises(InvalidTextError, match="cannot be empty"):
            self.pipeline._validate_text_input("")
    
    def test_validate_text_input_whitespace_only(self):
        """Test validation with whitespace-only text."""
        with pytest.raises(InvalidTextError, match="cannot be empty"):
            self.pipeline._validate_text_input("   \n\t  ")
    
    def test_validate_text_input_not_string(self):
        """Test validation with non-string input."""
        with pytest.raises(InvalidTextError, match="must be a string"):
            self.pipeline._validate_text_input(123)
    
    def test_validate_text_input_single_token(self):
        """Test validation with single token."""
        with pytest.raises(InvalidTextError, match="at least 2 tokens"):
            self.pipeline._validate_text_input("single")
    
    @patch('dynamic_chunking_hnet.core.pipeline.HAS_SKLEARN', True)
    def test_get_embeddings_tfidf_fallback(self):
        """Test embedding generation using TF-IDF fallback."""
        embeddings, tokens = self.pipeline.get_embeddings(self.simple_text)
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[1] == self.pipeline.embedding_dim
        assert len(tokens) == embeddings.shape[0]
        assert tokens == self.simple_text.split()
    
    def test_get_embeddings_random_fallback(self):
        """Test embedding generation using random fallback."""
        # Patch to disable sklearn
        with patch('dynamic_chunking_hnet.core.pipeline.HAS_SKLEARN', False):
            pipeline = DynamicChunkingPipeline()
            embeddings, tokens = pipeline.get_embeddings(self.simple_text)
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[1] == pipeline.embedding_dim
        assert len(tokens) == embeddings.shape[0]
    
    def test_get_embeddings_caching(self):
        """Test embedding caching functionality."""
        # First call
        embeddings1, tokens1 = self.pipeline.get_embeddings(self.simple_text)
        
        # Second call with same text
        embeddings2, tokens2 = self.pipeline.get_embeddings(self.simple_text)
        
        # Results should be identical (from cache)
        np.testing.assert_array_equal(embeddings1, embeddings2)
        assert tokens1 == tokens2
    
    def test_get_embeddings_cache_disabled(self):
        """Test with caching disabled."""
        pipeline = DynamicChunkingPipeline(cache_embeddings=False)
        
        embeddings1, tokens1 = pipeline.get_embeddings(self.simple_text)
        embeddings2, tokens2 = pipeline.get_embeddings(self.simple_text)
        
        # Should still work but may be different due to randomness
        assert embeddings1.shape == embeddings2.shape
        assert tokens1 == tokens2
    
    def test_process_text_basic(self):
        """Test basic text processing."""
        result = self.pipeline.process_text(self.simple_text)
        
        # Check required result keys
        required_keys = [
            'original_text', 'tokens', 'num_tokens', 'chunks', 'num_chunks',
            'compression_ratio_target', 'compression_ratio_achieved',
            'boundary_probs', 'boundaries', 'processing_time', 'timing'
        ]
        
        for key in required_keys:
            assert key in result
        
        # Check data validity
        assert result['original_text'] == self.simple_text
        assert result['num_tokens'] > 0
        assert result['num_chunks'] > 0
        assert result['compression_ratio_achieved'] > 0
        assert result['processing_time'] > 0
    
    def test_process_text_with_intermediate_results(self):
        """Test processing with intermediate results."""
        result = self.pipeline.process_text(
            self.simple_text,
            return_intermediate=True,
            return_metrics=False
        )
        
        # Should include intermediate processing results
        intermediate_keys = [
            'embeddings', 'chunk_embeddings', 'chunk_means',
            'smoothed_chunks', 'processed_chunks', 'reconstructed_embeddings'
        ]
        
        for key in intermediate_keys:
            assert key in result
    
    def test_process_text_with_metrics(self):
        """Test processing with quality metrics."""
        result = self.pipeline.process_text(
            self.simple_text,
            return_intermediate=False,
            return_metrics=True
        )
        
        assert 'quality_metrics' in result
        assert isinstance(result['quality_metrics'], dict)
    
    def test_process_text_different_lengths(self):
        """Test processing texts of different lengths."""
        texts = [
            "Short text here.",
            self.simple_text,
            self.long_text
        ]
        
        for text in texts:
            result = self.pipeline.process_text(text)
            
            assert result['num_tokens'] > 0
            assert result['num_chunks'] > 0
            assert result['compression_ratio_achieved'] > 0
    
    def test_process_text_timing_breakdown(self):
        """Test that timing information is provided."""
        result = self.pipeline.process_text(self.simple_text)
        
        timing = result['timing']
        timing_keys = ['encoding', 'routing', 'smoothing', 'dechunking', 'total']
        
        for key in timing_keys:
            assert key in timing
            assert timing[key] >= 0
        
        # Total should be sum of parts (approximately)
        parts_sum = sum(timing[key] for key in timing_keys[:-1])
        assert abs(timing['total'] - parts_sum) < 0.1  # Allow small discrepancy
    
    def test_batch_process_basic(self):
        """Test batch processing functionality."""
        texts = [self.simple_text, self.short_text, "Another test sentence here."]
        
        results = self.pipeline.batch_process(texts)
        
        assert len(results) == len(texts)
        
        for i, result in enumerate(results):
            if 'error' not in result:
                assert result['original_text'] == texts[i]
                assert result['num_chunks'] > 0
    
    def test_batch_process_empty_list(self):
        """Test batch processing with empty list."""
        results = self.pipeline.batch_process([])
        assert results == []
    
    def test_batch_process_error_handling(self):
        """Test batch processing with some invalid texts."""
        texts = [
            self.simple_text,
            "",  # Invalid - empty text
            self.short_text,
            "single",  # Invalid - single token
        ]
        
        results = self.pipeline.batch_process(texts)
        
        assert len(results) == len(texts)
        
        # First and third should succeed
        assert 'error' not in results[0]
        assert 'error' not in results[2]
        
        # Second and fourth should have errors
        assert 'error' in results[1]
        assert 'error' in results[3]
    
    def test_cache_management(self):
        """Test cache management functions."""
        # Process some text to populate cache
        self.pipeline.process_text(self.simple_text)
        
        # Check cache stats
        stats = self.pipeline.get_cache_stats()
        assert stats['cache_enabled'] is True
        assert stats['cache_size'] >= 0
        
        # Clear cache
        self.pipeline.clear_cache()
        
        # Check cache is cleared
        stats_after = self.pipeline.get_cache_stats()
        assert stats_after['cache_size'] == 0
    
    @pytest.mark.parametrize("compression_ratio", [3.0, 6.0, 9.0, 12.0])
    def test_different_compression_ratios(self, compression_ratio):
        """Test with different compression ratios."""
        pipeline = DynamicChunkingPipeline(compression_ratio=compression_ratio)
        result = pipeline.process_text(self.long_text)
        
        assert result['compression_ratio_target'] == compression_ratio
        
        # Higher compression should generally produce fewer chunks
        achieved_compression = result['compression_ratio_achieved']
        assert achieved_compression > 0
        
        # Should be roughly in the ballpark of target
        assert 0.3 * compression_ratio <= achieved_compression <= 3.0 * compression_ratio
    
    def test_compression_ratio_consistency(self):
        """Test that compression ratio affects chunk count as expected."""
        low_compression_pipeline = DynamicChunkingPipeline(compression_ratio=3.0)
        high_compression_pipeline = DynamicChunkingPipeline(compression_ratio=12.0)
        
        result_low = low_compression_pipeline.process_text(self.long_text)
        result_high = high_compression_pipeline.process_text(self.long_text)
        
        # Low compression should produce more chunks
        assert result_low['num_chunks'] >= result_high['num_chunks']
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic for same input."""
        result1 = self.pipeline.process_text(self.simple_text)
        result2 = self.pipeline.process_text(self.simple_text)
        
        # Key results should be identical
        assert result1['num_chunks'] == result2['num_chunks']
        assert result1['num_tokens'] == result2['num_tokens']
        np.testing.assert_array_equal(
            result1['boundary_probs'], 
            result2['boundary_probs']
        )
    
    def test_error_propagation(self):
        """Test that errors are properly propagated."""
        # Test with invalid text
        with pytest.raises(InvalidTextError):
            self.pipeline.process_text("")
    
    def test_memory_efficiency_large_text(self):
        """Test memory efficiency with large text."""
        # Create a large text
        large_text = " ".join([f"Sentence {i} with substantial content here." for i in range(500)])
        
        result = self.pipeline.process_text(large_text)
        
        # Should still work
        assert result['num_chunks'] > 0
        assert result['compression_ratio_achieved'] > 0
        assert len(result['chunks']) > 0
    
    @patch('dynamic_chunking_hnet.core.pipeline.HAS_TORCH', True)
    def test_transformer_model_integration(self):
        """Test integration with transformer models (mocked)."""
        # Mock transformer components
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[101, 102, 103, 104, 105]],  # Mock token IDs
            'attention_mask': [[1, 1, 1, 1, 1]]
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = ['token1', 'token2', 'token3']
        
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = [[np.random.randn(3, 384)]]  # Mock embeddings
        mock_model.return_value = mock_outputs
        
        with patch('dynamic_chunking_hnet.core.pipeline.AutoTokenizer') as mock_auto_tokenizer, \
             patch('dynamic_chunking_hnet.core.pipeline.AutoModel') as mock_auto_model:
            
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            mock_auto_model.from_pretrained.return_value = mock_model
            
            # Create pipeline with transformer model
            pipeline = DynamicChunkingPipeline(embedding_model="bert-base-uncased")
            
            # Should initialize without error
            assert pipeline.embedding_model == "bert-base-uncased"
    
    def test_performance_monitoring(self):
        """Test that performance metrics are captured."""
        result = self.pipeline.process_text(self.simple_text)
        
        # Check timing information is reasonable
        timing = result['timing']
        
        assert timing['total'] > 0
        assert timing['encoding'] >= 0
        assert timing['routing'] >= 0
        assert timing['smoothing'] >= 0
        assert timing['dechunking'] >= 0
        
        # Total processing time should be reasonable (not too slow)
        assert timing['total'] < 10.0  # Should complete within 10 seconds
    
    def test_chunk_quality_basic(self):
        """Test basic chunk quality properties."""
        result = self.pipeline.process_text(self.simple_text)
        
        chunks = result['chunks']
        
        # All chunks should be non-empty
        assert all(len(chunk) > 0 for chunk in chunks)
        
        # Total tokens should be preserved
        total_chunk_tokens = sum(len(chunk) for chunk in chunks)
        assert total_chunk_tokens == result['num_tokens']
        
        # Chunks should contain all original tokens
        all_chunk_tokens = []
        for chunk in chunks:
            all_chunk_tokens.extend(chunk)
        
        original_tokens = result['tokens']
        assert len(all_chunk_tokens) == len(original_tokens)
    
    def test_boundary_probability_properties(self):
        """Test properties of boundary probabilities."""
        result = self.pipeline.process_text(self.simple_text)
        
        boundary_probs = result['boundary_probs']
        
        # Should have probabilities for each token
        assert len(boundary_probs) == result['num_tokens']
        
        # All probabilities should be in [0, 1]
        assert all(0 <= p <= 1 for p in boundary_probs)
        
        # First position should always be a boundary
        assert boundary_probs[0] == 1.0
    
    def test_configuration_persistence(self):
        """Test that pipeline configuration is maintained."""
        pipeline = DynamicChunkingPipeline(
            compression_ratio=8.5,
            embedding_dim=512
        )
        
        result = pipeline.process_text(self.simple_text)
        
        assert result['compression_ratio_target'] == 8.5
        # Embedding dim should be reflected in intermediate results
        if 'embeddings' in result:
            assert result['embeddings'].shape[1] == 512