"""
Pytest configuration and fixtures for Dynamic ChunkingHNet tests.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import List, Tuple

from dynamic_chunking_hnet.core.boundary_detector import SimilarityBasedBoundaryDetector
from dynamic_chunking_hnet.core.routing_module import RoutingModule
from dynamic_chunking_hnet.core.smoothing_module import SmoothingModule
from dynamic_chunking_hnet.core.pipeline import DynamicChunkingPipeline


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(10, 384)


@pytest.fixture
def sample_tokens() -> List[str]:
    """Create sample tokens for testing."""
    return [f"token{i}" for i in range(10)]


@pytest.fixture
def sample_text() -> str:
    """Create sample text for testing."""
    return (
        "Machine learning is fascinating. "
        "Natural language processing enables computers to understand text. "
        "Deep learning has revolutionized artificial intelligence applications."
    )


@pytest.fixture
def structured_embeddings() -> np.ndarray:
    """Create embeddings with clear semantic structure."""
    np.random.seed(42)
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
    
    # Third semantic context (3 embeddings)
    base_emb3 = np.random.randn(384) * 0.5
    for _ in range(3):
        noise = np.random.randn(384) * 0.1
        embeddings.append(base_emb3 + noise)
    
    return np.array(embeddings)


@pytest.fixture
def boundary_detector() -> SimilarityBasedBoundaryDetector:
    """Create a boundary detector instance."""
    return SimilarityBasedBoundaryDetector(embedding_dim=384)


@pytest.fixture
def routing_module() -> RoutingModule:
    """Create a routing module instance."""
    return RoutingModule(target_compression_ratio=6.0)


@pytest.fixture
def smoothing_module() -> SmoothingModule:
    """Create a smoothing module instance."""
    return SmoothingModule()


@pytest.fixture
def pipeline() -> DynamicChunkingPipeline:
    """Create a pipeline instance."""
    return DynamicChunkingPipeline(compression_ratio=6.0)


@pytest.fixture
def temp_cache_dir() -> str:
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def long_text() -> str:
    """Create a longer text for stress testing."""
    sentences = [
        "Artificial intelligence has made significant progress in recent years.",
        "Machine learning algorithms can now process vast amounts of data efficiently.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision systems can recognize objects and scenes in images.",
        "Deep learning networks have achieved remarkable performance in many domains.",
        "Transformer architectures have revolutionized natural language processing.",
        "Self-attention mechanisms allow models to focus on relevant information.",
        "Pre-trained language models can be fine-tuned for specific tasks.",
        "Transfer learning has made AI more accessible to a broader audience.",
        "Ethical considerations are becoming increasingly important in AI development."
    ]
    return " ".join(sentences)


@pytest.fixture
def multi_language_text() -> str:
    """Create text with multiple languages for testing."""
    return (
        "Hello world. This is English text. "
        "Hola mundo. Este es texto en español. "
        "Bonjour le monde. Ceci est du texte français."
    )


@pytest.fixture
def technical_text() -> str:
    """Create technical text with domain-specific vocabulary."""
    return (
        "The convolutional neural network architecture consists of multiple layers. "
        "Each layer applies filters to extract features from the input data. "
        "Pooling operations reduce the spatial dimensions of feature maps. "
        "The final layers typically consist of fully connected networks. "
        "Backpropagation is used to train the network parameters through gradient descent."
    )


# Parametrized fixtures for testing with different configurations
@pytest.fixture(params=[3.0, 6.0, 9.0, 12.0])
def compression_ratios(request) -> float:
    """Provide different compression ratios for testing."""
    return request.param


@pytest.fixture(params=[128, 256, 384, 512, 768])
def embedding_dimensions(request) -> int:
    """Provide different embedding dimensions for testing."""
    return request.param


@pytest.fixture(params=[0.3, 0.5, 0.7])
def boundary_thresholds(request) -> float:
    """Provide different boundary thresholds for testing."""
    return request.param


# Performance testing fixtures
@pytest.fixture
def performance_text() -> str:
    """Create large text for performance testing."""
    base_sentence = "This is a test sentence with multiple words and some complexity. "
    return base_sentence * 1000  # Create very long text


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield initial_memory
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Warn if memory increase is significant
    if memory_increase > 100:  # More than 100MB increase
        pytest.warn(f"Significant memory increase: {memory_increase:.1f}MB")


# Error testing fixtures
@pytest.fixture
def invalid_texts() -> List[str]:
    """Provide various invalid text inputs for error testing."""
    return [
        "",  # Empty string
        "   ",  # Whitespace only
        "single",  # Single token
        None,  # None value (will cause type error)
    ]


@pytest.fixture
def malformed_embeddings() -> List[np.ndarray]:
    """Provide malformed embeddings for error testing."""
    return [
        np.array([]),  # Empty array
        np.random.randn(10),  # 1D instead of 2D
        np.random.randn(5, 100),  # Wrong embedding dimension
    ]


# Mock fixtures for external dependencies
@pytest.fixture
def mock_transformer_model():
    """Mock transformer model components."""
    from unittest.mock import Mock
    
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        'input_ids': [[101, 102, 103, 104]],
        'attention_mask': [[1, 1, 1, 1]]
    }
    mock_tokenizer.convert_ids_to_tokens.return_value = ['token1', 'token2', 'token3']
    
    mock_model = Mock()
    mock_outputs = Mock()
    mock_outputs.last_hidden_state = [np.random.randn(3, 384)]
    mock_model.return_value = mock_outputs
    
    return mock_tokenizer, mock_model


# Quality testing fixtures
@pytest.fixture
def quality_test_cases() -> List[Tuple[str, dict]]:
    """Provide test cases with expected quality metrics."""
    return [
        (
            "First sentence here. Second sentence follows. Third completes the set.",
            {
                'expected_min_chunks': 1,
                'expected_max_chunks': 5,
                'expected_min_compression': 2.0,
                'expected_max_compression': 15.0
            }
        ),
        (
            "A" * 100,  # Repetitive text
            {
                'expected_min_chunks': 1,
                'expected_max_chunks': 10,
                'expected_min_compression': 10.0,
                'expected_max_compression': 100.0
            }
        )
    ]


# Benchmark fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        'max_processing_time': 5.0,  # seconds
        'max_memory_usage': 500,  # MB
        'min_compression_ratio': 1.0,
        'max_compression_ratio': 20.0
    }