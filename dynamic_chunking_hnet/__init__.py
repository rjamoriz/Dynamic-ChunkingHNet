"""
H-Net Dynamic Chunking Implementation
=====================================

A comprehensive implementation of the H-Net (Hierarchical Network) dynamic chunking 
algorithm for Retrieval-Augmented Generation (RAG) systems.

This package provides:
- Dynamic boundary detection using cosine similarity
- Routing module for adaptive chunk creation
- Smoothing module for gradient flow stability
- Comprehensive evaluation metrics
- Interactive visualizations

Usage:
    from dynamic_chunking_hnet import DynamicChunkingPipeline
    
    pipeline = DynamicChunkingPipeline(compression_ratio=6.0)
    result = pipeline.process_text("Your text here...")
"""

__version__ = "1.0.0"
__author__ = "Dynamic ChunkingHNet Team"

# Core imports
from .core.pipeline import DynamicChunkingPipeline
from .core.boundary_detector import SimilarityBasedBoundaryDetector
from .core.routing_module import RoutingModule  
from .core.smoothing_module import SmoothingModule

# Evaluation imports
from .evaluation.metrics import ChunkingQualityMetrics

# Utility imports
from .utils.exceptions import ChunkingError, InvalidTextError, EmbeddingError

__all__ = [
    'DynamicChunkingPipeline',
    'SimilarityBasedBoundaryDetector', 
    'RoutingModule',
    'SmoothingModule',
    'ChunkingQualityMetrics',
    'ChunkingError',
    'InvalidTextError',
    'EmbeddingError',
]