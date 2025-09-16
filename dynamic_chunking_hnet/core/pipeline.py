"""
Dynamic Chunking Pipeline

Complete H-Net dynamic chunking pipeline combining routing and smoothing modules.
Implements the full algorithm steps from the paper.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Any
import logging

# Optional imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .routing_module import RoutingModule
from .smoothing_module import SmoothingModule
from .boundary_detector import SimilarityBasedBoundaryDetector
from ..utils.exceptions import InvalidTextError, EmbeddingError, ModelNotFoundError

logger = logging.getLogger(__name__)


class DynamicChunkingPipeline:
    """
    Complete H-Net dynamic chunking pipeline combining routing and smoothing modules.
    Implements the full algorithm steps from the paper.
    
    Algorithm steps:
    1. Encoding: Process raw text through encoder networks (embeddings)
    2. Routing: Predict boundaries based on representation similarity
    3. Chunking: Downsample by selecting boundary-marked vectors
    4. Smoothing: Apply smoothing for gradient flow
    5. Main Processing: (simulated - would apply Transformer/Mamba)
    6. Dechunking: Upsample using smoothing and confidence-weighted decompression
    """
    
    def __init__(
        self,
        compression_ratio: float = 6.0,
        embedding_model: Optional[str] = None,
        embedding_dim: int = 384,
        cache_embeddings: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize the dynamic chunking pipeline.
        
        Args:
            compression_ratio: Target compression ratio (tokens per chunk)
            embedding_model: Name of embedding model to use
            embedding_dim: Dimension of embeddings
            cache_embeddings: Whether to cache computed embeddings
            device: Device for computations ('cpu' or 'cuda')
        """
        if compression_ratio <= 1.0:
            raise InvalidTextError(f"Compression ratio must be > 1.0, got {compression_ratio}")
        
        self.compression_ratio = compression_ratio
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.cache_embeddings = cache_embeddings
        self.device = device
        
        # Initialize modules
        self.boundary_detector = SimilarityBasedBoundaryDetector(
            embedding_dim=embedding_dim,
            device=device
        )
        
        self.routing_module = RoutingModule(
            target_compression_ratio=compression_ratio,
            boundary_detector=self.boundary_detector
        )
        
        self.smoothing_module = SmoothingModule()
        
        # Initialize embedding model if available
        self._init_embedding_model()
        
        # Simple cache for embeddings
        self._embedding_cache = {} if cache_embeddings else None
        
        logger.info(
            f"Initialized pipeline with compression_ratio={compression_ratio}, "
            f"embedding_model={embedding_model}, device={device}"
        )
    
    def _init_embedding_model(self) -> None:
        """Initialize the embedding model."""
        if HAS_TORCH and self.embedding_model:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
                self.model = AutoModel.from_pretrained(self.embedding_model)
                self.model.eval()
                
                if self.device == 'cuda' and torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                logger.info(f"Loaded transformer model: {self.embedding_model}")
                
            except Exception as e:
                logger.warning(f"Could not load {self.embedding_model}: {e}")
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None
            
            if self.embedding_model and not HAS_TORCH:
                logger.warning("PyTorch not available, using TF-IDF fallback")
    
    def _validate_text_input(self, text: str) -> None:
        """Validate text input."""
        if not isinstance(text, str):
            raise InvalidTextError(f"Text must be a string, got {type(text)}")
        
        if not text or len(text.strip()) == 0:
            raise InvalidTextError("Text cannot be empty or whitespace-only")
        
        if len(text.split()) < 2:
            raise InvalidTextError("Text must contain at least 2 tokens for chunking")
    
    def get_embeddings(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """
        Get embeddings for input text.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (embeddings, tokens) where:
            - embeddings: Token-level embeddings of shape (num_tokens, embedding_dim)
            - tokens: List of text tokens
            
        Raises:
            InvalidTextError: If text is invalid
            EmbeddingError: If embedding computation fails
            ModelNotFoundError: If specified model is not available
        """
        try:
            self._validate_text_input(text)
            
            # Check cache first
            cache_key = None
            if self._embedding_cache is not None:
                cache_key = hash(text + str(self.embedding_model))
                if cache_key in self._embedding_cache:
                    logger.debug("Using cached embeddings")
                    return self._embedding_cache[cache_key]
            
            logger.debug("Computing embeddings for text")
            
            if self.model and self.tokenizer:
                # Use transformer model for embeddings
                embeddings, tokens = self._get_transformer_embeddings(text)
            elif HAS_SKLEARN:
                # Fallback to TF-IDF based embeddings
                embeddings, tokens = self._get_tfidf_embeddings(text)
            else:
                # Last resort: random embeddings
                embeddings, tokens = self._get_random_embeddings(text)
            
            # Cache result
            if self._embedding_cache is not None and cache_key is not None:
                self._embedding_cache[cache_key] = (embeddings, tokens)
            
            logger.debug(f"Generated embeddings: {embeddings.shape} for {len(tokens)} tokens")
            
            return embeddings, tokens
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            if isinstance(e, (InvalidTextError, EmbeddingError, ModelNotFoundError)):
                raise
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    def _get_transformer_embeddings(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Get embeddings using transformer model."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[0].cpu().numpy()  # Remove batch dimension
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Filter out special tokens
        filtered_embeddings = []
        filtered_tokens = []
        
        for emb, token in zip(embeddings, tokens):
            if not token.startswith('[') and not token.startswith('<'):
                filtered_embeddings.append(emb)
                filtered_tokens.append(token.replace('##', ''))  # Clean subword tokens
        
        if len(filtered_embeddings) == 0:
            raise EmbeddingError("No valid tokens after filtering special tokens")
        
        return np.array(filtered_embeddings), filtered_tokens
    
    def _get_tfidf_embeddings(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Get embeddings using TF-IDF vectorizer."""
        tokens = text.split()
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=self.embedding_dim,
            ngram_range=(1, 2),
            stop_words=None
        )
        
        # Create corpus for better embeddings
        corpus = [text] + [' '.join(tokens[i:i+3]) for i in range(0, len(tokens)-2, 3)]
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            base_embedding = tfidf_matrix[0].toarray()[0]
            
            # Pad or truncate to desired dimension
            if len(base_embedding) < self.embedding_dim:
                base_embedding = np.pad(base_embedding, (0, self.embedding_dim - len(base_embedding)))
            else:
                base_embedding = base_embedding[:self.embedding_dim]
            
            # Create token-level embeddings with variation
            embeddings = []
            for i, token in enumerate(tokens):
                # Add position and token-specific variation
                np.random.seed(hash(token) % 1000)  # Deterministic but varied
                noise = np.random.randn(len(base_embedding)) * 0.1
                position_bias = np.sin(np.arange(len(base_embedding)) * i / len(tokens)) * 0.05
                
                token_embedding = base_embedding + noise + position_bias
                embeddings.append(token_embedding)
            
            return np.array(embeddings), tokens
            
        except Exception as e:
            logger.warning(f"TF-IDF embedding failed: {e}, falling back to random embeddings")
            return self._get_random_embeddings(text)
    
    def _get_random_embeddings(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """Generate random embeddings as last resort."""
        logger.warning("Using random embeddings - results will be poor")
        
        tokens = text.split()
        embeddings = []
        
        for i, token in enumerate(tokens):
            # Deterministic but varied random embeddings
            np.random.seed(hash(token) % 1000)
            embedding = np.random.randn(self.embedding_dim) * 0.5
            embeddings.append(embedding)
        
        return np.array(embeddings), tokens
    
    def process_text(
        self, 
        text: str, 
        return_intermediate: bool = False,
        return_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Process text through the complete dynamic chunking pipeline.
        
        Args:
            text: Input text to process
            return_intermediate: Whether to return intermediate processing results
            return_metrics: Whether to compute and return quality metrics
            
        Returns:
            Dictionary with processing results including:
            - 'chunks': List of text chunks
            - 'num_chunks': Number of chunks created
            - 'compression_ratio_achieved': Actual compression ratio
            - 'processing_time': Time taken for processing
            - Additional metrics and intermediate results if requested
            
        Raises:
            InvalidTextError: If text is invalid
            EmbeddingError: If embedding computation fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing text with compression ratio {self.compression_ratio}")
            
            # Step 1: Encoding - Get embeddings
            embeddings, tokens = self.get_embeddings(text)
            step1_time = time.time() - start_time
            
            logger.debug(f"Step 1 - Encoding: {len(tokens)} tokens, embedding dim {embeddings.shape[1]}")
            
            # Step 2 & 3: Routing and Chunking
            routing_start = time.time()
            routing_result = self.routing_module.create_chunks(embeddings, tokens)
            routing_time = time.time() - routing_start
            
            boundary_probs = routing_result['boundary_probs']
            boundaries = routing_result['boundaries']
            chunk_embeddings = routing_result['chunk_embeddings']
            
            logger.debug(f"Step 2-3 - Routing & Chunking: {len(chunk_embeddings)} chunks created")
            
            # Step 4: Smoothing
            smoothing_start = time.time()
            
            # Average each chunk's embeddings for processing
            chunk_means = []
            for chunk_emb in chunk_embeddings:
                if len(chunk_emb) > 0:
                    chunk_means.append(np.mean(chunk_emb, axis=0))
            
            chunk_means = np.array(chunk_means) if chunk_means else np.array([]).reshape(0, embeddings.shape[1])
            
            if len(chunk_means) > 0:
                smoothed_chunks = self.smoothing_module.apply_smoothing(
                    chunk_means, 
                    boundary_probs[:len(chunk_means)]
                )
            else:
                smoothed_chunks = chunk_means
            
            smoothing_time = time.time() - smoothing_start
            
            logger.debug(f"Step 4 - Smoothing: Applied to {len(smoothed_chunks)} chunks")
            
            # Step 5: Main Processing (simulated)
            processed_chunks = smoothed_chunks.copy()  # Placeholder for actual processing
            
            # Step 6: Dechunking - Upsample back to original resolution
            dechunking_start = time.time()
            
            if len(processed_chunks) > 0:
                reconstructed = self.smoothing_module.upsample_with_confidence(
                    processed_chunks, boundaries, len(tokens), boundary_probs
                )
            else:
                reconstructed = np.zeros((len(tokens), embeddings.shape[1]))
            
            dechunking_time = time.time() - dechunking_start
            
            logger.debug(f"Step 6 - Dechunking: Reconstructed to {reconstructed.shape[0]} positions")
            
            # Calculate final metrics
            total_time = time.time() - start_time
            compression_achieved = len(tokens) / len(chunk_means) if len(chunk_means) > 0 else 1.0
            
            # Build result dictionary
            result = {
                'original_text': text,
                'tokens': tokens,
                'num_tokens': len(tokens),
                'chunks': routing_result['chunks'],
                'num_chunks': len(chunk_means),
                'compression_ratio_target': self.compression_ratio,
                'compression_ratio_achieved': compression_achieved,
                'boundary_probs': boundary_probs,
                'boundaries': boundaries,
                'processing_time': total_time,
                'timing': {
                    'encoding': step1_time,
                    'routing': routing_time,
                    'smoothing': smoothing_time,
                    'dechunking': dechunking_time,
                    'total': total_time
                }
            }
            
            # Add intermediate results if requested
            if return_intermediate:
                result.update({
                    'embeddings': embeddings,
                    'chunk_embeddings': chunk_embeddings,
                    'chunk_means': chunk_means,
                    'smoothed_chunks': smoothed_chunks,
                    'processed_chunks': processed_chunks,
                    'reconstructed_embeddings': reconstructed,
                    'routing_result': routing_result
                })
            
            # Add quality metrics if requested
            if return_metrics:
                from ..evaluation.metrics import ChunkingQualityMetrics
                evaluator = ChunkingQualityMetrics()
                
                quality_metrics = evaluator.evaluate_chunking_method(
                    routing_result['chunks'], text, boundary_probs
                )
                
                result['quality_metrics'] = quality_metrics
            
            logger.info(
                f"Processing completed: {len(chunk_means)} chunks, "
                f"compression {compression_achieved:.2f}, time {total_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            if isinstance(e, (InvalidTextError, EmbeddingError)):
                raise
            raise EmbeddingError(f"Pipeline processing failed: {e}")
    
    def batch_process(
        self, 
        texts: List[str], 
        return_intermediate: bool = False,
        return_metrics: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of text strings to process
            return_intermediate: Whether to return intermediate results
            return_metrics: Whether to compute quality metrics
            
        Returns:
            List of result dictionaries, one for each input text
        """
        if not texts:
            return []
        
        logger.info(f"Batch processing {len(texts)} texts")
        
        results = []
        for i, text in enumerate(texts):
            try:
                logger.debug(f"Processing text {i+1}/{len(texts)}")
                result = self.process_text(
                    text, 
                    return_intermediate=return_intermediate,
                    return_metrics=return_metrics
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process text {i+1}: {e}")
                # Add error result to maintain alignment
                results.append({
                    'error': str(e),
                    'text_index': i,
                    'original_text': text
                })
        
        logger.info(f"Batch processing completed: {len(results)} results")
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._embedding_cache is not None:
            self._embedding_cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if self._embedding_cache is None:
            return {'cache_enabled': False}
        
        return {
            'cache_enabled': True,
            'cache_size': len(self._embedding_cache),
            'cache_keys': list(self._embedding_cache.keys())[:5]  # First 5 keys for debugging
        }