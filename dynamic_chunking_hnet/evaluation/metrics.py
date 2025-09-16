"""
Chunking Quality Metrics

Comprehensive evaluation metrics for assessing chunking quality across different methods.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

# Optional imports
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from ..utils.exceptions import InvalidTextError

logger = logging.getLogger(__name__)


class ChunkingQualityMetrics:
    """
    Comprehensive evaluation metrics for chunking approaches.
    
    Provides various metrics to assess:
    - Compression efficiency
    - Semantic coherence 
    - Boundary detection accuracy
    - Chunk size consistency
    """
    
    def __init__(self):
        """Initialize the metrics evaluator."""
        logger.info("Initialized chunking quality metrics evaluator")
        
        # Download NLTK data if needed
        if HAS_NLTK:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
    
    def compression_ratio(self, original_tokens: int, num_chunks: int) -> float:
        """
        Calculate compression ratio (tokens per chunk).
        
        Args:
            original_tokens: Number of original tokens
            num_chunks: Number of chunks created
            
        Returns:
            Compression ratio (higher = more compression)
        """
        if num_chunks <= 0:
            return 1.0
        
        return original_tokens / num_chunks
    
    def chunk_size_variance(self, chunks: List[List[str]]) -> float:
        """
        Calculate variance in chunk sizes (lower indicates more consistent chunking).
        
        Args:
            chunks: List of chunks, each chunk is a list of tokens
            
        Returns:
            Variance in chunk sizes
        """
        if not chunks or len(chunks) <= 1:
            return 0.0
        
        sizes = [len(chunk) for chunk in chunks]
        return float(np.var(sizes))
    
    def semantic_boundary_score(self, chunks: List[List[str]]) -> float:
        """
        Score based on how well chunks respect semantic boundaries.
        
        Evaluates:
        - Starting with capital letters (sentence beginnings)
        - Ending with sentence punctuation
        - Not breaking mid-sentence
        
        Args:
            chunks: List of chunks, each chunk is a list of tokens
            
        Returns:
            Semantic boundary score (0-1, higher is better)
        """
        if not chunks:
            return 0.0
        
        total_score = 0.0
        
        for chunk in chunks:
            if not chunk:
                continue
                
            chunk_text = ' '.join(chunk).strip()
            
            if not chunk_text:
                continue
            
            chunk_score = 0.0
            
            # Points for starting with capital letter (sentence beginning)
            if chunk_text[0].isupper():
                chunk_score += 0.3
            
            # Points for ending with sentence punctuation
            if chunk_text.endswith(('.', '!', '?')):
                chunk_score += 0.5
            
            # Points for not breaking mid-sentence (no internal sentence breaks unless ending properly)
            sentence_breaks = sum(1 for i in range(len(chunk_text)-1) 
                                if chunk_text[i:i+2] in ['. ', '! ', '? '])
            
            if sentence_breaks == 0 or chunk_text.endswith(('.', '!', '?')):
                chunk_score += 0.2
            
            total_score += chunk_score
        
        return total_score / len(chunks)
    
    def boundary_precision_score(
        self, 
        boundary_probs: np.ndarray, 
        text: str
    ) -> float:
        """
        Evaluate boundary prediction quality by comparing with natural sentence boundaries.
        
        Args:
            boundary_probs: Predicted boundary probabilities
            text: Original text for extracting true sentence boundaries
            
        Returns:
            F1 score for boundary detection (0-1, higher is better)
        """
        if not HAS_NLTK:
            logger.warning("NLTK not available, cannot compute boundary precision score")
            return 0.0
        
        if boundary_probs is None or len(boundary_probs) == 0:
            return 0.0
        
        try:
            tokens = text.split()
            sentences = sent_tokenize(text)
            
            if len(tokens) == 0 or len(sentences) <= 1:
                return 1.0  # Perfect if no boundaries to find
            
            # Find actual sentence boundaries in token positions
            actual_boundaries = set()
            token_idx = 0
            
            for sentence in sentences:
                sentence_tokens = sentence.split()
                token_idx += len(sentence_tokens)
                if token_idx < len(tokens):
                    actual_boundaries.add(token_idx)
            
            # Ensure boundary_probs length matches tokens
            if len(boundary_probs) != len(tokens):
                logger.warning(
                    f"Boundary probs length ({len(boundary_probs)}) "
                    f"doesn't match tokens ({len(tokens)})"
                )
                return 0.0
            
            # Use top-k boundaries based on probabilities
            k = len(actual_boundaries)
            if k == 0:
                return 1.0  # Perfect score if no sentence boundaries to match
            
            # Get top-k predicted boundaries
            if k <= len(boundary_probs):
                top_k_indices = np.argsort(boundary_probs)[-k:]
            else:
                top_k_indices = range(len(boundary_probs))
            
            predicted_boundaries = set(top_k_indices)
            
            # Calculate precision, recall, and F1
            true_positives = len(predicted_boundaries.intersection(actual_boundaries))
            
            precision = true_positives / len(predicted_boundaries) if predicted_boundaries else 0.0
            recall = true_positives / len(actual_boundaries) if actual_boundaries else 1.0
            
            # F1 score
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            logger.debug(
                f"Boundary precision: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}"
            )
            
            return f1
            
        except Exception as e:
            logger.error(f"Failed to compute boundary precision score: {e}")
            return 0.0
    
    def chunk_coherence_score(self, chunks: List[List[str]]) -> float:
        """
        Measure semantic coherence within chunks based on token patterns.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Coherence score (0-1, higher is better)
        """
        if not chunks:
            return 0.0
        
        total_coherence = 0.0
        
        for chunk in chunks:
            if len(chunk) <= 1:
                total_coherence += 1.0  # Single tokens are perfectly coherent
                continue
            
            chunk_text = ' '.join(chunk)
            
            # Simple heuristics for coherence
            coherence = 0.0
            
            # Bonus for consistent capitalization pattern
            capitals = sum(1 for token in chunk if token and token[0].isupper())
            if capitals <= 2:  # At most sentence start + proper nouns
                coherence += 0.3
            
            # Bonus for balanced punctuation
            punct_count = sum(chunk_text.count(p) for p in '.!?')
            if punct_count <= 1:  # At most one sentence
                coherence += 0.4
            
            # Bonus for reasonable length (not too short or long)
            if 3 <= len(chunk) <= 15:
                coherence += 0.3
            
            total_coherence += coherence
        
        return total_coherence / len(chunks)
    
    def compression_efficiency_score(
        self, 
        original_tokens: int, 
        num_chunks: int,
        target_compression: float
    ) -> float:
        """
        Score how close the achieved compression is to the target.
        
        Args:
            original_tokens: Number of original tokens
            num_chunks: Number of chunks created
            target_compression: Target compression ratio
            
        Returns:
            Efficiency score (0-1, higher is better)
        """
        if num_chunks <= 0 or target_compression <= 0:
            return 0.0
        
        achieved_compression = self.compression_ratio(original_tokens, num_chunks)
        
        # Calculate how close to target (perfect = 1.0, further away = lower)
        ratio = min(achieved_compression, target_compression) / max(achieved_compression, target_compression)
        
        return ratio
    
    def evaluate_chunking_method(
        self, 
        chunks: List[List[str]], 
        original_text: str,
        boundary_probs: Optional[np.ndarray] = None,
        target_compression: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a chunking method.
        
        Args:
            chunks: List of chunks from the chunking method
            original_text: Original text that was chunked
            boundary_probs: Optional boundary probabilities for precision scoring
            target_compression: Optional target compression ratio
            
        Returns:
            Dictionary with various quality metrics
        """
        if not chunks:
            logger.warning("Empty chunks provided for evaluation")
            return {
                'compression_ratio': 0.0,
                'chunk_size_variance': 0.0,
                'semantic_boundary_score': 0.0,
                'num_chunks': 0,
                'avg_chunk_size': 0.0,
                'chunk_sizes': []
            }
        
        tokens = original_text.split()
        
        # Core metrics
        metrics = {
            'compression_ratio': self.compression_ratio(len(tokens), len(chunks)),
            'chunk_size_variance': self.chunk_size_variance(chunks),
            'semantic_boundary_score': self.semantic_boundary_score(chunks),
            'chunk_coherence_score': self.chunk_coherence_score(chunks),
            'num_chunks': len(chunks),
            'avg_chunk_size': np.mean([len(chunk) for chunk in chunks]) if chunks else 0.0,
            'median_chunk_size': np.median([len(chunk) for chunk in chunks]) if chunks else 0.0,
            'min_chunk_size': min(len(chunk) for chunk in chunks) if chunks else 0,
            'max_chunk_size': max(len(chunk) for chunk in chunks) if chunks else 0,
            'chunk_sizes': [len(chunk) for chunk in chunks]
        }
        
        # Optional boundary precision score
        if boundary_probs is not None:
            metrics['boundary_precision_score'] = self.boundary_precision_score(boundary_probs, original_text)
        
        # Optional compression efficiency
        if target_compression is not None:
            metrics['compression_efficiency_score'] = self.compression_efficiency_score(
                len(tokens), len(chunks), target_compression
            )
        
        # Composite quality score (weighted average of key metrics)
        quality_components = [
            metrics['semantic_boundary_score'] * 0.4,
            metrics['chunk_coherence_score'] * 0.3,
            (1.0 - min(1.0, metrics['chunk_size_variance'] / 10.0)) * 0.2,  # Normalize variance
        ]
        
        if 'boundary_precision_score' in metrics:
            quality_components.append(metrics['boundary_precision_score'] * 0.1)
        
        metrics['composite_quality_score'] = sum(quality_components)
        
        logger.debug(f"Evaluation completed: {len(chunks)} chunks, composite score: {metrics['composite_quality_score']:.3f}")
        
        return metrics
    
    def compare_methods(
        self, 
        results: Dict[str, Dict], 
        text: str
    ) -> Dict[str, Dict]:
        """
        Compare multiple chunking methods.
        
        Args:
            results: Dictionary mapping method names to their chunking results
            text: Original text
            
        Returns:
            Dictionary with comparison metrics and rankings
        """
        if not results:
            return {}
        
        comparison = {}
        all_scores = {}
        
        # Evaluate each method
        for method_name, result in results.items():
            chunks = result.get('chunks', [])
            boundary_probs = result.get('boundary_probs')
            target_compression = result.get('target_compression_ratio')
            
            metrics = self.evaluate_chunking_method(
                chunks, text, boundary_probs, target_compression
            )
            
            comparison[method_name] = metrics
            
            # Collect scores for ranking
            for metric_name, score in metrics.items():
                if isinstance(score, (int, float)) and metric_name != 'num_chunks':
                    if metric_name not in all_scores:
                        all_scores[metric_name] = {}
                    all_scores[metric_name][method_name] = score
        
        # Create rankings
        rankings = {}
        for metric_name, scores in all_scores.items():
            # Higher is better for most metrics (except variance)
            reverse = 'variance' not in metric_name.lower()
            
            sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=reverse)
            rankings[metric_name] = {
                method: rank + 1 
                for rank, (method, score) in enumerate(sorted_methods)
            }
        
        # Overall ranking based on composite score
        if 'composite_quality_score' in all_scores:
            overall_ranking = rankings['composite_quality_score']
        else:
            # Fallback ranking based on semantic boundary score
            overall_ranking = rankings.get('semantic_boundary_score', {})
        
        return {
            'method_metrics': comparison,
            'rankings': rankings,
            'overall_ranking': overall_ranking,
            'best_method': min(overall_ranking.items(), key=lambda x: x[1])[0] if overall_ranking else None
        }