"""
Advanced Caching System

Implements LRU cache with persistence and statistics for embeddings and boundary calculations.
"""

import os
import pickle
import hashlib
import time
import logging
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import json

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Least Recently Used (LRU) cache with optional persistence.
    """
    
    def __init__(
        self, 
        max_size: int = 1000, 
        cache_dir: Optional[str] = None,
        enable_persistence: bool = False
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
            cache_dir: Directory to store persistent cache files
            enable_persistence: Whether to persist cache to disk
        """
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.enable_persistence = enable_persistence
        
        self._cache = OrderedDict()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Create cache directory if needed
        if self.enable_persistence and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_persistent_cache()
        
        logger.info(f"Initialized LRU cache with max_size={max_size}, persistence={enable_persistence}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a hash key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_persistent_cache(self) -> None:
        """Load cache from persistent storage."""
        if not self.cache_dir:
            return
        
        cache_file = os.path.join(self.cache_dir, 'cache.pkl')
        stats_file = os.path.join(self.cache_dir, 'stats.json')
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} items from persistent cache")
            
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self._stats = json.load(f)
                logger.info("Loaded cache statistics from disk")
                
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
            self._cache = OrderedDict()
    
    def _save_persistent_cache(self) -> None:
        """Save cache to persistent storage."""
        if not self.cache_dir or not self.enable_persistence:
            return
        
        try:
            cache_file = os.path.join(self.cache_dir, 'cache.pkl')
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            
            stats_file = os.path.join(self.cache_dir, 'stats.json')
            with open(stats_file, 'w') as f:
                json.dump(self._stats, f, indent=2)
                
            logger.debug("Saved cache to persistent storage")
            
        except Exception as e:
            logger.error(f"Failed to save persistent cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        self._stats['total_requests'] += 1
        
        if key in self._cache:
            # Move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            
            self._stats['hits'] += 1
            logger.debug(f"Cache hit for key: {key[:16]}...")
            
            return value
        
        self._stats['misses'] += 1
        logger.debug(f"Cache miss for key: {key[:16]}...")
        
        return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove if already exists
        if key in self._cache:
            self._cache.pop(key)
        
        # Add to end
        self._cache[key] = value
        
        # Evict oldest if over capacity
        if len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            self._stats['evictions'] += 1
            logger.debug(f"Evicted key: {oldest_key[:16]}...")
        
        logger.debug(f"Cached key: {key[:16]}...")
        
        # Periodically save to disk
        if self.enable_persistence and len(self._cache) % 100 == 0:
            self._save_persistent_cache()
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (
            self._stats['hits'] / max(1, self._stats['total_requests'])
        )
        
        return {
            **self._stats,
            'cache_size': len(self._cache),
            'hit_rate': hit_rate,
            'miss_rate': 1.0 - hit_rate
        }
    
    def __len__(self) -> int:
        """Return current cache size."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache


class EmbeddingCache:
    """
    Specialized cache for embedding computations with size-aware eviction.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
            max_memory_mb: Maximum memory usage in MB
            cache_dir: Directory for persistent storage
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self._cache = LRUCache(
            max_size=max_size,
            cache_dir=cache_dir,
            enable_persistence=cache_dir is not None
        )
        
        self._memory_usage = 0
        self._item_sizes = {}
        
        logger.info(f"Initialized embedding cache: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory usage of a value in bytes."""
        try:
            import sys
            return sys.getsizeof(pickle.dumps(value))
        except:
            # Fallback estimation
            if hasattr(value, 'nbytes'):  # numpy arrays
                return value.nbytes
            elif hasattr(value, '__len__'):
                return len(value) * 8  # rough estimate
            else:
                return 1024  # default estimate
    
    def get_embeddings(self, text: str, model_name: str = "default") -> Optional[Any]:
        """
        Get cached embeddings for text.
        
        Args:
            text: Input text
            model_name: Name of embedding model
            
        Returns:
            Cached embeddings or None
        """
        cache_key = self._generate_embedding_key(text, model_name)
        return self._cache.get(cache_key)
    
    def cache_embeddings(self, text: str, embeddings: Any, model_name: str = "default") -> None:
        """
        Cache embeddings for text.
        
        Args:
            text: Input text
            embeddings: Computed embeddings
            model_name: Name of embedding model
        """
        cache_key = self._generate_embedding_key(text, model_name)
        
        # Estimate size
        size = self._estimate_size(embeddings)
        
        # Check memory limit
        if size > self.max_memory_bytes:
            logger.warning(f"Embedding too large to cache: {size / 1024 / 1024:.1f}MB")
            return
        
        # Evict items if memory limit would be exceeded
        while (self._memory_usage + size > self.max_memory_bytes and 
               len(self._cache) > 0):
            self._evict_oldest()
        
        # Cache the embeddings
        self._cache.put(cache_key, embeddings)
        self._memory_usage += size
        self._item_sizes[cache_key] = size
        
        logger.debug(f"Cached embeddings: {size / 1024:.1f}KB, total memory: {self._memory_usage / 1024 / 1024:.1f}MB")
    
    def _generate_embedding_key(self, text: str, model_name: str) -> str:
        """Generate cache key for embeddings."""
        # Include text hash and model name
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"emb_{model_name}_{text_hash}"
    
    def _evict_oldest(self) -> None:
        """Evict oldest item from cache."""
        if not self._cache._cache:
            return
        
        oldest_key = next(iter(self._cache._cache))
        self._cache._cache.pop(oldest_key)
        
        # Update memory usage
        if oldest_key in self._item_sizes:
            self._memory_usage -= self._item_sizes.pop(oldest_key)
        
        logger.debug(f"Evicted oldest embedding: {oldest_key[:16]}...")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'memory_usage_mb': self._memory_usage / 1024 / 1024,
            'memory_limit_mb': self.max_memory_bytes / 1024 / 1024,
            'memory_usage_percent': (self._memory_usage / self.max_memory_bytes) * 100,
            'num_cached_embeddings': len(self._cache),
            'avg_embedding_size_kb': (
                self._memory_usage / len(self._cache) / 1024 
                if len(self._cache) > 0 else 0
            ),
            **self._cache.get_stats()
        }
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._memory_usage = 0
        self._item_sizes.clear()
        logger.info("Embedding cache cleared")


class BoundaryCache:
    """
    Specialized cache for boundary probability calculations.
    """
    
    def __init__(self, max_size: int = 500):
        """
        Initialize boundary cache.
        
        Args:
            max_size: Maximum number of cached boundary calculations
        """
        self._cache = LRUCache(max_size=max_size)
        logger.info(f"Initialized boundary cache with max_size={max_size}")
    
    def get_boundary_probs(
        self, 
        embedding_hash: str, 
        compression_ratio: float
    ) -> Optional[Any]:
        """
        Get cached boundary probabilities.
        
        Args:
            embedding_hash: Hash of input embeddings
            compression_ratio: Compression ratio used
            
        Returns:
            Cached boundary probabilities or None
        """
        cache_key = f"boundary_{embedding_hash}_{compression_ratio}"
        return self._cache.get(cache_key)
    
    def cache_boundary_probs(
        self,
        embedding_hash: str,
        compression_ratio: float,
        boundary_probs: Any
    ) -> None:
        """
        Cache boundary probabilities.
        
        Args:
            embedding_hash: Hash of input embeddings
            compression_ratio: Compression ratio used
            boundary_probs: Computed boundary probabilities
        """
        cache_key = f"boundary_{embedding_hash}_{compression_ratio}"
        self._cache.put(cache_key, boundary_probs)
        
        logger.debug(f"Cached boundary probs for hash: {embedding_hash[:16]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()
    
    def clear(self) -> None:
        """Clear boundary cache."""
        self._cache.clear()
        logger.info("Boundary cache cleared")


class CacheManager:
    """
    Unified cache manager for all caching needs.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        embedding_cache_size: int = 1000,
        embedding_memory_mb: float = 100.0,
        boundary_cache_size: int = 500,
        enable_persistence: bool = False
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for persistent cache storage
            embedding_cache_size: Maximum embeddings to cache
            embedding_memory_mb: Maximum memory for embeddings (MB)
            boundary_cache_size: Maximum boundary calculations to cache
            enable_persistence: Whether to persist caches to disk
        """
        self.cache_dir = cache_dir
        self.enable_persistence = enable_persistence
        
        # Initialize caches
        self.embedding_cache = EmbeddingCache(
            max_size=embedding_cache_size,
            max_memory_mb=embedding_memory_mb,
            cache_dir=os.path.join(cache_dir, 'embeddings') if cache_dir else None
        )
        
        self.boundary_cache = BoundaryCache(max_size=boundary_cache_size)
        
        logger.info("Initialized cache manager")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all caches."""
        return {
            'embedding_cache': self.embedding_cache.get_memory_stats(),
            'boundary_cache': self.boundary_cache.get_stats(),
            'cache_enabled': True,
            'persistence_enabled': self.enable_persistence
        }
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.embedding_cache.clear()
        self.boundary_cache.clear()
        logger.info("All caches cleared")
    
    def save_all_caches(self) -> None:
        """Save all persistent caches."""
        if self.enable_persistence:
            self.embedding_cache._cache._save_persistent_cache()
            logger.info("All caches saved to disk")