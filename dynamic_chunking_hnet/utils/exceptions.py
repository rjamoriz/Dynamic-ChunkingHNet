"""
Custom exceptions for the Dynamic ChunkingHNet package.
"""

class ChunkingError(Exception):
    """Base exception for all chunking operations."""
    pass


class InvalidTextError(ChunkingError):
    """Raised when input text is invalid or malformed."""
    
    def __init__(self, message: str = "Invalid text input"):
        self.message = message
        super().__init__(self.message)


class EmbeddingError(ChunkingError):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str = "Failed to generate embeddings"):
        self.message = message
        super().__init__(self.message)


class ConfigurationError(ChunkingError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str = "Invalid configuration"):
        self.message = message
        super().__init__(self.message)


class ModelNotFoundError(ChunkingError):
    """Raised when a requested model is not available."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.message = f"Model '{model_name}' not found or not available"
        super().__init__(self.message)