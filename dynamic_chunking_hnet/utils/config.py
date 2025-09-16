"""
Configuration Management System

Provides centralized configuration management with YAML support, 
environment variable overrides, and validation.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
from copy import deepcopy

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration manager with YAML support and validation.
    
    Supports:
    - Loading from YAML files
    - Environment variable overrides
    - Configuration validation
    - Nested configuration access
    - Default value fallbacks
    """
    
    def __init__(
        self, 
        config_file: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        enable_env_overrides: bool = True
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)
            enable_env_overrides: Whether to enable environment variable overrides
        """
        self.config_file = config_file
        self.enable_env_overrides = enable_env_overrides
        self._config = {}
        self._env_prefix = "CHUNKING_"
        
        # Load configuration
        if config_dict:
            self._config = deepcopy(config_dict)
        elif config_file:
            self._load_config_file(config_file)
        else:
            self._load_default_config()
        
        # Apply environment overrides
        if enable_env_overrides:
            self._apply_env_overrides()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Configuration loaded from {config_file or 'defaults'}")
    
    def _load_default_config(self) -> None:
        """Load default configuration."""
        # Try to find default config file
        possible_paths = [
            "config/default.yaml",
            "../config/default.yaml", 
            os.path.join(os.path.dirname(__file__), "../config/default.yaml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self._load_config_file(path)
                return
        
        # Fallback to hardcoded defaults
        logger.warning("No configuration file found, using hardcoded defaults")
        self._config = self._get_hardcoded_defaults()
    
    def _load_config_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            
            logger.info(f"Loaded configuration from {config_file}")
            
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _get_hardcoded_defaults(self) -> Dict[str, Any]:
        """Get hardcoded default configuration."""
        return {
            'pipeline': {
                'compression_ratio': 6.0,
                'embedding_dim': 384,
                'device': 'cpu',
                'cache_embeddings': True
            },
            'boundary_detection': {
                'embedding_model': None,
                'threshold_mode': 'adaptive',
                'fixed_threshold': 0.5,
                'min_threshold': 0.1
            },
            'routing': {
                'target_compression_ratio': 6.0,
                'enable_ratio_loss': True
            },
            'smoothing': {
                'alpha': 0.7,
                'use_confidence_weighting': True,
                'noise_threshold': 0.3
            },
            'caching': {
                'enabled': True,
                'max_embedding_cache_size': 1000,
                'max_memory_mb': 100.0,
                'max_boundary_cache_size': 500,
                'enable_persistence': False
            },
            'evaluation': {
                'compute_metrics': True,
                'include_boundary_precision': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_vars = {}
        
        # Collect relevant environment variables
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                config_key = key[len(self._env_prefix):].lower()
                env_vars[config_key] = value
        
        # Apply overrides
        for env_key, env_value in env_vars.items():
            # Convert environment key to nested path
            key_parts = env_key.split('_')
            
            try:
                # Convert value to appropriate type
                typed_value = self._convert_env_value(env_value)
                
                # Set nested configuration value
                self._set_nested_value(key_parts, typed_value)
                
                logger.debug(f"Applied environment override: {env_key} = {typed_value}")
                
            except Exception as e:
                logger.warning(f"Failed to apply environment override {env_key}: {e}")
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Handle boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Handle numeric values
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, key_parts: List[str], value: Any) -> None:
        """Set a nested configuration value."""
        current = self._config
        
        # Navigate to the parent of the target key
        for part in key_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the final value
        current[key_parts[-1]] = value
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        try:
            # Validate pipeline settings
            pipeline_config = self.get('pipeline', {})
            
            compression_ratio = pipeline_config.get('compression_ratio', 6.0)
            if not 1.0 < compression_ratio <= 50.0:
                raise ConfigurationError(
                    f"Invalid compression_ratio: {compression_ratio}. Must be between 1.0 and 50.0"
                )
            
            embedding_dim = pipeline_config.get('embedding_dim', 384)
            if not 32 <= embedding_dim <= 2048:
                raise ConfigurationError(
                    f"Invalid embedding_dim: {embedding_dim}. Must be between 32 and 2048"
                )
            
            # Validate boundary detection settings
            boundary_config = self.get('boundary_detection', {})
            
            threshold_mode = boundary_config.get('threshold_mode', 'adaptive')
            if threshold_mode not in ['adaptive', 'fixed']:
                raise ConfigurationError(
                    f"Invalid threshold_mode: {threshold_mode}. Must be 'adaptive' or 'fixed'"
                )
            
            fixed_threshold = boundary_config.get('fixed_threshold', 0.5)
            if not 0.0 <= fixed_threshold <= 1.0:
                raise ConfigurationError(
                    f"Invalid fixed_threshold: {fixed_threshold}. Must be between 0.0 and 1.0"
                )
            
            # Validate smoothing settings
            smoothing_config = self.get('smoothing', {})
            
            alpha = smoothing_config.get('alpha', 0.7)
            if not 0.0 < alpha <= 1.0:
                raise ConfigurationError(
                    f"Invalid smoothing alpha: {alpha}. Must be between 0.0 and 1.0"
                )
            
            logger.debug("Configuration validation passed")
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).
        
        Args:
            key: Configuration key (e.g., 'pipeline.compression_ratio')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports nested keys with dot notation).
        
        Args:
            key: Configuration key (e.g., 'pipeline.compression_ratio') 
            value: Value to set
        """
        keys = key.split('.')
        current = self._config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set value
        current[keys[-1]] = value
        
        logger.debug(f"Set configuration: {key} = {value}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary of values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'pipeline')
            
        Returns:
            Configuration section as dictionary
        """
        return self.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Return complete configuration as dictionary."""
        return deepcopy(self._config)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            file_path: Path to save configuration file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def validate_section(self, section: str, schema: Dict[str, Any]) -> bool:
        """
        Validate a configuration section against a schema.
        
        Args:
            section: Section name to validate
            schema: Validation schema
            
        Returns:
            True if valid, False otherwise
        """
        section_config = self.get_section(section)
        
        try:
            for key, constraints in schema.items():
                if key not in section_config:
                    if constraints.get('required', False):
                        raise ConfigurationError(f"Required configuration key missing: {section}.{key}")
                    continue
                
                value = section_config[key]
                value_type = constraints.get('type')
                
                # Type checking
                if value_type and not isinstance(value, value_type):
                    raise ConfigurationError(
                        f"Invalid type for {section}.{key}: expected {value_type.__name__}, got {type(value).__name__}"
                    )
                
                # Range checking for numeric values
                if isinstance(value, (int, float)):
                    min_val = constraints.get('min')
                    max_val = constraints.get('max')
                    
                    if min_val is not None and value < min_val:
                        raise ConfigurationError(f"Value too small for {section}.{key}: {value} < {min_val}")
                    
                    if max_val is not None and value > max_val:
                        raise ConfigurationError(f"Value too large for {section}.{key}: {value} > {max_val}")
                
                # Choice validation
                choices = constraints.get('choices')
                if choices and value not in choices:
                    raise ConfigurationError(f"Invalid choice for {section}.{key}: {value} not in {choices}")
            
            return True
            
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Schema validation failed for section {section}: {e}")
    
    def create_pipeline_config(self) -> Dict[str, Any]:
        """Create configuration dictionary for DynamicChunkingPipeline."""
        return {
            'compression_ratio': self.get('pipeline.compression_ratio', 6.0),
            'embedding_model': self.get('boundary_detection.embedding_model'),
            'embedding_dim': self.get('pipeline.embedding_dim', 384),
            'cache_embeddings': self.get('caching.enabled', True),
            'device': self.get('pipeline.device', 'cpu')
        }
    
    def create_boundary_detector_config(self) -> Dict[str, Any]:
        """Create configuration dictionary for SimilarityBasedBoundaryDetector."""
        return {
            'embedding_dim': self.get('pipeline.embedding_dim', 384),
            'device': self.get('pipeline.device', 'cpu')
        }
    
    def create_routing_config(self) -> Dict[str, Any]:
        """Create configuration dictionary for RoutingModule."""
        return {
            'target_compression_ratio': self.get('routing.target_compression_ratio', 6.0)
        }
    
    def create_smoothing_config(self) -> Dict[str, Any]:
        """Create configuration dictionary for SmoothingModule."""
        return {
            'alpha': self.get('smoothing.alpha', 0.7),
            'use_confidence_weighting': self.get('smoothing.use_confidence_weighting', True)
        }


# Global configuration instance
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigManager()
    
    return _global_config


def set_config(config: ConfigManager) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config


def load_config(config_file: str) -> ConfigManager:
    """Load configuration from file and set as global."""
    config = ConfigManager(config_file=config_file)
    set_config(config)
    return config


def create_config_from_dict(config_dict: Dict[str, Any]) -> ConfigManager:
    """Create configuration from dictionary and set as global."""
    config = ConfigManager(config_dict=config_dict)
    set_config(config)
    return config


# Configuration schemas for validation
PIPELINE_SCHEMA = {
    'compression_ratio': {'type': float, 'min': 1.0, 'max': 50.0, 'required': True},
    'embedding_dim': {'type': int, 'min': 32, 'max': 2048, 'required': True},
    'device': {'type': str, 'choices': ['cpu', 'cuda'], 'required': True},
    'cache_embeddings': {'type': bool, 'required': True}
}

BOUNDARY_DETECTION_SCHEMA = {
    'threshold_mode': {'type': str, 'choices': ['adaptive', 'fixed'], 'required': True},
    'fixed_threshold': {'type': float, 'min': 0.0, 'max': 1.0},
    'min_threshold': {'type': float, 'min': 0.0, 'max': 1.0}
}

SMOOTHING_SCHEMA = {
    'alpha': {'type': float, 'min': 0.0, 'max': 1.0, 'required': True},
    'use_confidence_weighting': {'type': bool, 'required': True},
    'noise_threshold': {'type': float, 'min': 0.0, 'max': 1.0}
}