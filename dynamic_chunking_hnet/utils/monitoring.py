"""
Logging and Performance Monitoring System

Provides structured logging, performance metrics, and monitoring capabilities
for the Dynamic ChunkingHNet system.
"""

import logging
import logging.handlers
import time
import threading
import psutil
import os
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps

from .config import get_config


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    cpu_percent: float
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceMonitor:
    """
    Performance monitoring system that tracks execution time, memory usage,
    and other system metrics.
    """
    
    def __init__(self):
        self._metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
    
    @contextmanager
    def monitor_operation(
        self, 
        operation_name: str,
        input_size: Optional[int] = None
    ):
        """
        Context manager for monitoring an operation.
        
        Args:
            operation_name: Name of the operation being monitored
            input_size: Size of input data (optional)
            
        Yields:
            Dictionary that can be updated with additional metrics
        """
        # Get process for memory monitoring
        process = psutil.Process(os.getpid())
        
        # Initial measurements
        start_time = time.time()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        
        # Additional metrics that can be set during operation
        operation_metrics = {}
        
        success = True
        error_message = None
        
        try:
            yield operation_metrics
            
        except Exception as e:
            success = False
            error_message = str(e)
            raise
            
        finally:
            # Final measurements
            end_time = time.time()
            duration = end_time - start_time
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before
            cpu_after = process.cpu_percent()
            cpu_avg = (cpu_before + cpu_after) / 2
            
            # Create metrics object
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_delta_mb=memory_delta,
                cpu_percent=cpu_avg,
                input_size=input_size,
                output_size=operation_metrics.get('output_size'),
                success=success,
                error_message=error_message
            )
            
            # Store metrics
            with self._lock:
                self._metrics.append(metrics)
            
            # Log performance info
            if success:
                self.logger.info(
                    f"Operation '{operation_name}' completed: "
                    f"duration={duration:.3f}s, "
                    f"memory_delta={memory_delta:+.1f}MB"
                )
            else:
                self.logger.error(
                    f"Operation '{operation_name}' failed: "
                    f"duration={duration:.3f}s, error={error_message}"
                )
    
    def get_metrics(
        self, 
        operation_name: Optional[str] = None,
        last_n: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """
        Get collected performance metrics.
        
        Args:
            operation_name: Filter by operation name (optional)
            last_n: Return only last N metrics (optional)
            
        Returns:
            List of performance metrics
        """
        with self._lock:
            metrics = self._metrics.copy()
        
        # Filter by operation name
        if operation_name:
            metrics = [m for m in metrics if m.operation_name == operation_name]
        
        # Limit to last N
        if last_n:
            metrics = metrics[-last_n:]
        
        return metrics
    
    def get_summary_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for operations.
        
        Args:
            operation_name: Filter by operation name (optional)
            
        Returns:
            Dictionary with summary statistics
        """
        metrics = self.get_metrics(operation_name=operation_name)
        
        if not metrics:
            return {}
        
        durations = [m.duration for m in metrics]
        memory_deltas = [m.memory_delta_mb for m in metrics]
        successes = [m.success for m in metrics]
        
        return {
            'total_operations': len(metrics),
            'successful_operations': sum(successes),
            'failed_operations': len(metrics) - sum(successes),
            'success_rate': sum(successes) / len(metrics),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
            'max_memory_delta_mb': max(memory_deltas),
            'min_memory_delta_mb': min(memory_deltas),
        }
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        with self._lock:
            self._metrics.clear()
        self.logger.info("Performance metrics cleared")
    
    def export_metrics(self, file_path: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            file_path: Path to export file
        """
        metrics_data = [m.to_dict() for m in self.get_metrics()]
        
        try:
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported {len(metrics_data)} metrics to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


def performance_monitor(operation_name: Optional[str] = None):
    """
    Decorator for monitoring function performance.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        actual_operation_name = operation_name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            
            # Try to determine input size
            input_size = None
            if args:
                first_arg = args[0]
                if hasattr(first_arg, '__len__'):
                    input_size = len(first_arg)
                elif isinstance(first_arg, str):
                    input_size = len(first_arg.split())
            
            with monitor.monitor_operation(actual_operation_name, input_size) as metrics:
                result = func(*args, **kwargs)
                
                # Try to determine output size
                if hasattr(result, '__len__'):
                    metrics['output_size'] = len(result)
                elif isinstance(result, dict) and 'chunks' in result:
                    metrics['output_size'] = len(result['chunks'])
                
                return result
        
        return wrapper
    return decorator


class StructuredLogger:
    """
    Structured logging system with JSON formatting and multiple handlers.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config or get_config().get_section('logging')
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        
        if self.config.get('enable_file_logging', False):
            self._setup_file_handler()
    
    def _setup_console_handler(self) -> None:
        """Setup console logging handler."""
        handler = logging.StreamHandler()
        
        # Use structured formatter
        formatter = StructuredFormatter(
            fmt=self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def _setup_file_handler(self) -> None:
        """Setup file logging handler with rotation."""
        log_file = self.config.get('log_file', 'dynamic_chunking.log')
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # JSON formatter for file logs
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with structured data."""
        self.logger.info(message, extra={'structured_data': kwargs})
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with structured data."""
        self.logger.debug(message, extra={'structured_data': kwargs})
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with structured data."""
        self.logger.warning(message, extra={'structured_data': kwargs})
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with structured data."""
        self.logger.error(message, extra={'structured_data': kwargs})
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with structured data."""
        self.logger.critical(message, extra={'structured_data': kwargs})


class StructuredFormatter(logging.Formatter):
    """Custom formatter that includes structured data."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Get base formatted message
        base_msg = super().format(record)
        
        # Add structured data if present
        if hasattr(record, 'structured_data') and record.structured_data:
            structured_parts = []
            for key, value in record.structured_data.items():
                structured_parts.append(f"{key}={value}")
            
            if structured_parts:
                base_msg += f" [{', '.join(structured_parts)}]"
        
        return base_msg


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add structured data if present
        if hasattr(record, 'structured_data') and record.structured_data:
            log_entry['data'] = record.structured_data
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


# Global instances
_performance_monitor: Optional[PerformanceMonitor] = None
_loggers: Dict[str, StructuredLogger] = {}


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor


def get_logger(name: str) -> StructuredLogger:
    """
    Get structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    
    return _loggers[name]


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup global logging configuration.
    
    Args:
        config: Logging configuration (uses global config if None)
    """
    if config is None:
        config = get_config().get_section('logging')
    
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    
    # Clear existing loggers cache
    global _loggers
    _loggers.clear()


# Convenience functions for the main logger
main_logger = get_logger('dynamic_chunking_hnet')

def log_info(message: str, **kwargs) -> None:
    """Log info message."""
    main_logger.info(message, **kwargs)

def log_debug(message: str, **kwargs) -> None:
    """Log debug message."""
    main_logger.debug(message, **kwargs)

def log_warning(message: str, **kwargs) -> None:
    """Log warning message."""
    main_logger.warning(message, **kwargs)

def log_error(message: str, **kwargs) -> None:
    """Log error message."""
    main_logger.error(message, **kwargs)

def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance information."""
    main_logger.info(
        f"Performance: {operation}",
        duration=duration,
        **kwargs
    )