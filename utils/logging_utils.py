"""
Logging utilities for the Chinese Spelling Correction project.
Provides centralized logging configuration and utilities.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name. If None, logs to console only
        log_dir: Directory to store log files
        include_timestamp: Whether to include timestamp in log messages
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("llmcsc")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped filename if not provided
        if log_file == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"llmcsc_{timestamp}.log"
        
        file_handler = logging.FileHandler(
            log_file,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "llmcsc") -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_system_info(logger: logging.Logger):
    """Log system information for debugging purposes."""
    import platform
    import torch
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA available: False")

class ProgressLogger:
    """Utility class for logging progress with intervals."""
    
    def __init__(self, logger: logging.Logger, total: int, interval: int = 100):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance
            total: Total number of items
            interval: Log progress every N items
        """
        self.logger = logger
        self.total = total
        self.interval = interval
        self.current = 0
        
    def update(self, increment: int = 1):
        """Update progress counter and log if needed."""
        self.current += increment
        if self.current % self.interval == 0 or self.current == self.total:
            progress = (self.current / self.total) * 100
            self.logger.info(f"Progress: {self.current}/{self.total} ({progress:.1f}%)")
    
    def reset(self):
        """Reset progress counter."""
        self.current = 0
