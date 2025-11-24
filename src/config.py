"""
Model Configuration Classes

This module contains configuration classes for model generation parameters.
Separated from model.py for better modularity and reusability.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model generation parameters."""
    max_tokens: int = 1024
    temperature: Optional[float] = None  # None means use model's default temperature
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop: Optional[List[str]] = None
    stream: bool = False