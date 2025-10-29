"""Mind Evolution: Evolutionary Search for LLM-based Problem Solving.

This package implements Mind Evolution, a novel approach to improve Large Language Model
problem-solving by using evolutionary search strategies during inference time.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .core import (
    MindEvolutionConfig,
    Problem,
    Solution,
    Population,
)
from .core.evolutionary_engine import MindEvolution
from .core.island_model import IslandModel
from .evaluation import BaseEvaluator, create_evaluator
from .llm import BaseLLM, create_llm, PromptManager
from .utils import ConfigManager, setup_logging

__all__ = [
    # Core classes
    "MindEvolution",
    "IslandModel", 
    "MindEvolutionConfig",
    "Problem",
    "Solution",
    "Population",
    
    # LLM interface
    "BaseLLM",
    "create_llm",
    "PromptManager",
    
    # Evaluation
    "BaseEvaluator", 
    "create_evaluator",
    
    # Utilities
    "ConfigManager",
    "setup_logging",
]