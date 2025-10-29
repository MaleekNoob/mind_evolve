"""Core module initialization."""

from .evolutionary_engine import MindEvolution
from .island_model import IslandModel
from .models import (
    ConversationThread,
    ConversationTurn,
    EvaluationResult,
    MindEvolutionConfig,
    PopulationStats,
    Problem,
    Solution,
)
from .population import Population
from .selection import (
    BoltzmannTournamentSelection,
    RouletteWheelSelection,
    SelectionStrategy,
    TournamentSelection,
    create_selection_strategy,
)

__all__ = [
    # Main classes
    "MindEvolution",
    "IslandModel",
    
    # Models
    "Solution",
    "ConversationTurn",
    "ConversationThread", 
    "EvaluationResult",
    "Problem",
    "MindEvolutionConfig",
    "PopulationStats",
    
    # Population
    "Population",
    
    # Selection
    "SelectionStrategy",
    "BoltzmannTournamentSelection",
    "TournamentSelection",
    "RouletteWheelSelection",
    "create_selection_strategy",
]