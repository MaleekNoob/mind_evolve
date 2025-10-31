"""Selection algorithms for Mind Evolution."""

import random

import numpy as np
from loguru import logger

from ..core.models import Solution


class SelectionStrategy:
    """Base class for selection strategies."""

    def select_parents(self,
                      selection_pool: list[Solution],
                      num_parents: int,
                      config: dict) -> list[Solution]:
        """Select parents from the population.
        
        Args:
            selection_pool: Available solutions for selection
            num_parents: Number of parents to select
            config: Configuration parameters
            
        Returns:
            List of selected parent solutions
        """
        raise NotImplementedError


class BoltzmannTournamentSelection(SelectionStrategy):
    """Boltzmann tournament selection with softmax probabilities."""

    def select_parents(self,
                      selection_pool: list[Solution],
                      num_parents: int,
                      config: dict) -> list[Solution]:
        """Select parents using Boltzmann tournament selection.
        
        Args:
            selection_pool: Available solutions for selection
            num_parents: Number of parents to select (0 to N_parent)
            config: Configuration with Pr_no_parents, temperature, etc.
            
        Returns:
            List of selected parent solutions (may be empty)
        """
        # Handle edge cases
        if not selection_pool:
            return []

        # Decide if we use parents at all
        if random.random() < config.get('Pr_no_parents', 1/6):
            return []  # Pure mutation - no parents

        # Limit number of parents to available solutions
        num_parents = min(num_parents, len(selection_pool))
        if num_parents <= 0:
            return []

        # Calculate selection probabilities via softmax
        scores = np.array([s.score for s in selection_pool])

        # Handle case where all scores are identical
        if np.std(scores) == 0:
            # Random selection when all scores are equal
            selected_indices = random.sample(range(len(selection_pool)), num_parents)
            return [selection_pool[i] for i in selected_indices]

        # Apply temperature scaling
        temperature = config.get('temperature', 1.0)
        if temperature == 0:
            # Greedy selection - always pick best
            sorted_indices = np.argsort(scores)[::-1]
            return [selection_pool[i] for i in sorted_indices[:num_parents]]

        # Softmax with temperature
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / np.sum(exp_scores)

        # Sample without replacement
        try:
            selected_indices = np.random.choice(
                len(selection_pool),
                size=num_parents,
                replace=False,
                p=probabilities
            )
            parents = [selection_pool[i] for i in selected_indices]

            logger.debug(f"Selected {len(parents)} parents with scores: "
                        f"{[p.score for p in parents]}")
            return parents

        except ValueError as e:
            logger.warning(f"Selection failed: {e}. Using random fallback.")
            # Fallback to random selection
            selected_indices = random.sample(range(len(selection_pool)), num_parents)
            return [selection_pool[i] for i in selected_indices]


class TournamentSelection(SelectionStrategy):
    """Traditional tournament selection."""

    def __init__(self, tournament_size: int = 3):
        """Initialize tournament selection.
        
        Args:
            tournament_size: Number of candidates per tournament
        """
        self.tournament_size = tournament_size

    def select_parents(self,
                      selection_pool: list[Solution],
                      num_parents: int,
                      config: dict) -> list[Solution]:
        """Select parents using tournament selection.
        
        Args:
            selection_pool: Available solutions for selection
            num_parents: Number of parents to select
            config: Configuration parameters
            
        Returns:
            List of selected parent solutions
        """
        if not selection_pool:
            return []

        # Decide if we use parents at all
        if random.random() < config.get('Pr_no_parents', 1/6):
            return []

        num_parents = min(num_parents, len(selection_pool))
        if num_parents <= 0:
            return []

        parents = []
        for _ in range(num_parents):
            # Run tournament
            tournament_size = min(self.tournament_size, len(selection_pool))
            tournament = random.sample(selection_pool, tournament_size)
            winner = max(tournament, key=lambda s: s.score)
            parents.append(winner)

        return parents


class RouletteWheelSelection(SelectionStrategy):
    """Fitness-proportionate selection (roulette wheel)."""

    def select_parents(self,
                      selection_pool: list[Solution],
                      num_parents: int,
                      config: dict) -> list[Solution]:
        """Select parents using roulette wheel selection.
        
        Args:
            selection_pool: Available solutions for selection
            num_parents: Number of parents to select
            config: Configuration parameters
            
        Returns:
            List of selected parent solutions
        """
        if not selection_pool:
            return []

        # Decide if we use parents at all
        if random.random() < config.get('Pr_no_parents', 1/6):
            return []

        num_parents = min(num_parents, len(selection_pool))
        if num_parents <= 0:
            return []

        # Calculate fitness proportions
        scores = np.array([s.score for s in selection_pool])

        # Handle negative scores by shifting
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score + 1e-6

        # Handle case where all scores are zero
        if np.sum(scores) == 0:
            scores = np.ones_like(scores)

        probabilities = scores / np.sum(scores)

        # Sample with replacement
        selected_indices = np.random.choice(
            len(selection_pool),
            size=num_parents,
            replace=True,
            p=probabilities
        )

        return [selection_pool[i] for i in selected_indices]


def create_selection_strategy(strategy_name: str, **kwargs) -> SelectionStrategy:
    """Factory function to create selection strategies.
    
    Args:
        strategy_name: Name of the strategy
        **kwargs: Strategy-specific parameters
        
    Returns:
        Initialized selection strategy
    """
    strategies = {
        'boltzmann': BoltzmannTournamentSelection,
        'tournament': TournamentSelection,
        'roulette': RouletteWheelSelection,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown selection strategy: {strategy_name}")

    return strategies[strategy_name](**kwargs)
