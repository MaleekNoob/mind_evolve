"""Population management for Mind Evolution."""

import numpy as np
from loguru import logger

from .models import PopulationStats, Solution


class Population:
    """Population of solutions on a single island."""

    def __init__(self, island_id: int, max_size: int | None = None) -> None:
        """Initialize population.

        Args:
            island_id: Unique identifier for this island
            max_size: Maximum population size (None for unlimited)
        """
        self.island_id = island_id
        self.solutions: dict[str, Solution] = {}
        self.max_size = max_size
        self.generation = 0
        self.best_solution: Solution | None = None

    def add_solution(self, solution: Solution) -> None:
        """Add solution to population, checking for duplicates.

        Args:
            solution: Solution to add
        """
        # Check for duplicate content
        if self._is_duplicate(solution):
            logger.debug(f"Skipping duplicate solution: {solution.id[:8]}")
            return

        # Add to population
        self.solutions[solution.id] = solution
        self._update_best(solution)

        # Enforce size limit if specified
        if self.max_size and len(self.solutions) > self.max_size:
            self._remove_worst()

        logger.debug(f"Added solution {solution.id[:8]} to island {self.island_id}")

    def _is_duplicate(self, solution: Solution) -> bool:
        """Check if solution content already exists in population.

        Args:
            solution: Solution to check

        Returns:
            True if duplicate exists
        """
        normalized_content = solution.content.strip().lower()
        for existing in self.solutions.values():
            if existing.content.strip().lower() == normalized_content:
                return True
        return False

    def _update_best(self, solution: Solution) -> None:
        """Update best solution if this one is better.

        Args:
            solution: Candidate best solution
        """
        if self.best_solution is None or solution.score > self.best_solution.score:
            self.best_solution = solution
            logger.debug(
                f"New best solution on island {self.island_id}: "
                f"score={solution.score:.3f}"
            )

    def _remove_worst(self) -> None:
        """Remove worst solution to maintain size limit."""
        if not self.solutions:
            return

        worst_id = min(self.solutions.keys(), key=lambda k: self.solutions[k].score)
        removed = self.solutions.pop(worst_id)
        logger.debug(
            f"Removed worst solution {worst_id[:8]} "
            f"(score={removed.score:.3f}) from island {self.island_id}"
        )

    def get_selection_pool(self) -> list[Solution]:
        """Return all solutions available for selection.

        Returns:
            List of all solutions in population
        """
        return list(self.solutions.values())

    def get_top_solutions(self, n: int) -> list[Solution]:
        """Get top N solutions by score.

        Args:
            n: Number of solutions to return

        Returns:
            List of top N solutions
        """
        sorted_solutions = sorted(
            self.solutions.values(), key=lambda s: s.score, reverse=True
        )
        return sorted_solutions[:n]

    def clear(self) -> None:
        """Clear all solutions from population."""
        logger.info(f"Clearing population on island {self.island_id}")
        self.solutions.clear()
        self.best_solution = None

    def calculate_statistics(self) -> PopulationStats:
        """Calculate population statistics.

        Returns:
            PopulationStats object with computed metrics
        """
        if not self.solutions:
            return PopulationStats(
                island_id=self.island_id,
                generation=self.generation,
                size=0,
                mean_score=0.0,
                max_score=0.0,
                min_score=0.0,
                std_score=0.0,
                valid_solutions=0,
                best_solution_id=None,
            )

        scores = [s.score for s in self.solutions.values()]
        valid_count = sum(1 for s in self.solutions.values() if s.is_valid())

        return PopulationStats(
            island_id=self.island_id,
            generation=self.generation,
            size=len(self.solutions),
            mean_score=float(np.mean(scores)),
            max_score=float(np.max(scores)),
            min_score=float(np.min(scores)),
            std_score=float(np.std(scores)),
            valid_solutions=valid_count,
            best_solution_id=self.best_solution.id if self.best_solution else None,
        )

    def get_diversity_score(self) -> float:
        """Calculate diversity score based on solution content.

        Returns:
            Diversity score (higher = more diverse)
        """
        if len(self.solutions) < 2:
            return 0.0

        # Simple diversity metric based on content length variance
        lengths = [len(s.content) for s in self.solutions.values()]
        return float(np.std(lengths))

    def __len__(self) -> int:
        """Return number of solutions in population."""
        return len(self.solutions)

    def __repr__(self) -> str:
        """String representation of population."""
        return (
            f"Population(island_id={self.island_id}, "
            f"size={len(self.solutions)}, "
            f"generation={self.generation}, "
            f"best_score={self.best_solution.score if self.best_solution else 'N/A'})"
        )
