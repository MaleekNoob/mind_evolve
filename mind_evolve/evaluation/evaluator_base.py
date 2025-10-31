"""Base evaluator interface for Mind Evolution."""

from abc import ABC, abstractmethod
from typing import Any

from ..core.models import EvaluationResult, Problem


class BaseEvaluator(ABC):
    """Abstract base class for solution evaluators."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize evaluator with configuration.

        Args:
            **kwargs: Evaluator-specific configuration
        """
        self.config = kwargs

    @abstractmethod
    def evaluate(self, solution_content: str, problem: Problem) -> EvaluationResult:
        """Evaluate a candidate solution.

        Args:
            solution_content: The solution text to evaluate
            problem: The problem definition

        Returns:
            EvaluationResult with score, feedback, and validity
        """
        pass

    @abstractmethod
    def parse_solution(self, solution_content: str) -> Any:
        """Parse solution from natural language to structured format.

        Args:
            solution_content: Raw solution text

        Returns:
            Structured representation of the solution
        """
        pass

    def get_evaluator_info(self) -> dict[str, Any]:
        """Get information about this evaluator.

        Returns:
            Dictionary with evaluator metadata
        """
        return {
            "type": self.__class__.__name__,
            "config": self.config,
        }


class SimpleScoreEvaluator(BaseEvaluator):
    """Simple evaluator that assigns random scores for testing."""

    def __init__(
        self, min_score: float = 0.0, max_score: float = 10.0, **kwargs: Any
    ) -> None:
        """Initialize simple evaluator.

        Args:
            min_score: Minimum possible score
            max_score: Maximum possible score
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.min_score = min_score
        self.max_score = max_score

    def evaluate(self, solution_content: str, problem: Problem) -> EvaluationResult:
        """Evaluate solution with simple heuristics.

        Args:
            solution_content: Solution to evaluate
            problem: Problem definition

        Returns:
            EvaluationResult with basic scoring
        """
        import random

        # Simple heuristics
        content_length = len(solution_content.strip())
        has_content = content_length > 0

        # Basic scoring
        if not has_content:
            score = self.min_score
            feedback = ["Solution is empty or contains no meaningful content"]
            is_valid = False
            violations = 1
        else:
            # Random score weighted by content length
            length_factor = min(content_length / 100, 1.0)  # Normalize to [0, 1]
            score = (
                self.min_score
                + (self.max_score - self.min_score) * length_factor * random.random()
            )

            feedback = []
            violations = 0

            # Generate some feedback based on content
            if content_length < 50:
                feedback.append("Solution appears too brief")
                violations += 1
            elif content_length > 1000:
                feedback.append("Solution may be unnecessarily verbose")

            if not any(
                constraint.lower() in solution_content.lower()
                for constraint in problem.constraints[:3]
            ):  # Check first 3 constraints
                feedback.append("Solution does not explicitly address key constraints")
                violations += 1

            is_valid = violations == 0

        return EvaluationResult(
            score=score,
            feedback=feedback,
            is_valid=is_valid,
            constraint_violations=violations,
            metadata={
                "content_length": content_length,
                "evaluator": "SimpleScoreEvaluator",
            },
        )

    def parse_solution(self, solution_content: str) -> dict[str, Any]:
        """Parse solution into simple dictionary.

        Args:
            solution_content: Raw solution text

        Returns:
            Dictionary with parsed solution
        """
        return {
            "content": solution_content.strip(),
            "word_count": len(solution_content.split()),
            "char_count": len(solution_content),
        }


class ConstraintBasedEvaluator(BaseEvaluator):
    """Evaluator that checks solutions against explicit constraints."""

    def __init__(
        self,
        constraint_weights: dict[str, float] = None,
        base_score: float = 10.0,
        **kwargs: Any,
    ) -> None:
        """Initialize constraint-based evaluator.

        Args:
            constraint_weights: Weights for different constraint types
            base_score: Base score before constraint violations
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.constraint_weights = constraint_weights or {}
        self.base_score = base_score

    def evaluate(self, solution_content: str, problem: Problem) -> EvaluationResult:
        """Evaluate solution based on constraint satisfaction.

        Args:
            solution_content: Solution to evaluate
            problem: Problem with constraints

        Returns:
            EvaluationResult with constraint-based scoring
        """
        if not solution_content.strip():
            return EvaluationResult(
                score=0.0,
                feedback=["Empty solution"],
                is_valid=False,
                constraint_violations=1,
                metadata={"evaluator": "ConstraintBasedEvaluator"},
            )

        score = self.base_score
        feedback = []
        violations = 0

        # Check each constraint
        for i, constraint in enumerate(problem.constraints):
            satisfied = self._check_constraint(solution_content, constraint)

            if not satisfied:
                violations += 1
                penalty = self.constraint_weights.get(f"constraint_{i}", 1.0)
                score -= penalty
                feedback.append(f"Constraint violated: {constraint}")
            else:
                feedback.append(f"Constraint satisfied: {constraint}")

        # Ensure score doesn't go below 0
        score = max(0.0, score)
        is_valid = violations == 0

        return EvaluationResult(
            score=score,
            feedback=feedback,
            is_valid=is_valid,
            constraint_violations=violations,
            metadata={
                "base_score": self.base_score,
                "total_constraints": len(problem.constraints),
                "evaluator": "ConstraintBasedEvaluator",
            },
        )

    def _check_constraint(self, solution: str, constraint: str) -> bool:
        """Check if solution satisfies a constraint.

        Args:
            solution: Solution content
            constraint: Constraint to check

        Returns:
            True if constraint is satisfied
        """
        # Simple keyword-based checking
        solution_lower = solution.lower()
        constraint_lower = constraint.lower()

        # Extract key terms from constraint
        key_terms = [
            term.strip()
            for term in constraint_lower.split()
            if len(term.strip()) > 3
            and term.strip()
            not in ["must", "should", "need", "have", "include", "contain", "require"]
        ]

        # Check if at least half of key terms are mentioned
        mentioned_terms = sum(1 for term in key_terms if term in solution_lower)
        return mentioned_terms >= max(1, len(key_terms) // 2)

    def parse_solution(self, solution_content: str) -> dict[str, Any]:
        """Parse solution into structured format.

        Args:
            solution_content: Raw solution text

        Returns:
            Parsed solution dictionary
        """
        lines = solution_content.strip().split("\n")

        return {
            "content": solution_content.strip(),
            "line_count": len(lines),
            "word_count": len(solution_content.split()),
            "has_structure": any(
                line.strip().startswith(("-", "*", "1.", "2.")) for line in lines
            ),
        }


def create_evaluator(evaluator_type: str, **kwargs: Any) -> BaseEvaluator:
    """Factory function to create evaluators.

    Args:
        evaluator_type: Type of evaluator to create
        **kwargs: Evaluator-specific configuration

    Returns:
        Initialized evaluator instance
    """
    evaluators = {
        "simple": SimpleScoreEvaluator,
        "constraint": ConstraintBasedEvaluator,
    }

    if evaluator_type not in evaluators:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")

    return evaluators[evaluator_type](**kwargs)
