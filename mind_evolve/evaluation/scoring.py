"""Scoring utilities for solution evaluation."""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.models import Solution


class ScoringFunction:
    """Base class for scoring functions."""
    
    def __init__(self, name: str, weight: float = 1.0):
        """Initialize scoring function.
        
        Args:
            name: Name of the scoring function
            weight: Weight for this scoring component
        """
        self.name = name
        self.weight = weight
        
    def score(self, solution_content: str, **kwargs: Any) -> float:
        """Calculate score for solution content.
        
        Args:
            solution_content: Solution to score
            **kwargs: Additional context
            
        Returns:
            Score value
        """
        raise NotImplementedError
        

class LengthBasedScoring(ScoringFunction):
    """Scoring based on solution length."""
    
    def __init__(self, 
                 target_length: int = 200, 
                 min_length: int = 50,
                 max_length: int = 1000,
                 **kwargs: Any):
        """Initialize length-based scoring.
        
        Args:
            target_length: Optimal solution length in characters
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length before penalty
            **kwargs: Additional parameters
        """
        super().__init__("length_based", **kwargs)
        self.target_length = target_length
        self.min_length = min_length
        self.max_length = max_length
        
    def score(self, solution_content: str, **kwargs: Any) -> float:
        """Score based on length proximity to target.
        
        Args:
            solution_content: Solution to score
            **kwargs: Additional context
            
        Returns:
            Score between 0 and 10
        """
        length = len(solution_content.strip())
        
        if length < self.min_length:
            # Penalty for too short
            return 2.0 * (length / self.min_length)
        elif length > self.max_length:
            # Penalty for too long
            excess_ratio = (length - self.max_length) / self.max_length
            return max(1.0, 8.0 - 3.0 * excess_ratio)
        else:
            # Score based on proximity to target
            distance = abs(length - self.target_length) / self.target_length
            return 8.0 + 2.0 * (1.0 - distance)


class StructureBasedScoring(ScoringFunction):
    """Scoring based on solution structure."""
    
    def __init__(self, **kwargs: Any):
        """Initialize structure-based scoring."""
        super().__init__("structure_based", **kwargs)
        
    def score(self, solution_content: str, **kwargs: Any) -> float:
        """Score based on solution structure quality.
        
        Args:
            solution_content: Solution to score
            **kwargs: Additional context
            
        Returns:
            Score between 0 and 10
        """
        lines = solution_content.strip().split('\n')
        score = 5.0  # Base score
        
        # Reward multi-line structure
        if len(lines) > 1:
            score += 1.0
            
        # Reward lists or numbered items
        structured_lines = sum(1 for line in lines 
                             if line.strip().startswith(('-', '*', '1.', '2.', '3.')))
        if structured_lines > 0:
            score += min(2.0, structured_lines * 0.5)
            
        # Reward paragraph structure
        if '\n\n' in solution_content:
            score += 1.0
            
        # Check for section headers (lines ending with ':')
        headers = sum(1 for line in lines if line.strip().endswith(':'))
        if headers > 0:
            score += min(1.0, headers * 0.3)
            
        return min(10.0, score)


class KeywordBasedScoring(ScoringFunction):
    """Scoring based on keyword presence."""
    
    def __init__(self, keywords: List[str], **kwargs: Any):
        """Initialize keyword-based scoring.
        
        Args:
            keywords: List of important keywords to check for
            **kwargs: Additional parameters
        """
        super().__init__("keyword_based", **kwargs)
        self.keywords = [kw.lower() for kw in keywords]
        
    def score(self, solution_content: str, **kwargs: Any) -> float:
        """Score based on keyword coverage.
        
        Args:
            solution_content: Solution to score
            **kwargs: Additional context
            
        Returns:
            Score between 0 and 10
        """
        if not self.keywords:
            return 5.0
            
        content_lower = solution_content.lower()
        found_keywords = sum(1 for kw in self.keywords if kw in content_lower)
        
        coverage_ratio = found_keywords / len(self.keywords)
        return 3.0 + 7.0 * coverage_ratio


class CompositeScorer:
    """Combines multiple scoring functions."""
    
    def __init__(self, scoring_functions: List[ScoringFunction]):
        """Initialize composite scorer.
        
        Args:
            scoring_functions: List of scoring functions to combine
        """
        self.scoring_functions = scoring_functions
        
    def calculate_score(self, solution_content: str, **kwargs: Any) -> Dict[str, float]:
        """Calculate composite score.
        
        Args:
            solution_content: Solution to score
            **kwargs: Additional context
            
        Returns:
            Dictionary with individual and composite scores
        """
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        for func in self.scoring_functions:
            individual_score = func.score(solution_content, **kwargs)
            scores[func.name] = individual_score
            
            weighted_sum += individual_score * func.weight
            total_weight += func.weight
            
        # Calculate final composite score
        composite_score = weighted_sum / max(total_weight, 1.0)
        scores['composite'] = composite_score
        scores['total_weight'] = total_weight
        
        return scores
        
    def add_scoring_function(self, scoring_function: ScoringFunction) -> None:
        """Add a new scoring function.
        
        Args:
            scoring_function: Scoring function to add
        """
        self.scoring_functions.append(scoring_function)
        
    def remove_scoring_function(self, name: str) -> bool:
        """Remove scoring function by name.
        
        Args:
            name: Name of scoring function to remove
            
        Returns:
            True if function was removed, False if not found
        """
        for i, func in enumerate(self.scoring_functions):
            if func.name == name:
                self.scoring_functions.pop(i)
                return True
        return False


class PopulationScorer:
    """Utilities for scoring populations of solutions."""
    
    @staticmethod
    def calculate_diversity_score(solutions: List[Solution]) -> float:
        """Calculate diversity score for a population.
        
        Args:
            solutions: List of solutions
            
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(solutions) < 2:
            return 0.0
            
        # Simple diversity metric based on content differences
        contents = [sol.content for sol in solutions]
        pairwise_distances = []
        
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                distance = PopulationScorer._calculate_text_distance(contents[i], contents[j])
                pairwise_distances.append(distance)
                
        return float(np.mean(pairwise_distances))
        
    @staticmethod
    def _calculate_text_distance(text1: str, text2: str) -> float:
        """Calculate simple text distance between two strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Distance score between 0 and 1
        """
        # Simple Jaccard distance based on word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 0.0
        if not words1 or not words2:
            return 1.0
            
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return 1.0 - (intersection / union)
        
    @staticmethod
    def rank_solutions(solutions: List[Solution], 
                      rank_by: str = "score") -> List[Solution]:
        """Rank solutions by specified criteria.
        
        Args:
            solutions: List of solutions to rank
            rank_by: Ranking criteria (score, generation, timestamp)
            
        Returns:
            Sorted list of solutions (best first)
        """
        if rank_by == "score":
            return sorted(solutions, key=lambda s: s.score, reverse=True)
        elif rank_by == "generation":
            return sorted(solutions, key=lambda s: s.generation, reverse=True)
        elif rank_by == "timestamp":
            return sorted(solutions, key=lambda s: s.timestamp, reverse=True)
        else:
            raise ValueError(f"Unknown ranking criteria: {rank_by}")
            
    @staticmethod
    def calculate_population_stats(solutions: List[Solution]) -> Dict[str, float]:
        """Calculate statistics for a population of solutions.
        
        Args:
            solutions: List of solutions
            
        Returns:
            Dictionary with population statistics
        """
        if not solutions:
            return {}
            
        scores = [s.score for s in solutions]
        
        return {
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "range_score": float(np.max(scores) - np.min(scores)),
            "valid_solutions": sum(1 for s in solutions if s.is_valid()),
            "total_solutions": len(solutions),
        }