"""Metrics collection and analysis utilities."""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.models import PopulationStats, Solution


@dataclass
class ExperimentMetrics:
    """Comprehensive metrics for an experiment run."""

    # Basic info
    experiment_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    # Evolution metrics
    generations_completed: int = 0
    total_solutions_generated: int = 0
    valid_solutions_found: int = 0
    best_score_achieved: float = 0.0

    # Performance metrics
    avg_generation_time: float = 0.0
    total_llm_calls: int = 0
    total_evaluation_calls: int = 0
    avg_llm_response_time: float = 0.0

    # Population metrics
    population_stats_history: list[dict[str, PopulationStats]] = field(
        default_factory=list
    )
    diversity_history: list[float] = field(default_factory=list)

    # Island metrics
    island_performance: dict[int, dict[str, float]] = field(default_factory=dict)
    migration_statistics: list[dict[str, Any]] = field(default_factory=list)
    reset_statistics: list[dict[str, Any]] = field(default_factory=list)

    # Solution quality metrics
    score_history: list[float] = field(default_factory=list)
    constraint_violation_history: list[int] = field(default_factory=list)

    @property
    def runtime_minutes(self) -> float:
        """Get total runtime in minutes."""
        end = self.end_time or time.time()
        return (end - self.start_time) / 60.0

    @property
    def solutions_per_minute(self) -> float:
        """Get solutions generated per minute."""
        runtime = self.runtime_minutes
        return self.total_solutions_generated / max(runtime, 0.01)

    @property
    def success_rate(self) -> float:
        """Get percentage of valid solutions."""
        if self.total_solutions_generated == 0:
            return 0.0
        return (self.valid_solutions_found / self.total_solutions_generated) * 100.0


class MetricsCollector:
    """Collects and analyzes metrics during Mind Evolution runs."""

    def __init__(self, experiment_name: str):
        """Initialize metrics collector.

        Args:
            experiment_name: Name of the experiment
        """
        self.metrics = ExperimentMetrics(experiment_name=experiment_name)
        self.generation_start_times: dict[int, float] = {}
        self.llm_call_times: list[float] = []
        self.evaluation_times: list[float] = []

    def start_generation(self, generation: int) -> None:
        """Mark start of a generation.

        Args:
            generation: Generation number
        """
        self.generation_start_times[generation] = time.time()

    def end_generation(
        self,
        generation: int,
        population_stats: dict[int, PopulationStats],
        best_solution: Solution | None = None,
    ) -> None:
        """Mark end of a generation and collect metrics.

        Args:
            generation: Generation number
            population_stats: Statistics for each island
            best_solution: Best solution found this generation
        """
        # Calculate generation time
        if generation in self.generation_start_times:
            generation_time = time.time() - self.generation_start_times[generation]
            self._update_avg_generation_time(generation_time)

        # Update generation count
        self.metrics.generations_completed = generation + 1

        # Store population statistics
        self.metrics.population_stats_history.append(population_stats.copy())

        # Update best score
        if best_solution:
            self.metrics.best_score_achieved = max(
                self.metrics.best_score_achieved, best_solution.score
            )

        # Calculate population diversity
        all_solutions = []
        for stats in population_stats.values():
            # This is a placeholder - in real implementation,
            # we'd need access to actual solutions for diversity calculation
            pass

    def record_solution_generated(self, solution: Solution) -> None:
        """Record generation of a new solution.

        Args:
            solution: Generated solution
        """
        self.metrics.total_solutions_generated += 1

        if solution.is_valid():
            self.metrics.valid_solutions_found += 1

        self.metrics.score_history.append(solution.score)

        # Count constraint violations
        violation_count = 0
        for feedback in solution.feedback:
            if "violat" in feedback.lower() or "fail" in feedback.lower():
                violation_count += 1
        self.metrics.constraint_violation_history.append(violation_count)

    def record_llm_call(self, response_time: float) -> None:
        """Record an LLM API call.

        Args:
            response_time: Time taken for the call in seconds
        """
        self.metrics.total_llm_calls += 1
        self.llm_call_times.append(response_time)

        # Update average response time
        self.metrics.avg_llm_response_time = np.mean(self.llm_call_times)

    def record_evaluation_call(self, evaluation_time: float) -> None:
        """Record a solution evaluation call.

        Args:
            evaluation_time: Time taken for evaluation in seconds
        """
        self.metrics.total_evaluation_calls += 1
        self.evaluation_times.append(evaluation_time)

    def record_migration(self, migration_stats: dict[str, Any]) -> None:
        """Record migration statistics.

        Args:
            migration_stats: Migration statistics
        """
        migration_record = {"timestamp": time.time(), **migration_stats}
        self.metrics.migration_statistics.append(migration_record)

    def record_island_reset(self, reset_islands: list[int], elite_count: int) -> None:
        """Record island reset operation.

        Args:
            reset_islands: Islands that were reset
            elite_count: Number of elite solutions used
        """
        reset_record = {
            "timestamp": time.time(),
            "reset_islands": reset_islands,
            "elite_count": elite_count,
        }
        self.metrics.reset_statistics.append(reset_record)

    def update_island_performance(
        self, island_id: int, performance_metrics: dict[str, float]
    ) -> None:
        """Update performance metrics for an island.

        Args:
            island_id: Island identifier
            performance_metrics: Performance metrics dictionary
        """
        if island_id not in self.metrics.island_performance:
            self.metrics.island_performance[island_id] = {}

        self.metrics.island_performance[island_id].update(performance_metrics)

    def finalize_experiment(self) -> ExperimentMetrics:
        """Finalize experiment and return complete metrics.

        Returns:
            Complete experiment metrics
        """
        self.metrics.end_time = time.time()
        return self.metrics

    def _update_avg_generation_time(self, generation_time: float) -> None:
        """Update average generation time.

        Args:
            generation_time: Time for this generation
        """
        current_avg = self.metrics.avg_generation_time
        completed = self.metrics.generations_completed

        # Running average calculation
        self.metrics.avg_generation_time = (
            current_avg * completed + generation_time
        ) / (completed + 1)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary.

        Returns:
            Dictionary with key performance metrics
        """
        return {
            "runtime_minutes": self.metrics.runtime_minutes,
            "generations_completed": self.metrics.generations_completed,
            "solutions_per_minute": self.metrics.solutions_per_minute,
            "success_rate": self.metrics.success_rate,
            "best_score": self.metrics.best_score_achieved,
            "avg_generation_time": self.metrics.avg_generation_time,
            "total_llm_calls": self.metrics.total_llm_calls,
            "avg_llm_response_time": self.metrics.avg_llm_response_time,
        }

    def get_convergence_analysis(self) -> dict[str, Any]:
        """Analyze convergence patterns.

        Returns:
            Convergence analysis results
        """
        if not self.metrics.score_history:
            return {}

        scores = np.array(self.metrics.score_history)

        # Calculate moving averages
        window_size = min(50, len(scores) // 4)
        if window_size < 2:
            moving_avg = scores
        else:
            moving_avg = np.convolve(
                scores, np.ones(window_size) / window_size, mode="valid"
            )

        # Find best improvement periods
        if len(moving_avg) > 1:
            improvements = np.diff(moving_avg)
            best_improvement_idx = np.argmax(improvements)
            best_improvement = improvements[best_improvement_idx]
        else:
            best_improvement_idx = 0
            best_improvement = 0.0

        return {
            "initial_score": float(scores[0]) if len(scores) > 0 else 0.0,
            "final_score": float(scores[-1]) if len(scores) > 0 else 0.0,
            "max_score": float(np.max(scores)),
            "score_variance": float(np.var(scores)),
            "best_improvement": float(best_improvement),
            "best_improvement_at": int(best_improvement_idx),
            "convergence_trend": self._calculate_convergence_trend(moving_avg),
        }

    def _calculate_convergence_trend(self, moving_avg: np.ndarray) -> str:
        """Calculate overall convergence trend.

        Args:
            moving_avg: Moving average of scores

        Returns:
            Trend description
        """
        if len(moving_avg) < 3:
            return "insufficient_data"

        # Calculate slope of linear regression
        x = np.arange(len(moving_avg))
        slope = np.polyfit(x, moving_avg, 1)[0]

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"

    def export_metrics(self, output_path: str, format: str = "json") -> None:
        """Export metrics to file.

        Args:
            output_path: Output file path
            format: Export format (json, csv)
        """
        import json
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            # Convert metrics to JSON-serializable format
            metrics_dict = {
                "experiment_name": self.metrics.experiment_name,
                "runtime_minutes": self.metrics.runtime_minutes,
                "generations_completed": self.metrics.generations_completed,
                "total_solutions_generated": self.metrics.total_solutions_generated,
                "valid_solutions_found": self.metrics.valid_solutions_found,
                "best_score_achieved": self.metrics.best_score_achieved,
                "success_rate": self.metrics.success_rate,
                "performance_summary": self.get_performance_summary(),
                "convergence_analysis": self.get_convergence_analysis(),
                "score_history": self.metrics.score_history,
                "constraint_violation_history": self.metrics.constraint_violation_history,
            }

            with open(output_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)

        elif format.lower() == "csv":
            import pandas as pd

            # Create DataFrame with key metrics over time
            df_data = {
                "generation": list(range(len(self.metrics.score_history))),
                "score": self.metrics.score_history,
                "constraint_violations": self.metrics.constraint_violation_history,
            }

            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}")
