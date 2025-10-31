"""Logging utilities for Mind Evolution."""

import sys
from pathlib import Path

from loguru import logger


class MindEvolutionLogger:
    """Custom logger for Mind Evolution with structured logging."""

    def __init__(
        self,
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_dir: Path | None = None,
        experiment_name: str | None = None,
    ):
        """Initialize Mind Evolution logger.

        Args:
            log_level: Logging level
            log_to_file: Whether to log to file
            log_dir: Directory for log files
            experiment_name: Name of current experiment
        """
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_dir = log_dir or Path("logs")
        self.experiment_name = experiment_name or "mind_evolution"

        # Remove default logger
        logger.remove()

        # Configure console logging
        self._configure_console_logging()

        # Configure file logging if enabled
        if self.log_to_file:
            self._configure_file_logging()

    def _configure_console_logging(self) -> None:
        """Configure console logging with colors and formatting."""
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stdout,
            format=console_format,
            level=self.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    def _configure_file_logging(self) -> None:
        """Configure file logging with rotation."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        main_log_file = self.log_dir / f"{self.experiment_name}.log"
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            main_log_file,
            format=file_format,
            level=self.log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

        # Error log file (separate file for errors)
        error_log_file = self.log_dir / f"{self.experiment_name}_errors.log"
        logger.add(
            error_log_file,
            format=file_format,
            level="ERROR",
            rotation="5 MB",
            retention="60 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    def log_experiment_start(self, config: dict) -> None:
        """Log experiment start with configuration.

        Args:
            config: Experiment configuration
        """
        logger.info("=" * 80)
        logger.info(f"MIND EVOLUTION EXPERIMENT STARTED: {self.experiment_name}")
        logger.info("=" * 80)

        # Log key configuration parameters
        for key, value in config.items():
            if not key.startswith("_"):  # Skip private fields
                logger.info(f"Config - {key}: {value}")

        logger.info("-" * 80)

    def log_experiment_end(self, results: dict) -> None:
        """Log experiment end with results summary.

        Args:
            results: Experiment results
        """
        logger.info("-" * 80)
        logger.info(f"MIND EVOLUTION EXPERIMENT COMPLETED: {self.experiment_name}")

        # Log key results
        for key, value in results.items():
            if not key.startswith("_"):
                logger.info(f"Result - {key}: {value}")

        logger.info("=" * 80)

    def log_generation_start(self, generation: int, num_islands: int) -> None:
        """Log start of a generation.

        Args:
            generation: Generation number
            num_islands: Number of islands
        """
        logger.info(f"🧬 Generation {generation} starting with {num_islands} islands")

    def log_generation_end(self, generation: int, stats: dict) -> None:
        """Log end of a generation with statistics.

        Args:
            generation: Generation number
            stats: Generation statistics
        """
        best_score = stats.get("best_score", "N/A")
        mean_score = stats.get("mean_score", "N/A")
        valid_solutions = stats.get("valid_solutions", "N/A")

        logger.info(
            f"✅ Generation {generation} completed - "
            f"Best: {best_score}, Mean: {mean_score}, "
            f"Valid: {valid_solutions}"
        )

    def log_island_evolution(
        self, island_id: int, generation: int, before_stats: dict, after_stats: dict
    ) -> None:
        """Log island evolution results.

        Args:
            island_id: Island identifier
            generation: Generation number
            before_stats: Statistics before evolution
            after_stats: Statistics after evolution
        """
        before_best = before_stats.get("max_score", 0)
        after_best = after_stats.get("max_score", 0)
        improvement = after_best - before_best

        logger.debug(
            f"🏝️ Island {island_id} G{generation}: "
            f"{before_best:.3f} → {after_best:.3f} "
            f"({improvement:+.3f})"
        )

    def log_migration(self, migration_stats: dict) -> None:
        """Log migration statistics.

        Args:
            migration_stats: Migration statistics
        """
        total_migrants = migration_stats.get("total_migrants", 0)
        receiving_islands = migration_stats.get("receiving_islands", 0)

        logger.debug(
            f"🔄 Migration: {total_migrants} solutions to {receiving_islands} islands"
        )

    def log_island_reset(self, reset_islands: list, elite_count: int) -> None:
        """Log island reset operation.

        Args:
            reset_islands: List of islands being reset
            elite_count: Number of elite solutions used
        """
        logger.info(
            f"🔄 Island reset: Islands {reset_islands} reset with {elite_count} elites"
        )

    def log_solution_generated(
        self,
        solution_id: str,
        score: float,
        island_id: int,
        generation: int,
        solution_type: str = "solution",
    ) -> None:
        """Log solution generation.

        Args:
            solution_id: Solution identifier
            score: Solution score
            island_id: Island identifier
            generation: Generation number
            solution_type: Type of solution (initial, refined, crossover, etc.)
        """
        logger.debug(
            f"💡 {solution_type.title()} {solution_id[:8]} "
            f"generated: score={score:.3f} "
            f"(I{island_id}, G{generation})"
        )

    def log_evaluation_error(self, solution_id: str, error: Exception) -> None:
        """Log evaluation error.

        Args:
            solution_id: Solution identifier
            error: Exception that occurred
        """
        logger.error(f"❌ Evaluation failed for solution {solution_id[:8]}: {error}")

    def log_llm_error(
        self, operation: str, error: Exception, retry_count: int = 0
    ) -> None:
        """Log LLM API error.

        Args:
            operation: Operation that failed
            error: Exception that occurred
            retry_count: Current retry count
        """
        if retry_count > 0:
            logger.warning(f"🔄 LLM {operation} failed (retry {retry_count}): {error}")
        else:
            logger.error(f"❌ LLM {operation} failed: {error}")

    def log_performance_metrics(self, metrics: dict) -> None:
        """Log performance metrics.

        Args:
            metrics: Performance metrics dictionary
        """
        logger.info("📊 Performance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.3f}")
            else:
                logger.info(f"  {metric}: {value}")

    def create_progress_logger(self, total_steps: int, description: str = "Progress"):
        """Create a progress logger for long operations.

        Args:
            total_steps: Total number of steps
            description: Description of the operation

        Returns:
            Progress logger function
        """

        def log_progress(current_step: int, extra_info: str = ""):
            percentage = (current_step / total_steps) * 100
            logger.info(
                f"⏳ {description}: {current_step}/{total_steps} "
                f"({percentage:.1f}%) {extra_info}"
            )

        return log_progress


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Path | None = None,
    experiment_name: str | None = None,
) -> MindEvolutionLogger:
    """Setup logging for Mind Evolution.

    Args:
        log_level: Logging level
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        experiment_name: Name of current experiment

    Returns:
        Configured logger instance
    """
    return MindEvolutionLogger(
        log_level=log_level,
        log_to_file=log_to_file,
        log_dir=log_dir,
        experiment_name=experiment_name,
    )


def get_logger() -> "loguru.Logger":
    """Get the configured loguru logger instance.

    Returns:
        Loguru logger instance
    """
    return logger
