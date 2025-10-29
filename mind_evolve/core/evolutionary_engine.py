"""Main evolutionary engine for Mind Evolution."""

import time
from typing import Optional

from loguru import logger

from .island_model import IslandModel
from .models import MindEvolutionConfig, Problem, Solution
from ..evaluation.evaluator_base import BaseEvaluator
from ..llm.llm_interface import BaseLLM
from ..llm.prompt_manager import PromptManager
from ..utils.metrics import MetricsCollector


class MindEvolution:
    """Main evolutionary search engine for LLM-based problem solving."""
    
    def __init__(self,
                 config: MindEvolutionConfig,
                 llm: BaseLLM,
                 evaluator: BaseEvaluator,
                 prompt_manager: PromptManager):
        """Initialize Mind Evolution engine.
        
        Args:
            config: Evolution configuration
            llm: LLM interface
            evaluator: Solution evaluator
            prompt_manager: Prompt management system
        """
        self.config = config
        self.llm = llm
        self.evaluator = evaluator
        self.prompt_manager = prompt_manager
        
        # Initialize island model
        self.island_model = IslandModel(config, llm, evaluator, prompt_manager)
        
        # Metrics collection
        self.metrics_collector: Optional[MetricsCollector] = None
        
    def solve(self, problem: Problem, 
             experiment_name: Optional[str] = None) -> Solution:
        """Solve a problem using Mind Evolution.
        
        Args:
            problem: Problem definition
            experiment_name: Optional experiment name for tracking
            
        Returns:
            Best solution found
        """
        # Initialize metrics collection
        if experiment_name:
            self.metrics_collector = MetricsCollector(experiment_name)
            
        logger.info(f"Starting Mind Evolution for problem: {problem.title}")
        logger.info(f"Configuration: {self.config.N_gens} generations, "
                   f"{self.config.N_island} islands, {self.config.N_convs} conversations")
        
        try:
            # Initialize populations
            logger.info("Initializing populations...")
            self.island_model.initialize_populations(problem)
            
            # Main evolution loop
            for generation in range(self.config.N_gens):
                self._run_generation(generation, problem)
                
                # Check early stopping
                if self.config.early_stopping and self.island_model.has_valid_solution():
                    logger.info(f"Valid solution found at generation {generation}. "
                               f"Early stopping enabled.")
                    break
                    
        except KeyboardInterrupt:
            logger.warning("Evolution interrupted by user")
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            raise
            
        # Get final result
        best_solution = self.island_model.global_best
        if best_solution is None:
            logger.warning("No solutions generated during evolution")
            # Create dummy solution
            best_solution = Solution(
                content="No solution found",
                score=0.0,
                feedback=["Evolution failed to generate solutions"],
                generation=0,
                island_id=0,
                conversation_id="failed"
            )
            
        logger.info(f"Evolution completed. Best solution score: {best_solution.score:.3f}")
        
        # Finalize metrics
        if self.metrics_collector:
            final_metrics = self.metrics_collector.finalize_experiment()
            logger.info(f"Experiment metrics: {self.metrics_collector.get_performance_summary()}")
            
        return best_solution
        
    def _run_generation(self, generation: int, problem: Problem) -> None:
        """Run a single generation of evolution.
        
        Args:
            generation: Generation number
            problem: Problem being solved
        """
        start_time = time.time()
        
        if self.metrics_collector:
            self.metrics_collector.start_generation(generation)
            
        logger.info(f"🧬 Starting Generation {generation}")
        
        # Evolve all islands
        island_stats = self.island_model.evolve_generation(problem)
        
        # Perform migration after each island evolution
        migration_stats = self.island_model.perform_migration()
        if self.metrics_collector and migration_stats:
            for island_id, migrants in migration_stats.items():
                for migrant in migrants:
                    self.metrics_collector.record_solution_generated(migrant)
                    
        # Perform island reset if needed
        reset_islands = self.island_model.perform_island_reset()
        if reset_islands and self.metrics_collector:
            self.metrics_collector.record_island_reset(
                reset_islands, self.config.N_top
            )
            
        # Update generation counter
        self.island_model.increment_generation()
        
        # Log generation summary
        generation_time = time.time() - start_time
        self._log_generation_summary(generation, island_stats, generation_time)
        
        # Update metrics
        if self.metrics_collector:
            population_stats = {}
            for island_id, stats in island_stats.items():
                # Convert to PopulationStats format
                from .models import PopulationStats
                after_stats = stats['after']
                pop_stats = PopulationStats(**after_stats)
                population_stats[island_id] = pop_stats
                
            self.metrics_collector.end_generation(
                generation, population_stats, self.island_model.global_best
            )
            
    def _log_generation_summary(self, generation: int, 
                              island_stats: dict, 
                              generation_time: float) -> None:
        """Log summary of generation results.
        
        Args:
            generation: Generation number
            island_stats: Statistics from island evolution
            generation_time: Time taken for generation
        """
        # Calculate aggregate statistics
        total_improvement = sum(stats['improvement'] for stats in island_stats.values())
        best_island_score = max(stats['after']['max_score'] for stats in island_stats.values())
        avg_island_score = sum(stats['after']['mean_score'] for stats in island_stats.values()) / len(island_stats)
        
        logger.info(f"✅ Generation {generation} completed in {generation_time:.2f}s")
        logger.info(f"   Best score: {best_island_score:.3f}")
        logger.info(f"   Avg score: {avg_island_score:.3f}")
        logger.info(f"   Total improvement: {total_improvement:+.3f}")
        
        # Log individual island performance
        for island_id, stats in island_stats.items():
            before_score = stats['before']['max_score']
            after_score = stats['after']['max_score']
            improvement = stats['improvement']
            
            logger.debug(f"   Island {island_id}: {before_score:.3f} → "
                        f"{after_score:.3f} ({improvement:+.3f})")
                        
    def get_evolution_statistics(self) -> dict:
        """Get comprehensive evolution statistics.
        
        Returns:
            Dictionary with evolution statistics
        """
        stats = {
            'generations_completed': self.island_model.generation,
            'total_islands': len(self.island_model.islands),
            'global_best_score': self.island_model.global_best.score if self.island_model.global_best else 0.0,
            'population_statistics': self.island_model.get_population_statistics(),
            'best_solutions': [s.model_dump() for s in self.island_model.get_best_solutions(10)],
        }
        
        if self.metrics_collector:
            stats['performance_metrics'] = self.metrics_collector.get_performance_summary()
            stats['convergence_analysis'] = self.metrics_collector.get_convergence_analysis()
            
        return stats
        
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save evolution checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        import json
        from pathlib import Path
        
        checkpoint_data = {
            'generation': self.island_model.generation,
            'config': self.config.model_dump(),
            'global_best': self.island_model.global_best.model_dump() if self.island_model.global_best else None,
            'population_statistics': self.island_model.get_population_statistics(),
            'timestamp': time.time(),
        }
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
            
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load evolution checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        import json
        from pathlib import Path
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            
        # Restore generation counter
        self.island_model.generation = checkpoint_data['generation']
        
        # Restore global best if available
        if checkpoint_data.get('global_best'):
            self.island_model.global_best = Solution(**checkpoint_data['global_best'])
            
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from generation {self.island_model.generation}")
        
    def export_results(self, output_path: str, format: str = "json") -> None:
        """Export evolution results.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv)
        """
        results = self.get_evolution_statistics()
        
        if format.lower() == "json":
            import json
            from pathlib import Path
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
        elif format.lower() == "csv":
            # Export key metrics to CSV
            if self.metrics_collector:
                self.metrics_collector.export_metrics(output_path, "csv")
            else:
                logger.warning("No metrics collector available for CSV export")
                
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Results exported to {output_path}")
        
    def run_analysis(self) -> dict:
        """Run post-evolution analysis.
        
        Returns:
            Analysis results dictionary
        """
        analysis = {}
        
        # Convergence analysis
        if self.metrics_collector:
            analysis['convergence'] = self.metrics_collector.get_convergence_analysis()
            
        # Population diversity analysis
        population_stats = self.island_model.get_population_statistics()
        analysis['population_diversity'] = {}
        
        for island_id, stats in population_stats.items():
            analysis['population_diversity'][island_id] = {
                'score_variance': stats.get('std_score', 0.0),
                'score_range': stats.get('max_score', 0.0) - stats.get('min_score', 0.0),
            }
            
        # Solution quality distribution
        best_solutions = self.island_model.get_best_solutions(50)
        if best_solutions:
            scores = [s.score for s in best_solutions]
            analysis['solution_quality'] = {
                'top_10_avg': sum(scores[:10]) / min(10, len(scores)),
                'top_50_avg': sum(scores) / len(scores),
                'score_spread': max(scores) - min(scores) if scores else 0,
            }
            
        return analysis