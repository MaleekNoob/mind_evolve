"""Island model implementation for Mind Evolution."""

import random

from loguru import logger

from ..evaluation.evaluator_base import BaseEvaluator
from ..llm.llm_interface import BaseLLM
from ..llm.prompt_manager import PromptManager
from ..operators.crossover import CrossoverOperator
from ..operators.initialization import PopulationInitializer
from ..operators.migration import MigrationOperator
from ..operators.mutation import MutationOperator
from .models import MindEvolutionConfig, Solution
from .population import Population
from .selection import BoltzmannTournamentSelection


class IslandModel:
    """Manages multiple islands with independent evolution and migration."""

    def __init__(self,
                 config: MindEvolutionConfig,
                 llm: BaseLLM,
                 evaluator: BaseEvaluator,
                 prompt_manager: PromptManager):
        """Initialize island model.
        
        Args:
            config: Mind Evolution configuration
            llm: LLM interface
            evaluator: Solution evaluator
            prompt_manager: Prompt management system
        """
        self.config = config
        self.llm = llm
        self.evaluator = evaluator
        self.prompt_manager = prompt_manager

        # Initialize populations for each island
        self.islands = [Population(island_id=i) for i in range(config.N_island)]

        # Initialize operators
        self.initializer = PopulationInitializer(llm, evaluator, prompt_manager)
        self.crossover_op = CrossoverOperator(llm, evaluator, prompt_manager)
        self.mutation_op = MutationOperator(llm, evaluator, prompt_manager)
        self.migration_op = MigrationOperator(config.N_emigrate)

        # Selection strategy
        self.selector = BoltzmannTournamentSelection()

        # Global tracking
        self.global_best: Solution | None = None
        self.generation = 0

    def initialize_populations(self, problem) -> None:
        """Initialize all island populations.
        
        Args:
            problem: Problem to solve
        """
        logger.info(f"Initializing {len(self.islands)} island populations")

        for island in self.islands:
            solutions = self.initializer.initialize_population(
                problem=problem,
                island_id=island.island_id,
                num_conversations=self.config.N_convs,
                num_refinement_turns=self.config.N_seq,
                temperature=self.config.temperature
            )

            # Add solutions to island population
            for solution in solutions:
                island.add_solution(solution)

            # Update global best
            if island.best_solution:
                self._update_global_best(island.best_solution)

        logger.info(f"Initialized populations with {sum(len(island) for island in self.islands)} solutions")

    def evolve_generation(self, problem) -> dict[int, dict[str, float]]:
        """Evolve all islands for one generation.
        
        Args:
            problem: Problem to solve
            
        Returns:
            Dictionary of island statistics
        """
        island_stats = {}

        for island in self.islands:
            stats_before = island.calculate_statistics()

            # Evolve this island
            self._evolve_island(island, problem)

            # Update generation number
            island.generation = self.generation

            stats_after = island.calculate_statistics()
            island_stats[island.island_id] = {
                'before': stats_before.model_dump(),
                'after': stats_after.model_dump(),
                'improvement': stats_after.max_score - stats_before.max_score
            }

            # Update global best
            if island.best_solution:
                self._update_global_best(island.best_solution)

        return island_stats

    def _evolve_island(self, island: Population, problem) -> None:
        """Evolve a single island for one generation.
        
        Args:
            island: Population to evolve
            problem: Problem to solve
        """
        selection_pool = island.get_selection_pool()

        if not selection_pool:
            logger.warning(f"Island {island.island_id} has no solutions for evolution")
            return

        # Generate new solutions through conversations
        for conv_idx in range(self.config.N_convs):
            conversation_id = f"G{self.generation}_I{island.island_id}_C{conv_idx}"

            # Select parents
            num_parents = random.randint(0, self.config.N_parent)
            parents = self.selector.select_parents(
                selection_pool=selection_pool,
                num_parents=num_parents,
                config={
                    'Pr_no_parents': self.config.Pr_no_parents,
                    'temperature': self.config.temperature
                }
            )

            # Create conversation thread
            from .models import ConversationThread
            conversation = ConversationThread(
                id=conversation_id,
                island_id=island.island_id,
                generation=self.generation,
                parent_solutions=parents,
                children=[],
                turns=[]
            )

            # Generate initial child solution
            if len(parents) == 0:
                # Pure mutation: generate from scratch
                initial_child = self._generate_initial_solution(problem, conversation)
            else:
                # Crossover: combine parent insights
                initial_child = self.crossover_op.crossover(
                    problem=problem,
                    parents=parents,
                    conversation=conversation,
                    turn=1,
                    temperature=self.config.temperature,
                    enable_critic=self.config.enable_critic
                )

            island.add_solution(initial_child)

            # Sequential refinement through mutation
            current_solution = initial_child
            for turn in range(2, self.config.N_seq + 1):
                refined_solution = self.mutation_op.mutate(
                    problem=problem,
                    solution=current_solution,
                    conversation=conversation,
                    turn=turn,
                    temperature=self.config.temperature,
                    enable_critic=self.config.enable_critic
                )

                island.add_solution(refined_solution)
                current_solution = refined_solution

    def _generate_initial_solution(self, problem, conversation):
        """Generate initial solution when no parents are selected.
        
        Args:
            problem: Problem to solve
            conversation: Conversation context
            
        Returns:    
            Generated initial solution
        """
        return self.initializer._generate_initial_solution(
            problem=problem,
            conversation=conversation,
            temperature=self.config.temperature
        )

    def perform_migration(self) -> dict[str, float]:
        """Perform migration between islands.
        
        Returns:
            Migration statistics
        """
        return self.migration_op.ring_migration(self.islands)

    def perform_island_reset(self) -> list[int]:
        """Perform island reset operation.
        
        Returns:
            List of islands that were reset
        """
        if self.generation % self.config.N_reset_interval != 0:
            return []

        # Calculate island fitness
        island_fitness = []
        for island in self.islands:
            stats = island.calculate_statistics()
            island_fitness.append((island.island_id, stats.mean_score))

        # Sort by fitness (ascending - worst first)
        island_fitness.sort(key=lambda x: x[1])

        # Select worst islands for reset
        reset_islands = [island_fitness[i][0] for i in range(self.config.N_reset)]

        # Gather global elite solutions
        all_solutions = []
        for island in self.islands:
            all_solutions.extend(island.get_selection_pool())

        # Sort and get top candidates
        all_solutions.sort(key=lambda s: s.score, reverse=True)
        elite_candidates = all_solutions[:self.config.N_candidate]

        # Select diverse elites
        if self.config.use_llm_for_reset and len(elite_candidates) > self.config.N_top:
            selected_elites = self._select_diverse_elites_with_llm(
                elite_candidates, self.config.N_top
            )
        else:
            selected_elites = elite_candidates[:self.config.N_top]

        # Reset selected islands
        for island_id in reset_islands:
            island = self.islands[island_id]
            island.clear()

            # Add elite solutions
            for elite in selected_elites:
                cloned_elite = self._clone_solution_for_reset(
                    elite, island_id, self.generation
                )
                island.add_solution(cloned_elite)

        logger.info(f"Reset islands {reset_islands} with {len(selected_elites)} elites")
        return reset_islands

    def _select_diverse_elites_with_llm(self, candidates: list[Solution],
                                      num_to_select: int) -> list[Solution]:
        """Use LLM to select diverse elite solutions.
        
        Args:
            candidates: Candidate elite solutions
            num_to_select: Number of solutions to select
            
        Returns:
            Selected diverse elite solutions
        """
        try:
            selection_prompt = self.prompt_manager.create_elite_selection_prompt(
                candidates=candidates,
                num_to_select=num_to_select
            )

            llm_response = self.llm.generate(
                prompt=selection_prompt,
                temperature=0.5
            )

            # Parse selection response
            selected_indices = self._parse_selection_response(llm_response, len(candidates))
            selected = [candidates[i] for i in selected_indices if i < len(candidates)]

            # Fallback if parsing failed
            if len(selected) < num_to_select:
                logger.warning("LLM elite selection failed, using top-N fallback")
                selected = candidates[:num_to_select]

            return selected[:num_to_select]

        except Exception as e:
            logger.error(f"LLM-based elite selection failed: {e}")
            return candidates[:num_to_select]

    def _parse_selection_response(self, response: str, max_index: int) -> list[int]:
        """Parse LLM selection response to extract indices.
        
        Args:
            response: LLM response
            max_index: Maximum valid index
            
        Returns:
            List of selected indices
        """
        import re

        # Look for list-like patterns
        patterns = [
            r'\[([0-9,\s]+)\]',  # [1, 2, 3]
            r'([0-9]+(?:,\s*[0-9]+)*)',  # 1, 2, 3
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    indices_str = match.group(1)
                    indices = [int(x.strip()) for x in indices_str.split(',')]
                    # Filter valid indices
                    valid_indices = [i for i in indices if 0 <= i < max_index]
                    if valid_indices:
                        return valid_indices
                except ValueError:
                    continue

        # Fallback: look for individual numbers
        numbers = re.findall(r'\b([0-9]+)\b', response)
        try:
            indices = [int(num) for num in numbers]
            return [i for i in indices if 0 <= i < max_index]
        except ValueError:
            return []

    def _clone_solution_for_reset(self, solution: Solution,
                                target_island_id: int, generation: int) -> Solution:
        """Clone solution for island reset.
        
        Args:
            solution: Solution to clone
            target_island_id: Target island ID
            generation: Current generation
            
        Returns:
            Cloned solution
        """
        import uuid
        from datetime import datetime

        return Solution(
            id=str(uuid.uuid4()),
            content=solution.content,
            score=solution.score,
            feedback=solution.feedback.copy(),
            generation=generation,
            island_id=target_island_id,
            conversation_id=solution.conversation_id,
            parent_ids=solution.parent_ids.copy(),
            metadata={
                **solution.metadata,
                'reset_elite': True,
                'original_island': solution.island_id,
                'original_id': solution.id,
            },
            timestamp=datetime.now()
        )

    def _update_global_best(self, solution: Solution) -> None:
        """Update global best solution.
        
        Args:
            solution: Candidate best solution
        """
        if self.global_best is None or solution.score > self.global_best.score:
            self.global_best = solution
            logger.debug(f"New global best: {solution.score:.3f} "
                        f"from island {solution.island_id}")

    def get_population_statistics(self) -> dict[int, dict[str, float]]:
        """Get statistics for all island populations.
        
        Returns:
            Dictionary mapping island_id to statistics
        """
        stats = {}
        for island in self.islands:
            island_stats = island.calculate_statistics()
            stats[island.island_id] = island_stats.model_dump()
        return stats

    def get_best_solutions(self, n: int = 10) -> list[Solution]:
        """Get top N solutions across all islands.
        
        Args:
            n: Number of solutions to return
            
        Returns:
            List of best solutions
        """
        all_solutions = []
        for island in self.islands:
            all_solutions.extend(island.get_selection_pool())

        all_solutions.sort(key=lambda s: s.score, reverse=True)
        return all_solutions[:n]

    def has_valid_solution(self) -> bool:
        """Check if any island has a valid solution.
        
        Returns:
            True if valid solution exists
        """
        return self.global_best is not None and self.global_best.is_valid()

    def increment_generation(self) -> None:
        """Increment generation counter."""
        self.generation += 1
