"""Migration operators for inter-island solution transfer."""

import uuid

from loguru import logger

from ..core.models import Solution
from ..core.population import Population


class MigrationOperator:
    """Handles migration of solutions between islands."""

    def __init__(self, migration_rate: int = 5):
        """Initialize migration operator.

        Args:
            migration_rate: Number of solutions to migrate per operation
        """
        self.migration_rate = migration_rate

    def migrate_solutions(
        self,
        source_population: Population,
        target_population: Population,
        num_migrants: int = None,
    ) -> list[Solution]:
        """Migrate solutions from source to target population.

        Args:
            source_population: Population to migrate from
            target_population: Population to migrate to
            num_migrants: Number of solutions to migrate (uses migration_rate if None)

        Returns:
            List of migrated solutions
        """
        if num_migrants is None:
            num_migrants = self.migration_rate

        # Get top solutions from source population
        emigrants = source_population.get_top_solutions(num_migrants)

        if not emigrants:
            logger.debug(
                f"No solutions available for migration from island "
                f"{source_population.island_id}"
            )
            return []

        # Clone solutions for target population
        migrants = []
        for solution in emigrants:
            migrant = self._clone_solution_for_migration(
                solution=solution,
                target_island_id=target_population.island_id,
                source_island_id=source_population.island_id,
            )

            target_population.add_solution(migrant)
            migrants.append(migrant)

        logger.debug(
            f"Migrated {len(migrants)} solutions from island "
            f"{source_population.island_id} to island "
            f"{target_population.island_id}"
        )

        return migrants

    def _clone_solution_for_migration(
        self, solution: Solution, target_island_id: int, source_island_id: int
    ) -> Solution:
        """Clone solution for migration to new island.

        Args:
            solution: Original solution
            target_island_id: ID of target island
            source_island_id: ID of source island

        Returns:
            Cloned solution with updated metadata
        """
        from datetime import datetime

        # Create new solution with updated metadata
        migrant = Solution(
            id=str(uuid.uuid4()),  # New ID for the migrant
            content=solution.content,
            score=solution.score,
            feedback=solution.feedback.copy(),
            generation=solution.generation,
            island_id=target_island_id,  # Update island ID
            conversation_id=solution.conversation_id,
            parent_ids=solution.parent_ids.copy(),
            metadata={
                **solution.metadata,
                "migrated": True,
                "source_island": source_island_id,
                "target_island": target_island_id,
                "original_id": solution.id,
                "migration_timestamp": datetime.now().isoformat(),
            },
            timestamp=datetime.now(),
        )

        return migrant

    def ring_migration(
        self, populations: list[Population]
    ) -> dict[int, list[Solution]]:
        """Perform ring migration between populations.

        Args:
            populations: List of populations to migrate between

        Returns:
            Dictionary mapping island_id to list of received migrants
        """
        migration_results = {}

        for i, source_pop in enumerate(populations):
            target_idx = (i + 1) % len(populations)
            target_pop = populations[target_idx]

            migrants = self.migrate_solutions(source_pop, target_pop)
            migration_results[target_pop.island_id] = migrants

        logger.info(f"Completed ring migration across {len(populations)} islands")
        return migration_results

    def tournament_migration(
        self, populations: list[Population], tournament_size: int = 3
    ) -> dict[int, list[Solution]]:
        """Perform tournament-based migration.

        Args:
            populations: List of populations
            tournament_size: Number of populations to compete for migration

        Returns:
            Dictionary mapping island_id to list of received migrants
        """
        import random

        migration_results = {}

        for target_pop in populations:
            # Select random populations for tournament
            tournament_pops = random.sample(
                [p for p in populations if p.island_id != target_pop.island_id],
                min(tournament_size, len(populations) - 1),
            )

            # Find best population in tournament (by best solution score)
            best_pop = max(
                tournament_pops,
                key=lambda p: p.best_solution.score if p.best_solution else 0,
            )

            # Migrate from best population
            migrants = self.migrate_solutions(best_pop, target_pop)
            migration_results[target_pop.island_id] = migrants

        logger.info(
            f"Completed tournament migration with tournament size {tournament_size}"
        )
        return migration_results

    def adaptive_migration(
        self, populations: list[Population], diversity_threshold: float = 0.5
    ) -> dict[int, list[Solution]]:
        """Perform adaptive migration based on population diversity.

        Args:
            populations: List of populations
            diversity_threshold: Threshold for triggering migration

        Returns:
            Dictionary mapping island_id to list of received migrants
        """
        migration_results = {}

        # Calculate diversity for each population
        population_diversities = []
        for pop in populations:
            diversity = pop.get_diversity_score()
            population_diversities.append((pop, diversity))

        # Sort by diversity (lowest first)
        population_diversities.sort(key=lambda x: x[1])

        # Migrate to populations with low diversity
        for pop, diversity in population_diversities:
            if diversity < diversity_threshold:
                # Find most diverse population as source
                source_pop = max(population_diversities, key=lambda x: x[1])[0]

                if source_pop.island_id != pop.island_id:
                    migrants = self.migrate_solutions(source_pop, pop)
                    migration_results[pop.island_id] = migrants

        logger.info(
            f"Completed adaptive migration for {len(migration_results)} islands"
        )
        return migration_results

    def elitist_migration(
        self, populations: list[Population], elite_ratio: float = 0.1
    ) -> dict[int, list[Solution]]:
        """Perform elitist migration sharing best solutions globally.

        Args:
            populations: List of populations
            elite_ratio: Ratio of population to consider as elite

        Returns:
            Dictionary mapping island_id to list of received migrants
        """
        # Collect all solutions from all populations
        all_solutions = []
        for pop in populations:
            all_solutions.extend(pop.get_selection_pool())

        if not all_solutions:
            return {}

        # Sort by score and get global elites
        all_solutions.sort(key=lambda s: s.score, reverse=True)
        num_elites = max(1, int(len(all_solutions) * elite_ratio))
        global_elites = all_solutions[:num_elites]

        migration_results = {}

        # Distribute elites to all populations
        for pop in populations:
            migrants = []

            for elite in global_elites:
                # Don't migrate solution to its own island
                if elite.island_id != pop.island_id:
                    migrant = self._clone_solution_for_migration(
                        solution=elite,
                        target_island_id=pop.island_id,
                        source_island_id=elite.island_id,
                    )

                    # Add elite marker
                    migrant.metadata["elite_migration"] = True
                    migrant.metadata["global_rank"] = global_elites.index(elite) + 1

                    pop.add_solution(migrant)
                    migrants.append(migrant)

            migration_results[pop.island_id] = migrants

        logger.info(f"Completed elitist migration distributing {num_elites} elites")
        return migration_results

    def get_migration_statistics(
        self, migration_results: dict[int, list[Solution]]
    ) -> dict[str, float]:
        """Calculate statistics for migration operation.

        Args:
            migration_results: Results from migration operation

        Returns:
            Dictionary with migration statistics
        """
        if not migration_results:
            return {}

        total_migrants = sum(len(migrants) for migrants in migration_results.values())
        receiving_islands = len(migration_results)

        # Calculate average scores of migrants
        all_migrant_scores = []
        for migrants in migration_results.values():
            all_migrant_scores.extend([m.score for m in migrants])

        avg_migrant_score = (
            sum(all_migrant_scores) / len(all_migrant_scores)
            if all_migrant_scores
            else 0
        )

        return {
            "total_migrants": total_migrants,
            "receiving_islands": receiving_islands,
            "avg_migrants_per_island": total_migrants / max(receiving_islands, 1),
            "avg_migrant_score": avg_migrant_score,
        }
