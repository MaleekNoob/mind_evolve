"""Crossover operators for combining parent solutions."""

import uuid
from datetime import datetime
from typing import List

from loguru import logger

from ..core.models import ConversationThread, Problem, Solution
from ..evaluation.evaluator_base import BaseEvaluator
from ..llm.llm_interface import BaseLLM
from ..llm.prompt_manager import PromptManager


class CrossoverOperator:
    """Handles crossover operations between parent solutions."""
    
    def __init__(self,
                 llm: BaseLLM,
                 evaluator: BaseEvaluator,
                 prompt_manager: PromptManager):
        """Initialize crossover operator.
        
        Args:
            llm: LLM interface for generation
            evaluator: Solution evaluator
            prompt_manager: Prompt management system
        """
        self.llm = llm
        self.evaluator = evaluator
        self.prompt_manager = prompt_manager
        
    def crossover(self,
                  problem: Problem,
                  parents: List[Solution],
                  conversation: ConversationThread,
                  turn: int = 1,
                  temperature: float = 1.0,
                  enable_critic: bool = True) -> Solution:
        """Perform crossover operation on parent solutions.
        
        Args:
            problem: Problem to solve
            parents: Parent solutions to combine
            conversation: Conversation context
            turn: Turn number
            temperature: LLM temperature
            enable_critic: Whether to use critic analysis
            
        Returns:
            New child solution from crossover
        """
        if not parents:
            raise ValueError("Cannot perform crossover with no parents")
            
        logger.debug(f"Performing crossover with {len(parents)} parents")
        
        # Generate critic analysis if enabled
        critic_response = ""
        if enable_critic:
            critic_response = self._generate_multi_parent_critique(
                problem=problem,
                parents=parents,
                temperature=temperature * 0.8
            )
            
        # Generate child solution through synthesis
        child_solution = self._synthesize_child_solution(
            problem=problem,
            parents=parents,
            critic_analysis=critic_response,
            conversation=conversation,
            turn=turn,
            temperature=temperature
        )
        
        logger.debug(f"Crossover produced child {child_solution.id[:8]} "
                    f"with score {child_solution.score:.3f}")
        
        return child_solution
        
    def _generate_multi_parent_critique(self,
                                      problem: Problem,
                                      parents: List[Solution],
                                      temperature: float) -> str:
        """Generate critique analyzing multiple parent solutions.
        
        Args:
            problem: Problem context
            parents: Parent solutions to analyze
            temperature: LLM temperature
            
        Returns:
            Critic's analysis of parent solutions
        """
        critic_prompt = self.prompt_manager.create_multi_parent_critic_prompt(
            problem=problem,
            parents=parents
        )
        
        return self.llm.generate(
            prompt=critic_prompt,
            temperature=temperature
        )
        
    def _synthesize_child_solution(self,
                                 problem: Problem,
                                 parents: List[Solution],
                                 critic_analysis: str,
                                 conversation: ConversationThread,
                                 turn: int,
                                 temperature: float) -> Solution:
        """Synthesize child solution from parent insights.
        
        Args:
            problem: Problem to solve
            parents: Parent solutions
            critic_analysis: Critic's synthesis analysis
            conversation: Conversation context
            turn: Turn number
            temperature: LLM temperature
            
        Returns:
            Synthesized child solution
        """
        # Generate synthesis prompt
        author_prompt = self.prompt_manager.create_multi_parent_author_prompt(
            problem=problem,
            parents=parents,
            critic_analysis=critic_analysis
        )
        
        # Generate child solution
        author_response = self.llm.generate(
            prompt=author_prompt,
            temperature=temperature
        )
        
        # Parse solution content
        solution_content = self._parse_solution_content(author_response)
        
        # Evaluate child solution
        evaluation = self.evaluator.evaluate(solution_content, problem)
        
        # Create child solution object
        child_solution = Solution(
            id=str(uuid.uuid4()),
            content=solution_content,
            score=evaluation.score,
            feedback=evaluation.feedback,
            generation=conversation.generation,
            island_id=conversation.island_id,
            conversation_id=conversation.id,
            parent_ids=[parent.id for parent in parents],
            metadata={
                'turn': turn,
                'crossover': True,
                'num_parents': len(parents),
                'parent_scores': [p.score for p in parents],
                'critic_analysis_length': len(critic_analysis),
                'temperature': temperature
            },
            timestamp=datetime.now()
        )
        
        return child_solution
        
    def _parse_solution_content(self, llm_response: str) -> str:
        """Parse solution content from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Cleaned solution content
        """
        content = llm_response.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's my synthesized solution:",
            "Here is my synthesized solution:",
            "My synthesized solution:",
            "Synthesized solution:",
            "Combined solution:",
            "New solution:",
            "Solution:",
        ]
        
        for prefix in prefixes_to_remove:
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()
                break
                
        return content
        
    def uniform_crossover(self,
                         problem: Problem,
                         parent1: Solution,
                         parent2: Solution,
                         crossover_rate: float = 0.5) -> str:
        """Perform uniform crossover between two parent solutions.
        
        Args:
            problem: Problem context
            parent1: First parent solution
            parent2: Second parent solution
            crossover_rate: Probability of selecting from parent1
            
        Returns:
            Combined solution content
        """
        import random
        
        # Split solutions into sentences
        sentences1 = self._split_into_sentences(parent1.content)
        sentences2 = self._split_into_sentences(parent2.content)
        
        # Perform uniform crossover
        max_length = max(len(sentences1), len(sentences2))
        combined_sentences = []
        
        for i in range(max_length):
            # Choose which parent to take from
            if random.random() < crossover_rate and i < len(sentences1):
                combined_sentences.append(sentences1[i])
            elif i < len(sentences2):
                combined_sentences.append(sentences2[i])
            elif i < len(sentences1):
                combined_sentences.append(sentences1[i])
                
        return " ".join(combined_sentences)
        
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences.
        
        Args:
            content: Text content to split
            
        Returns:
            List of sentences
        """
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
        
    def multi_point_crossover(self,
                            problem: Problem,
                            parents: List[Solution],
                            num_points: int = 2) -> str:
        """Perform multi-point crossover among multiple parents.
        
        Args:
            problem: Problem context
            parents: Parent solutions
            num_points: Number of crossover points
            
        Returns:
            Combined solution content
        """
        import random
        
        if not parents:
            return ""
        if len(parents) == 1:
            return parents[0].content
            
        # Get all content as word lists
        parent_words = []
        max_length = 0
        
        for parent in parents:
            words = parent.content.split()
            parent_words.append(words)
            max_length = max(max_length, len(words))
            
        # Generate crossover points
        crossover_points = sorted(random.sample(range(1, max_length), 
                                               min(num_points, max_length - 1)))
        crossover_points = [0] + crossover_points + [max_length]
        
        # Combine segments from different parents
        combined_words = []
        current_parent = 0
        
        for i in range(len(crossover_points) - 1):
            start = crossover_points[i]
            end = crossover_points[i + 1]
            
            # Use words from current parent if available
            if current_parent < len(parent_words) and start < len(parent_words[current_parent]):
                segment_end = min(end, len(parent_words[current_parent]))
                combined_words.extend(parent_words[current_parent][start:segment_end])
                
            # Switch to next parent
            current_parent = (current_parent + 1) % len(parent_words)
            
        return " ".join(combined_words)
        
    def semantic_crossover(self,
                          problem: Problem,
                          parents: List[Solution],
                          temperature: float = 1.0) -> Solution:
        """Perform semantic crossover using LLM understanding.
        
        Args:
            problem: Problem to solve
            parents: Parent solutions
            temperature: LLM temperature
            
        Returns:
            Semantically combined solution
        """
        # Create a dummy conversation for this operation
        conversation = ConversationThread(
            id=f"semantic_crossover_{uuid.uuid4().hex[:8]}",
            island_id=parents[0].island_id if parents else 0,
            generation=parents[0].generation if parents else 0,
            parent_solutions=parents,
            children=[],
            turns=[]
        )
        
        return self.crossover(
            problem=problem,
            parents=parents,
            conversation=conversation,
            turn=1,
            temperature=temperature,
            enable_critic=True
        )