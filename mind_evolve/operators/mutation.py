"""Mutation operators for solution refinement."""

import uuid
from datetime import datetime
from typing import List, Optional

from loguru import logger

from ..core.models import ConversationThread, Problem, Solution
from ..evaluation.evaluator_base import BaseEvaluator
from ..llm.llm_interface import BaseLLM
from ..llm.prompt_manager import PromptManager


class MutationOperator:
    """Handles mutation operations on solutions."""
    
    def __init__(self,
                 llm: BaseLLM,
                 evaluator: BaseEvaluator,
                 prompt_manager: PromptManager):
        """Initialize mutation operator.
        
        Args:
            llm: LLM interface for generation
            evaluator: Solution evaluator
            prompt_manager: Prompt management system
        """
        self.llm = llm
        self.evaluator = evaluator
        self.prompt_manager = prompt_manager
        
    def mutate(self,
               problem: Problem,
               solution: Solution,
               conversation: ConversationThread,
               turn: int = 1,
               temperature: float = 1.0,
               mutation_strength: float = 0.5,
               enable_critic: bool = True) -> Solution:
        """Perform mutation operation on a solution.
        
        Args:
            problem: Problem to solve
            solution: Solution to mutate
            conversation: Conversation context
            turn: Turn number
            temperature: LLM temperature
            mutation_strength: Strength of mutation (0.0 to 1.0)
            enable_critic: Whether to use critic analysis
            
        Returns:
            Mutated solution
        """
        logger.debug(f"Mutating solution {solution.id[:8]} "
                    f"with strength {mutation_strength}")
        
        # Adjust temperature based on mutation strength
        mutation_temperature = temperature * (1.0 + mutation_strength)
        
        # Generate critic analysis if enabled
        critic_response = ""
        if enable_critic:
            critic_response = self._generate_mutation_critique(
                problem=problem,
                solution=solution,
                temperature=mutation_temperature * 0.8
            )
            
        # Generate mutated solution
        mutated_solution = self._generate_mutated_solution(
            problem=problem,
            original_solution=solution,
            critic_analysis=critic_response,
            conversation=conversation,
            turn=turn,
            temperature=mutation_temperature,
            mutation_strength=mutation_strength
        )
        
        logger.debug(f"Mutation produced solution {mutated_solution.id[:8]} "
                    f"with score {mutated_solution.score:.3f}")
        
        return mutated_solution
        
    def _generate_mutation_critique(self,
                                  problem: Problem,
                                  solution: Solution,
                                  temperature: float) -> str:
        """Generate critique for mutation guidance.
        
        Args:
            problem: Problem context
            solution: Solution to analyze
            temperature: LLM temperature
            
        Returns:
            Critic's analysis
        """
        critic_prompt = self.prompt_manager.create_critic_prompt(
            problem=problem,
            solution=solution,
            feedback=solution.feedback
        )
        
        return self.llm.generate(
            prompt=critic_prompt,
            temperature=temperature
        )
        
    def _generate_mutated_solution(self,
                                 problem: Problem,
                                 original_solution: Solution,
                                 critic_analysis: str,
                                 conversation: ConversationThread,
                                 turn: int,
                                 temperature: float,
                                 mutation_strength: float) -> Solution:
        """Generate mutated version of original solution.
        
        Args:
            problem: Problem to solve
            original_solution: Original solution
            critic_analysis: Critic's analysis
            conversation: Conversation context
            turn: Turn number
            temperature: LLM temperature
            mutation_strength: Mutation strength
            
        Returns:
            Mutated solution
        """
        # Create mutation-specific prompt
        mutation_prompt = self._create_mutation_prompt(
            problem=problem,
            solution=original_solution,
            critic_analysis=critic_analysis,
            mutation_strength=mutation_strength
        )
        
        # Generate mutated solution
        author_response = self.llm.generate(
            prompt=mutation_prompt,
            temperature=temperature
        )
        
        # Parse solution content
        solution_content = self._parse_solution_content(author_response)
        
        # Evaluate mutated solution
        evaluation = self.evaluator.evaluate(solution_content, problem)
        
        # Create mutated solution object
        mutated_solution = Solution(
            id=str(uuid.uuid4()),
            content=solution_content,
            score=evaluation.score,
            feedback=evaluation.feedback,
            generation=conversation.generation,
            island_id=conversation.island_id,
            conversation_id=conversation.id,
            parent_ids=[original_solution.id],
            metadata={
                'turn': turn,
                'mutation': True,
                'mutation_strength': mutation_strength,
                'original_score': original_solution.score,
                'temperature': temperature,
                'critic_analysis_length': len(critic_analysis)
            },
            timestamp=datetime.now()
        )
        
        return mutated_solution
        
    def _create_mutation_prompt(self,
                              problem: Problem,
                              solution: Solution,
                              critic_analysis: str,
                              mutation_strength: float) -> str:
        """Create prompt for mutation operation.
        
        Args:
            problem: Problem context
            solution: Original solution
            critic_analysis: Critic's analysis
            mutation_strength: Mutation strength
            
        Returns:
            Mutation prompt
        """
        if mutation_strength < 0.3:
            mutation_instruction = "Make MINOR modifications to improve the solution while keeping the core approach."
        elif mutation_strength < 0.7:
            mutation_instruction = "Make MODERATE changes to significantly improve the solution."
        else:
            mutation_instruction = "Make MAJOR changes or try a completely different approach."
            
        # Use the author prompt as base and add mutation-specific instructions
        base_prompt = self.prompt_manager.create_author_prompt(
            problem=problem,
            solution=solution,
            feedback=solution.feedback,
            critic_analysis=critic_analysis
        )
        
        mutation_prompt = f"""{base_prompt}

MUTATION INSTRUCTIONS:
{mutation_instruction}

Mutation Strength: {mutation_strength:.1f} (0.0 = minimal change, 1.0 = maximum change)

Focus on creating a meaningfully different solution while addressing the identified issues.

Generate your mutated solution now:"""
        
        return mutation_prompt
        
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
            "Here's my mutated solution:",
            "Here is my mutated solution:",
            "My mutated solution:",
            "Mutated solution:",
            "Improved solution:",
            "Modified solution:",
            "Solution:",
        ]
        
        for prefix in prefixes_to_remove:
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()
                break
                
        return content
        
    def random_mutation(self,
                       problem: Problem,
                       solution: Solution,
                       num_changes: int = 1) -> str:
        """Perform random textual mutations on solution.
        
        Args:
            problem: Problem context
            solution: Solution to mutate
            num_changes: Number of random changes to make
            
        Returns:
            Randomly mutated solution content
        """
        import random
        
        words = solution.content.split()
        if not words:
            return solution.content
            
        mutated_words = words.copy()
        
        for _ in range(min(num_changes, len(words) // 4)):
            mutation_type = random.choice(['replace', 'insert', 'delete', 'swap'])
            
            if mutation_type == 'replace' and len(words) > 0:
                idx = random.randint(0, len(mutated_words) - 1)
                # Replace with a similar word (simplified)
                synonyms = ['improved', 'enhanced', 'better', 'optimized', 'refined']
                mutated_words[idx] = random.choice(synonyms)
                
            elif mutation_type == 'insert':
                idx = random.randint(0, len(mutated_words))
                insertions = ['additionally', 'furthermore', 'moreover', 'also']
                mutated_words.insert(idx, random.choice(insertions))
                
            elif mutation_type == 'delete' and len(mutated_words) > 5:
                idx = random.randint(0, len(mutated_words) - 1)
                mutated_words.pop(idx)
                
            elif mutation_type == 'swap' and len(mutated_words) > 1:
                idx1 = random.randint(0, len(mutated_words) - 1)
                idx2 = random.randint(0, len(mutated_words) - 1)
                mutated_words[idx1], mutated_words[idx2] = mutated_words[idx2], mutated_words[idx1]
                
        return ' '.join(mutated_words)
        
    def guided_mutation(self,
                       problem: Problem,
                       solution: Solution,
                       feedback_focus: Optional[List[str]] = None,
                       temperature: float = 1.0) -> Solution:
        """Perform guided mutation based on specific feedback.
        
        Args:
            problem: Problem to solve
            solution: Solution to mutate
            feedback_focus: Specific feedback items to address
            temperature: LLM temperature
            
        Returns:
            Guided mutated solution
        """
        # Create a dummy conversation for this operation
        conversation = ConversationThread(
            id=f"guided_mutation_{uuid.uuid4().hex[:8]}",
            island_id=solution.island_id,
            generation=solution.generation,
            parent_solutions=[solution],
            children=[],
            turns=[]
        )
        
        # Create focused feedback if provided
        if feedback_focus:
            focused_solution = Solution(
                **solution.model_dump(),
                feedback=feedback_focus
            )
        else:
            focused_solution = solution
            
        return self.mutate(
            problem=problem,
            solution=focused_solution,
            conversation=conversation,
            temperature=temperature,
            mutation_strength=0.5,
            enable_critic=True
        )
        
    def adaptive_mutation(self,
                         problem: Problem,
                         solution: Solution,
                         population_diversity: float,
                         temperature: float = 1.0) -> Solution:
        """Perform adaptive mutation based on population diversity.
        
        Args:
            problem: Problem to solve
            solution: Solution to mutate
            population_diversity: Current population diversity score
            temperature: LLM temperature
            
        Returns:
            Adaptively mutated solution
        """
        # Adjust mutation strength based on diversity
        # Low diversity -> higher mutation strength to explore
        # High diversity -> lower mutation strength to exploit
        if population_diversity < 0.3:
            mutation_strength = 0.8  # High exploration
        elif population_diversity < 0.6:
            mutation_strength = 0.5  # Moderate
        else:
            mutation_strength = 0.3  # Low, focus on exploitation
            
        # Create conversation context
        conversation = ConversationThread(
            id=f"adaptive_mutation_{uuid.uuid4().hex[:8]}",
            island_id=solution.island_id,
            generation=solution.generation,
            parent_solutions=[solution],
            children=[],
            turns=[]
        )
        
        return self.mutate(
            problem=problem,
            solution=solution,
            conversation=conversation,
            temperature=temperature,
            mutation_strength=mutation_strength,
            enable_critic=True
        )