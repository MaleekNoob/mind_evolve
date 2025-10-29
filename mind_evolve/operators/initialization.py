"""Population initialization operators."""

from typing import List

from loguru import logger

from ..core.models import ConversationThread, Problem, Solution
from ..evaluation.evaluator_base import BaseEvaluator
from ..llm.llm_interface import BaseLLM
from ..llm.prompt_manager import PromptManager


class PopulationInitializer:
    """Handles initialization of populations with solutions."""
    
    def __init__(self, 
                 llm: BaseLLM,
                 evaluator: BaseEvaluator,
                 prompt_manager: PromptManager):
        """Initialize population initializer.
        
        Args:
            llm: LLM interface for generation
            evaluator: Solution evaluator
            prompt_manager: Prompt management system
        """
        self.llm = llm
        self.evaluator = evaluator
        self.prompt_manager = prompt_manager
        
    def initialize_population(self, 
                            problem: Problem,
                            island_id: int, 
                            num_conversations: int,
                            num_refinement_turns: int,
                            temperature: float = 1.0) -> List[Solution]:
        """Initialize population for an island.
        
        Args:
            problem: Problem to solve
            island_id: Island identifier
            num_conversations: Number of conversations to create
            num_refinement_turns: Number of refinement turns per conversation
            temperature: LLM temperature
            
        Returns:
            List of generated solutions
        """
        logger.info(f"Initializing population for island {island_id}")
        
        all_solutions = []
        
        for conv_idx in range(num_conversations):
            conversation_id = f"G0_I{island_id}_C{conv_idx}"
            
            # Create conversation thread
            conversation = ConversationThread(
                id=conversation_id,
                island_id=island_id,
                generation=0,
                parent_solutions=[],  # No parents for initialization
                children=[],
                turns=[]
            )
            
            # Generate initial solution
            initial_solution = self._generate_initial_solution(
                problem=problem,
                conversation=conversation,
                temperature=temperature
            )
            
            conversation.children.append(initial_solution)
            all_solutions.append(initial_solution)
            
            # Sequential refinement
            current_solution = initial_solution
            for turn in range(2, num_refinement_turns + 1):
                refined_solution = self._refine_solution(
                    problem=problem,
                    previous_solution=current_solution,
                    conversation=conversation,
                    turn=turn,
                    temperature=temperature
                )
                
                conversation.children.append(refined_solution)
                all_solutions.append(refined_solution)
                current_solution = refined_solution
                
        logger.info(f"Generated {len(all_solutions)} solutions for island {island_id}")
        return all_solutions
        
    def _generate_initial_solution(self, 
                                 problem: Problem,
                                 conversation: ConversationThread,
                                 temperature: float) -> Solution:
        """Generate initial solution for a conversation.
        
        Args:
            problem: Problem to solve
            conversation: Conversation context
            temperature: LLM temperature
            
        Returns:
            Generated solution
        """
        import uuid
        from datetime import datetime
        
        # Create initial prompt
        initial_prompt = self.prompt_manager.create_initial_prompt(problem)
        
        # Generate solution
        response = self.llm.generate(
            prompt=initial_prompt,
            temperature=temperature
        )
        
        # Parse solution content
        solution_content = self._parse_solution_content(response)
        
        # Evaluate solution
        evaluation = self.evaluator.evaluate(solution_content, problem)
        
        # Create solution object
        solution = Solution(
            id=str(uuid.uuid4()),
            content=solution_content,
            score=evaluation.score,
            feedback=evaluation.feedback,
            generation=conversation.generation,
            island_id=conversation.island_id,
            conversation_id=conversation.id,
            parent_ids=[],
            metadata={
                'turn': 1,
                'initialization': True,
                'temperature': temperature
            },
            timestamp=datetime.now()
        )
        
        logger.debug(f"Generated initial solution {solution.id[:8]} "
                    f"with score {solution.score:.3f}")
        
        return solution
        
    def _refine_solution(self, 
                        problem: Problem,
                        previous_solution: Solution,
                        conversation: ConversationThread,
                        turn: int,
                        temperature: float) -> Solution:
        """Refine a solution using critic-author dialog.
        
        Args:
            problem: Problem to solve
            previous_solution: Previous solution to refine
            conversation: Conversation context
            turn: Turn number
            temperature: LLM temperature
            
        Returns:
            Refined solution
        """
        import uuid
        from datetime import datetime
        
        # Generate critic analysis
        critic_prompt = self.prompt_manager.create_critic_prompt(
            problem=problem,
            solution=previous_solution,
            feedback=previous_solution.feedback
        )
        
        critic_response = self.llm.generate(
            prompt=critic_prompt,
            temperature=temperature * 0.8  # Slightly lower temperature for analysis
        )
        
        # Generate refined solution
        author_prompt = self.prompt_manager.create_author_prompt(
            problem=problem,
            solution=previous_solution,
            feedback=previous_solution.feedback,
            critic_analysis=critic_response
        )
        
        author_response = self.llm.generate(
            prompt=author_prompt,
            temperature=temperature
        )
        
        # Parse solution content
        solution_content = self._parse_solution_content(author_response)
        
        # Evaluate refined solution
        evaluation = self.evaluator.evaluate(solution_content, problem)
        
        # Create refined solution object
        refined_solution = Solution(
            id=str(uuid.uuid4()),
            content=solution_content,
            score=evaluation.score,
            feedback=evaluation.feedback,
            generation=conversation.generation,
            island_id=conversation.island_id,
            conversation_id=conversation.id,
            parent_ids=[previous_solution.id],
            metadata={
                'turn': turn,
                'refinement': True,
                'critic_analysis': critic_response[:200] + "..." if len(critic_response) > 200 else critic_response,
                'temperature': temperature
            },
            timestamp=datetime.now()
        )
        
        logger.debug(f"Refined solution {refined_solution.id[:8]} "
                    f"(turn {turn}) with score {refined_solution.score:.3f}")
        
        return refined_solution
        
    def _parse_solution_content(self, llm_response: str) -> str:
        """Parse solution content from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Cleaned solution content
        """
        # Remove common prefixes/suffixes
        content = llm_response.strip()
        
        # Remove common LLM response patterns
        prefixes_to_remove = [
            "Here's my solution:",
            "Here is my solution:",
            "My solution:",
            "Solution:",
            "Answer:",
            "Response:",
        ]
        
        for prefix in prefixes_to_remove:
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()
                break
                
        # Remove markdown code blocks if present
        if content.startswith("```") and content.endswith("```"):
            lines = content.split('\n')
            if len(lines) > 2:
                content = '\n'.join(lines[1:-1])
                
        return content.strip()
        
    def generate_diverse_initial_solutions(self, 
                                         problem: Problem,
                                         num_solutions: int,
                                         temperature_range: tuple = (0.7, 1.3)) -> List[Solution]:
        """Generate diverse initial solutions with varying parameters.
        
        Args:
            problem: Problem to solve
            num_solutions: Number of solutions to generate
            temperature_range: Range of temperatures to use
            
        Returns:
            List of diverse initial solutions
        """
        import numpy as np
        
        solutions = []
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_solutions)
        
        for i, temp in enumerate(temperatures):
            # Create dummy conversation for generation
            conversation = ConversationThread(
                id=f"diverse_{i}",
                island_id=0,
                generation=0,
                parent_solutions=[],
                children=[],
                turns=[]
            )
            
            solution = self._generate_initial_solution(
                problem=problem,
                conversation=conversation,
                temperature=float(temp)
            )
            
            # Add diversity metadata
            solution.metadata['diversity_generation'] = True
            solution.metadata['temperature_used'] = float(temp)
            
            solutions.append(solution)
            
        logger.info(f"Generated {len(solutions)} diverse initial solutions")
        return solutions