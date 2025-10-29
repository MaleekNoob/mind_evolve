"""Prompt management system for Mind Evolution."""

from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader, TemplateNotFound
from pydantic import BaseModel

from ..core.models import Problem, Solution


class PromptTemplate(BaseModel):
    """Template for generating prompts."""
    
    name: str
    template: str
    description: str = ""
    required_variables: List[str] = []
    

class PromptManager:
    """Manages prompt templates and generation for Mind Evolution."""
    
    def __init__(self, task_type: str = "general"):
        """Initialize prompt manager.
        
        Args:
            task_type: Type of task (travel_planning, coding, etc.)
        """
        self.task_type = task_type
        self.env = Environment(loader=BaseLoader())
        self.templates = self._load_default_templates()
        
    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """Load default prompt templates."""
        return {
            "initial": PromptTemplate(
                name="initial",
                template=self._get_initial_template(),
                description="Generate initial solution",
                required_variables=["problem"]
            ),
            "critic": PromptTemplate(
                name="critic",
                template=self._get_critic_template(),
                description="Analyze solution critically",
                required_variables=["problem", "solution", "feedback"]
            ),
            "author": PromptTemplate(
                name="author",
                template=self._get_author_template(),
                description="Generate improved solution",
                required_variables=["problem", "solution", "feedback", "critic_analysis"]
            ),
            "multi_parent_critic": PromptTemplate(
                name="multi_parent_critic",
                template=self._get_multi_parent_critic_template(),
                description="Analyze multiple parent solutions",
                required_variables=["problem", "parents"]
            ),
            "multi_parent_author": PromptTemplate(
                name="multi_parent_author",
                template=self._get_multi_parent_author_template(),
                description="Synthesize from multiple parents",
                required_variables=["problem", "parents", "critic_analysis"]
            ),
            "elite_selection": PromptTemplate(
                name="elite_selection",
                template=self._get_elite_selection_template(),
                description="Select diverse elite solutions",
                required_variables=["candidates", "num_to_select"]
            )
        }
        
    def create_initial_prompt(self, problem: Problem) -> str:
        """Generate prompt for initial solution generation.
        
        Args:
            problem: Problem to solve
            
        Returns:
            Formatted prompt string
        """
        template = self.env.from_string(self.templates["initial"].template)
        return template.render(
            problem=problem,
            problem_description=problem.description,
            constraints=self._format_constraints(problem.constraints),
            examples=self._format_examples(problem.examples)
        )
        
    def create_critic_prompt(self, 
                           problem: Problem, 
                           solution: Solution,
                           feedback: List[str]) -> str:
        """Generate critic analysis prompt.
        
        Args:
            problem: Original problem
            solution: Solution to analyze
            feedback: Evaluation feedback
            
        Returns:
            Formatted prompt string
        """
        template = self.env.from_string(self.templates["critic"].template)
        return template.render(
            problem=problem,
            problem_description=problem.description,
            constraints=self._format_constraints(problem.constraints),
            solution_content=solution.content,
            feedback_list=self._format_feedback(feedback),
            score=solution.score
        )
        
    def create_author_prompt(self, 
                           problem: Problem,
                           solution: Solution,
                           feedback: List[str],
                           critic_analysis: str) -> str:
        """Generate author refinement prompt.
        
        Args:
            problem: Original problem
            solution: Previous solution
            feedback: Evaluation feedback
            critic_analysis: Critic's analysis
            
        Returns:
            Formatted prompt string
        """
        template = self.env.from_string(self.templates["author"].template)
        return template.render(
            problem=problem,
            problem_description=problem.description,
            constraints=self._format_constraints(problem.constraints),
            solution_content=solution.content,
            feedback_list=self._format_feedback(feedback),
            critic_analysis=critic_analysis,
            score=solution.score
        )
        
    def create_multi_parent_critic_prompt(self, 
                                        problem: Problem,
                                        parents: List[Solution]) -> str:
        """Generate critic prompt for multiple parents.
        
        Args:
            problem: Original problem
            parents: Parent solutions to analyze
            
        Returns:
            Formatted prompt string
        """
        template = self.env.from_string(self.templates["multi_parent_critic"].template)
        return template.render(
            problem=problem,
            problem_description=problem.description,
            constraints=self._format_constraints(problem.constraints),
            parents=parents,
            parent_summaries=self._format_parent_summaries(parents)
        )
        
    def create_multi_parent_author_prompt(self, 
                                        problem: Problem,
                                        parents: List[Solution],
                                        critic_analysis: str) -> str:
        """Generate author prompt for crossover.
        
        Args:
            problem: Original problem
            parents: Parent solutions
            critic_analysis: Critic's analysis
            
        Returns:
            Formatted prompt string
        """
        template = self.env.from_string(self.templates["multi_parent_author"].template)
        return template.render(
            problem=problem,
            problem_description=problem.description,
            constraints=self._format_constraints(problem.constraints),
            parents=parents,
            parent_summaries=self._format_parent_summaries(parents),
            critic_analysis=critic_analysis
        )
        
    def create_elite_selection_prompt(self, 
                                    candidates: List[Solution],
                                    num_to_select: int) -> str:
        """Generate prompt for diverse elite selection.
        
        Args:
            candidates: Candidate solutions
            num_to_select: Number to select
            
        Returns:
            Formatted prompt string
        """
        template = self.env.from_string(self.templates["elite_selection"].template)
        return template.render(
            candidates=candidates,
            num_to_select=num_to_select,
            candidate_summaries=self._format_candidate_summaries(candidates)
        )
        
    def _format_constraints(self, constraints: List[str]) -> str:
        """Format constraints for prompt inclusion."""
        if not constraints:
            return "No specific constraints provided."
        return "\n".join(f"- {constraint}" for constraint in constraints)
        
    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples for prompt inclusion."""
        if not examples:
            return "No examples provided."
        
        formatted = []
        for i, example in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            for key, value in example.items():
                formatted.append(f"  {key}: {value}")
        return "\n".join(formatted)
        
    def _format_feedback(self, feedback: List[str]) -> str:
        """Format feedback for prompt inclusion."""
        if not feedback:
            return "No specific feedback provided."
        return "\n".join(f"- {fb}" for fb in feedback)
        
    def _format_parent_summaries(self, parents: List[Solution]) -> str:
        """Format parent solution summaries."""
        summaries = []
        for i, parent in enumerate(parents, 1):
            summaries.append(f"Parent {i} (Score: {parent.score:.2f}):")
            summaries.append(f"  Content: {parent.content[:200]}...")
            if parent.feedback:
                summaries.append(f"  Feedback: {'; '.join(parent.feedback[:2])}")
        return "\n".join(summaries)
        
    def _format_candidate_summaries(self, candidates: List[Solution]) -> str:
        """Format candidate solution summaries."""
        summaries = []
        for i, candidate in enumerate(candidates):
            summaries.append(f"Candidate {i} (ID: {candidate.id[:8]}, Score: {candidate.score:.2f}):")
            summaries.append(f"  Content: {candidate.content[:150]}...")
        return "\n".join(summaries)
        
    # Template definitions
    def _get_initial_template(self) -> str:
        """Get initial solution generation template."""
        return """You are an expert problem solver tasked with creating a high-quality solution.

PROBLEM:
{{ problem_description }}

CONSTRAINTS:
{{ constraints }}

{% if examples %}
EXAMPLES:
{{ examples }}
{% endif %}

Your task is to generate a comprehensive solution that satisfies all constraints and addresses the problem effectively.

IMPORTANT GUIDELINES:
- Read the problem carefully and understand all requirements
- Consider all constraints when developing your solution
- Be specific and detailed in your response
- Ensure your solution is feasible and practical
- Think step-by-step through your approach

Generate your solution now:"""

    def _get_critic_template(self) -> str:
        """Get critic analysis template."""
        return """You are a critical analyst reviewing a proposed solution.

PROBLEM:
{{ problem_description }}

CONSTRAINTS:
{{ constraints }}

PROPOSED SOLUTION (Score: {{ score }}):
{{ solution_content }}

EVALUATION FEEDBACK:
{{ feedback_list }}

Your task is to provide a thorough critical analysis:

1. CONSTRAINT ANALYSIS:
   - Which constraints are satisfied?
   - Which constraints are violated and why?
   - What are the specific issues with constraint violations?

2. SOLUTION QUALITY:
   - What aspects of the solution work well?
   - What are the main weaknesses or gaps?
   - How could the approach be improved?

3. IMPROVEMENT SUGGESTIONS:
   - What specific changes would address the feedback?
   - Are there alternative approaches to consider?
   - What trade-offs should be considered?

Provide your critical analysis:"""

    def _get_author_template(self) -> str:
        """Get author refinement template."""
        return """You are an expert problem solver creating an improved solution.

PROBLEM:
{{ problem_description }}

CONSTRAINTS:
{{ constraints }}

PREVIOUS SOLUTION (Score: {{ score }}):
{{ solution_content }}

EVALUATION FEEDBACK:
{{ feedback_list }}

CRITIC'S ANALYSIS:
{{ critic_analysis }}

Your task is to generate an IMPROVED solution that addresses the issues identified.

IMPROVEMENT STRATEGY:
- Focus on fixing the specific problems mentioned in the feedback
- Incorporate insights from the critic's analysis
- Maintain the good aspects of the previous solution
- Ensure all constraints are properly addressed

QUALITY CHECKLIST:
- Does the solution address all constraint violations?
- Have you incorporated the critic's suggestions?
- Is the solution more comprehensive than before?
- Are all requirements clearly satisfied?

Generate your improved solution now:"""

    def _get_multi_parent_critic_template(self) -> str:
        """Get multi-parent critic template."""
        return """You are analyzing multiple solution proposals to identify their strengths and weaknesses.

PROBLEM:
{{ problem_description }}

CONSTRAINTS:
{{ constraints }}

PARENT SOLUTIONS:
{{ parent_summaries }}

Your task is to analyze these solutions collectively:

1. COMPARATIVE ANALYSIS:
   - What does each solution do well?
   - What are the unique strengths of each approach?
   - What common weaknesses do they share?

2. SYNTHESIS OPPORTUNITIES:
   - Which elements from different solutions could be combined?
   - How could the best aspects be integrated?
   - What novel approaches could emerge from this combination?

3. IMPROVEMENT STRATEGY:
   - What would an ideal solution incorporate from each parent?
   - How can the weaknesses be addressed in a new solution?
   - What creative approaches haven't been tried yet?

Provide your analysis:"""

    def _get_multi_parent_author_template(self) -> str:
        """Get multi-parent author template."""
        return """You are synthesizing insights from multiple solutions to create a superior approach.

PROBLEM:
{{ problem_description }}

CONSTRAINTS:
{{ constraints }}

PARENT SOLUTIONS:
{{ parent_summaries }}

CRITIC'S SYNTHESIS ANALYSIS:
{{ critic_analysis }}

Your task is to create a NEW solution that combines the best elements while avoiding the weaknesses.

SYNTHESIS STRATEGY:
- Identify the strongest elements from each parent solution
- Combine complementary approaches effectively
- Address weaknesses present in the parent solutions
- Introduce novel improvements where possible

INTEGRATION PRINCIPLES:
- Ensure coherence in the combined approach
- Maintain feasibility while maximizing quality
- Address all constraints comprehensively
- Create something better than any individual parent

Generate your synthesized solution now:"""

    def _get_elite_selection_template(self) -> str:
        """Get elite selection template."""
        return """You are selecting diverse, high-quality solutions for population reset.

CANDIDATES:
{{ candidate_summaries }}

Your task is to select {{ num_to_select }} solutions that are:
1. HIGH QUALITY: Among the best performing candidates
2. DIVERSE: Substantially different approaches from each other
3. COMPLEMENTARY: Covering different aspects or strategies

SELECTION CRITERIA:
- Prioritize solutions with different core approaches
- Avoid selecting solutions that are too similar
- Consider both quality and diversity
- Aim for a balanced set that explores different solution spaces

Please provide your selection as a list of candidate numbers (0-indexed):
Example format: [0, 3, 7, 12, 15]

Your selection:"""