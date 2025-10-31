"""Core data models for Mind Evolution system."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, validator


class Solution(BaseModel):
    """Individual candidate solution in the population."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="The actual solution content")
    score: float = Field(..., description="Fitness score from evaluator")
    feedback: list[str] = Field(default_factory=list, description="Textual feedback")
    generation: int = Field(..., description="Generation number created")
    island_id: int = Field(..., description="Origin island ID")
    conversation_id: str = Field(..., description="Parent conversation thread ID")
    parent_ids: list[str] = Field(default_factory=list, description="Parent solution IDs")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional info")
    timestamp: datetime = Field(default_factory=datetime.now)

    def is_valid(self) -> bool:
        """Check if solution satisfies basic validity constraints."""
        return self.score >= 0 and len(self.content.strip()) > 0

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationTurn(BaseModel):
    """Single turn in the refinement conversation."""

    turn_number: int = Field(..., description="Turn number in conversation")
    critic_prompt: str = Field(default="", description="Critic's analysis prompt")
    critic_response: str = Field(default="", description="Critic's response")
    author_prompt: str = Field(..., description="Author's generation prompt")
    author_response: str = Field(..., description="Author's response")
    generated_solution: Solution = Field(..., description="Solution generated this turn")
    evaluation_result: EvaluationResult = Field(..., description="Evaluation outcome")


class ConversationThread(BaseModel):
    """Tracks a single evolutionary conversation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    island_id: int = Field(..., description="Island where conversation occurs")
    generation: int = Field(..., description="Generation number")
    parent_solutions: list[Solution] = Field(default_factory=list)
    children: list[Solution] = Field(default_factory=list)
    turns: list[ConversationTurn] = Field(default_factory=list)

    @property
    def num_parents(self) -> int:
        """Number of parent solutions."""
        return len(self.parent_solutions)

    @property
    def best_child(self) -> Solution | None:
        """Best child solution by score."""
        if not self.children:
            return None
        return max(self.children, key=lambda s: s.score)


class EvaluationResult(BaseModel):
    """Result of evaluating a solution."""

    score: float = Field(..., description="Numeric fitness score")
    feedback: list[str] = Field(default_factory=list, description="Textual feedback")
    is_valid: bool = Field(..., description="Whether solution satisfies constraints")
    constraint_violations: int = Field(default=0, description="Number of violations")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metrics")

    @validator('score')
    def score_must_be_numeric(cls, v):
        """Ensure score is a valid number."""
        if not isinstance(v, (int, float)):
            raise ValueError('Score must be numeric')
        return float(v)


class Problem(BaseModel):
    """Problem definition for Mind Evolution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Problem title")
    description: str = Field(..., description="Detailed problem description")
    constraints: list[str] = Field(default_factory=list, description="Problem constraints")
    examples: list[dict[str, Any]] = Field(default_factory=list, description="Example inputs/outputs")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional problem info")

    @validator('description')
    def description_not_empty(cls, v):
        """Ensure description is not empty."""
        if not v.strip():
            raise ValueError('Problem description cannot be empty')
        return v


class MindEvolutionConfig(BaseModel):
    """Hyperparameters for Mind Evolution system."""

    # Generation control
    N_gens: int = Field(10, description="Maximum generations", ge=1)

    # Island model
    N_island: int = Field(4, description="Number of islands", ge=1)
    N_reset_interval: int = Field(3, description="Generations between resets", ge=1)
    N_reset: int = Field(2, description="Islands to reset each time", ge=1)

    # Population structure
    N_convs: int = Field(5, description="Conversations per island per generation", ge=1)
    N_seq: int = Field(4, description="Sequential refinement turns", ge=1)

    # Genetic operators
    N_parent: int = Field(5, description="Max parents for crossover", ge=0)
    Pr_no_parents: float = Field(1/6, description="Probability of zero parents", ge=0.0, le=1.0)

    # Migration
    N_emigrate: int = Field(5, description="Solutions to migrate per island", ge=0)

    # Island reset
    N_top: int = Field(5, description="Elite solutions for reset", ge=1)
    N_candidate: int = Field(15, description="Candidates to consider for reset", ge=1)
    use_llm_for_reset: bool = Field(True, description="Use LLM to select diverse elites")

    # LLM settings
    model_name: str = Field("gemini-1.5-flash-001", description="LLM model name")
    temperature: float = Field(1.0, description="LLM temperature", ge=0.0, le=2.0)
    max_retries: int = Field(5, description="Retry failed generations", ge=1)

    # Optimization
    early_stopping: bool = Field(True, description="Stop when valid solution found")
    enable_critic: bool = Field(True, description="Use critic-author dialog")
    enable_feedback: bool = Field(True, description="Include textual feedback")

    # Logging
    log_level: str = Field("INFO", description="Logging level")
    save_all_solutions: bool = Field(True, description="Save all generated solutions")
    checkpoint_frequency: int = Field(1, description="Save every N generations", ge=1)

    @validator('N_reset')
    def reset_not_exceed_islands(cls, v, values):
        """Ensure N_reset doesn't exceed N_island."""
        if 'N_island' in values and v > values['N_island']:
            raise ValueError('N_reset cannot exceed N_island')
        return v

    @validator('N_top')
    def top_not_exceed_candidates(cls, v, values):
        """Ensure N_top doesn't exceed N_candidate."""
        if 'N_candidate' in values and v > values['N_candidate']:
            raise ValueError('N_top cannot exceed N_candidate')
        return v


class PopulationStats(BaseModel):
    """Statistics for a population."""

    island_id: int = Field(..., description="Island ID")
    generation: int = Field(..., description="Generation number")
    size: int = Field(..., description="Population size")
    mean_score: float = Field(..., description="Mean fitness score")
    max_score: float = Field(..., description="Maximum fitness score")
    min_score: float = Field(..., description="Minimum fitness score")
    std_score: float = Field(..., description="Standard deviation of scores")
    valid_solutions: int = Field(..., description="Number of valid solutions")
    best_solution_id: str | None = Field(None, description="ID of best solution")
