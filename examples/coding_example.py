"""
Coding problem example demonstrating Mind Evolution on algorithm design.

This example shows how to use Mind Evolution to solve complex programming
challenges that require algorithmic thinking and optimization.
"""

import os
from pathlib import Path

from mind_evolve import (
    MindEvolution,
    MindEvolutionConfig,
    Problem,
    create_evaluator,
    create_llm,
    setup_logging,
)
from mind_evolve.llm import PromptManager


def main():
    """Run coding problem example with Mind Evolution."""

    # Setup logging
    setup_logging(log_level="INFO", experiment_name="coding_example")

    # Define a complex coding problem
    problem = Problem(
        title="Dynamic Programming Optimization Challenge",
        description="""
        Design and implement an efficient algorithm to solve the following problem:
        
        You have a robot that needs to collect treasures in a 2D grid. The robot starts 
        at the top-left corner (0,0) and must reach the bottom-right corner (m-1, n-1). 
        The robot can only move right or down. Each cell contains a treasure with a 
        certain value (positive, negative, or zero). The robot has limited energy and 
        each move consumes 1 unit of energy.
        
        Find the path that maximizes total treasure value while ensuring the robot 
        has enough energy to complete the journey.
        """,
        constraints=[
            "Provide complete working Python solution",
            "Use dynamic programming approach",
            "Handle edge cases (empty grid, insufficient energy)",
            "Include time and space complexity analysis",
            "Code must be well-commented and follow best practices",
            "Include test cases with expected outputs",
            "Solution should be efficient for grids up to 100x100",
            "Explain the optimal substructure property",
            "Must include both recursive and iterative approaches",
        ],
        examples=[
            {
                "input": "Grid: [[1, -3, 3], [1, 5, -2], [4, -1, 1]], Energy: 4",
                "output": "Maximum treasure path with dynamic programming solution",
            }
        ],
    )

    # Create configuration optimized for coding problems
    config = MindEvolutionConfig(
        N_gens=8,  # More generations for complex problems
        N_island=3,  # More islands for diverse approaches
        N_convs=4,  # More conversations for thorough exploration
        N_seq=4,  # More refinement turns for code quality
        temperature=0.7,  # Lower temperature for more focused code generation
        model_name="gpt-4",  # Use more capable model for coding
        early_stopping=False,  # Don't stop early for coding problems
        enable_critic=True,
        enable_feedback=True,
        max_retries=3,
    )

    # Initialize components
    print("Initializing Mind Evolution for coding challenge...")

    # Create LLM
    llm = create_llm("openai", config.model_name)

    # Create specialized evaluator for code
    evaluator = create_evaluator(
        "constraint",
        constraint_weights={
            "completeness": 2.0,  # Complete working solution
            "correctness": 2.5,  # Algorithmic correctness
            "efficiency": 2.0,  # Time/space complexity
            "code_quality": 1.5,  # Comments, style, practices
            "edge_cases": 1.5,  # Handling of edge cases
            "testing": 1.0,  # Test cases included
            "explanation": 1.5,  # Clear explanations
        },
    )

    # Create prompt manager for coding tasks
    prompt_manager = PromptManager(task_type="coding")

    # Initialize Mind Evolution
    mind_evolution = MindEvolution(
        config=config, llm=llm, evaluator=evaluator, prompt_manager=prompt_manager
    )

    # Solve the coding problem
    print("Starting Mind Evolution on coding challenge...")
    print(f"Problem: {problem.title}")
    print(f"Configuration: {config.N_gens} generations, {config.N_island} islands")
    print("This may take several minutes due to the complexity...")

    best_solution = mind_evolution.solve(problem, "coding_example")

    # Display results
    print("\n" + "=" * 80)
    print("CODING CHALLENGE EVOLUTION COMPLETE!")
    print("=" * 80)
    print(f"Best Solution Score: {best_solution.score:.3f}")
    print(f"Valid Solution: {'Yes' if best_solution.is_valid() else 'No'}")
    print(f"Generation: {best_solution.generation}")
    print(f"Island: {best_solution.island_id}")

    print("\nKey Feedback:")
    for feedback in best_solution.feedback[:5]:
        print(f"• {feedback}")

    print("\nBest Algorithm Solution:")
    print("-" * 80)
    # Truncate very long solutions for display
    content = best_solution.content
    if len(content) > 2000:
        print(content[:2000] + f"\n... [truncated, total length: {len(content)} chars]")
    else:
        print(content)
    print("-" * 80)

    # Get evolution statistics
    stats = mind_evolution.get_evolution_statistics()
    print("\nEvolution Statistics:")
    print(f"• Generations completed: {stats['generations_completed']}")
    print(f"• Global best score: {stats['global_best_score']:.3f}")

    if "performance_metrics" in stats:
        perf = stats["performance_metrics"]
        print(f"• Runtime: {perf['runtime_minutes']:.2f} minutes")
        print(f"• Solutions generated: {perf.get('total_solutions_generated', 'N/A')}")
        print(f"• Success rate: {perf.get('success_rate', 0):.1f}%")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Export detailed results
    mind_evolution.export_results("results/coding_example_results.json")

    # Save best solution as Python file
    with open("results/coding_example_solution.py", "w") as f:
        f.write(f"# Dynamic Programming Solution - Score: {best_solution.score:.3f}\n")
        f.write("# Generated by Mind Evolution\n")
        f.write(f"# Feedback: {'; '.join(best_solution.feedback[:3])}\n\n")
        f.write(best_solution.content)

    # Save analysis
    with open("results/coding_example_analysis.txt", "w") as f:
        f.write("Coding Challenge Analysis\n")
        f.write("========================\n\n")
        f.write(f"Problem: {problem.title}\n")
        f.write(f"Best Score: {best_solution.score:.3f}\n")
        f.write(f"Generation Found: {best_solution.generation}\n")
        f.write(f"Island: {best_solution.island_id}\n\n")
        f.write("Feedback:\n")
        for i, feedback in enumerate(best_solution.feedback, 1):
            f.write(f"{i}. {feedback}\n")
        f.write(f"\nEvolution took {stats['generations_completed']} generations\n")

    print("\nResults saved to 'results/' directory")
    print("• Full solution: results/coding_example_solution.py")
    print("• Analysis: results/coding_example_analysis.txt")

    return best_solution


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        exit(1)

    # Run the coding example
    main()
