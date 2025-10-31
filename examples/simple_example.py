"""
Simple example demonstrating Mind Evolution usage.

This example shows how to solve a basic problem using Mind Evolution
with minimal configuration.
"""

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
    """Run simple Mind Evolution example."""

    # Setup logging
    setup_logging(log_level="INFO", experiment_name="simple_example")

    # Define a simple problem
    problem = Problem(
        title="Creative Writing Task",
        description="""
        Write a short creative story (200-300 words) about a robot who discovers emotions.
        The story should be engaging, well-structured, and explore themes of consciousness
        and what it means to be human.
        """,
        constraints=[
            "Story must be 200-300 words long",
            "Must feature a robot as the main character",
            "Must explore the theme of emotions or consciousness",
            "Should have a clear beginning, middle, and end",
            "Writing should be engaging and creative"
        ]
    )

    # Create configuration
    config = MindEvolutionConfig(
        N_gens=5,           # 5 generations (quick example)
        N_island=2,         # 2 islands
        N_convs=3,          # 3 conversations per island per generation
        N_seq=3,            # 3 refinement turns per conversation
        temperature=0.8,    # Slightly lower temperature for more focused responses
        model_name="gpt-3.5-turbo",  # Use cheaper model for example
        early_stopping=True,
        enable_critic=True,
        enable_feedback=True,
    )

    # Initialize components
    print("Initializing Mind Evolution components...")

    # Create LLM (will use OPENAI_API_KEY environment variable)
    llm = create_llm("openai", config.model_name)

    # Create evaluator
    evaluator = create_evaluator("constraint", constraint_weights={
        "length": 1.0,
        "character": 2.0,
        "theme": 2.0,
        "structure": 1.5,
        "creativity": 1.0,
    })

    # Create prompt manager
    prompt_manager = PromptManager(task_type="creative_writing")

    # Initialize Mind Evolution
    mind_evolution = MindEvolution(
        config=config,
        llm=llm,
        evaluator=evaluator,
        prompt_manager=prompt_manager
    )

    # Solve the problem
    print("Starting Mind Evolution...")
    print(f"Problem: {problem.title}")
    print(f"Configuration: {config.N_gens} generations, {config.N_island} islands")

    best_solution = mind_evolution.solve(problem, "simple_example")

    # Display results
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE!")
    print("="*60)
    print(f"Best Solution Score: {best_solution.score:.3f}")
    print(f"Valid Solution: {'Yes' if best_solution.is_valid() else 'No'}")
    print(f"Generation: {best_solution.generation}")
    print(f"Island: {best_solution.island_id}")

    print("\nFeedback:")
    for feedback in best_solution.feedback[:3]:
        print(f"• {feedback}")

    print("\nBest Story:")
    print("-" * 40)
    print(best_solution.content)
    print("-" * 40)

    # Get evolution statistics
    stats = mind_evolution.get_evolution_statistics()
    print("\nEvolution Statistics:")
    print(f"• Generations completed: {stats['generations_completed']}")
    print(f"• Global best score: {stats['global_best_score']:.3f}")

    if 'performance_metrics' in stats:
        perf = stats['performance_metrics']
        print(f"• Runtime: {perf['runtime_minutes']:.2f} minutes")
        print(f"• Solutions generated: {perf.get('total_solutions_generated', 'N/A')}")
        print(f"• Success rate: {perf.get('success_rate', 0):.1f}%")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Export results
    mind_evolution.export_results("results/simple_example_results.json")

    # Save best solution to file
    with open("results/simple_example_best_story.txt", "w") as f:
        f.write(f"Score: {best_solution.score:.3f}\n")
        f.write(f"Valid: {best_solution.is_valid()}\n")
        f.write(f"Feedback: {'; '.join(best_solution.feedback)}\n\n")
        f.write("Story:\n")
        f.write(best_solution.content)

    print("\nResults saved to 'results/' directory")

    return best_solution


if __name__ == "__main__":
    # Run the example
    import os

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        exit(1)

    # Run the example
    main()
