"""
Chemistry problem example demonstrating Mind Evolution on complex chemical challenges.

This example shows how to use Mind Evolution to solve advanced chemistry problems
that require deep understanding of chemical principles, mechanisms, and calculations.
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
    """Run chemistry problem example with Mind Evolution."""

    # Setup logging
    setup_logging(log_level="INFO", experiment_name="chemistry_example")

    # Define a challenging chemistry problem
    problem = Problem(
        title="Complex Organic Synthesis and Mechanism Analysis",
        description="""
        Design a complete synthetic route and analyze the mechanism for the following 
        multi-step organic synthesis challenge:
        
        Starting Material: 2-methylcyclohexanone
        Target Product: (S)-2-(4-methoxyphenyl)-2-methylcyclohexanol
        
        Requirements:
        1. Design an efficient synthetic route (≤ 8 steps)
        2. Ensure stereoselectivity to obtain the (S) enantiomer
        3. Provide detailed mechanism for each key step
        4. Identify all intermediates and transition states
        5. Discuss regioselectivity and stereoselectivity issues
        6. Calculate theoretical yield and predict major side products
        7. Suggest purification methods for each step
        8. Consider green chemistry principles where possible
        9. Estimate costs and scalability for industrial production
        10. Include safety considerations and handling procedures
        """,
        constraints=[
            "Provide complete synthetic scheme with all reagents and conditions",
            "Show detailed arrow-pushing mechanisms for key transformations",
            "Explain stereochemical outcomes using orbital theory",
            "Include transition state analysis for selectivity-determining steps",
            "Calculate yields and identify major side reactions",
            "Suggest analytical methods to confirm product identity",
            "Consider alternative synthetic approaches and compare efficiency",
            "Include proper chemical nomenclature and IUPAC naming",
            "Discuss reaction thermodynamics and kinetics where relevant",
            "Address potential scalability and industrial considerations",
            "Include safety data and handling precautions",
            "Solution should be at graduate-level organic chemistry depth",
        ],
        examples=[
            {
                "input": "Synthesis of chiral alcohol from ketone precursor",
                "output": "Complete synthetic route with mechanism and stereochemical analysis",
            }
        ],
    )

    # Create configuration optimized for chemistry problems
    config = MindEvolutionConfig(
        N_gens=12,  # More generations for complex synthesis design
        N_island=4,  # Multiple approaches to synthesis
        N_convs=4,  # Thorough exploration of synthetic routes
        N_seq=4,  # Refinement of mechanisms and selectivity
        temperature=0.7,  # Balanced creativity and precision
        model_name="gpt-4",  # Use most capable model for chemistry
        early_stopping=False,  # Don't stop early for complex synthesis
        enable_critic=True,
        enable_feedback=True,
        max_retries=3,
    )

    # Initialize components
    print("Initializing Mind Evolution for organic synthesis challenge...")

    # Create LLM
    llm = create_llm("openai", config.model_name)

    # Create specialized evaluator for chemistry
    evaluator = create_evaluator(
        "constraint",
        constraint_weights={
            "synthetic_efficiency": 2.5,  # Route efficiency and step count
            "mechanism_accuracy": 2.5,  # Correct mechanisms and arrows
            "stereochemistry": 2.0,  # Proper stereochemical analysis
            "regioselectivity": 1.5,  # Selectivity explanations
            "yield_calculation": 1.5,  # Realistic yield estimates
            "safety_considerations": 1.0,  # Safety and handling
            "analytical_methods": 1.0,  # Product confirmation methods
            "green_chemistry": 1.0,  # Environmental considerations
            "scalability": 1.0,  # Industrial applicability
            "side_reactions": 1.5,  # Side product prediction
            "nomenclature": 1.0,  # Proper chemical naming
        },
    )

    # Create prompt manager for chemistry tasks
    prompt_manager = PromptManager(task_type="scientific")

    # Initialize Mind Evolution
    mind_evolution = MindEvolution(
        config=config, llm=llm, evaluator=evaluator, prompt_manager=prompt_manager
    )

    # Solve the chemistry problem
    print("Starting Mind Evolution on organic synthesis problem...")
    print(f"Problem: {problem.title}")
    print(f"Configuration: {config.N_gens} generations, {config.N_island} islands")
    print("This may take several minutes due to synthesis complexity...")

    best_solution = mind_evolution.solve(problem, "chemistry_example")

    # Display results
    print("\n" + "=" * 80)
    print("ORGANIC SYNTHESIS CHALLENGE EVOLUTION COMPLETE!")
    print("=" * 80)
    print(f"Best Solution Score: {best_solution.score:.3f}")
    print(f"Valid Solution: {'Yes' if best_solution.is_valid() else 'No'}")
    print(f"Generation: {best_solution.generation}")
    print(f"Island: {best_solution.island_id}")

    print("\nKey Chemistry Insights:")
    for feedback in best_solution.feedback[:5]:
        print(f"• {feedback}")

    print("\nBest Synthesis Solution:")
    print("-" * 80)
    # Display solution with proper formatting
    content = best_solution.content
    if len(content) > 3000:
        # Show beginning and end for very long solutions
        print(content[:1500])
        print("\n... [middle section truncated for display] ...\n")
        print(content[-1500:])
        print(f"\n[Total solution length: {len(content)} characters]")
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
    mind_evolution.export_results("results/chemistry_example_results.json")

    # Save best solution as chemistry document
    with open("results/chemistry_example_solution.md", "w") as f:
        f.write(f"# Organic Synthesis Analysis - Score: {best_solution.score:.3f}\n\n")
        f.write("*Generated by Mind Evolution*\n\n")
        f.write("## Synthesis Challenge\n")
        f.write(f"{problem.description}\n\n")
        f.write("## Complete Solution\n\n")
        f.write(best_solution.content)
        f.write("\n\n## Evolution Feedback\n\n")
        for i, feedback in enumerate(best_solution.feedback, 1):
            f.write(f"{i}. {feedback}\n")

    # Save chemical data file
    with open("results/chemistry_example_data.txt", "w") as f:
        f.write("CHEMICAL SYNTHESIS DATA\n")
        f.write("======================\n\n")
        f.write("Starting Material: 2-methylcyclohexanone\n")
        f.write("Target Product: (S)-2-(4-methoxyphenyl)-2-methylcyclohexanol\n")
        f.write(f"Evolution Score: {best_solution.score:.3f}\n")
        f.write(f"Generation: {best_solution.generation}\n")
        f.write(f"Island: {best_solution.island_id}\n\n")
        f.write("KEY SYNTHETIC CONSIDERATIONS:\n")
        for feedback in best_solution.feedback:
            f.write(f"- {feedback}\n")

    # Save analysis
    with open("results/chemistry_example_analysis.txt", "w") as f:
        f.write("Organic Chemistry Challenge Analysis\n")
        f.write("====================================\n\n")
        f.write(f"Problem: {problem.title}\n")
        f.write(f"Best Score: {best_solution.score:.3f}\n")
        f.write(f"Generation Found: {best_solution.generation}\n")
        f.write(f"Island: {best_solution.island_id}\n\n")
        f.write("Chemistry Quality Assessment:\n")
        for i, feedback in enumerate(best_solution.feedback, 1):
            f.write(f"{i}. {feedback}\n")
        f.write("\nEvolution Statistics:\n")
        f.write(f"• Generations: {stats['generations_completed']}\n")
        f.write(f"• Best score: {stats['global_best_score']:.3f}\n")
        if "performance_metrics" in stats:
            perf = stats["performance_metrics"]
            f.write(f"• Runtime: {perf['runtime_minutes']:.2f} minutes\n")

    print("\nResults saved to 'results/' directory")
    print("• Chemistry solution: results/chemistry_example_solution.md")
    print("• Chemical data: results/chemistry_example_data.txt")
    print("• Analysis: results/chemistry_example_analysis.txt")

    return best_solution


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        exit(1)

    # Run the chemistry example
    main()
