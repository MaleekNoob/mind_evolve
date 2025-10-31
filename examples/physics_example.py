"""
Physics problem example demonstrating Mind Evolution on complex scientific challenges.

This example shows how to use Mind Evolution to solve advanced physics problems
that require deep understanding, mathematical derivation, and conceptual reasoning.
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
    """Run physics problem example with Mind Evolution."""

    # Setup logging
    setup_logging(log_level="INFO", experiment_name="physics_example")

    # Define a challenging physics problem
    problem = Problem(
        title="Quantum Tunneling and Barrier Penetration Analysis",
        description="""
        A particle with mass m and energy E approaches a rectangular potential barrier 
        of height V₀ and width a. The particle has energy E < V₀, so classically it 
        cannot pass through the barrier. However, quantum mechanics allows tunneling.
        
        Derive and analyze the complete quantum mechanical solution including:
        1. The time-independent Schrödinger equation in all three regions
        2. Boundary conditions and continuity requirements
        3. Transmission and reflection coefficients
        4. The dependence of tunneling probability on barrier parameters
        5. Physical interpretation and real-world applications
        Specific case: Consider an electron (m = 9.11 × 10⁻³¹ kg) with energy 
        E = 2.0 eV approaching a barrier with V₀ = 3.0 eV and width a = 0.5 nm.
        """,
        constraints=[
            "Provide complete mathematical derivation with all steps shown",
            "Include proper wave function forms in all three regions",
            "Apply boundary conditions correctly at x = 0 and x = a",
            "Derive transmission coefficient T and reflection coefficient R",
            "Calculate numerical values for the specific electron case",
            "Verify that R + T = 1 (probability conservation)",
            "Include physical interpretation of results",
            "Discuss quantum vs classical behavior",
            "Mention real-world applications (STM, tunnel diodes, etc.)",
            "Use proper physics notation and units throughout",
            "Include at least one energy diagram or conceptual illustration",
            "Solution should be at graduate-level physics depth",
        ],
        examples=[
            {
                "input": "Electron tunneling through 1 eV barrier, width 1 nm",
                "output": "Complete quantum mechanical analysis with transmission coefficient",
            }
        ],
    )

    # Create configuration optimized for physics problems
    config = MindEvolutionConfig(
        N_gens=10,  # More generations for complex derivations
        N_island=4,  # Multiple islands for different approaches
        N_convs=3,  # Balanced conversations
        N_seq=5,  # More refinement for mathematical accuracy
        temperature=0.6,  # Lower temperature for precise mathematical work
        model_name="gpt-4",  # Use most capable model for physics
        early_stopping=False,  # Don't stop early for complex problems
        enable_critic=True,
        enable_feedback=True,
        max_retries=3,
    )

    # Initialize components
    print("Initializing Mind Evolution for quantum physics challenge...")

    # Create LLM
    llm = create_llm("openai", config.model_name)

    # Create specialized evaluator for physics
    evaluator = create_evaluator(
        "constraint",
        constraint_weights={
            "mathematical_rigor": 2.5,  # Correct equations and derivations
            "physical_insight": 2.0,  # Deep physics understanding
            "completeness": 2.0,  # All required components
            "numerical_accuracy": 1.5,  # Correct calculations
            "notation": 1.0,  # Proper physics notation
            "interpretation": 1.5,  # Physical meaning explained
            "applications": 1.0,  # Real-world relevance
            "boundary_conditions": 2.0,  # Proper application of BCs
            "conservation_laws": 1.5,  # R + T = 1 verification
        },
    )

    # Create prompt manager for physics tasks
    prompt_manager = PromptManager(task_type="scientific")

    # Initialize Mind Evolution
    mind_evolution = MindEvolution(
        config=config, llm=llm, evaluator=evaluator, prompt_manager=prompt_manager
    )

    # Solve the physics problem
    print("Starting Mind Evolution on quantum tunneling problem...")
    print(f"Problem: {problem.title}")
    print(f"Configuration: {config.N_gens} generations, {config.N_island} islands")
    print("This may take several minutes due to mathematical complexity...")

    best_solution = mind_evolution.solve(problem, "physics_example")

    # Display results
    print("\n" + "=" * 80)
    print("QUANTUM PHYSICS CHALLENGE EVOLUTION COMPLETE!")
    print("=" * 80)
    print(f"Best Solution Score: {best_solution.score:.3f}")
    print(f"Valid Solution: {'Yes' if best_solution.is_valid() else 'No'}")
    print(f"Generation: {best_solution.generation}")
    print(f"Island: {best_solution.island_id}")

    print("\nKey Physics Insights:")
    for feedback in best_solution.feedback[:5]:
        print(f"• {feedback}")

    print("\nBest Physics Solution:")
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
    mind_evolution.export_results("results/physics_example_results.json")

    # Save best solution as physics document
    with open("results/physics_example_solution.md", "w") as f:
        f.write(f"# Quantum Tunneling Analysis - Score: {best_solution.score:.3f}\n\n")
        f.write("*Generated by Mind Evolution*\n\n")
        f.write("## Problem Statement\n")
        f.write(f"{problem.description}\n\n")
        f.write("## Solution\n\n")
        f.write(best_solution.content)
        f.write("\n\n## Evolution Feedback\n\n")
        for i, feedback in enumerate(best_solution.feedback, 1):
            f.write(f"{i}. {feedback}\n")

    # Save LaTeX version for academic use
    with open("results/physics_example_solution.tex", "w") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{amsmath,amssymb,physics}\n")
        f.write("\\title{Quantum Tunneling Analysis}\n")
        f.write("\\author{Mind Evolution System}\n")
        f.write("\\begin{document}\n")
        f.write("\\maketitle\n\n")
        # Convert markdown-style content to basic LaTeX
        latex_content = best_solution.content.replace("**", "\\textbf{").replace(
            "**", "}"
        )
        f.write(latex_content)
        f.write("\n\\end{document}\n")

    # Save analysis
    with open("results/physics_example_analysis.txt", "w") as f:
        f.write("Quantum Physics Challenge Analysis\n")
        f.write("==================================\n\n")
        f.write(f"Problem: {problem.title}\n")
        f.write(f"Best Score: {best_solution.score:.3f}\n")
        f.write(f"Generation Found: {best_solution.generation}\n")
        f.write(f"Island: {best_solution.island_id}\n\n")
        f.write("Physics Quality Assessment:\n")
        for i, feedback in enumerate(best_solution.feedback, 1):
            f.write(f"{i}. {feedback}\n")
        f.write("\nEvolution Statistics:\n")
        f.write(f"• Generations: {stats['generations_completed']}\n")
        f.write(f"• Best score: {stats['global_best_score']:.3f}\n")
        if "performance_metrics" in stats:
            perf = stats["performance_metrics"]
            f.write(f"• Runtime: {perf['runtime_minutes']:.2f} minutes\n")

    print("\nResults saved to 'results/' directory")
    print("• Physics solution: results/physics_example_solution.md")
    print("• LaTeX version: results/physics_example_solution.tex")
    print("• Analysis: results/physics_example_analysis.txt")

    return best_solution


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        exit(1)

    # Run the physics example
    main()
