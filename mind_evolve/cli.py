"""Command-line interface for Mind Evolution."""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import MindEvolution, create_evaluator, create_llm, setup_logging
from .core.models import MindEvolutionConfig, Problem
from .llm.prompt_manager import PromptManager
from .utils.config import ConfigManager

app = typer.Typer(
    name="mind-evolve",
    help="Mind Evolution: Evolutionary Search for LLM-based Problem Solving"
)
console = Console()


@app.command()
def run(
    problem_file: Path = typer.Argument(..., help="Path to problem definition file"),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="Configuration file path"),
    output_dir: Path = typer.Option(Path("results"), "--output", "-o", help="Output directory"),
    experiment_name: str | None = typer.Option(None, "--name", "-n", help="Experiment name"),
    llm_provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider (openai, anthropic, google)"),
    llm_model: str = typer.Option("gpt-4", "--model", "-m", help="LLM model name"),
    generations: int = typer.Option(10, "--generations", "-g", help="Number of generations"),
    islands: int = typer.Option(4, "--islands", "-i", help="Number of islands"),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="LLM temperature"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
) -> None:
    """Run Mind Evolution on a problem."""

    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    exp_name = experiment_name or f"experiment_{int(time.time())}"
    setup_logging(log_level=log_level, experiment_name=exp_name)

    console.print(f"[bold blue]Mind Evolution[/bold blue] - {exp_name}")
    console.print(f"Problem: {problem_file}")
    console.print(f"Provider: {llm_provider} ({llm_model})")
    console.print(f"Config: {generations} generations, {islands} islands")

    try:
        # Load configuration
        config_manager = ConfigManager()
        system_config = config_manager.load_system_config()

        if config_file and config_file.exists():
            exp_config = config_manager.load_experiment_config(config_file)
            evolution_config = exp_config.mind_evolution
        else:
            # Create default configuration
            evolution_config = MindEvolutionConfig(
                N_gens=generations,
                N_island=islands,
                temperature=temperature,
                model_name=llm_model,
            )

        # Load problem
        problem = load_problem_from_file(problem_file)

        # Initialize components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            task = progress.add_task("Initializing components...", total=None)

            # Create LLM interface
            api_key = config_manager.get_api_key(llm_provider)
            llm = create_llm(llm_provider, llm_model, api_key=api_key)

            # Create evaluator
            evaluator = create_evaluator("simple")  # Use simple evaluator as default

            # Create prompt manager
            prompt_manager = PromptManager()

            progress.update(task, description="Starting evolution...")

            # Run evolution
            mind_evolution = MindEvolution(
                config=evolution_config,
                llm=llm,
                evaluator=evaluator,
                prompt_manager=prompt_manager
            )

            progress.update(task, description="Running evolution...")

        # Run the actual evolution (without progress bar to avoid conflicts)
        console.print("\n[yellow]Running Mind Evolution...[/yellow]")

        import time
        start_time = time.time()

        best_solution = mind_evolution.solve(problem, exp_name)

        end_time = time.time()
        runtime = end_time - start_time

        # Display results
        display_results(best_solution, mind_evolution, runtime)

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export detailed results
        results_file = output_dir / f"{exp_name}_results.json"
        mind_evolution.export_results(str(results_file))

        # Save best solution
        solution_file = output_dir / f"{exp_name}_best_solution.txt"
        with open(solution_file, 'w') as f:
            f.write(f"Score: {best_solution.score}\n")
            f.write(f"Valid: {best_solution.is_valid()}\n")
            f.write(f"Feedback: {'; '.join(best_solution.feedback)}\n\n")
            f.write("Solution:\n")
            f.write(best_solution.content)

        console.print(f"\n[green]Results saved to {output_dir}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_config(
    output_path: Path = typer.Argument(..., help="Output path for configuration template"),
    format: str = typer.Option("yaml", "--format", "-f", help="Config format (yaml, json)"),
) -> None:
    """Create a configuration template."""

    config_manager = ConfigManager()

    if format.lower() not in ["yaml", "yml", "json"]:
        console.print(f"[red]Error: Unsupported format '{format}'. Use 'yaml' or 'json'.[/red]")
        raise typer.Exit(1)

    # Ensure correct file extension
    if format.lower() in ["yaml", "yml"]:
        output_path = output_path.with_suffix(".yml")
    else:
        output_path = output_path.with_suffix(".json")

    try:
        config_manager.create_experiment_config_template(output_path)
        console.print(f"[green]Configuration template created at {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error creating config: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    results_file: Path = typer.Argument(..., help="Path to results JSON file"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory for analysis"),
) -> None:
    """Analyze results from a completed experiment."""

    import json

    if not results_file.exists():
        console.print(f"[red]Error: Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    try:
        with open(results_file) as f:
            results = json.load(f)

        # Display analysis
        display_analysis(results)

        # Save detailed analysis if output directory specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            analysis_file = output_dir / "analysis_report.txt"
            with open(analysis_file, 'w') as f:
                f.write("Mind Evolution Analysis Report\n")
                f.write("=" * 40 + "\n\n")

                # Write key metrics
                if 'performance_metrics' in results:
                    f.write("Performance Metrics:\n")
                    for key, value in results['performance_metrics'].items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

                if 'convergence_analysis' in results:
                    f.write("Convergence Analysis:\n")
                    for key, value in results['convergence_analysis'].items():
                        f.write(f"  {key}: {value}\n")

            console.print(f"[green]Analysis saved to {analysis_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error analyzing results: {e}[/red]")
        raise typer.Exit(1)


def load_problem_from_file(problem_file: Path) -> Problem:
    """Load problem definition from file.
    
    Args:
        problem_file: Path to problem file
        
    Returns:
        Problem instance
    """
    import json

    if not problem_file.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_file}")

    if problem_file.suffix.lower() == ".json":
        with open(problem_file) as f:
            problem_data = json.load(f)
    elif problem_file.suffix.lower() in [".txt", ".md"]:
        # Simple text file - use content as description
        with open(problem_file) as f:
            content = f.read().strip()
        problem_data = {
            "title": problem_file.stem,
            "description": content,
        }
    else:
        raise ValueError(f"Unsupported problem file format: {problem_file.suffix}")

    return Problem(**problem_data)


def display_results(best_solution, mind_evolution, runtime: float) -> None:
    """Display evolution results.
    
    Args:
        best_solution: Best solution found
        mind_evolution: MindEvolution instance
        runtime: Total runtime in seconds
    """
    console.print("\n[bold green]Evolution Complete![/bold green]")

    # Create results table
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Runtime", f"{runtime:.2f} seconds")
    table.add_row("Best Score", f"{best_solution.score:.3f}")
    table.add_row("Valid Solution", "✅ Yes" if best_solution.is_valid() else "❌ No")
    table.add_row("Generation", str(best_solution.generation))
    table.add_row("Island", str(best_solution.island_id))

    # Get evolution statistics
    stats = mind_evolution.get_evolution_statistics()
    table.add_row("Generations", str(stats['generations_completed']))

    if 'performance_metrics' in stats:
        perf = stats['performance_metrics']
        table.add_row("Solutions/min", f"{perf.get('solutions_per_minute', 0):.1f}")
        table.add_row("Success Rate", f"{perf.get('success_rate', 0):.1f}%")

    console.print(table)

    # Display solution content
    console.print("\n[bold cyan]Best Solution:[/bold cyan]")
    console.print(best_solution.content)

    if best_solution.feedback:
        console.print("\n[bold yellow]Feedback:[/bold yellow]")
        for feedback in best_solution.feedback[:3]:  # Show top 3 feedback items
            console.print(f"• {feedback}")


def display_analysis(results: dict) -> None:
    """Display analysis of results.
    
    Args:
        results: Results dictionary
    """
    console.print("\n[bold blue]Results Analysis[/bold blue]")

    if 'performance_metrics' in results:
        console.print("\n[cyan]Performance Metrics:[/cyan]")
        perf = results['performance_metrics']

        for key, value in perf.items():
            if isinstance(value, float):
                console.print(f"  {key}: {value:.3f}")
            else:
                console.print(f"  {key}: {value}")

    if 'convergence_analysis' in results:
        console.print("\n[cyan]Convergence Analysis:[/cyan]")
        conv = results['convergence_analysis']

        for key, value in conv.items():
            if isinstance(value, float):
                console.print(f"  {key}: {value:.3f}")
            else:
                console.print(f"  {key}: {value}")


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
