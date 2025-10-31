# Mind Evolution: Evolutionary Search for LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Mind Evolution is a novel approach to improve Large Language Model (LLM) problem-solving by using evolutionary search strategies during inference time. Instead of generating single answers, it evolves populations of candidate solutions over multiple generations, combining ideas through crossover and refining them through mutation.

## 🧬 Key Features

- **Evolutionary Search**: Solutions evolve over generations through selection, crossover, and mutation
- **Island Model**: Multiple independent populations with periodic migration for diversity
- **Refinement through Critical Conversation (RCC)**: Critic-author dialog for solution improvement
- **LLM Agnostic**: Support for OpenAI, Anthropic, and Google models
- **Modular Architecture**: Clean, extensible design with comprehensive testing
- **Rich CLI**: Easy-to-use command-line interface with progress tracking
- **Comprehensive Logging**: Detailed experiment tracking and analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mind-evolve.git
cd mind-evolve

# Install with UV (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Basic Usage

```python
from mind_evolve import MindEvolution, Problem, MindEvolutionConfig, create_llm, create_evaluator
from mind_evolve.llm import PromptManager

# Define your problem
problem = Problem(
    title="Creative Writing Task",
    description="Write a short story about AI discovering emotions",
    constraints=[
        "200-300 words long",
        "Feature an AI character",
        "Explore themes of consciousness"
    ]
)

# Configure evolution parameters
config = MindEvolutionConfig(
    N_gens=10,        # 10 generations
    N_island=4,       # 4 islands
    N_convs=5,        # 5 conversations per island
    temperature=0.8   # LLM temperature
)

# Initialize components
llm = create_llm("openai", "gpt-4")
evaluator = create_evaluator("constraint")
prompt_manager = PromptManager()

# Run evolution
mind_evolution = MindEvolution(config, llm, evaluator, prompt_manager)
best_solution = mind_evolution.solve(problem)

print(f"Best solution (score: {best_solution.score:.3f}):")
print(best_solution.content)
```

### Command Line Usage

```bash
# Run evolution on a problem file
mind-evolve run problem.txt --generations 10 --islands 4

# Create configuration template
mind-evolve create-config config.yml

# Analyze results
mind-evolve analyze results.json --output analysis/
```

## 📖 Core Concepts

### Evolution Workflow

1. **Initialization**: Generate initial population using LLM prompts
2. **Selection**: Choose parent solutions using Boltzmann tournament selection
3. **Crossover**: Combine insights from multiple parent solutions
4. **Mutation**: Refine solutions through critic-author dialog
5. **Migration**: Share solutions between islands periodically
6. **Reset**: Replace weak islands with diverse elite solutions

### Island Model

Multiple independent populations evolve in parallel:

- **Diversity**: Different islands explore different solution spaces
- **Migration**: Periodic sharing of best solutions between islands
- **Reset**: Weak islands are periodically reset with elite solutions
- **Parallel Evolution**: Islands evolve independently for most generations

### Refinement through Critical Conversation (RCC)

Solutions are improved through a two-step dialog:

1. **Critic**: Analyzes current solution and identifies weaknesses
2. **Author**: Generates improved solution based on criticism

## 🏗️ Architecture

```
mind_evolve/
├── core/                  # Core evolutionary components
│   ├── evolutionary_engine.py    # Main evolution orchestrator
│   ├── island_model.py           # Multi-island management
│   ├── models.py                 # Pydantic data models
│   ├── population.py             # Population management
│   └── selection.py              # Selection algorithms
├── llm/                   # LLM interface and prompt management
│   ├── llm_interface.py          # Abstract LLM interface
│   ├── prompt_manager.py         # Prompt templates and formatting
│   └── conversation_manager.py   # Multi-turn dialog handling
├── evaluation/            # Solution evaluation
│   ├── evaluator_base.py         # Abstract evaluator interface
│   ├── feedback_generator.py     # Textual feedback generation
│   └── scoring.py                # Scoring utilities
├── operators/             # Genetic operators
│   ├── initialization.py         # Population initialization
│   ├── crossover.py              # Parent combination
│   ├── mutation.py               # Solution refinement
│   └── migration.py              # Inter-island transfer
└── utils/                 # Utilities
    ├── config.py                 # Configuration management
    ├── logging.py                # Structured logging
    └── metrics.py                # Performance metrics
```

## 🔧 Configuration

Mind Evolution uses Pydantic for configuration management. Key parameters:

```python
config = MindEvolutionConfig(
    # Generation control
    N_gens=10,                 # Maximum generations

    # Island model
    N_island=4,                # Number of islands
    N_reset_interval=3,        # Generations between resets
    N_reset=2,                 # Islands to reset each time

    # Population structure
    N_convs=5,                 # Conversations per island per generation
    N_seq=4,                   # Sequential refinement turns

    # Genetic operators
    N_parent=5,                # Max parents for crossover
    Pr_no_parents=1/6,         # Probability of zero parents (mutation only)

    # LLM settings
    model_name="gpt-4",        # LLM model name
    temperature=1.0,           # LLM temperature

    # Optimization
    early_stopping=True,       # Stop when valid solution found
    enable_critic=True,        # Use critic-author dialog
    enable_feedback=True,      # Include textual feedback
)
```

## 📊 Examples

### Creative Writing

Evolve creative stories, poems, or scripts with thematic constraints.

### Problem Solving

Solve complex reasoning problems through evolutionary refinement.

### Code Generation

Generate and refine code solutions with correctness constraints.

### Research Tasks

Develop research hypotheses or experimental designs.

## 🧪 Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run type checking
mypy mind_evolve

# Format code
black mind_evolve
ruff mind_evolve
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mind_evolve --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v
```

## 📚 Research Background

Mind Evolution is based on research in evolutionary computation and large language models:

1. **Evolutionary Algorithms**: Uses selection, crossover, and mutation operators adapted for text generation
2. **Island Model**: Maintains diversity through multiple independent populations
3. **LLM Prompting**: Leverages advanced prompting techniques for solution generation and refinement
4. **Critical Conversation**: Implements structured self-reflection for solution improvement

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation]([https://mind-evolve.readthedocs.io/](https://github.com/MaleekNoob/mind_evolve))
- [Paper](https://arxiv.org/abs/2501.09891)
- [Examples](examples/)
- [API Reference](docs/api.md)

## 📧 Contact

Research Team - [maleekhussainalii@gmail.com](maleekhussainalii@gmail.com)

Project Link: [https://github.com/your-org/mind-evolve](https://github.com/MaleekNoob/mind_evolve)

## 🙏 Acknowledgments

- OpenAI for GPT models and API
- Anthropic for Claude models
- Google for Gemini models
- The evolutionary computation research community
- Contributors and beta testers
