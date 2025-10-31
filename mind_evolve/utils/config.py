"""Configuration management for Mind Evolution."""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator

from ..core.models import MindEvolutionConfig


class SystemConfig(BaseModel):
    """System-level configuration."""

    # Paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default="data")
    logs_dir: Path = Field(default="logs")
    checkpoints_dir: Path = Field(default="checkpoints")
    cache_dir: Path = Field(default="cache")

    # API Keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # Logging
    log_level: str = Field(default="INFO")
    log_to_file: bool = Field(default=True)
    log_format: str = Field(default="{time} | {level} | {module} | {message}")

    # Performance
    max_workers: int = Field(default=4, ge=1, le=32)
    enable_parallel_evaluation: bool = Field(default=True)
    cache_evaluations: bool = Field(default=True)

    # Safety
    max_tokens_per_request: int = Field(default=4096, ge=100, le=32000)
    request_timeout: int = Field(default=60, ge=10, le=300)
    max_retries: int = Field(default=3, ge=1, le=10)

    @validator('data_dir', 'logs_dir', 'checkpoints_dir', 'cache_dir')
    def make_paths_absolute(cls, v, values):
        """Convert relative paths to absolute paths."""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute() and 'project_root' in values:
            return values['project_root'] / v
        return v

    def create_directories(self) -> None:
        """Create necessary directories."""
        for dir_path in [self.data_dir, self.logs_dir, self.checkpoints_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExperimentConfig(BaseModel):
    """Configuration for a specific experiment."""

    name: str = Field(..., description="Experiment name")
    description: str = Field(default="", description="Experiment description")

    # Core Mind Evolution config
    mind_evolution: MindEvolutionConfig = Field(default_factory=MindEvolutionConfig)

    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider (openai, anthropic, google)")
    llm_model: str = Field(default="gpt-4", description="LLM model name")

    # Evaluator Configuration
    evaluator_type: str = Field(default="simple", description="Evaluator type")
    evaluator_config: dict[str, Any] = Field(default_factory=dict)

    # Problem Configuration
    problem_type: str = Field(default="general", description="Type of problem to solve")
    problem_config: dict[str, Any] = Field(default_factory=dict)

    # Output Configuration
    save_solutions: bool = Field(default=True)
    save_conversations: bool = Field(default=True)
    save_statistics: bool = Field(default=True)
    output_format: str = Field(default="json", description="Output format (json, csv, both)")

    # Experiment Control
    random_seed: int | None = Field(default=None, description="Random seed for reproducibility")
    max_runtime_minutes: int | None = Field(default=None, description="Maximum runtime")

    @validator('llm_provider')
    def validate_llm_provider(cls, v):
        """Validate LLM provider."""
        valid_providers = ["openai", "anthropic", "google"]
        if v not in valid_providers:
            raise ValueError(f"LLM provider must be one of {valid_providers}")
        return v


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: Path | None = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.system_config: SystemConfig | None = None
        self.experiment_config: ExperimentConfig | None = None

    def load_system_config(self, config_path: Path | None = None) -> SystemConfig:
        """Load system configuration.
        
        Args:
            config_path: Optional config file path
            
        Returns:
            Loaded system configuration
        """
        config_data = {}

        # Load from file if provided
        if config_path and config_path.exists():
            config_data = self._load_config_file(config_path)

        # Override with environment variables
        env_overrides = self._load_env_config()
        config_data.update(env_overrides)

        # Create and validate config
        self.system_config = SystemConfig(**config_data)

        # Create necessary directories
        self.system_config.create_directories()

        return self.system_config

    def load_experiment_config(self, config_path: Path) -> ExperimentConfig:
        """Load experiment configuration.
        
        Args:
            config_path: Path to experiment config file
            
        Returns:
            Loaded experiment configuration
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config file not found: {config_path}")

        config_data = self._load_config_file(config_path)
        self.experiment_config = ExperimentConfig(**config_data)

        return self.experiment_config

    def _load_config_file(self, config_path: Path) -> dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        import json

        import yaml

        if config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        elif config_path.suffix.lower() == '.json':
            with open(config_path) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def _load_env_config(self) -> dict[str, Any]:
        """Load configuration from environment variables.
        
        Returns:
            Environment configuration dictionary
        """
        env_config = {}

        # API Keys
        if os.getenv('OPENAI_API_KEY'):
            env_config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('ANTHROPIC_API_KEY'):
            env_config['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
        if os.getenv('GOOGLE_API_KEY'):
            env_config['google_api_key'] = os.getenv('GOOGLE_API_KEY')

        # Logging
        if os.getenv('LOG_LEVEL'):
            env_config['log_level'] = os.getenv('LOG_LEVEL')

        # Performance
        if os.getenv('MAX_WORKERS'):
            env_config['max_workers'] = int(os.getenv('MAX_WORKERS'))

        return env_config

    def save_config(self, config: BaseModel, output_path: Path) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration object
            output_path: Output file path
        """
        import json

        import yaml

        config_dict = config.model_dump()

        # Convert Path objects to strings for serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        config_dict = convert_paths(config_dict)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() in ['.yml', '.yaml']:
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

    def create_experiment_config_template(self, output_path: Path) -> None:
        """Create a template experiment configuration file.
        
        Args:
            output_path: Path to save template
        """
        template_config = ExperimentConfig(
            name="example_experiment",
            description="Example Mind Evolution experiment configuration",
        )

        self.save_config(template_config, output_path)

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for specified provider.
        
        Args:
            provider: LLM provider name
            
        Returns:
            API key if available
        """
        if not self.system_config:
            self.load_system_config()

        key_mapping = {
            'openai': self.system_config.openai_api_key,
            'anthropic': self.system_config.anthropic_api_key,
            'google': self.system_config.google_api_key,
        }

        return key_mapping.get(provider)
