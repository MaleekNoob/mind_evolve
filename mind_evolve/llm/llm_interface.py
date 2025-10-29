"""LLM interface abstraction for Mind Evolution."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Response from LLM API call."""
    
    content: str
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""
    
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize LLM interface.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        
    @abstractmethod
    async def generate_async(self, 
                           prompt: str, 
                           temperature: float = 1.0,
                           max_tokens: Optional[int] = None,
                           **kwargs: Any) -> LLMResponse:
        """Generate response asynchronously.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with generated content
        """
        pass
        
    def generate(self, 
                prompt: str, 
                temperature: float = 1.0,
                max_tokens: Optional[int] = None,
                **kwargs: Any) -> str:
        """Generate response synchronously.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature 
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text content
        """
        import asyncio
        
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, run in thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._sync_generate, prompt, temperature, max_tokens, **kwargs)
                response = future.result()
        except RuntimeError:
            # No event loop running, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    self.generate_async(prompt, temperature, max_tokens, **kwargs)
                )
            finally:
                loop.close()
                
        return response.content
    
    def _sync_generate(self, prompt: str, temperature: float = 1.0, 
                      max_tokens: Optional[int] = None, **kwargs: Any) -> Any:
        """Helper method to run async generation in a new event loop."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate_async(prompt, temperature, max_tokens, **kwargs)
            )
        finally:
            loop.close()
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        pass


class OpenAILLM(BaseLLM):
    """OpenAI GPT interface."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize OpenAI LLM.
        
        Args:
            model_name: OpenAI model name
            api_key: API key (if None, uses environment variable)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required for OpenAI LLM")
            
    async def generate_async(self, 
                           prompt: str, 
                           temperature: float = 1.0,
                           max_tokens: Optional[int] = None,
                           **kwargs: Any) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                finish_reason=choice.finish_reason,
                usage=response.usage.model_dump() if response.usage else {},
                metadata={"model": self.model_name}
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model": self.model_name,
            "type": "chat",
        }


class AnthropicLLM(BaseLLM):
    """Anthropic Claude interface."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize Anthropic LLM.
        
        Args:
            model_name: Anthropic model name
            api_key: API key (if None, uses environment variable)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required for Anthropic LLM")
            
    async def generate_async(self, 
                           prompt: str, 
                           temperature: float = 1.0,
                           max_tokens: Optional[int] = None,
                           **kwargs: Any) -> LLMResponse:
        """Generate response using Anthropic API."""
        try:
            # Default max_tokens for Claude
            if max_tokens is None:
                max_tokens = 4096
                
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            content = ""
            if response.content:
                content = response.content[0].text if response.content else ""
                
            return LLMResponse(
                content=content,
                finish_reason=response.stop_reason,
                usage={"input_tokens": response.usage.input_tokens, 
                      "output_tokens": response.usage.output_tokens} if response.usage else {},
                metadata={"model": self.model_name}
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        return {
            "provider": "anthropic",
            "model": self.model_name,
            "type": "chat",
        }


class GoogleLLM(BaseLLM):
    """Google Gemini interface."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-001", api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize Google LLM.
        
        Args:
            model_name: Google model name
            api_key: API key (if None, uses environment variable)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        
        try:
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError("google-generativeai package required for Google LLM")
            
    async def generate_async(self, 
                           prompt: str, 
                           temperature: float = 1.0,
                           max_tokens: Optional[int] = None,
                           **kwargs: Any) -> LLMResponse:
        """Generate response using Google Gemini API."""
        try:
            # Configure generation parameters
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
                
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config,
                **kwargs
            )
            
            return LLMResponse(
                content=response.text or "",
                finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
                usage={"input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                      "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0},
                metadata={"model": self.model_name}
            )
            
        except Exception as e:
            raise RuntimeError(f"Google API error: {e}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get Google model information."""
        return {
            "provider": "google",
            "model": self.model_name,
            "type": "generative",
        }


def create_llm(provider: str, model_name: str, api_key: Optional[str] = None, **kwargs: Any) -> BaseLLM:
    """Factory function to create LLM instances.
    
    Args:
        provider: LLM provider (openai, anthropic, google)
        model_name: Model name
        api_key: API key
        **kwargs: Additional configuration
        
    Returns:
        Initialized LLM instance
    """
    providers = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "google": GoogleLLM,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")
        
    return providers[provider](model_name=model_name, api_key=api_key, **kwargs)