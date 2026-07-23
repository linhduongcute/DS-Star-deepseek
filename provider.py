import os
from abc import ABC, abstractmethod
import google.generativeai as genai
import openai
import ollama

class ModelProvider(ABC):
    """Abstract base class for model providers."""

    def _record_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.total_input_tokens = getattr(self, "total_input_tokens", 0) + (input_tokens or 0)
        self.total_output_tokens = getattr(self, "total_output_tokens", 0) + (output_tokens or 0)

    def get_usage(self) -> dict:
        input_tokens = getattr(self, "total_input_tokens", 0)
        output_tokens = getattr(self, "total_output_tokens", 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    @classmethod
    @abstractmethod
    def provider_instance(cls, model_name: str) -> bool:
        """Check if this provider can handle the given model name."""
        pass

    @property
    @abstractmethod
    def env_var_name(self) -> str:
        """The name of the environment variable required for the API key."""
        pass

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """Generates content based on the prompt."""
        pass


class GeminiProvider(ModelProvider):
    """Provider for Google's Gemini models."""
    
    def __init__(self, config_api_key: str, model_name: str):
        # In GeminiProvider the order is first config_api_key for backward compatibility
        self.api_key = config_api_key or os.getenv(self.env_var_name)
        if not self.api_key:
            raise ValueError(f"Missing API key for {model_name}. Env var = {self.env_var_name}.")

        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        """For backward compatibility, Gemini is the default provider."""
        return True

    @property
    def env_var_name(self) -> str:
        return "GEMINI_API_KEY"
        
    def generate_content(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self._record_usage(
                getattr(usage, "prompt_token_count", 0),
                getattr(usage, "candidates_token_count", 0),
            )
        return response.text


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""
    
    def __init__(self, config_api_key: str, model_name: str):
        self.api_key = os.getenv(self.env_var_name, config_api_key)
        if not self.api_key:
            raise ValueError(f"Missing API key for {model_name}. Env var = {self.env_var_name}.")

        self.model_name = model_name
        self.client = openai.OpenAI(api_key=self.api_key)

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return model_name.startswith("gpt") or model_name.startswith("o1")

    @property
    def env_var_name(self) -> str:
        return "OPENAI_API_KEY"
        
    def generate_content(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        if response.usage:
            self._record_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content


class OpenRouterProvider(ModelProvider):
    """Provider for models served through OpenRouter's OpenAI-compatible API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config_api_key: str, model_name: str):
        self.api_key = os.getenv(self.env_var_name) or config_api_key
        if not self.api_key:
            raise ValueError(
                f"Missing API key for {model_name}. Env var = {self.env_var_name}."
            )

        self.model_name = model_name.removeprefix("openrouter/")
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", self.BASE_URL),
        )

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        # The explicit prefix also allows OpenRouter models whose slugs overlap
        # with another provider, e.g. openrouter/openai/gpt-4.1.
        return model_name.startswith("openrouter/") or model_name.startswith("deepseek/")

    @property
    def env_var_name(self) -> str:
        return "OPENROUTER_API_KEY"

    def generate_content(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.usage:
            self._record_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content


class OllamaProvider(ModelProvider):
    """Provider for Ollama models."""

    def __init__(self, config_api_key: str, model_name: str):
        self.api_key = os.getenv(self.env_var_name, config_api_key)
        self.model_name = model_name.lstrip("ollama/")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = ollama.Client(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            headers=headers
        )

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return model_name.startswith("ollama/")

    @property
    def env_var_name(self) -> str:
        return "OLLAMA_API_KEY"

    def generate_content(self, prompt: str) -> str:
        response = self.client.chat(
            self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        self._record_usage(
            getattr(response, "prompt_eval_count", 0),
            getattr(response, "eval_count", 0),
        )
        return response.message.content
