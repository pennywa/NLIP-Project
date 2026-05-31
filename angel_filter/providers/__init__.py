"""Provider adapters.

Each adapter takes a user query and returns a list of ProviderResult objects.
All adapters implement the same async interface so they can be fanned out in parallel.
"""

from angel_filter.providers.base import BaseProvider, ProviderResult
from angel_filter.providers.brave import BraveProvider
from angel_filter.providers.duckduckgo import DuckDuckGoProvider
from angel_filter.providers.gemini import GeminiProvider
from angel_filter.providers.mock import MockProvider
from angel_filter.providers.ollama_provider import OllamaProvider
from angel_filter.providers.openai_provider import OpenAIProvider
from angel_filter.providers.watsonx import WatsonXProvider

__all__ = [
    "BaseProvider",
    "ProviderResult",
    "BraveProvider",
    "DuckDuckGoProvider",
    "GeminiProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "WatsonXProvider",
]
