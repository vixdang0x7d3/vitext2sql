"""
Tokenizer utilities for LLM clients.
Provides tokenizers that can be injected into LLMClient for token counting.

Usage: client = LLMClient(tokenizer=create_tokenizer("openai", "gpt-4"))
"""

import os
import re


class TokenizerInterface:
    """Base interface for tokenizers used by LLMClient"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs"""
        raise NotImplementedError

    def __call__(self, text: str) -> int:
        """Count tokens when called directly"""
        return len(self.encode(text, add_special_tokens=False))


class TikTokenizer(TokenizerInterface):
    """OpenAI tiktoken-based tokenizer"""

    def __init__(self, model_name: str):
        try:
            import tiktoken

            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback for unknown models
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            raise ImportError("tiktoken required for OpenAI models")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.encoding.encode(text)


class HuggingFaceTokenizer(TokenizerInterface):
    """Hugging Face transformers tokenizer"""

    def __init__(self, model_name: str, **kwargs):
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        except ImportError:
            raise ImportError("transformers required for HuggingFace models")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)


class GeminiTokenizer(TokenizerInterface):
    """Google Gemini tokenizer"""

    def __init__(self, model_name: str = "gemini-pro"):
        try:
            import google.generativeai as genai

            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError("google-generativeai required for Gemini models")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Gemini doesn't expose token IDs, return placeholder
        return list(range(self.__call__(text)))

    def __call__(self, text: str) -> int:
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception:
            return max(1, len(text) // 4)


class OpenSourceTokenizer(TokenizerInterface):
    """Tokenizer for open source models via HuggingFace"""

    # Model family mappings for common open source models
    MODEL_MAPPINGS = {
        "llama": "meta-llama/Llama-2-7b-hf",
        "llama-2": "meta-llama/Llama-2-7b-hf",
        "llama-3": "meta-llama/Meta-Llama-3-8B",
        "llama-3.1": "meta-llama/Meta-Llama-3.1-8B",
        "code-llama": "codellama/CodeLlama-7b-hf",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "mixtral": "mistralai/Mixtral-8x7B-v0.1",
        "qwen": "Qwen/Qwen2-7B",
        "qwen2": "Qwen/Qwen2-7B",
        "yi": "01-ai/Yi-6B",
        "deepseek": "deepseek-ai/deepseek-llm-7b-base",
        "phi": "microsoft/phi-2",
        "phi-3": "microsoft/Phi-3-mini-4k-instruct",
        "gemma": "google/gemma-7b",
        "starcoder": "bigcode/starcoder",
    }

    def __init__(self, model_name: str, fallback_ratio: float = 4.0):
        self.model_name = model_name
        self.fallback_ratio = fallback_ratio
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> HuggingFaceTokenizer | None:
        """Try to load appropriate HuggingFace tokenizer"""
        hf_model = self._get_hf_model_name()
        if hf_model:
            try:
                return HuggingFaceTokenizer(hf_model)
            except Exception:
                pass
        return None

    def _get_hf_model_name(self) -> str | None:
        """Map model name to HuggingFace model"""
        model_lower = self.model_name.lower()
        model_lower = re.sub(r"(-chat|-instruct|-base|-v\d+.*)", "", model_lower)

        # Exact match
        if model_lower in self.MODEL_MAPPINGS:
            return self.MODEL_MAPPINGS[model_lower]

        # Partial match
        for key, value in self.MODEL_MAPPINGS.items():
            if key in model_lower:
                return value

        return None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self.tokenizer:
            return self.tokenizer.encode(text, add_special_tokens)
        # Fallback - return placeholder tokens
        return list(range(self.__call__(text)))

    def __call__(self, text: str) -> int:
        if self.tokenizer:
            return self.tokenizer(text)
        # Character-based fallback
        return max(1, int(len(text) / self.fallback_ratio))


def create_tokenizer(provider: str, model_name: str, **kwargs) -> TokenizerInterface:
    """Factory function to create appropriate tokenizer"""
    provider = provider.lower()

    if provider in ["openai", "azure"]:
        return TikTokenizer(model_name)
    elif provider == "gemini":
        return GeminiTokenizer(model_name)
    elif provider == "huggingface":
        return HuggingFaceTokenizer(model_name, **kwargs)
    elif provider in ["openai_compat", "vllm", "ollama"]:
        return OpenSourceTokenizer(model_name, **kwargs)
    else:
        # Fallback - simple character counting
        class FallbackTokenizer(TokenizerInterface):
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return list(range(len(text) // 4))

            def __call__(self, text: str) -> int:
                return max(1, len(text) // 4)

        return FallbackTokenizer()
