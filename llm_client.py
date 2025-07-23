import sys
import os

from abc import ABC, abstractmethod
from dataclasses import dataclass
from utils import extract_all_blocks

from openai import OpenAI, AzureOpenAI


@dataclass
class ChatMessage:
    """Standardized chat message format across all clients"""

    role: str
    content: str


class LLMClient(ABC):
    def __init__(
        self,
        model: str = "o4-mini",
        temperature: float = 1,
    ):
        self.model = model
        self.temperature = temperature
        self.messages: list[ChatMessage] = []

    @abstractmethod
    def _generate_respond():
        pass

    def get_response(self, prompt: str) -> str:
        self.messages.append(ChatMessage(role="user", content=prompt))
        response_content = self._generate_response(self.messages)
        self.messages.append(ChatMessage(role="assistant", content=response_content))
        return response_content

    def get_model_response(
        self, prompt: str, code_format: str | None = None
    ) -> list[str]:
        """Get response and extraact code blocks with retry logic"""
        code_blocks = []
        max_try = 3
        while code_blocks == [] and max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
            code_blocks = extract_all_blocks(response, code_format)

        if max_try == 0 or code_blocks == []:
            print(
                f"get_model_response() exit, max_try: {max_try}, code_block: {code_blocks}"
            )
            sys.exit(0)

        return code_blocks

    def get_model_response_txt(self, prompt: str) -> str:
        """Get text response with retry logic"""

        max_try = 3
        while max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                continue
            break

        if max_try == 0:
            print(f"get_model_response_txt() exit, max_try: {max_try}")
            sys.exit(0)

        return response

    def get_message_len(self):
        """Get conversation statistics"""
        return {
            "prompt_len": sum(
                len(item["content"]) for item in self.messages if item["role"] == "user"
            ),
            "response_len": sum(
                len(item["content"])
                for item in self.messages
                if item["role"] == "assistant"
            ),
            "num_calls": len(self.messages) // 2,
        }

    def init_message(self):
        """Reset conversation history"""
        self.messages = []


class OpenAIClient(LLMClient):
    """OpenAI/Azure OpenAI chat client implementation"""

    def __init__(
        self, model: str = "o4-mini", temperature: float = 1.0, azure: bool = False
    ):
        super().__init__(model, temperature)
        self.azure = azure
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI or Azure OpenAI client"""
        try:
            from openai import OpenAI, AzureOpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAI client")

        if not self.azure:
            if self.model in ["o1-preview", "o1-mini"]:
                self.client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    api_version="2024-12-01-preview",
                )
            elif self.model in ["deepseek-reasoner"]:
                self.client = OpenAI(
                    base_url="https://api.deepseek.com",
                    api_key=os.environ.get("DS_API_KEY"),
                )
            else:
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            if self.model in ["o1-preview", "o1-mini", "o3", "o4-mini"]:
                api_version = "2024-12-01-preview"
            elif self.model in ["o3-pro"]:
                api_version = "2025-03-01-preview"
            else:
                api_version = "2024-05-01-preview"

            self.client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_KEY"),
                api_version=api_version,
            )

    def _generate_response(self, messages: list[ChatMessage]) -> str:
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        if self.model == "o3-pro":
            response = self.client.response.create(
                model=self.model, input=openai_messages, temperature=self.temperature
            )
            return response.output_text
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=openai_messages, temperature=self.temperature
            )
            return response.choices[0].message.content


class HuggingFaceClient(LLMClient):
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        device: str = "auto",
        max_new_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__(model, temperature)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.kwargs = kwargs
        self._init_client()

    def _init_client(self):
        """Initialize Hugging Face model and tokenizer"""

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch packages required for HuggingFaceClient"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, **self.kwargs)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model,
            device=self.device,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            **self.kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _generate_response(self, messages: list[ChatMessage]) -> str:
        """Generate response using Hugging Face model"""

        # Format messages as conversation
        conversation = ""
        for msg in messages:
            if msg.role == "user":
                conversation += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n"
        conversation += "Assistant: "

        inputs = self.tokenizer.encode(conversation, return_tensors="pt")
        if self.device != "cpu":
            inputs = inputs.to(self.model_obj.device)

        with torch.no_grad():  # ty: ignore # noqa: F821
            outputs = self.model_obj.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][len(inputs[0]) :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()


class LlamaClient(LLMClient):
    def __init__(
        self,
        model_path: str,
        temperature: float = 1.0,
        n_ctx: int = 4096,
        n_gpu_layers=0,
        **kwargs,
    ):
        super().__init__(model_path, temperature)
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.kwargs = kwargs
        self._init_client()

    def _init_client(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python package required for LlamaCpp client")

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
            **self.kwargs,
        )

    def _generate_response(self, messages: list[ChatMessage]):
        conversation = ""
        for msg in messages:
            if msg.role == "user":
                conversation += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n"
        conversation += "Assistant: "

        output = self.llm(
            conversation,
            max_tokens=2048,
            temperature=self.temperature,
            stop=["User:", "\n\n"],
        )

        return output["choice"][0]["text"].strip()


class LLMClientFactory:
    """Factory for creating llm clients"""

    @staticmethod
    def __init__(client_type: str, **kwargs) -> LLMClient:
        client_type = client_type.lower()

        if client_type in ["openai", "azure"]:
            return OpenAIClient(**kwargs)
        elif client_type in ["huggingface", "hf", "transformers"]:
            return HuggingFaceClient(**kwargs)  # ty: ignore
        elif client_type in ["llamacpp", "llama_cpp", "llama-cpp"]:
            return LlamaClient(**kwargs)  # ty: ignore
        else:
            raise ValueError(f"Unsupported client type: {client_type}")
