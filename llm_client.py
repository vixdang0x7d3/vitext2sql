import os
import requests

from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai import OpenAI, AzureOpenAI
# from llama_cpp import Llama

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils import extract_all_blocks
from tokenizer import TokenizerInterface

from dotenv import load_dotenv
import os
import sys

@dataclass
class ChatMessage:
    """Standardized chat message format across all clients"""

    role: str
    content: str


class LLMClient(ABC):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1,
        tokenizer: TokenizerInterface | None = None,
        max_context_length: int = 4096,
        context_buffer_ratio: float = 0.1,
        preserve_recent_exchanges: int = 2,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.messages: list[ChatMessage] = []
        self.messages_trimmed_count = 0
        self.preserve_recent_exchanges = preserve_recent_exchanges
        self.effective_max_tokens = int(max_context_length * (1 - context_buffer_ratio))

        self.system_prompt = system_prompt
        if system_prompt:
            self.messages.append(ChatMessage(role="system", content=system_prompt))

    def _count_with_custom_tokenizer(self, text: str) -> int:
        """Count tokens with provider-specific optimization"""
        if hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        elif callable(self.tokenizer):
            return self.tokenizer(text)

        return max(1, len(text) // 4)

    def _count_tokens(self, text: str) -> int:
        """Use injected tokenizer"""
        if hasattr(self, "tokenizer") and self.tokenizer:
            return self._count_with_custom_tokenizer(text)

        # Try provider-specific counting
        if hasattr(self, "_provider_count_tokens"):
            return self._provider_count_tokens(text)

        return max(1, len(text) // 4)

    def _estimate_message_tokens(self, messages: list[ChatMessage]) -> int:
        """Estimate total tokens for message list"""
        total = 0
        for msg in messages:
            # Count content tokens
            total += self._count_tokens(msg.content)
            # Add overhead for role/formatting (typically 2-4 tokens per message)
            total += 3

        # Add overhead for conversation structure
        total += 10
        return total

    def _get_preserved_messages(self) -> list[ChatMessage]:
        """Get messages that should always be preserved"""
        preserved = []

        # Always preserve system message
        if self.messages and self.messages[0].role == "system":
            preserved.append(self.messages[0])

        non_system_messages = [m for m in self.messages if m.role != "system"]

        # Take last N*2 messages (N user-assiten)
        recent_count = min(len(non_system_messages), self.preserve_recent_exchanges * 2)
        preserved.extend(non_system_messages[-recent_count:])

        return preserved

    def _get_trimmable_messages(
        self, preserved_messages: list[ChatMessage]
    ) -> list[ChatMessage]:
        """Get messages that can be trimmed"""
        preserved_set = set(id(msg) for msg in preserved_messages)
        return [msg for msg in self.messages if id(msg) not in preserved_set]

    def _trim_to_fit(
        self,
        messages: list[ChatMessage],
        target_tokens: int,
    ) -> list[ChatMessage]:
        """Trim messages to fit, preserving conversation pairs"""
        if not messages:
            return []

        result = []
        current_tokens = 0
        i = len(messages) - 1

        # Work backwards, keeping complete user-assistant pairs
        while i >= 0:
            if (
                messages[i].role == "assistant"
                and i > 0
                and messages[i - 1].role == "user"
            ):
                # Try to add user-assistant pair
                pair_tokens = (
                    self._count_tokens(messages[i - 1].content)
                    + self._count_tokens(messages[i].content)
                    + 6  # +6 for formatting
                )

                if current_tokens + pair_tokens <= target_tokens:
                    result.insert(0, messages[i - 1])
                    result.insert(1, messages[i])
                    current_tokens += pair_tokens
                    i -= 2
                else:
                    break
            else:
                # Single message
                msg_tokens = self._count_tokens(messages[i].content) + 3
                if current_tokens + msg_tokens <= target_tokens:
                    result.insert(0, messages[i])
                    current_tokens += msg_tokens
                    i -= 1
                else:
                    break

        return result

    def _reconstruct_messages(
        self, preserved: list[ChatMessage], trimmed: list[ChatMessage]
    ) -> list[ChatMessage]:
        """Reconstruct message list maintaining chronological order"""
        result = []

        # Add system message if present
        if preserved and preserved[0].role == "system":
            result.append(preserved[0])
            preserved = preserved[1:]

        # Add trimmed messages (middle part)
        result.extend(trimmed)

        # Add preserved recent messages
        result.extend(preserved)

        return result

    def _trim_context_window(self) -> None:
        """Trim messages to fit within context window"""
        current_tokens = self._estimate_message_tokens(self.messages)

        if current_tokens <= self.effective_max_tokens:
            return

        original_message_count = len(self.messages)

        # Preserve system message and recent exchanges
        preserved_messages = self._get_preserved_messages()
        trimmable_messages = self._get_trimmable_messages(preserved_messages)

        # Calculate tokens for preserved messages
        preserved_tokens = self._estimate_message_tokens(preserved_messages)
        available_tokens = self.effective_max_tokens - preserved_tokens

        # Trim from oldest trimmable messages
        trimmed_messages = self._trim_to_fit(trimmable_messages, available_tokens)

        # Reconstruct message list
        self.messages = self._reconstruct_messages(preserved_messages, trimmed_messages)

        messages_removed = original_message_count - len(self.messages)
        self.messages_trimmed_count += messages_removed

    @abstractmethod
    def _generate_response():
        pass

    def get_response(self, prompt: str) -> str:
        self.messages.append(ChatMessage(role="user", content=prompt))
        # breakpoint()
        self._trim_context_window()
        response_content = self._generate_response(self.messages)
        self.messages.append(ChatMessage(role="assistant", content=response_content))
        return response_content

    def get_model_response(
        self, prompt: str, code_format: str | None = None
    ) -> list[str]:
        """Get response and extract code blocks with retry logic"""
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
            raise RuntimeError(
                f"Failed to get valid response after retries. max_try: {max_try}, code_block: {code_blocks}"
            )

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
            raise RuntimeError(
                f"Failed to get text response after retries. max_try: {max_try}"
            )

        return response

    def get_message_len(self):
        """Get conversation statistics with context info"""
        return {
            "prompt_len": sum(
                len(item.content) for item in self.messages if item.role == "user"
            ),
            "response_len": sum(
                len(item.content) for item in self.messages if item.role == "assistant"
            ),
            "num_calls": len([m for m in self.messages if m.role == "user"]),
            "estimated_tokens": self._estimate_message_tokens(self.messages),
            "context_utilization": self._estimate_message_tokens(self.messages)
            / self.effective_max_tokens,
            "messages_trimmed": self.messages_trimmed_count,
            "max_context_length": self.max_context_length,
            "effective_max_tokens": self.effective_max_tokens,
        }

    def init_messages(self):
        """Reset conversation history"""
        self.messages = []
        if self.system_prompt:
            self.messages.append(ChatMessage(role="system", content=self.system_prompt))


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "sk_dummy",
        timeout: int = 60,
        max_retries: int = 3,
        tokenizer: TokenizerInterface | None = None,
        max_context_length: int = 4096,
        context_buffer_ratio: float = 0.1,
        preserve_recent_exchanges: int = 2,
        system_prompt: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            context_buffer_ratio=context_buffer_ratio,
            preserve_recent_exchanges=preserve_recent_exchanges,
            system_prompt=system_prompt,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_kwargs = kwargs
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client with custom base URL"""
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _generate_response(self, messages: list[ChatMessage]) -> str:
        """Generate response using OpenAI-compatible API"""
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.temperature,
                **self.extra_kwargs,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            print(f"Error generating response: {e}")
            raise

    def list_models(self) -> list[str]:
        """List available models from the server"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error generating response: {e}")
            return []

    def health_check(self) -> bool:
        """Check if the server is healthy"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False


class OpenAIClient(LLMClient):
    """OpenAI/Azure OpenAI chat client implementation"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        azure: bool = False,
        tokenizer: TokenizerInterface | None = None,
        max_context_length: int = 4096,
        context_buffer_ratio: float = 0.1,
        preserve_recent_exchanges: int = 2,
        system_prompt: str | None = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            preserve_recent_exchanges=preserve_recent_exchanges,
            system_prompt=system_prompt,
        )
        self.azure = azure
        self._init_client()

    def _init_client(self):
        if not self.azure:
            self.client = OpenAI(

                api_key=os.environ.get("OPENAI_API_KEY")
            )
        else:
            if self.model in ["o1-preview", "o1-mini", "o3", "o4-mini"]:
                api_version = "2024-12-01-preview"
            elif self.model in ["o3-pro"]:
                api_version = "2025-03-01-preview"
            else:
                api_version = "2024-05-01-preview"

            self.client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_ENDPOINT"),  # ty: ignore
                api_key=os.environ.get("AZURE_OPENAI_KEY"),
                api_version=api_version,
            )

    def _provider_count_tokens(self, text: str) -> int:
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except (ImportError, KeyError):
            return max(1, len(text) // 4)

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

    def health_check(self) -> bool:
        """Check if the OpenAI client is healthy"""
        try:
            # Test with a simple models list call
            self.client.models.list()
            return True
        except Exception:
            return False


# class GPTChat_sl:
#     def __init__(self, azure=False, model="openai/gpt-4.1", temperature=1,system_prompt: str | None = None) -> None:
#         load_dotenv()
#         self.client = OpenAI(
#             base_url="https://models.github.ai/inference",
#             api_key=os.environ.get("OPENAI_API_KEY"),
#         )
#         self.messages = []
#         self.model = model
#         self.temperature = float(temperature)

#         # if system_prompt:
#         #     self.system_prompt=system_prompt

    
#     def get_response(self, prompt) -> str:
#         self.messages.append({"role": "user", "content": prompt})
#         if self.model in ["o3-pro"]:
#             response = self.client.responses.create(
#                 model=self.model,
#                 input=self.messages,
#                 temperature=self.temperature
#             )
#             main_content = response.output_text
#         else:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=self.messages,
#                 temperature=self.temperature
#             )
#             main_content = response.choices[0].message.content
#         self.messages.append({"role": "assistant", "content": main_content})
#          # Xóa nội dung user vừa gửi (set về rỗng)
#         for msg in reversed(self.messages):
#             if msg["role"] == "user" and msg["content"] == prompt:
#                 msg["content"] = ""
#                 break
#         return main_content

#     def get_model_response(self, prompt, code_format=None) -> list:
#         code_blocks = []
#         max_try = 3
#         while code_blocks == [] and max_try > 0:
#             max_try -= 1
#             try:
#                 response = self.get_response(prompt)
#             except Exception as e:
#                 print(f"max_try: {max_try}, exception: {e}")
#                 continue
#             print("response: "+ response)
#             code_blocks = extract_all_blocks(response, code_format)
#         if max_try == 0 or code_blocks == []:
#             print(f"get_model_response() exit, max_try: {max_try}, code_blocks: {code_blocks}")
#             sys.exit(0)
            
#         return code_blocks

#     def get_model_response_txt(self, prompt):
#         max_try = 3
#         while max_try > 0:
#             max_try -= 1
#             try:
#                 response = self.get_response(prompt)
#             except Exception as e:
#                 print(f"max_try: {max_try}, exception: {e}")
#                 continue
#             break
#         if max_try == 0:
#             print(f"get_model_response_txt() exit, max_try: {max_try}")
#             sys.exit(0)
        
#         return response

#     def get_message_len(self):
#         return {
#             "prompt_len": sum(len(item["content"]) for item in self.messages if item["role"] == "user"),
#             "response_len": sum(len(item["content"]) for item in self.messages if item["role"] == "assistant"),
#             "num_calls": len(self.messages) // 2
#         }
    
#     def init_messages(self):
#         self.messages = []
#         # if self.system_prompt:
#         #     self.messages.append({"role": "system", "content": self.system_prompt})

class GPTChat_sl:
    def __init__(
        self,
        base_url="https://models.github.ai/inference",
        model="openai/gpt-4.1",
        temperature=1,
        system_prompt: str | None = None,
    ) -> None:
        load_dotenv()
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.messages = []
        self.model = model
        self.temperature = float(temperature)

        # if system_prompt:
        #     self.system_prompt=system_prompt

    def get_response(self, prompt) -> str:
        self.messages.append({"role": "user", "content": prompt})

        if self.model in ["o3-pro"]:
            response = self.client.responses.create(
                model=self.model, input=self.messages, temperature=self.temperature
            )
            main_content = response.output_text
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages, temperature=self.temperature
            )
            main_content = response.choices[0].message.content

        self.messages.append({"role": "assistant", "content": main_content})

        for msg in reversed(self.messages):
            if msg["role"] == "user" and msg["content"] == prompt:
                msg["content"] = ""
                break
        return main_content

    def get_model_response(self, prompt, code_format=None) -> list:
        code_blocks = []
        max_try = 3
        while code_blocks == [] and max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                continue
            print("response: " + response)
            code_blocks = extract_all_blocks(response, code_format)
        if max_try == 0 or code_blocks == []:
            print(
                f"get_model_response() exit, max_try: {max_try}, code_blocks: {code_blocks}"
            )
            sys.exit(0)

        return code_blocks

    def get_model_response_txt(self, prompt):
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

    def init_messages(self):
        self.messages = []
        # if self.system_prompt:
        #     self.messages.append({"role": "system", "content": self.system_prompt})


class GPTChat:
    def __init__(self, azure=False, model="openai/gpt-4.1", temperature=1,system_prompt: str | None = None) -> None:
        load_dotenv()
        self.client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.messages = []
        self.model = model
        self.temperature = float(temperature)
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    def get_response(self, prompt) -> str:
        self.messages.append({"role": "user", "content": prompt})
        if self.model in ["o3-pro"]:
            response = self.client.responses.create(
                model=self.model,
                input=self.messages,
                temperature=self.temperature
            )
            main_content = response.output_text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature
            )
            main_content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": main_content})
        return main_content

    def get_model_response(self, prompt, code_format=None) -> list:
        code_blocks = []
        max_try = 3
        while code_blocks == [] and max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                continue
            code_blocks = extract_all_blocks(response, code_format)
        if max_try == 0 or code_blocks == []:
            print(f"get_model_response() exit, max_try: {max_try}, code_blocks: {code_blocks}")
            sys.exit(0)
            
        return code_blocks

    def get_model_response_txt(self, prompt):
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
        return {
            "prompt_len": sum(len(item["content"]) for item in self.messages if item["role"] == "user"),
            "response_len": sum(len(item["content"]) for item in self.messages if item["role"] == "assistant"),
            "num_calls": len(self.messages) // 2
        }
    
    # def init_messages(self, system_prompt=None):
    #     self.messages = []
    #     if self.system_prompt:
    #         self.messages.append({"role": "system", "content": self.system_prompt})

class HuggingFaceClient(LLMClient):
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        tokenizer: TokenizerInterface | None = None,
        max_context_length: int = 4096,
        context_buffer_ratio: float = 0.1,
        preserve_recent_exchanges: int = 2,
        system_prompt: str | None = None,
        max_new_tokens: int = 2048,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            context_buffer_ratio=context_buffer_ratio,
            preserve_recent_exchanges=preserve_recent_exchanges,
            system_prompt=system_prompt,
        )
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.kwargs = kwargs
        self._init_client()

    def _init_client(self):
        """Initialize Hugging Face model and tokenizer"""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, **self.kwargs)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            **self.kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _provider_count_tokens(self, text: str) -> int:
        if hasattr(self, "tokenizer") and self.tokenizer:
            tokens = self.tokenizer.encode(text, add_special_token=False)
            return len(tokens)
        return max(1, len(text) // 4)

    def _generate_response(self, messages: list[ChatMessage]) -> str:
        """Generate response using Hugging Face model"""

        conversation = [{"role": msg.role, "content": msg.content} for msg in messages]

        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation=conversation, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback to manual formatting if apply_chat_template fails
            formatted_prompt = ""
            for msg in messages:
                if msg.role == "system":
                    formatted_prompt += f"System: {msg.content}\n"
                elif msg.role == "user":
                    formatted_prompt += f"User: {msg.content}\n"
                elif msg.role == "assistant":
                    formatted_prompt += f"Assistant: {msg.content}\n"
            formatted_prompt += "Assistant: "

        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
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


# class LlamaClient(LLMClient):
#     def __init__(
#         self,
#         model_path: str,  # path to GGUF file
#         model_name: str,  # model's ID on HuggingFace
#         temperature: float = 1.0,
#         tokenizer: TokenizerInterface | None = None,
#         n_ctx: int = 4096,
#         context_buffer_ratio: float = 0.1,
#         preserve_recent_exchanges: int = 2,
#         n_gpu_layers=0,
#         system_prompt: str | None = None,
#         **kwargs,
#     ):
#         super().__init__(
#             model=model_name,
#             temperature=temperature,
#             tokenizer=tokenizer,
#             max_context_length=n_ctx,
#             context_buffer_ratio=context_buffer_ratio,
#             preserve_recent_exchanges=preserve_recent_exchanges,
#             system_prompt=system_prompt,
#         )
#         self.model_path = model_path
#         self.n_ctx = n_ctx
#         self.n_gpu_layers = n_gpu_layers
#         self.kwargs = kwargs
#         self._init_client()

#     def _init_client(self):
#         self.llm = Llama(
#             model_path=self.model_path,
#             n_ctx=self.n_ctx,
#             n_gpu_layers=self.n_gpu_layers,
#             verbose=False,
#             **self.kwargs,
#         )

#     def _generate_response(self, messages: list[ChatMessage]):
#         conversation = ""
#         for msg in messages:
#             if msg.role == "system":
#                 conversation += f"System: {msg.content}\n"
#             elif msg.role == "user":
#                 conversation += f"User: {msg.content}\n"
#             elif msg.role == "assistant":
#                 conversation += f"Assistant: {msg.content}\n"
#         conversation += "Assistant: "

#         output = self.llm(
#             conversation,
#             max_tokens=2048,
#             temperature=self.temperature,
#             stop=["User:", "\n\n"],
#         )

#         return output["choices"][0]["text"].strip()


class LLMClientFactory:
    """Factory for creating llm clients"""

    @staticmethod
    def create_client(client_type: str, **kwargs) -> LLMClient:
        client_type = client_type.lower()

        if client_type in ["openai", "azure"]:
            return OpenAIClient(**kwargs)
        elif client_type in ["azure"]:
            return OpenAIClient(azure=True, **kwargs)
        elif client_type in ["openai_compat"]:
            return OpenAICompatClient(**kwargs)
        elif client_type in ["huggingface", "hf", "transformers"]:
            return HuggingFaceClient(**kwargs)  # ty: ignore
        # elif client_type in ["llamacpp", "llama_cpp", "llama-cpp"]:
        #     return LlamaClient(**kwargs)  # ty: ignore
        else:
            raise ValueError(f"Unsupported client type: {client_type}")


def create_vllm_client(base_url: str, model: str, **kwargs) -> OpenAICompatClient:
    """Create client for vLLM server"""
    return OpenAICompatClient(
        base_url=f"{base_url}/v1",
        model=model,
        api_key="dummy",
        **kwargs,
    )


def create_ollama_client(
    base_url: str = "http://localhost:11434", model: str = "llama2", **kwargs
) -> OpenAICompatClient:
    """Create client for Ollama server"""
    return OpenAICompatClient(
        base_url=f"{base_url}/v1",
        model=model,
        api_key="dummy",
        **kwargs,
    )
