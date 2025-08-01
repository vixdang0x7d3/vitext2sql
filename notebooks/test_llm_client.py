import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    from unittest.mock import Mock, patch, MagicMock
    from dataclasses import dataclass

    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from llm_client import (
        ChatMessage,
        LLMClient,
        OpenAIClient,
        HuggingFaceClient,
        LlamaClient,
        LLMClientFactory,
        create_ollama_client,
    )
    from utils import extract_all_blocks

    test_results = {"passed": 0, "failed": 0, "errors": []}
    return (
        ChatMessage,
        HuggingFaceClient,
        LLMClient,
        LLMClientFactory,
        LlamaClient,
        OpenAIClient,
        create_ollama_client,
        extract_all_blocks,
        mo,
        patch,
        test_results,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### ChatMessage tests""")
    return


@app.cell
def _(ChatMessage, mo, test_results):
    def test_chat_message():
        """Test ChatMessage dataclass functionality"""
        try:
            # Test basic creation
            msg = ChatMessage(role="user", content="Hello world")
            assert msg.role == "user"
            assert msg.content == "Hello world"

            # Test equality
            msg1 = ChatMessage(role="user", content="Hello")
            msg2 = ChatMessage(role="user", content="Hello")
            msg3 = ChatMessage(role="assistant", content="Hello")

            assert msg1 == msg2
            assert msg1 != msg3

            # Test field types
            assert isinstance(msg.role, str)
            assert isinstance(msg.content, str)

            test_results["passed"] += 1
            return mo.md("✅ **ChatMessage tests passed**")

        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"ChatMessage test failed: {str(e)}")
            return mo.md(f"❌ **ChatMessage tests failed**: {str(e)}")


    test_chat_message()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### LLMClient base class tests""")
    return


@app.cell
def _(LLMClient, mo, test_results):
    # Create concrete implementation for testing
    class ConcreteLLMClient(LLMClient):
        def _generate_response(self, messages):
            return "Test response"


    def test_llm_client_base():
        """Test LLMClient abstract base class"""
        try:
            # Test initialization
            client = ConcreteLLMClient(model="test-model", temperature=0.5)
            assert client.model == "test-model"
            assert client.temperature == 0.5
            assert client.messages == []

            # Test default values
            client_default = ConcreteLLMClient()
            assert client_default.model == "o4-mini"
            assert client_default.temperature == 1

            # Test get_response
            response = client.get_response("Hello")
            assert response == "Test response"
            assert len(client.messages) == 2
            assert client.messages[0].role == "user"
            assert client.messages[0].content == "Hello"
            assert client.messages[1].role == "assistant"
            assert client.messages[1].content == "Test response"

            # Test get_message_len
            stats = client.get_message_len()
            expected = {
                "prompt_len": len("Hello"),
                "response_len": len("Test response"),
                "num_calls": 1,
            }
            assert stats == expected

            # Test init_message
            client.init_message()
            assert client.messages == []

            test_results["passed"] += 1
            return mo.md("✅ **LLMClient base class tests passed**")

        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"LLMClient base test failed: {str(e)}")
            return mo.md(f"❌ **LLMClient base class tests failed**: {str(e)}")


    test_llm_client_base()
    return (ConcreteLLMClient,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### LLMClient system prompt tests""")
    return


@app.cell
def _(ConcreteLLMClient, mo, test_results):
    def test_llm_client_system_prompt():
        """Test LLMClient system prompt functionality"""
        try:
            # Test initialization with system prompt
            client = ConcreteLLMClient(system_prompt="You are a helpful assistant.")
            assert client.system_prompt == "You are a helpful assistant."
            assert len(client.messages) == 1
            assert client.messages[0].role == "system"
            assert client.messages[0].content == "You are a helpful assistant."

            # Test initialization without system prompt
            client_no_system = ConcreteLLMClient()
            assert client_no_system.system_prompt is None
            assert client_no_system.messages == []

            # Test get_response with system prompt
            response = client.get_response("Hello")
            assert response == "Test response"
            assert len(client.messages) == 3  # system + user + assistant
            assert client.messages[0].role == "system"
            assert client.messages[1].role == "user"
            assert client.messages[2].role == "assistant"

            # Test init_message preserves system prompt
            client.init_message()
            assert len(client.messages) == 1
            assert client.messages[0].role == "system"
            assert client.messages[0].content == "You are a helpful assistant."

            # Test message length calculation with system prompt
            client.get_response("Test message")
            stats = client.get_message_len()
            expected = {
                "prompt_len": len("Test message"),
                "response_len": len("Test response"),
                "num_calls": 1,
            }
            assert stats == expected

            test_results["passed"] += 1
            return mo.md("✅ **LLMClient system prompt tests passed**")

        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"LLMClient system prompt test failed: {str(e)}")
            return mo.md(f"❌ **LLMClient system prompt tests failed**: {str(e)}")

    test_llm_client_system_prompt()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### LLMClientFactory tests""")
    return


@app.cell
def _(LLMClientFactory, mo, patch, test_results):
    def test_llm_client_factory():
        """Test LLMClientFactory functionality"""
        try:
            # Test OpenAI client creation
            with patch('llm_client.OpenAIClient') as mock_openai_client:
                LLMClientFactory.__init__("openai", model="gpt-4", temperature=0.5)
                mock_openai_client.assert_called_once_with(model="gpt-4", temperature=0.5)

            # Test Azure client creation
            with patch('llm_client.OpenAIClient') as mock_openai_client:
                LLMClientFactory.__init__("azure", model="gpt-4", azure=True)
                mock_openai_client.assert_called_once_with(model="gpt-4", azure=True)

            # Test HuggingFace client creation
            with patch('llm_client.HuggingFaceClient') as mock_hf_client:
                LLMClientFactory.__init__("huggingface", model="test-model")
                mock_hf_client.assert_called_once_with(model="test-model")

            # Test HF alias
            with patch('llm_client.HuggingFaceClient') as mock_hf_client:
                LLMClientFactory.__init__("hf", model="test-model")
                mock_hf_client.assert_called_once_with(model="test-model")

            # Test Llama client creation
            with patch('llm_client.LlamaClient') as mock_llama_client:
                LLMClientFactory.__init__("llamacpp", model_path="/path/to/model")
                mock_llama_client.assert_called_once_with(model_path="/path/to/model")

            # Test case insensitivity
            with patch('llm_client.OpenAIClient') as mock_client:
                LLMClientFactory.__init__("OPENAI", model="gpt-4")
                mock_client.assert_called_once_with(model="gpt-4")

            # Test unsupported client type
            try:
                LLMClientFactory.__init__("invalid")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Unsupported client type: invalid" in str(e)

            test_results["passed"] += 1
            return mo.md("✅ **LLMClientFactory tests passed**")

        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"LLMClientFactory test failed: {str(e)}")
            return mo.md(f"❌ **LLMClientFactory tests failed**: {str(e)}")

    test_llm_client_factory()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Utitilty functions and edge cases tests""")
    return


@app.cell
def _(extract_all_blocks, mo, test_results):
    def test_utils_and_edge_cases():
        """Test utility functions and edge cases"""
        try:
            # Test extract_all_blocks utility function
            content = """
            Here is some SQL:
            ```sql
            SELECT * FROM users;
            ```

            And another query:
            ```sql
            SELECT name FROM products;
            ```
            """

            blocks = extract_all_blocks(content, "sql")
            expected = ["SELECT * FROM users;", "SELECT name FROM products;"]  # Due to bug in utils.py - early return
            assert blocks == expected, f"Expected {expected}, got {blocks}"

            # Test with no code blocks
            content_no_blocks = "Just plain text with no code blocks"
            blocks_empty = extract_all_blocks(content_no_blocks, "sql")
            assert blocks_empty == [], "Expected to return an empty list"

            # Test with no format specified
            content_no_format = "```SELECT * FROM users;```"
            blocks_no_format = extract_all_blocks(content_no_format, None)
            assert blocks_no_format == ["SELECT * FROM users;"], "Expected to handle codeblock with unspecified"

            test_results["passed"] += 1
            return mo.md("✅ **Utility functions and edge cases tests passed**")

        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append(f"Utils/edge cases test failed: {str(e)}")
            return mo.md(f"❌ **Utility functions and edge cases tests failed**: {str(e)}")

    test_utils_and_edge_cases()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Manual Testing""")
    return


@app.cell
def _(OpenAIClient):
    from dotenv import load_dotenv

    def _():
        load_dotenv()
        openai_client = OpenAIClient(model="gpt-4o-mini", temperature=1)
        response = openai_client.get_response("Hello, how are you?")
        print(f"OpenAI Response: {response}")


    _()

    return


@app.cell
def _(HuggingFaceClient):
    def _():
        hf_client = HuggingFaceClient(model="Qwen/Qwen2.5-1.5B-Instruct", device="cpu")
        response = hf_client.get_response("")
        print(f"HuggingFace Response: {response}")

    _()
    return


@app.cell
def _(LlamaClient):
    def _():
        llama_client = LlamaClient(model_path="models/Qwen3-1.7B-BF16.gguf")
        response = llama_client.get_response("What is AI? Speak Vietnamese")
        print(f"Llama response: {response}")

    _()
    return


@app.cell
def _(LlamaClient):
    # System prompt test
    def _():
        sql_client = LlamaClient(
            model_path="models/Qwen3-1.7B-BF16.gguf",
            system_prompt="you're a SQL expert, only answer in valid SQL",
        )

        sql_response = sql_client.get_response("get the number of users that logged in more than 150 times")
        print(f"SQL response: {sql_response}")

    _()
    return


@app.cell
def _(LlamaClient):
    # Multi-turn convo test
    def _():
        llama_client = LlamaClient(
            model_path="models/Qwen3-1.7B-BF16.gguf",
            n_gpu_layers=10,
            system_prompt="You're a international communication agent, reponse with English only."
        )

        llama_client.get_response("My name is John")
        response_2 = llama_client.get_response("What's my name?")
        print(f"Memory test: {response_2}")

    _()
    return


@app.cell(disabled=True)
def _(OpenAIClient):
    def _():
        # Conversation length test
        client = OpenAIClient(model="gpt-4o-mini")
        for i in range(10):
            response = client.get_response(f"Message {i}: Tell me a random fact")
            stats = client.get_message_len()
            print(f"Turn {i}: {stats}")

        # Memory cleanup test
        client.init_message()  # Should reset conversation
        assert len(client.messages) <= 1  # Only system prompt if present

    _()
    return


@app.cell
def _(HuggingFaceClient):

      # Invalid model test
    def _():
        try:
            bad_client = HuggingFaceClient(model="nonexistent/model")
        except Exception as e:
            print(f"Expected error: {e}")

        client = HuggingFaceClient(model="Qwen/Qwen2.5-1.5B-Instruct")

        empty_response = client.get_response("")
    _()
    return


@app.cell
def _(create_ollama_client):
    print("Connecting to Ollama server:")
    try:
        ollama_client = create_ollama_client(
            base_url="http://localhost:11434",
            model="llama2",
            temperature=0.8
        )
    
        if ollama_client.health_check():
            response = ollama_client.get_response("Explain neural networks briefly.")
            print(f"Ollama Response: {response[:100]}...")
        else:
            print("Ollama server not available")
    except Exception as e:
        print(f"Ollama Error: {e}")
    return


if __name__ == "__main__":
    app.run()
