from openai import OpenAI
from dotenv import load_dotenv
from .utils import extract_all_blocks
import os
import sys

class GPTChat:
    def __init__(self, azure=False, model="openai/gpt-4.1", temperature=1) -> None:
        load_dotenv()
        self.client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.messages = []
        self.model = model
        self.temperature = float(temperature)

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
    
    def init_messages(self):
        self.messages = []