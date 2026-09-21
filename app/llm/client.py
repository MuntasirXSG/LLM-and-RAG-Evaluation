import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

env_path = Path(__file__).resolve().parents[2] / ".env" # here .resolves() gets the absoult path . and parent[i] is for choosing desired steps for the location
load_dotenv(env_path)


class LLMClient:

    def __init__(self, model: str = "openai/gpt-oss-20b"):

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY  is expired")

        self.llm = ChatGroq(
            model=model,
            temperature=0,
            api_key=groq_api_key,
        )

    def generate(self, prompt: str) -> str:

        response = self.llm.invoke(prompt)

        return response.content