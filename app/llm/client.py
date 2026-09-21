import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ============================================================
# Load .env
# ============================================================

env_path = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(env_path)


# ============================================================
# LLM Client
# ============================================================

class LLMClient:

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b"
    ):

        # Store model name
        self.model_name = model

        # Get Groq API key
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in the .env file"
            )

        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=model,
            temperature=0,
            api_key=groq_api_key,
        )

    # ========================================================
    # Generate response
    # ========================================================

    def generate(self, prompt: str):

        response = self.llm.invoke(prompt)

        # Get token usage
        usage = response.usage_metadata

        return {

            # Generated text
            "text": response.content,

            # Input tokens
            "input_tokens": usage.get(
                "input_tokens",
                0
            ),

            # Output tokens
            "output_tokens": usage.get(
                "output_tokens",
                0
            ),

            # Total tokens
            "total_tokens": usage.get(
                "total_tokens",
                0
            ),
        }