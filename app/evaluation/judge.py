import json

from langchain_core.prompts import ChatPromptTemplate


JUDGE_PROMPT = """
You are an expert evaluator of LLM responses.

Evaluate the generated answer against the reference answer.

Question:
{question}

Reference Answer:
{reference}

Generated Answer:
{generated}

Evaluate the generated answer on:

1. correctness
2. relevance
3. completeness

Use a score from 1 to 5.

Return ONLY valid JSON:

{{
    "correctness": 1,
    "relevance": 1,
    "completeness": 1,
    "reason": "brief explanation"
}}
"""


class LLMJudge:

    def __init__(self, llm):

        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template(
            JUDGE_PROMPT
        )

    def evaluate(
        self,
        question: str,
        reference: str,
        generated: str,
    ):

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "question": question,
                "reference": reference,
                "generated": generated,
            }
        )

        content = response.content

        try:

            return json.loads(content)

        except json.JSONDecodeError:

            return {
                "correctness": 0,
                "relevance": 0,
                "completeness": 0,
                "reason": "Judge returned invalid JSON.",
                "raw_response": content,
            }