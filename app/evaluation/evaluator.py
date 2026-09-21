from app.evaluation.correctness import exact_match
from app.evaluation.semantic import SemanticEvaluator
from app.evaluation.judge import LLMJudge

# using llm as a judge concept 
class Evaluator:

    def __init__(self, llm):

        self.llm = llm

        self.semantic = SemanticEvaluator()

        self.judge = LLMJudge(
            llm=self.llm.llm
        )

    def evaluate_case(
        self,
        case: dict,
        generated_answer: str,
    ):

        expected = case["expected"]

        results = {}

        # --------------------------------
        # Exact Match
        # --------------------------------

        if isinstance(expected, str):

            results["exact_match"] = exact_match(
                generated_answer,
                expected,
            )

        # --------------------------------
        # Semantic Similarity
        # --------------------------------

        if isinstance(expected, str):

            results["semantic"] = (
                self.semantic.evaluate(
                    generated_answer,
                    expected,
                )
            )

        # --------------------------------
        # LLM Judge
        # --------------------------------

        if isinstance(expected, str):

            results["llm_judge"] = (
                self.judge.evaluate(
                    question=case["input"],
                    reference=expected,
                    generated=generated_answer,
                )
            )

        return results