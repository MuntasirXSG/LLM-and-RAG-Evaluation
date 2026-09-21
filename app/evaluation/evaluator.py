from app.evaluation.correctness import exact_match
from app.evaluation.semantic import SemanticEvaluator
from app.evaluation.judge import LLMJudge
from app.evaluation.structured import structured_output_evaluate


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
        generated_answer: str
    ):

        expected = case["expected"]

        results = {}

        # -----------------------------------------
        # Normal text evaluation
        # -----------------------------------------

        if isinstance(expected, str):

            results["exact_match"] = exact_match(
                generated_answer,
                expected
            )

            results["semantic"] = (
                self.semantic.evaluate(
                    generated_answer,
                    expected
                )
            )

            results["llm_judge"] = (
                self.judge.evaluate(
                    question=case["input"],
                    reference=expected,
                    generated=generated_answer
                )
            )

        # -----------------------------------------
        # Structured output evaluation
        # -----------------------------------------

        elif isinstance(expected, dict):

            results["structured"] = (
                structured_output_evaluate(
                    generated_text=generated_answer,
                    expected=expected
                )
            )

        return results