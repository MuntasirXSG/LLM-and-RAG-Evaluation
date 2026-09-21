from app.data.loader import load_dataset
from app.llm.client import LLMClient
from app.evaluation.evaluator import Evaluator


def main():

    dataset = load_dataset(
        "datasets/eval_dataset.json"
    )

    llm = LLMClient()

    evaluator = Evaluator(llm)

    for case in dataset:

        print(
            f"\nRunning {case['id']}..."
        )

        answer = llm.generate(
            case["input"]
        )

        results = evaluator.evaluate_case(
            case,
            answer,
        )

        print(
            f"Generated: {answer}"
        )

        print(
            f"Exact Match: "
            f"{results.get('exact_match')}"
        )

        if "semantic" in results:

            print(
                f"Semantic Score: "
                f"{results['semantic']['score']:.3f}"
            )

        if "llm_judge" in results:

            judge = results["llm_judge"]

            print(
                f"Correctness: "
                f"{judge['correctness']}/5"
            )

            print(
                f"Relevance: "
                f"{judge['relevance']}/5"
            )

            print(
                f"Completeness: "
                f"{judge['completeness']}/5"
            )

            print(
                f"Reason: "
                f"{judge['reason']}"
            )

        print("-" * 60)


if __name__ == "__main__":
    main()