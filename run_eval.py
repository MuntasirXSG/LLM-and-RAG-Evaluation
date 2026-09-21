import json
import time

from app.data.loader import load_dataset
from app.llm.client import LLMClient
from app.evaluation.evaluator import Evaluator
from app.evaluation.run import EvaluationRun


def main():

    # ============================================
    # Load dataset
    # ============================================

    dataset = load_dataset(
        "datasets/eval_dataset.json"
    )

    # ============================================
    # Create LLM
    # ============================================

    llm = LLMClient()

    # ============================================
    # Create evaluator
    # ============================================

    evaluator = Evaluator(llm)

    # ============================================
    # Create evaluation run
    # ============================================

    evaluation_run = EvaluationRun(
        model_name=llm.model_name
    )

    # ============================================
    # Run evaluation
    # ============================================

    for case in dataset:

        print(
            f"\nRunning {case['id']}..."
        )

        print("-" * 60)

        # ----------------------------------------
        # Start latency timer
        # ----------------------------------------

        start_time = time.perf_counter()

        # ----------------------------------------
        # Generate response
        # ----------------------------------------

        response = llm.generate(
            case["input"]
        )

        # ----------------------------------------
        # Stop latency timer
        # ----------------------------------------

        end_time = time.perf_counter()

        latency = (
            end_time - start_time
        )

        # ----------------------------------------
        # Extract response
        # ----------------------------------------

        answer = response["text"]

        # ----------------------------------------
        # Extract token usage
        # ----------------------------------------

        input_tokens = (
            response["input_tokens"]
        )

        output_tokens = (
            response["output_tokens"]
        )

        total_tokens = (
            response["total_tokens"]
        )

        # ----------------------------------------
        # Evaluate answer
        # ----------------------------------------

        results = evaluator.evaluate_case(
            case,
            answer
        )

        # ----------------------------------------
        # Save case
        # ----------------------------------------

        evaluation_run.add_case(

            case_id=case["id"],

            question=case["input"],

            generated=answer,

            results=results,

            latency=latency,

            input_tokens=input_tokens,

            output_tokens=output_tokens,

            total_tokens=total_tokens
        )

        # ========================================
        # Display result
        # ========================================

        print(
            f"Generated: {answer}"
        )

        # ----------------------------------------
        # Exact Match
        # ----------------------------------------

        if "exact_match" in results:

            print(
                f"Exact Match: "
                f"{results['exact_match']}"
            )

        # ----------------------------------------
        # Semantic
        # ----------------------------------------

        if "semantic" in results:

            semantic = results["semantic"]

            print(
                f"Semantic Score: "
                f"{semantic['score']:.3f}"
            )

            print(
                f"Semantic Passed: "
                f"{semantic['passed']}"
            )

        # ----------------------------------------
        # LLM Judge
        # ----------------------------------------

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

        # ----------------------------------------
        # Structured Output
        # ----------------------------------------

        if "structured" in results:

            structured = results["structured"]

            print(
                f"JSON Valid: "
                f"{structured['json_valid']}"
            )

            print(
                f"Field Accuracy: "
                f"{structured['field_accuracy']:.3f}"
            )

            print(
                f"Overall Structured Score: "
                f"{structured['overall_score']:.3f}"
            )

            print("\nField Results:")

            for field, result in (
                structured["fields"].items()
            ):

                print(
                    f"  {field}: "
                    f"{result['generated']} "
                    f"→ "
                    f"{result['correct']}"
                )

        # ----------------------------------------
        # Latency
        # ----------------------------------------

        print(
            f"Latency: "
            f"{latency:.3f} seconds"
        )

        # ----------------------------------------
        # Token usage
        # ----------------------------------------

        print(
            f"Input Tokens: "
            f"{input_tokens}"
        )

        print(
            f"Output Tokens: "
            f"{output_tokens}"
        )

        print(
            f"Total Tokens: "
            f"{total_tokens}"
        )

        print("-" * 60)

    # ============================================
    # Summary
    # ============================================

    summary = evaluation_run.summary()

    print("\n")
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Model: "
        f"{summary['model']}"
    )

    print(
        f"Total Cases: "
        f"{summary['total_cases']}"
    )

    # --------------------------------------------
    # Exact Match
    # --------------------------------------------

    if (
        summary["exact_match_accuracy"]
        is not None
    ):

        print(
            f"Exact Match Accuracy: "
            f"{summary['exact_match_accuracy']:.3f}"
        )

    # --------------------------------------------
    # Semantic
    # --------------------------------------------

    if (
        summary["average_semantic_score"]
        is not None
    ):

        print(
            f"Average Semantic Score: "
            f"{summary['average_semantic_score']:.3f}"
        )

    # --------------------------------------------
    # LLM Judge
    # --------------------------------------------

    if (
        summary["average_correctness"]
        is not None
    ):

        print(
            f"Average Correctness: "
            f"{summary['average_correctness']:.3f}/5"
        )

    if (
        summary["average_relevance"]
        is not None
    ):

        print(
            f"Average Relevance: "
            f"{summary['average_relevance']:.3f}/5"
        )

    if (
        summary["average_completeness"]
        is not None
    ):

        print(
            f"Average Completeness: "
            f"{summary['average_completeness']:.3f}/5"
        )

    # --------------------------------------------
    # Structured Output
    # --------------------------------------------

    if (
        summary["structured_json_validity"]
        is not None
    ):

        print(
            f"Structured JSON Validity: "
            f"{summary['structured_json_validity']:.3f}"
        )

    if (
        summary["average_field_accuracy"]
        is not None
    ):

        print(
            f"Average Field Accuracy: "
            f"{summary['average_field_accuracy']:.3f}"
        )

    if (
        summary["average_structured_score"]
        is not None
    ):

        print(
            f"Average Structured Score: "
            f"{summary['average_structured_score']:.3f}"
        )

    # --------------------------------------------
    # Performance
    # --------------------------------------------

    if (
        summary["average_latency_seconds"]
        is not None
    ):

        print(
            f"Average Latency: "
            f"{summary['average_latency_seconds']:.3f} seconds"
        )

    # --------------------------------------------
    # Token Usage
    # --------------------------------------------

    if (
        summary["average_input_tokens"]
        is not None
    ):

        print(
            f"Average Input Tokens: "
            f"{summary['average_input_tokens']:.2f}"
        )

    if (
        summary["average_output_tokens"]
        is not None
    ):

        print(
            f"Average Output Tokens: "
            f"{summary['average_output_tokens']:.2f}"
        )

    if (
        summary["average_total_tokens"]
        is not None
    ):

        print(
            f"Average Total Tokens: "
            f"{summary['average_total_tokens']:.2f}"
        )

    # ============================================
    # Save results
    # ============================================

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_run.to_dict(),
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nResults saved to "
        "evaluation_results.json"
    )


if __name__ == "__main__":
    main()