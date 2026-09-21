from datetime import datetime


class EvaluationRun:

    def __init__(self, model_name: str):

        self.model_name = model_name

        self.started_at = (
            datetime.now().isoformat()
        )

        self.cases = []

    def add_case(
        self,
        case_id: str,
        question: str,
        generated: str,
        results: dict,
        latency: float,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int
    ):

        self.cases.append({

            "case_id": case_id,

            "question": question,

            "generated": generated,

            "latency_seconds": latency,

            "input_tokens": input_tokens,

            "output_tokens": output_tokens,

            "total_tokens": total_tokens,

            "results": results,
        })

    def summary(self):

        total_cases = len(self.cases)

        if total_cases == 0:
            return {}

        # =========================================
        # Normal evaluation metrics
        # =========================================

        exact_matches = []

        semantic_scores = []

        correctness_scores = []

        relevance_scores = []

        completeness_scores = []

        # =========================================
        # Structured output metrics
        # =========================================

        json_valid_results = []

        field_accuracy_scores = []

        structured_scores = []

        # =========================================
        # Performance metrics
        # =========================================

        latencies = []

        input_token_counts = []

        output_token_counts = []

        total_token_counts = []

        # =========================================
        # Collect results
        # =========================================

        for case in self.cases:

            results = case["results"]

            # -------------------------------------
            # Latency
            # -------------------------------------

            latencies.append(
                case["latency_seconds"]
            )

            # -------------------------------------
            # Token usage
            # -------------------------------------

            input_token_counts.append(
                case["input_tokens"]
            )

            output_token_counts.append(
                case["output_tokens"]
            )

            total_token_counts.append(
                case["total_tokens"]
            )

            # -------------------------------------
            # Exact Match
            # -------------------------------------

            if "exact_match" in results:

                exact_matches.append(
                    1
                    if results["exact_match"]
                    else 0
                )

            # -------------------------------------
            # Semantic Similarity
            # -------------------------------------

            if "semantic" in results:

                semantic_scores.append(
                    results["semantic"]["score"]
                )

            # -------------------------------------
            # LLM Judge
            # -------------------------------------

            if "llm_judge" in results:

                judge = results["llm_judge"]

                correctness_scores.append(
                    judge["correctness"]
                )

                relevance_scores.append(
                    judge["relevance"]
                )

                completeness_scores.append(
                    judge["completeness"]
                )

            # -------------------------------------
            # Structured Output
            # -------------------------------------

            if "structured" in results:

                structured = results["structured"]

                json_valid_results.append(
                    1
                    if structured["json_valid"]
                    else 0
                )

                field_accuracy_scores.append(
                    structured["field_accuracy"]
                )

                structured_scores.append(
                    structured["overall_score"]
                )

        # =========================================
        # Summary
        # =========================================

        return {

            "model": self.model_name,

            "total_cases": total_cases,

            # -------------------------------------
            # Exact Match
            # -------------------------------------

            "exact_match_accuracy": (

                sum(exact_matches)
                / len(exact_matches)

                if exact_matches

                else None
            ),

            # -------------------------------------
            # Semantic
            # -------------------------------------

            "average_semantic_score": (

                sum(semantic_scores)
                / len(semantic_scores)

                if semantic_scores

                else None
            ),

            # -------------------------------------
            # LLM Judge
            # -------------------------------------

            "average_correctness": (

                sum(correctness_scores)
                / len(correctness_scores)

                if correctness_scores

                else None
            ),

            "average_relevance": (

                sum(relevance_scores)
                / len(relevance_scores)

                if relevance_scores

                else None
            ),

            "average_completeness": (

                sum(completeness_scores)
                / len(completeness_scores)

                if completeness_scores

                else None
            ),

            # -------------------------------------
            # Structured Output
            # -------------------------------------

            "structured_json_validity": (

                sum(json_valid_results)
                / len(json_valid_results)

                if json_valid_results

                else None
            ),

            "average_field_accuracy": (

                sum(field_accuracy_scores)
                / len(field_accuracy_scores)

                if field_accuracy_scores

                else None
            ),

            "average_structured_score": (

                sum(structured_scores)
                / len(structured_scores)

                if structured_scores

                else None
            ),

            # -------------------------------------
            # Latency
            # -------------------------------------

            "average_latency_seconds": (

                sum(latencies)
                / len(latencies)

                if latencies

                else None
            ),

            # -------------------------------------
            # Token Usage
            # -------------------------------------

            "average_input_tokens": (

                sum(input_token_counts)
                / len(input_token_counts)

                if input_token_counts

                else None
            ),

            "average_output_tokens": (

                sum(output_token_counts)
                / len(output_token_counts)

                if output_token_counts

                else None
            ),

            "average_total_tokens": (

                sum(total_token_counts)
                / len(total_token_counts)

                if total_token_counts

                else None
            ),
        }

    def to_dict(self):

        return {

            "model": self.model_name,

            "started_at": self.started_at,

            "summary": self.summary(),

            "cases": self.cases,
        }