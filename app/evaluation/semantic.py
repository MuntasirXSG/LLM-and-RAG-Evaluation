from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticEvaluator:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):

        self.model = SentenceTransformer(model_name)

    def score(
        self,
        generated: str,
        expected: str,
    ) -> float:

        embeddings = self.model.encode(
            [generated, expected]
        )

        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]],
        )[0][0]

        return float(similarity)

    def evaluate(
        self,
        generated: str,
        expected: str,
        threshold: float = 0.80,
    ):

        score = self.score(
            generated,
            expected,
        )

        return {
            "score": score,
            "passed": score >= threshold,
        }
 # testing 
evaluator = SemanticEvaluator()

result = evaluator.evaluate(
    "France's capital city is Paris.",
    "Paris is the capital of France."
)

print(result)