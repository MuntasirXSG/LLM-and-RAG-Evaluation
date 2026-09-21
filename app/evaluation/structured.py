import json
import re


def parse_json(text: str):
    """
    Parse JSON from an LLM response.

    Handles both:
    1. Pure JSON
    2. JSON wrapped in ```json ... ```
    """

    text = text.strip()

    # Remove Markdown code fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        return None


def field_accuracy(
    generated: dict,
    expected: dict
) -> dict:

    results = {}

    for field, expected_value in expected.items():

        generated_value = generated.get(field)

        results[field] = {
            "expected": expected_value,
            "generated": generated_value,
            "correct": generated_value == expected_value,
        }

    return results


def structured_output_evaluate(
    generated_text: str,
    expected: dict
) -> dict:

    generated = parse_json(generated_text)

    # -------------------------
    # JSON validity
    # -------------------------

    if generated is None:

        return {
            "json_valid": False,
            "field_accuracy": 0.0,
            "overall_score": 0.0,
            "fields": {},
        }

    # Parsed JSON must be an object
    if not isinstance(generated, dict):

        return {
            "json_valid": False,
            "field_accuracy": 0.0,
            "overall_score": 0.0,
            "fields": {},
        }

    # -------------------------
    # Field-level accuracy
    # -------------------------

    fields = field_accuracy(
        generated,
        expected
    )

    # -------------------------
    # Overall score
    # -------------------------

    if len(expected) == 0:

        score = 1.0

    else:

        correct_fields = sum(
            1
            for result in fields.values()
            if result["correct"]
        )

        score = correct_fields / len(expected)

    return {
        "json_valid": True,
        "field_accuracy": score,
        "overall_score": score,
        "fields": fields,
    }