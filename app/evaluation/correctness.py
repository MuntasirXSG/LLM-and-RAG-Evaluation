import re


def normalize_text(text: str) -> str:

    text = text.lower().strip()

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text


def exact_match(
    generated: str,
    expected: str,
) -> bool:

    return (
        normalize_text(generated)
        == normalize_text(expected)
    )