import json
from pathlib import Path


def load_dataset(path: str):
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)