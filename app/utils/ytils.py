import os
import json
from pathlib import Path
from typing import Any, Dict

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON data from a file."""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json_file(file_path: Path, data: Dict[str, Any]) -> None:
    """Save data to a JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_text_file(file_path: Path) -> str:
    """Read text from a file."""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def write_text_file(file_path: Path, content: str) -> None:
    """Write text to a file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def ensure_directory(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)