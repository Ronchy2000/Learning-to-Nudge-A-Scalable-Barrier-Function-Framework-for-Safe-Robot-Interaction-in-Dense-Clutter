from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def resolve_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / raw)
        candidates.append(project_root / "configs" / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = "\n".join(f"  - {str(c)}" for c in candidates)
    raise FileNotFoundError(
        f"Cannot find file: {path}\n"
        f"Tried:\n{candidate_text}\n"
        "Tip: pass an explicit --config /abs/path/to/configs/env.yaml"
    )


def load_yaml(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
