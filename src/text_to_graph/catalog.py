"""Versioned model inventory shared by setup and inference."""

import json
import os
from pathlib import Path

DATA = Path(__file__).resolve().with_name("catalog.json")
CATALOG = json.loads(DATA.read_text())
ROUTES = CATALOG["routes"]
LANGUAGES = CATALOG["languages"]
ASSETS = CATALOG["assets"]


def model_dir():
    if override := os.environ.get("TEXT_TO_GRAPH_MODEL_DIR"):
        return Path(override).expanduser()
    # Editable installs load this file from <checkout>/src/text_to_graph.
    src = DATA.parent.parent
    root = src.parent if src.name == "src" else Path.cwd()
    return root / ".cache" / "text-to-graph"


def asset_path(name):
    if name.startswith("stanza/"):
        name = name.replace("/models/", "/", 1)
    if name.startswith("detector/") and os.environ.get("FTLANG_CACHE"):
        return Path(os.environ["FTLANG_CACHE"]) / Path(name).name
    return model_dir() / name
