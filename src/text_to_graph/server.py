"""Single-process local HTTP API."""

import argparse
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .catalog import LANGUAGES
from .models import language_cache, preload_languages
from .pipeline import extract_graph


class Document(BaseModel):
    text: str
    language: str | None = None


class PreloadRequest(BaseModel):
    languages: list[str]


app = FastAPI(title="Text to graph", version="1")


@app.get("/health")
async def health():
    cache = language_cache()
    return {
        "service": "text-to-graph",
        "api_version": 1,
        "max_loaded_languages": cache.limit,
        "loaded_languages": cache.resident(),
    }


@app.post("/v1/graph")
def graph(document: Document):
    if document.language is not None and document.language not in LANGUAGES:
        raise HTTPException(422, "Unknown detector language code")
    return extract_graph(document.text, language_code=document.language).to_dict()


@app.post("/v1/languages/preload")
def preload(request: PreloadRequest):
    try:
        return {"loaded_languages": preload_languages(request.languages)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


def main():
    import uvicorn

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument(
        "--preload",
        nargs="+",
        metavar="CODE",
        help="load these languages before serving",
    )
    cli.add_argument(
        "--max-loaded-languages",
        type=int,
        default=os.environ.get("TEXT_TO_GRAPH_MAX_LOADED_LANGUAGES", "4"),
    )
    args = cli.parse_args()
    if args.max_loaded_languages < 1:
        cli.error("--max-loaded-languages must be positive")
    os.environ["TEXT_TO_GRAPH_MAX_LOADED_LANGUAGES"] = str(args.max_loaded_languages)
    if args.preload:
        try:
            print(f"Preloading languages: {', '.join(args.preload)}", flush=True)
            preload_languages(args.preload)
        except (ValueError, RuntimeError) as exc:
            cli.exit(1, f"{exc}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)
