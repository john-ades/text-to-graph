"""Single-process local HTTP API."""

import argparse
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, SkipValidation, ValidationError

from . import backend, models
from .catalog import LANGUAGES
from .graph import DocumentGraph
from .models import language_cache, preload_languages
from .pipeline import extract_graph, extract_many


class Document(BaseModel):
    text: str
    language: str | None = None


class DocumentBatch(BaseModel):
    # Keep the Document schema in OpenAPI and validate individual items in the handler.
    documents: list[SkipValidation[Document]] = Field(min_length=1, max_length=32)


class DocumentError(BaseModel):
    code: str
    message: str


class GraphResult(BaseModel):
    index: int
    graph: DocumentGraph


class ErrorResult(BaseModel):
    index: int
    error: DocumentError


class BatchResponse(BaseModel):
    results: list[GraphResult | ErrorResult]


class PreloadRequest(BaseModel):
    languages: list[str]


app = FastAPI(title="Text to graph", version="1")


@app.get("/health")
def health():
    cache = language_cache()
    return {
        "service": "text-to-graph",
        "api_version": 1,
        "inference": backend.devices(),
        "max_loaded_languages": cache.limit,
        "loaded_languages": cache.resident(),
    }


@app.post("/v1/graph")
def graph(document: Document):
    if document.language is not None and document.language not in LANGUAGES:
        raise HTTPException(422, "Unknown detector language code")
    return extract_graph(document.text, language_code=document.language).to_dict()


@app.post("/v1/graph/batch", response_model=BatchResponse)
def graph_batch(batch: DocumentBatch):
    results = [{"index": i} for i in range(len(batch.documents))]
    valid = []
    indices = []
    for index, item in enumerate(batch.documents):
        try:
            document = Document.model_validate(item)
            if document.language is not None and document.language not in LANGUAGES:
                raise ValueError("Unknown detector language code")
        except (ValidationError, ValueError) as exc:
            if isinstance(exc, ValidationError):
                detail = exc.errors(include_input=False, include_url=False)[0]
                field = ".".join(map(str, detail["loc"]))
                reason = f"{field}: {detail['msg']}" if field else detail["msg"]
            else:
                reason = str(exc)
            results[index]["error"] = {"code": "invalid_document", "message": reason}
            continue
        valid.append((document.text, document.language))
        indices.append(index)
    for index, result in zip(indices, extract_many(valid), strict=True):
        if isinstance(result, Exception):
            results[index]["error"] = {
                "code": "extraction_failed",
                "message": str(result),
            }
        else:
            results[index]["graph"] = result
    return {"results": results}


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
    cli.add_argument(
        "--backend",
        choices=backend.BACKENDS,
        default=os.environ.get("TEXT_TO_GRAPH_BACKEND", "auto"),
    )
    cli.add_argument(
        "--warmup", action="store_true", help="run inference before accepting requests"
    )
    args = cli.parse_args()
    os.environ["TEXT_TO_GRAPH_BACKEND"] = args.backend
    try:
        backend.selected()
    except (ValueError, RuntimeError) as exc:
        cli.error(str(exc))
    if args.max_loaded_languages < 1:
        cli.error("--max-loaded-languages must be positive")
    os.environ["TEXT_TO_GRAPH_MAX_LOADED_LANGUAGES"] = str(args.max_loaded_languages)
    if args.preload:
        try:
            print(f"Preloading languages: {', '.join(args.preload)}", flush=True)
            preload_languages(args.preload)
        except (ValueError, RuntimeError) as exc:
            cli.exit(1, f"{exc}\n")
    if args.warmup:
        try:
            models.predict_many([("Jo lent a book to Kim.", ("borrowed", "lent"))])
            if "en" in (args.preload or []):
                with language_cache().acquire("en", models.parser) as bundle:
                    models.coreference_model(bundle).predict(
                        texts=["Jo wrote a book. She sold it."]
                    )
        except Exception as exc:  # noqa: BLE001 - report failure before starting the server
            cli.exit(1, f"Inference warmup failed: {exc}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)
