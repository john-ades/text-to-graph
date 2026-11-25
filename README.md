# Text to Graph

Text to Graph turns a UTF-8 document into a JSON graph of entities, events, and relationships. Each edge links back to the words that support it. For example, a sentence about a sale can answer “Who bought Beacon?” and show where the buyer was named.

## Install once

**Model setup downloads about 17.5 GiB into `.cache/text-to-graph/` at the root of your clone.** Git ignores this directory. Allow additional space for the Python environment, its spaCy packages, and temporary downloads.

1. Install `uv` using the instructions for your operating system. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
2. From the cloned project directory, let `uv` create the Python 3.12 environment and install the locked dependencies.

   ```sh
   uv sync
   ```

3. Run the model installer. If you already have models in `~/.cache/text-to-graph`, stop the server and move that folder to the project's `.cache/text-to-graph/` before this step. If you previously set `TEXT_TO_GRAPH_MODEL_DIR`, unset it to select the project default.

   ```sh
   uv run text-to-graph-setup
   ```

   Setup verifies each file against the inventory and finishes with `Ready. 89 language routes staged.` If interrupted, run the same command again to resume.

## Extract your first graph

Open two terminals in the project directory.

**Terminal 1 — start the local server** and leave it running. It listens on `127.0.0.1:8000` and loads the language models into memory on the first request.

```sh
uv run text-to-graph-server
```

The server automatically selects Metal on supported Apple Silicon. To pin a device or warm models before the first request, see [Inference devices](#inference-devices).

**Terminal 2 — submit a sentence.** The CLI prints the graph as JSON.

```sh
uv run text-to-graph "On Monday, Harbor Robotics sold Beacon to Northstar Labs."
```

Find the edge with `"predicate": "BOUGHT"`. Match its `source` and `target` IDs to `nodes` to read **Northstar Labs → BOUGHT → Beacon**. Its `source_event` identifies the `SALE`. Follow its `evidence` IDs to find “sold”, “Northstar Labs”, and “Beacon” in the original sentence.

### Process a file, call the API, or use Python

**File.** In a shell such as Bash or Zsh, redirect your UTF-8 document into the CLI and save the JSON output to a file.

```sh
uv run text-to-graph < document.txt > graph.json
```

**HTTP.** Send the same document directly to the running server. The request also accepts an optional `language` code.

```sh
curl http://127.0.0.1:8000/v1/graph \
  -H 'Content-Type: application/json' \
  -d '{"text":"On Monday, Harbor Robotics sold Beacon to Northstar Labs."}'
```

**Python.** Start the project interpreter with `uv run python`. The library runs inference in the current process with its own model cache, so you can use it independently of the server.

```python
from text_to_graph import extract_graph

graph = extract_graph("Blue Hour was written by Nia Voss.")
print(graph.edges)
```

## Batch HTTP API

`POST /v1/graph/batch` accepts **1–32 documents** and returns their graphs in one response. Supply a `text` string for each document. Set `language` when you know the detector code, or omit it to detect the language from that document. You can mix languages in one batch.

```sh
curl -sS http://127.0.0.1:8000/v1/graph/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"text": "Blue Hour was written by Nia Voss.", "language": "en"},
      {"text": "El museo prestó un telescopio a Lucía."}
    ]
  }'
```

Read `results` in input order. Each entry has a zero-based `index`. On success, `results[i].graph` has the same fields as the response from `POST /v1/graph`. Evidence offsets refer to that input's `text`. If a model stage was unavailable, the entry still contains a graph with `graph.profile.status` set to `"partial"`.

Check each entry even when the request returns **HTTP 200**. An invalid document or an extraction failure produces an `error` for that index, and the other documents keep their results. For example, sending an unknown language code on the second document produces the following entry in `results`.

```json
{"index": 1, "error": {"code": "invalid_document", "message": "Unknown detector language code"}}
```

A per-document processing failure uses the code `extraction_failed`. If `documents` is missing, is not an array, or falls outside the 1–32 limit, the server returns **HTTP 422** for the whole request. For larger inputs, send groups of up to 32 documents. The service groups passages by language, predicts up to four short sentences together, and processes long sentences in overlapping windows.

## Read the graph

| Field | What to look for |
| --- | --- |
| `nodes` | Entities, values, and events with document-local IDs. A recognized concept or event has a `canonical_id`, such as `SALE`. |
| `edges` | Directed relationships between nodes. Event roles identify participants such as the buyer. Derived relationships such as `BOUGHT` point to their source event. |
| `evidence` | Quoted source text with character offsets. In Python, `text[start:end]` reproduces the quote. |
| `unresolved` | Ambiguous references, unfamiliar predicates, and suggestions awaiting confirmation. |
| `profile` | Detected language, parser routes, model versions, and completion status. Inspect the routes when the status is `partial`. |
| `revision` | SHA-256 digest of the original UTF-8 input. |

Check `scope` when interpreting an event or a derived edge. For example, “offered to sell” can produce a sale with prospective modality. Scope also records negation, attribution, and conditions.

## Project map

| Part | Responsibility |
| --- | --- |
| [Project configuration](pyproject.toml) · [Python version](.python-version) · [Dependency lock](uv.lock) | Define the package and its three commands, select Python 3.12, and pin the dependency versions that `uv` installs. |
| [CLI and Python entry points](src/text_to_graph/__init__.py) · [HTTP server](src/text_to_graph/server.py) | Expose `extract_graph` to Python and let the CLI submit documents to a persistent local server. |
| [Installer](src/text_to_graph/setup.py) · [Catalog loader](src/text_to_graph/catalog.py) · [Model inventory](src/text_to_graph/catalog.json) | Map languages to parsers and locate, download, and verify the files used during inference. |
| [Extraction pipeline](src/text_to_graph/pipeline.py) · [Model loading](src/text_to_graph/models.py) · [Language cache](src/text_to_graph/cache.py) | Detect the language and route passages to a parser. GLiNER proposes entities and relationships, FastCoref resolves English references, and the cache reuses models across requests. |
| [Parser adapters](src/text_to_graph/native.py) | Adapt language-specific parsers to common token and sentence records. The package includes readers for [Amharic](src/text_to_graph/amharic.py), [Egyptian Arabic](src/text_to_graph/egyptian.py), and [Malay](src/text_to_graph/malay.py), plus small [Amharic feature tables](src/text_to_graph/amharic_features.json.gz) and [Stanza resource metadata](src/text_to_graph/stanza_resources.json). |
| [Semantics](src/text_to_graph/semantics.py) · [Vocabulary](src/text_to_graph/vocabulary.py) · [Graph records](src/text_to_graph/graph.py) | Assign event roles and scope, map them to shared predicates, and assemble nodes and edges with source evidence. |
| [Tests](tests/) · [Evaluation scripts and samples](evaluation/) | Check extraction and service behavior, compare predictions with hand-labeled readings, and exercise each parser on native text. |
| [Design proposal](docs/proj-proposal.md) | Explains the design rationale and the intended readings of the seven [example paragraphs](evaluation/example_paragraphs.txt). |

## Language coverage and server options

The detector recognizes **176 language codes** and **89** have a local parser. Relation coverage depends on the language and construction. Tests cover lending and borrowing in Arabic, Danish, English, French, German, Indonesian, Italian, Malay, Portuguese, Russian, and Spanish. English cases also cover sales, authorship, and reported events.

| Task | Command |
| --- | --- |
| List parsers and tested constructions by language | `uv run text-to-graph-setup --list` |
| Set the input language explicitly | `uv run text-to-graph --language id "Ani meminjamkan buku kepada Budi."` |
| Inspect server health and loaded languages | `curl http://127.0.0.1:8000/health` |
| Preload languages on the running server | `uv run text-to-graph --preload ar id` |
| Start with English preloaded and room for eight languages | `uv run text-to-graph-server --preload en --max-loaded-languages 8` |

The server keeps up to four languages in memory by default and evicts the least recently used idle language when the cache fills. English and its coreference model share a slot, as do both Chinese script variants. Set the default with `TEXT_TO_GRAPH_MAX_LOADED_LANGUAGES` or override it with the server flag. To use a different port, pass the same `--port` value to the server and CLI. HTTP clients can preload languages by posting `{"languages":["ar","id"]}` to `/v1/languages/preload`.

## Inference devices

The server and Python library select one backend per process. The default, `auto`, tries Apple Metal first, then CUDA, then CPU. Metal requires both MLX Metal and PyTorch MPS. If you explicitly request `metal` or `cuda` and it is unavailable, the server reports an error at startup.

| Backend | GLiNER encoder | GLiNER extraction heads | English coreference (FastCoref) | Language parsers |
| --- | --- | --- | --- | --- |
| `metal` | MLX on Apple GPU (FP32) | PyTorch CPU | PyTorch MPS | CPU |
| `cuda` | PyTorch CUDA | PyTorch CUDA | PyTorch CUDA | CPU |
| `cpu` | PyTorch CPU | PyTorch CPU | PyTorch CPU | CPU |

For an Apple Silicon deployment, use `--backend metal` to require the Apple GPU. Add `--warmup` to run GLiNER before accepting requests. Pair it with `--preload en` to load the English parser and warm FastCoref as well. The Metal encoder uses the existing checkpoint and caps MLX's reusable-buffer cache at 64 MiB.

```sh
uv run text-to-graph-server --backend metal --preload en --warmup
```

You can also set `TEXT_TO_GRAPH_BACKEND` before starting the server or a Python process that calls `extract_graph`. An explicit server flag overrides the environment setting. Both HTTP endpoints and the CLI use the device selected by the running server.

```sh
TEXT_TO_GRAPH_BACKEND=cpu uv run text-to-graph-server
```

To verify which device each stage uses, query the running server. With `jq` installed, this prints just the `inference` field from `GET /health`.

```sh
curl -sS http://127.0.0.1:8000/health | jq '.inference'
```

On Metal the field contains the following values.

```json
{"backend":"metal","gliner_encoder":"mlx","gliner_heads":"cpu","english_coreference":"mps","parser":"cpu"}
```

### Local HTTP benchmark

The benchmark starts a fresh localhost server with staged weights. It alternates batches and sequential requests over the same short English, French, and Spanish documents, with explicit language codes. Throughput includes JSON serialization and transport. Run it with the backend used in deployment.

```sh
uv run python evaluation/latency.py --backend metal --repeats 5
uv run python evaluation/latency.py --backend cpu --repeats 5
```

On an Apple A18 Pro Mac with 8 GiB of unified memory, a paired CPU check alternated the previous single endpoint (commit `7360338`) and the updated endpoint on the same loaded models. Across 64 requests per version, median latency went from 205 to 191 ms and p95 went from 358 to 321 ms.

Five repeats on the same Mac produced the following warm HTTP measurements. Each latency is for a complete batch request.

| Backend | Documents | Sequential docs/s | Batch docs/s | Batch p50 / p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| Metal | 1 | 8.19 | 8.26 | 0.121 / 0.143 |
| Metal | 8 | 5.77 | 6.76 | 1.184 / 1.342 |
| Metal | 32 | 6.0 | 6.54 | 4.891 / 5.185 |
| CPU | 1 | 5.38 | 5.51 | 0.182 / 0.186 |
| CPU | 8 | 5.03 | 5.55 | 1.441 / 1.456 |
| CPU | 32 | 4.88 | 5.4 | 5.922 / 5.962 |

The first request took 10.25 s on Metal and 12.86 s on CPU with models staged on disk and warmup disabled. Peak process RSS was 2622 MiB on Metal and 2823 MiB on CPU. Sampled Metal driver allocations peaked at 3252 MiB. RSS and driver allocations can overlap in unified memory.

CPU throughput varied between runs. In a second five-repeat trial, eight documents ran at 2.83 docs/s in batch and 2.90 sequentially. At 32 documents that trial reached 3.63 docs/s in batch and 3.30 sequentially. CUDA hardware qualification is pending.

## Tests and evaluation

Run these from the project directory after installation.

| Command | What it checks |
| --- | --- |
| `uv run python -m unittest discover -s tests` | Graph extraction, evidence offsets, CLI and API behavior, model setup, and cache eviction. |
| `uv run python evaluation/held_out.py` | Roles, relationships, and scope for six hand-labeled English, French, and Spanish documents. |
| `uv run python evaluation/languages.py ar id` | Parser annotations and offsets, with batch parity checks on routes with batched parsers. Omit the codes to check all 89 parser routes. |
