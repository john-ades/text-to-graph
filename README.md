# Text to Graph

Text to Graph turns a UTF-8 document into a JSON graph of entities, events, and relationships. Each edge links back to the words that support it. For example, a sentence about a sale can answer “Who bought Beacon?” and show where the buyer was named.

## Install once

**Model setup downloads about 17.5 GiB into `.cache/text-to-graph/` at the root of your clone.** Git ignores this directory. Allow additional space for the Python environment, its spaCy packages, and temporary downloads.

1. Install `uv` using the instructions for your operating system. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
2. From the cloned project directory, let `uv` create the Python 3.12 environment and install the locked dependencies.

   ```sh
   uv sync --locked
   ```

3. Run the model installer. If you already have models in `~/.cache/text-to-graph`, stop the server and move that folder to the project's `.cache/text-to-graph/` before this step. If you previously set `TEXT_TO_GRAPH_MODEL_DIR`, unset it to select the project default.

   ```sh
   uv run --locked text-to-graph-setup
   ```

   Setup verifies each file against the inventory and finishes with `Ready. 89 language routes staged.` If interrupted, run the same command again to resume.

## Extract your first graph

Open two terminals in the project directory.

**Terminal 1 — start the local server** and leave it running. It listens on `127.0.0.1:8000` and loads the language models into memory on the first request.

```sh
uv run --locked text-to-graph-server
```

**Terminal 2 — submit a sentence.** The CLI prints the graph as JSON.

```sh
uv run --locked text-to-graph "On Monday, Harbor Robotics sold Beacon to Northstar Labs."
```

Find the edge with `"predicate": "BOUGHT"`. Match its `source` and `target` IDs to `nodes` to read **Northstar Labs → BOUGHT → Beacon**. Its `source_event` identifies the `SALE`. Follow its `evidence` IDs to find “sold”, “Northstar Labs”, and “Beacon” in the original sentence.

### Process a file, call the API, or use Python

**File.** In a shell such as Bash or Zsh, redirect your UTF-8 document into the CLI and save the JSON output to a file.

```sh
uv run --locked text-to-graph < document.txt > graph.json
```

**HTTP.** Send the same document directly to the running server. The request also accepts an optional `language` code.

```sh
curl http://127.0.0.1:8000/v1/graph \
  -H 'Content-Type: application/json' \
  -d '{"text":"On Monday, Harbor Robotics sold Beacon to Northstar Labs."}'
```

**Python.** Start the project interpreter with `uv run --locked python`. The library runs inference in the current process with its own model cache, so you can use it independently of the server.

```python
from text_to_graph import extract_graph

graph = extract_graph("Blue Hour was written by Nia Voss.")
print(graph.edges)
```

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
| List parsers and tested constructions by language | `uv run --locked text-to-graph-setup --list` |
| Set the input language explicitly | `uv run --locked text-to-graph --language id "Ani meminjamkan buku kepada Budi."` |
| Inspect server health and loaded languages | `curl http://127.0.0.1:8000/health` |
| Preload languages on the running server | `uv run --locked text-to-graph --preload ar id` |
| Start with English preloaded and room for eight languages | `uv run --locked text-to-graph-server --preload en --max-loaded-languages 8` |

The server keeps up to four languages in memory by default and evicts the least recently used idle language when the cache fills. English and its coreference model share a slot, as do both Chinese script variants. Set the default with `TEXT_TO_GRAPH_MAX_LOADED_LANGUAGES` or override it with the server flag. To use a different port, pass the same `--port` value to the server and CLI. HTTP clients can preload languages by posting `{"languages":["ar","id"]}` to `/v1/languages/preload`.

## Tests and evaluation

Run these from the project directory after installation.

| Command | What it checks |
| --- | --- |
| `uv run --locked python -m unittest discover -s tests` | Graph extraction, evidence offsets, CLI and API behavior, model setup, and cache eviction. |
| `uv run --locked python evaluation/held_out.py` | Roles, relationships, and scope for six hand-labeled English, French, and Spanish documents. |
| `uv run --locked python evaluation/languages.py ar id` | Parser annotations and offsets for Arabic and Indonesian. Omit the codes to check all 89 parser routes. |
