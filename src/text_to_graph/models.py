"""Lazy local inference, with reproducible Hugging Face checkpoint revisions."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import batched, pairwise
from pathlib import Path
from threading import Lock, RLock

from . import backend
from .batching import isolated
from .cache import LanguageCache, default_limit
from .catalog import ASSETS, ROUTES, asset_path
from .vocabulary import SENSES

_inference_lock = RLock()
# MLX model creation, warmup, and inference share a thread-local Metal stream.
_metal_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gliner-metal")


_language_cache = None
_cache_lock = Lock()


def language_cache():
    global _language_cache
    with _cache_lock:
        if _language_cache is None:
            _language_cache = LanguageCache(default_limit())
        return _language_cache


PARSERS = {k: v["model"] for k, v in ROUTES.items()}
GLINER = ("fastino/gliner2.5-multi-v1", "2ca71aafb3446d9014e1c55c7ff51c9bc7209c47")
COREF = ("biu-nlp/f-coref", "e91dfff12879495d882ba9460d0c5d5dd44ade59")


def snapshot(model):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    folder = "hf/gliner" if model == GLINER else "hf/coref"
    required = [
        (asset_path(name), spec["size"])
        for name, spec in ASSETS.items()
        if name.startswith(folder + "/")
    ]
    if required and all(
        path.is_file() and path.stat().st_size == size for path, size in required
    ):
        return str(asset_path(folder))
    name, rev = model
    cache_dir = asset_path("hf/hub")
    try:
        cached = Path(
            snapshot_download(
                name, revision=rev, cache_dir=cache_dir, local_files_only=True
            )
        )
        required = (
            ["encoder_config/config.json", "model.safetensors"]
            if model == GLINER
            else ["pytorch_model.bin"]
        )
        if all(
            (cached / f).is_file()
            for f in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                *required,
            ]
        ):
            return str(cached)
    except LocalEntryNotFoundError:
        pass
    return snapshot_download(name, revision=rev, cache_dir=cache_dir)


def parser(lang):
    import spacy

    if ROUTES[lang]["backend"] == "installed":
        return spacy.load(PARSERS[lang])
    from .native import load_parser

    return load_parser(lang)


@lru_cache(maxsize=1)
def extractor():
    from gliner2 import AutoExtractor

    path = snapshot(GLINER)
    if backend.selected() == "metal":
        from .metal import load

        return load(path)
    return AutoExtractor.from_pretrained(
        path, local_files_only=True, map_location=backend.selected()
    )


def resolver(nlp):
    from fastcoref import FCoref

    return FCoref(
        model_name_or_path=snapshot(COREF),
        nlp=nlp,
        device=backend.devices()["english_coreference"],
        enable_progress_bar=False,
    )


@lru_cache(maxsize=128)
def schema(relations):
    result = (
        extractor()
        .create_schema()
        .entities(
            ["person", "organization", "place", "work", "object", "quantity", "time"]
        )
    )
    result.relations(list(relations))
    result.classification("predicate_senses", SENSES, multi_label=True, top_k=3)
    return result


def _predict_many(items):
    with _inference_lock:
        return extractor().batch_extract_long(
            [text for text, _ in items],
            [schema(relations) for _, relations in items],
            batch_size=len(items),
            include_spans=True,
            include_confidence=True,
        )


def predict_many(items):
    if backend.selected() == "metal":
        return _metal_worker.submit(_predict_many, items).result()
    return _predict_many(items)


def coreference_model(bundle):
    if bundle.coref is None:
        bundle.coref = resolver(bundle.parser)
    return bundle.coref


def preload_languages(languages):
    cache = language_cache()
    languages = list(dict.fromkeys(languages))
    if not languages:
        raise ValueError("Specify at least one language code")
    unsupported = [code for code in languages if code not in ROUTES]
    if unsupported:
        raise ValueError(f"No parser for language code(s): {', '.join(unsupported)}")
    if len(languages) > cache.limit:
        raise ValueError(
            f"Requested {len(languages)} languages but the cache limit is {cache.limit}"
        )
    for code in languages:
        try:
            with cache.acquire(code, parser) as bundle:
                if code == "en":
                    coreference_model(bundle)
                elif code == "zh":
                    bundle.parser.reader("traditional")
        except Exception as exc:
            raise RuntimeError(f"Failed to preload {code}: {exc}") from exc
    return cache.resident()


def _pronoun_windows(doc, clusters):
    sentences = list(doc.sents) if doc is not None else []
    if len(sentences) > 2:
        for left, right in pairwise(sentences):
            if any(
                t.pos_ == "PRON"
                and not any(a <= t.idx < z for cluster in clusters for a, z in cluster)
                for t in right
            ):
                yield left.start_char, right.end_char


def coreference_batch(sources, docs, bundle):
    coref = coreference_model(bundle)

    def predict(texts):
        predictions = coref.predict(texts=list(texts), max_tokens_in_batch=2048)
        rows = []
        for text, prediction in zip(texts, predictions, strict=True):
            mentions = prediction.get_clusters(as_strings=False)
            if any(
                not 0 <= start < end <= len(text)
                for cluster in mentions
                for start, end in cluster
            ):
                raise ValueError("Unaligned coreference span")
            rows.append(mentions)
        return rows

    clusters = isolated(sources, predict)

    def windows():
        for i, (source, doc, primary) in enumerate(
            zip(sources, docs, clusters, strict=True)
        ):
            if isinstance(primary, Exception):
                continue
            try:
                intervals = list(_pronoun_windows(doc, primary))
            except Exception as exc:  # noqa: BLE001 - isolate invalid sentence annotations
                clusters[i] = exc
                continue
            for start, end in intervals:
                yield i, start, source[start:end]

    for group in batched(windows(), 4):
        results = isolated([text for _, _, text in group], predict)
        for (i, base, _), result in zip(group, results, strict=True):
            if isinstance(result, Exception):
                clusters[i] = result
            elif not isinstance(clusters[i], Exception):
                clusters[i].extend(
                    [[(base + a, base + z) for a, z in mentions] for mentions in result]
                )
    return clusters
