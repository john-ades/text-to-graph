"""Lazy local inference, with reproducible Hugging Face checkpoint revisions."""

from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from threading import Lock, RLock

from .cache import LanguageCache, default_limit
from .catalog import ASSETS, ROUTES, asset_path
from .vocabulary import SENSES

_inference_lock = RLock()


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

    return AutoExtractor.from_pretrained(snapshot(GLINER), local_files_only=True)


def resolver(nlp):
    from fastcoref import FCoref

    return FCoref(
        model_name_or_path=snapshot(COREF),
        nlp=nlp,
        device="cpu",
        enable_progress_bar=False,
    )


def predict(text, relations):
    with _inference_lock:
        return _predict(text, relations)


def _predict(text, relations):
    model = extractor()
    schema = model.create_schema().entities(
        ["person", "organization", "place", "work", "object", "quantity", "time"]
    )
    schema.relations(sorted(relations))
    schema.classification("predicate_senses", SENSES, multi_label=True, top_k=3)
    return model.extract_long(text, schema, include_spans=True, include_confidence=True)


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


def coreference_clusters(source, doc, bundle):
    """Use adjacent sentences as a second context for unlinked pronouns."""
    coref = coreference_model(bundle)
    clusters = coref.predict(texts=[source])[0].get_clusters(as_strings=False)
    sentences = list(doc.sents) if doc is not None else []
    if len(sentences) <= 2:
        return clusters
    windows = []
    for left, right in pairwise(sentences):
        if any(
            t.pos_ == "PRON"
            and not any(a <= t.idx < z for cluster in clusters for a, z in cluster)
            for t in right
        ):
            windows.append((left.start_char, right.end_char))
    if windows:
        predictions = coref.predict(texts=[source[a:z] for a, z in windows])
        for (base, _), prediction in zip(windows, predictions):
            clusters.extend(
                [
                    [(base + a, base + z) for a, z in cluster]
                    for cluster in prediction.get_clusters(as_strings=False)
                ]
            )
    return clusters
