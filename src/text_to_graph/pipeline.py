"""Route original source passages and assemble their evidence in document coordinates."""

import os
import re
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, version

from . import models
from .catalog import LANGUAGES, ROUTES, asset_path
from .graph import Builder, DocumentGraph
from .semantics import Interpretation
from .vocabulary import FRAMES, PROJECTIONS


def language(text):
    os.environ.setdefault(
        "FTLANG_CACHE", str(asset_path("detector/lid.176.ftz").parent)
    )
    from ftlangdetect import detect

    first = detect(text=text, low_memory=True)
    if first["score"] < 0.5 and text.casefold() != text:
        folded = detect(text=text.casefold(), low_memory=True)
        if folded["score"] > first["score"]:
            return {**folded, "original_detection": first, "method": "casefold_recheck"}
    return first


def regions(text, primary):
    """Recheck substantive sentences for language switches, retaining contiguous context."""
    start = 0
    route = primary
    for block in re.finditer(r"[^\r\n]+", text):
        paragraph = block[0]
        paragraph_route = primary
        if (
            len(paragraph.split()) >= 6 or len(paragraph.strip()) >= 30
        ) and paragraph.strip() != text.strip():
            reading = language(paragraph)
            if reading["score"] >= 0.5:
                paragraph_route = reading
        for match in re.finditer(r"[^.!?。！？؟।]+[.!?。！？؟।]*", paragraph):
            chosen = paragraph_route
            if (len(match[0].split()) >= 10 or len(match[0].strip()) >= 48) and match[
                0
            ].strip() != paragraph.strip():
                reading = language(match[0])
                if reading["score"] >= 0.92:
                    chosen = reading
            offset = block.start() + match.start()
            if route["lang"] != chosen["lang"]:
                if start < offset:
                    yield start, offset, route
                start = offset
            route = chosen
    yield start, len(text), route


def bounded(text, start, end):
    # Keep parser and resolver inputs bounded even on a single unpunctuated line.
    while end - start > 12000:
        window = text[start : start + 12000]
        cuts = list(re.finditer(r"[.!?。！？؟।](?=\s)|\n", window))
        cut = cuts[-1].end() if cuts else max(1, window.rfind(" "))
        if cut < 1000:
            cut = len(window)
        yield start, start + cut
        start += cut
    if start < end:
        yield start, end


def extract_graph(text: str, *, language_code=None) -> DocumentGraph:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if language_code is not None and language_code not in LANGUAGES:
        raise ValueError("Unknown detector language code")
    b = Builder(text)
    profile = b.graph.profile
    profile.update(
        vocabulary_version="2",
        status="complete",
        routes=[],
        models={"gliner": list(models.GLINER), "coreference": list(models.COREF)},
        packages={},
    )
    for package in (
        "gliner2",
        "fastcoref",
        "spacy",
        "transformers",
        "fasttext-langdetect",
    ):
        try:
            profile["packages"][package] = version(package)
        except PackageNotFoundError:
            profile["packages"][package] = None
    if not text.strip():
        return b.graph
    try:
        if language_code:
            primary = {"lang": language_code, "score": None, "method": "explicit"}
            parts = [(0, len(text), primary)]
        else:
            primary = language(text)
            parts = list(regions(text, primary))
        profile["language"] = primary
    except Exception as exc:  # noqa: BLE001 - preserve failures at the model boundary
        primary = {"lang": "und", "score": 0}
        parts = [(0, len(text), primary)]
        profile.update(status="partial", language_error=str(exc))
    for a, z, reading in parts:
        for start, end in bounded(text, a, z):
            if text[start:end].strip():
                analyze(b, start, end, reading)
    return b.graph


def analyze(b, start, end, reading):
    lang = reading["lang"]
    bundle = None
    reason = f"No registered parser for {lang}"
    with ExitStack() as stack:
        if lang in models.PARSERS:
            cache = models.language_cache()
            try:
                bundle = stack.enter_context(cache.acquire(lang, models.parser))
            except Exception as exc:  # noqa: BLE001 - model loaders use provider-specific exceptions
                reason = str(exc)
        analyze_passage(b, start, end, reading, bundle, reason)


def analyze_passage(b, start, end, reading, bundle, reason):
    source = b.text[start:end]
    lang = reading["lang"]
    route = {
        "start": start,
        "end": end,
        "language": lang,
        "score": reading["score"],
        "parser": models.PARSERS.get(lang),
        "backend": ROUTES.get(lang, {}).get("backend"),
        "stages": {},
    }
    route["routing_status"] = (
        "explicit"
        if reading["score"] is None
        else "tentative"
        if reading["score"] < 0.5
        else "detected"
    )
    b.graph.profile["routes"].append(route)
    stages = route["stages"]

    def failed(stage, exc):
        stages[stage] = {"status": "unavailable", "reason": str(exc)}
        b.graph.profile["status"] = "partial"

    doc = None
    if bundle is not None:
        try:
            parser = bundle.parser
            doc = parser(source)
            if hasattr(doc, "parser_name"):
                route["parser"] = doc.parser_name
            stages["parser"] = {"status": "complete", "version": parser.meta["version"]}
        except Exception as exc:  # noqa: BLE001 - preserve failures at the model boundary
            failed("parser", exc)
    else:
        failed("parser", reason)
    clusters = []
    if lang == "en" and bundle is None:
        failed("coreference", reason)
    elif lang == "en":
        try:
            clusters = models.coreference_clusters(source, doc, bundle)
            stages["coreference"] = {"status": "complete"}
        except Exception as exc:  # noqa: BLE001 - preserve failures at the model boundary
            failed("coreference", exc)
    else:
        stages["coreference"] = {"status": "unqualified", "language": lang}
    relations = {p.lower() for triples in PROJECTIONS.values() for _, p, _ in triples}
    native = lang not in {"en", "fr", "es"}
    if native:
        from .native import QUALIFIED

        route["canonical_constructions"] = (
            ["loan"]
            if lang in QUALIFIED
            and doc is not None
            and route["routing_status"] != "tentative"
            else []
        )
        b.graph.profile["status"] = "partial"
    if doc is not None and not native:
        senses = {FRAMES[t.lemma_][0] for t in doc if t.lemma_ in FRAMES}
        relations = {
            p.lower() for s in senses for _, p, _ in PROJECTIONS.get(s, [])
        } or {"related_to"}
    candidates = {}
    proposals = []
    sense_candidates = {}
    passages = (
        [(s.start_char, s.text) for s in doc.sents]
        if doc is not None
        else [(0, source)]
    )
    try:
        for base, passage in passages:
            result = models.predict(passage, relations)
            collect_predictions(
                b, lang, start, base, passage, result, candidates, proposals
            )
            sense_candidates[base] = result.get("predicate_senses", [])
        stages["gliner"] = {"status": "complete"}
    except Exception as exc:  # noqa: BLE001 - preserve failures at the model boundary
        failed("gliner", exc)
    if doc is not None and native:
        from .native import interpret

        interpret(b, doc, start, lang, qualified=bool(route["canonical_constructions"]))
    if doc is not None and not native:
        analysis = Interpretation(b, doc, start, lang, clusters, candidates)
        analysis.run()
        for index, event in analysis.events.items():
            scores = sense_candidates.get(doc[index].sent.start_char, [])
            if scores:
                b.nodes[event]["sentence_sense_candidates"] = scores
    used_spans = {span for node in b.graph.nodes for span in node["evidence"]}
    for (a, z), prediction in candidates.items():
        ev = b.span(start + a, start + z, lang)
        if ev not in used_spans:
            if doc is None or native:
                b.node("mention", ev, prediction["type"], mapping_status="candidate")
            else:
                b.graph.unresolved.append(
                    dict(kind="mention_candidate", evidence=[ev], **prediction)
                )
    for proposal in proposals:
        head, tail = proposal["evidence"]
        for edge in b.graph.edges:
            if (
                edge["predicate"] == proposal["predicate"]
                and head in b.nodes[edge["source"]]["evidence"]
                and tail in b.nodes[edge["target"]]["evidence"]
                and set(proposal["evidence"]).issubset(edge["evidence"])
            ):
                edge["model_score"] = proposal["score"]
                break
        else:
            b.graph.unresolved.append(proposal)


def collect_predictions(b, lang, start, base, passage, result, candidates, proposals):
    for label, mentions in result.get("entities", {}).items():
        for m in mentions:
            a, z = m["start"], m["end"]
            if passage[a:z] == m["text"]:
                ev = b.span(start + base + a, start + base + z, lang)
                prediction = {"type": label.upper(), "score": m["confidence"]}
                b.spans[ev].setdefault("type_candidates", []).append(prediction)
                key = (base + a, base + z)
                if (
                    key not in candidates
                    or candidates[key]["score"] < prediction["score"]
                ):
                    candidates[key] = prediction
    for label, pairs in result.get("relation_extraction", {}).items():
        for pair in pairs:
            spans = [
                b.span(start + base + x["start"], start + base + x["end"], lang)
                for x in (pair["head"], pair["tail"])
                if passage[x["start"] : x["end"]] == x["text"]
            ]
            if len(spans) == 2:
                proposals.append(
                    {
                        "kind": "relation_candidate",
                        "predicate": label.upper(),
                        "evidence": spans,
                        "score": pair.get("confidence"),
                    }
                )
