"""Route original source passages and assemble their evidence in document coordinates."""

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from . import models
from .batching import isolated
from .catalog import LANGUAGES, ROUTES, asset_path
from .graph import Builder, DocumentGraph
from .semantics import Interpretation
from .vocabulary import FRAMES, PROJECTIONS

MAX_PASSAGE_CHARS = 12_000


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
    while end - start > MAX_PASSAGE_CHARS:
        window = text[start : start + MAX_PASSAGE_CHARS]
        cuts = list(re.finditer(r"[.!?。！？؟।](?=\s)|\n", window))
        cut = cuts[-1].end() if cuts else max(1, window.rfind(" "))
        if cut < 1000:
            cut = len(window)
        yield start, start + cut
        start += cut
    if start < end:
        yield start, end


@lru_cache(maxsize=1)
def package_versions():
    packages = {}
    for name in (
        "gliner2",
        "fastcoref",
        "spacy",
        "transformers",
        "fasttext-langdetect",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return packages


@dataclass
class Passage:
    builder: Builder
    index: int
    source: str
    route: dict
    doc: object = None
    clusters: list = field(default_factory=list)
    sentences: list = field(default_factory=list)
    predictions: list = field(default_factory=list)

    def fail(self, stage, error):
        self.route["stages"][stage] = {"status": "unavailable", "reason": str(error)}
        self.builder.graph.profile["status"] = "partial"


def prepare(b, index, language_code):
    text = b.text
    profile = b.graph.profile
    profile.update(
        vocabulary_version="2",
        status="complete",
        routes=[],
        models={"gliner": list(models.GLINER), "coreference": list(models.COREF)},
        packages=package_versions().copy(),
    )
    if not text.strip():
        return
    try:
        if language_code:
            primary = {"lang": language_code, "score": None, "method": "explicit"}
            parts = [(0, len(text), primary)]
        else:
            primary = language(text)
            parts = list(regions(text, primary))
        profile["language"] = primary
    except Exception as exc:  # noqa: BLE001 - record detector failures in the graph
        parts = [(0, len(text), {"lang": "und", "score": 0})]
        profile.update(status="partial", language_error=str(exc))
    for a, z, reading in parts:
        lang, score = reading["lang"], reading["score"]
        for start, end in bounded(text, a, z):
            source = text[start:end]
            if not source.strip():
                continue
            route = {
                "start": start,
                "end": end,
                "language": lang,
                "score": score,
                "parser": models.PARSERS.get(lang),
                "backend": ROUTES.get(lang, {}).get("backend"),
                "routing_status": "explicit"
                if score is None
                else "tentative"
                if score < 0.5
                else "detected",
                "stages": {},
            }
            profile["routes"].append(route)
            yield Passage(b, index, source, route)


def parse_group(lang, group):
    texts = [p.source for p in group]
    if lang != "en":
        for p in group:
            p.route["stages"]["coreference"] = {
                "status": "unqualified",
                "language": lang,
            }
    try:
        if lang not in models.PARSERS:
            raise ValueError(f"No registered parser for {lang}")
        with models.language_cache().acquire(lang, models.parser) as bundle:
            parser = bundle.parser

            def parse(items):
                if len(items) > 1 and hasattr(parser, "pipe"):
                    return parser.pipe(items, batch_size=len(items))
                return map(parser, items)

            for p, doc in zip(group, isolated(texts, parse), strict=True):
                if isinstance(doc, Exception):
                    p.fail("parser", doc)
                else:
                    p.doc = doc
                    p.route["parser"] = getattr(doc, "parser_name", p.route["parser"])
                    p.route["stages"]["parser"] = {
                        "status": "complete",
                        "version": parser.meta["version"],
                    }
            if lang == "en":
                results = models.coreference_batch(
                    texts, [p.doc for p in group], bundle
                )
                for p, result in zip(group, results, strict=True):
                    if isinstance(result, Exception):
                        p.fail("coreference", result)
                    else:
                        p.clusters = result
                        p.route["stages"]["coreference"] = {"status": "complete"}
    except Exception as exc:  # noqa: BLE001 - capture provider-specific failures under the active stage
        for p in group:
            for stage in ("parser", "coreference"):
                if stage not in p.route["stages"]:
                    p.fail(stage, exc)


def relation_labels(p):
    if p.doc is None or p.route["language"] not in {"en", "fr", "es"}:
        frames = PROJECTIONS
    else:
        frames = {FRAMES[t.lemma_][0] for t in p.doc if t.lemma_ in FRAMES}
    return tuple(
        sorted(
            {pred.lower() for f in frames for _, pred, _ in PROJECTIONS.get(f, [])}
            or {"related_to"}
        )
    )


def predict_sentences(group):
    jobs = []
    for p in group:
        try:
            labels = relation_labels(p)
            p.sentences = (
                [(s.start_char, s.text) for s in p.doc.sents]
                if p.doc is not None
                else [(0, p.source)]
            )
        except Exception as exc:  # noqa: BLE001 - a malformed parse can still use the whole-passage extractor
            p.fail("parser", exc)
            p.doc = None
            p.sentences = [(0, p.source)]
            labels = relation_labels(p)
        p.predictions = [None] * len(p.sentences)
        for i, (_, text) in enumerate(p.sentences):
            jobs.append((text, labels, p, i))

    def run(batch):
        results = isolated(
            [(text, labels) for text, labels, _, _ in batch], models.predict_many
        )
        for (_, _, p, i), result in zip(batch, results, strict=True):
            p.predictions[i] = result

    pending = []
    for job in sorted(jobs, key=lambda job: len(job[0])):
        # Long sentences may expand to multiple overlapping encoder windows.
        if len(job[0]) > 384:
            if pending:
                run(pending)
                pending = []
            run([job])
        else:
            pending.append(job)
            if len(pending) == 4:
                run(pending)
                pending = []
    if pending:
        run(pending)


def passage_batches(passages):
    group = []
    characters = 0
    for passage in passages:
        length = len(passage.source)
        if group and (len(group) == 8 or characters + length > MAX_PASSAGE_CHARS):
            yield group
            group = []
            characters = 0
        group.append(passage)
        characters += length
    if group:
        yield group


def extract_many(documents):
    """Return a DocumentGraph or an exception for each (text, language) pair."""
    results = [None] * len(documents)

    def passages():
        for index, (text, code) in enumerate(documents):
            try:
                if not isinstance(text, str):
                    raise TypeError("text must be a string")
                if code is not None and code not in LANGUAGES:
                    raise ValueError("Unknown detector language code")
                builder = Builder(text)
                results[index] = builder.graph
                yield from prepare(builder, index, code)
            except Exception as exc:  # noqa: BLE001 - isolate document preparation failures
                results[index] = exc

    for group in passage_batches(passages()):
        languages = defaultdict(list)
        for p in group:
            languages[p.route["language"]].append(p)
        for lang, items in languages.items():
            parse_group(lang, items)
        predict_sentences(group)
        for p in group:
            try:
                assemble(p)
            except Exception as exc:  # noqa: BLE001 - isolate graph assembly failures
                results[p.index] = exc
    return results


def extract_graph(text: str, *, language_code=None) -> DocumentGraph:
    graph = extract_many([(text, language_code)])[0]
    if isinstance(graph, Exception):
        raise graph
    return graph


def assemble(p):
    b, doc, clusters, route = p.builder, p.doc, p.clusters, p.route
    lang, start = route["language"], route["start"]
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
    candidates = {}
    proposals = []
    sense_candidates = {}
    route["stages"]["gliner"] = {"status": "complete"}
    for (base, passage), result in zip(p.sentences, p.predictions, strict=True):
        if isinstance(result, Exception):
            p.fail("gliner", result)
            continue
        try:
            collect_predictions(
                b, lang, start, base, passage, result, candidates, proposals
            )
            sense_candidates[base] = result.get("predicate_senses", [])
        except Exception as exc:  # noqa: BLE001 - record malformed provider output for this passage
            p.fail("gliner", exc)
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
