"""Source-aligned syntax from local parsers and qualified native loan constructions."""

import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .catalog import ROUTES, asset_path
from .vocabulary import LOAN_ITEMS


@dataclass
class Word:
    id: int
    head: int
    dep: str
    pos: str
    lemma: str
    start: int
    end: int
    feats: str = ""


@dataclass
class Sentence:
    start_char: int
    text: str
    words: list


def sentence(text, words):
    if not words or any(not 0 <= w.start < w.end <= len(text) for w in words):
        raise ValueError("Parser returned unaligned tokens")
    start, end = words[0].start, words[-1].end
    return Sentence(start, text[start:end], words)


def udpipe_reader(path):
    from ufal.udpipe import Model, ProcessingError
    from ufal.udpipe import Sentence as UDSentence

    model = Model.load(str(path))
    if model is None:
        raise OSError(f"Cannot open {path}. Run text-to-graph-setup.")

    def read(text):
        tokenizer = model.newTokenizer("ranges")
        tokenizer.setText(text)
        sent, error = UDSentence(), ProcessingError()
        while tokenizer.nextSentence(sent, error):
            if not model.tag(sent, Model.DEFAULT) or not model.parse(
                sent, Model.DEFAULT
            ):
                raise ValueError("UDPipe failed to parse sentence")
            surface = {
                i: m
                for m in sent.multiwordTokens
                for i in range(m.idFirst, m.idLast + 1)
            }
            words = []
            for w in list(sent.words)[1:]:
                token = surface.get(w.id, w)
                words.append(
                    Word(
                        w.id,
                        w.head,
                        w.deprel,
                        w.upostag,
                        w.lemma,
                        token.getTokenRangeStart(),
                        token.getTokenRangeEnd(),
                        w.feats,
                    )
                )
            yield sentence(text, words)
        if error.occurred():
            raise ValueError(error.message)

    return read


def stanza_pipeline(route):
    from stanza import Pipeline
    from stanza.pipeline.core import DownloadMethod

    options = {}
    if route["model"] == "or":
        from stanza.models.common.foundation_cache import FoundationCache

        cache = FoundationCache(local_files_only=True)
        local = str(asset_path("hf/muril"))
        cache.load_bert(local)
        cache.bert["google/muril-large-cased"] = cache.bert.pop(local)
        options["foundation_cache"] = cache
    # The standalone Turkic checkpoints predate Stanza's removal of the four
    # reserved vocabulary symbols from the relation classifier.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp:
        if route["model"] in {"az", "ba", "tk", "tt"}:
            import torch
            from stanza.models.common.vocab import VOCAB_PREFIX_SIZE

            source = asset_path(f"stanza/{route['model']}/depparse/native.pt")
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
            state = checkpoint["model"]
            prefix = "deprel.scorer.W_bilin."
            labels = checkpoint["vocab"]["deprel"]["_id2unit"]
            if state[prefix + "bias"].numel() == len(labels):
                state[prefix + "weight"] = state[prefix + "weight"][
                    ..., VOCAB_PREFIX_SIZE:
                ].contiguous()
                state[prefix + "bias"] = state[prefix + "bias"][
                    VOCAB_PREFIX_SIZE:
                ].contiguous()
                converted = Path(temp) / "parser.pt"
                torch.save(checkpoint, converted)
                options["depparse_model_path"] = str(converted)
            del checkpoint, state
        return Pipeline(
            lang=route["model"],
            dir=str(asset_path("stanza")),
            package=None,
            processors=route["processors"],
            use_gpu=False,
            resources_filepath=str(Path(__file__).with_name("stanza_resources.json")),
            download_method=DownloadMethod.NONE,
            verbose=False,
            **options,
        )


def stanza_parser(route):
    nlp = stanza_pipeline(route)

    def convert(text, doc):
        for sent in doc.sentences:
            words = [
                Word(
                    w.id,
                    w.head,
                    w.deprel,
                    w.upos,
                    w.lemma or w.text,
                    w.start_char if w.start_char is not None else t.start_char,
                    w.end_char if w.end_char is not None else t.end_char,
                    w.feats or "",
                )
                for t in sent.tokens
                for w in t.words
            ]
            yield sentence(text, words)

    return NativeParser(
        lambda text: convert(text, nlp(text)),
        route["revision"],
        lambda texts: (
            convert(text, doc)
            for text, doc in zip(texts, nlp.bulk_process(texts), strict=True)
        ),
    )


def spacy_reader(route):
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(
        route["model"]
        if route["backend"] == "spacy_package"
        else asset_path(route["model"])
    )
    segment = stanza_pipeline(route["segmenter"]) if "segmenter" in route else None

    def tokenize(text):
        if segment:
            tokens = [t for s in segment(text).sentences for t in s.tokens]
            spans = [(t.start_char, t.end_char) for t in tokens]
            doc = Doc(
                nlp.vocab,
                words=[text[a:z] for a, z in spans],
                spaces=[False] * len(spans),
            )
        else:
            doc, spans = text, None
        return doc, spans

    def convert(text, doc, spans):
        if spans is None:
            spans = [(t.idx, t.idx + len(t)) for t in doc]
        for sent in doc.sents:
            words = [
                Word(
                    t.i + 1,
                    0 if t.head == t else t.head.i + 1,
                    t.dep_.lower(),
                    t.pos_,
                    t.lemma_ or t.text,
                    *spans[t.i],
                    str(t.morph),
                )
                for t in sent
            ]
            yield sentence(text, words)

    def read(text):
        value, spans = tokenize(text)
        return convert(text, nlp(value), spans)

    def bulk(texts):
        values, offsets = zip(*(tokenize(text) for text in texts))
        docs = nlp.pipe(values, batch_size=len(texts))
        return (
            convert(text, doc, spans)
            for text, doc, spans in zip(texts, docs, offsets, strict=True)
        )

    version = route.get("revision")
    if route["backend"] == "spacy_package":
        from importlib.metadata import version as package_version

        version = package_version(route["model"])
    return NativeParser(read, version, bulk)


def cantonese_reader(route):
    from transformers import pipeline

    nlp = pipeline(
        "universal-dependencies",
        model=str(asset_path(route["model"])),
        trust_remote_code=True,
        device=-1,
        aggregation_strategy="simple",
    )

    def read(text):
        for part in re.finditer(r"[^。！？\r\n]{1,120}[。！？]?|[。！？]", text):
            cursor = part.start()
            words = []
            for line in nlp(part[0]).splitlines():
                if not line or line.startswith("#"):
                    continue
                i, form, _lemma, pos, _, feats, head, dep, _, _ = line.split("\t")
                start = text.index(form, cursor, part.end())
                cursor = start + len(form)
                words.append(
                    Word(int(i), int(head), dep, pos, form, start, cursor, feats)
                )
            if words:
                yield sentence(text, words)

    return read


class NativeParser:
    def __init__(self, reader, model, bulk=None):
        self.bulk = bulk
        self.read = reader
        self.meta = {"version": model}

    def __call__(self, text):
        return SimpleNamespace(sents=list(self.read(text)))

    def pipe(self, texts, *, batch_size):
        if self.bulk is None:
            return map(self, texts)
        return (
            SimpleNamespace(sents=list(sentences)) for sentences in self.bulk(texts)
        )


class ChineseParser:
    """Keep both script variants under one language lease."""

    def __init__(self, route):
        self.route = route
        self.readers = {"model": udpipe_reader(asset_path(route["model"]))}
        self.meta = {"version": "UDPipe UD 2.5"}

    def reader(self, variant):
        if variant not in self.readers:
            self.readers[variant] = udpipe_reader(asset_path(self.route[variant]))
        return self.readers[variant]

    def __call__(self, text):
        import hanzidentifier

        variant = (
            "traditional"
            if hanzidentifier.identify(text) == hanzidentifier.TRADITIONAL
            else "model"
        )
        return SimpleNamespace(
            sents=list(self.reader(variant)(text)), parser_name=self.route[variant]
        )


def load_parser(lang):
    route = ROUTES[lang]
    if lang == "zh":
        return ChineseParser(route)
    backend = route["backend"]
    if backend == "stanza":
        return stanza_parser(route)
    if backend in {"spacy", "spacy_package"}:
        return spacy_reader(route)
    if backend == "udpipe":
        reader = udpipe_reader(asset_path(route["model"]))
    elif backend == "camel":
        from .egyptian import reader as egyptian_reader

        reader = egyptian_reader(route)
    elif backend == "amharic":
        from .amharic import reader as amharic_reader

        reader = amharic_reader(route)
    elif backend == "malaya":
        from .malay import reader as malay_reader

        reader = malay_reader(route)
    elif backend == "cantonese":
        reader = cantonese_reader(route)
    else:
        raise ValueError(f"Unknown parser backend {backend}")
    return NativeParser(reader, route["revision"])


# Qualified entries specify the lending predicate, recipient marker, and negation.
LOANS = {
    "ar": ("أعار أعارت تُعِر", "إلى", "لم لا ما"),
    "ms": ("meminjamkan", "kepada", "tidak"),
    "id": ("meminjamkan", "kepada", "tidak"),
    "da": ("låne", "til", "ikke"),
    "ru": ("одолжить", "", "не"),
    "de": ("verleihen", "an", "nicht kein keine keinen keinem"),
    "it": ("prestare", "a", "non"),
    "pt": ("emprestar empresta", "a para", "não"),
}
QUALIFIED = set(LOANS)


def interpret(b, doc, base, lang, *, qualified):
    def evidence(w):
        return b.span(base + w.start, base + w.end, lang)

    for sent in doc.sents:
        for verb in sent.words:
            if verb.pos != "VERB" and not (verb.pos == "X" and verb.head == 0):
                continue
            anchor = evidence(verb)
            binding = loan_roles(sent, verb, lang, b.text, base) if qualified else None
            if binding is None:
                event = b.node(
                    "event",
                    anchor,
                    mapping_status="unresolved",
                    lemma=verb.lemma,
                    scope={"polarity": "unknown", "modality": "unknown"},
                )
                arguments = [
                    {"dependency": w.dep, "evidence": [evidence(w)]}
                    for w in sent.words
                    if w.head == verb.id
                    and w.dep.split(":")[0] in {"obj", "iobj", "obl", "nsubj", "csubj"}
                ]
                b.graph.unresolved.append(
                    {
                        "kind": "predicate",
                        "event": event,
                        "lemma": verb.lemma,
                        "evidence": [anchor],
                        "arguments": arguments,
                    }
                )
                continue
            roles, neg = binding
            scope = {
                "polarity": "negative" if neg else "positive",
                "modality": "asserted",
            }
            event = b.node("event", anchor, "LOAN", lemma=verb.lemma, scope=scope)
            if neg:
                b.nodes[event]["evidence"].append(evidence(neg))
            entities = {}
            for role, w in roles.items():
                span = evidence(w)
                concept = None
                if role == "ITEM":
                    forms = LOAN_ITEMS[lang]
                    concept = forms.get(w.lemma.casefold()) or forms.get(
                        b.spans[span]["text"].casefold()
                    )
                entities[role] = b.node("entity", span, concept)
                b.edge(event, role, entities[role], [anchor, span])
            for role, pred in (("LENDER", "LENT"), ("BORROWER", "BORROWED")):
                b.edge(
                    entities[role],
                    pred,
                    entities["ITEM"],
                    list(b.nodes[event]["evidence"])
                    + [evidence(roles[role]), evidence(roles["ITEM"])],
                    source_event=event,
                    rule=f"{role}.ITEM",
                    scope=scope,
                )


def loan_roles(sent, verb, lang, text, base):
    if lang not in LOANS or verb.head != 0:
        return
    verbs, markers, negatives = [v.split() for v in LOANS[lang]]

    def form(w):
        return text[base + w.start : base + w.end].casefold()

    if verb.lemma.casefold() not in verbs and form(verb) not in verbs:
        return
    children = [w for w in sent.words if w.head == verb.id]
    # Qualify simple single-clause assertions. Exclude coordination, quotation,
    # questions and modifiers whose scope the language rules cannot establish.
    if any(c in sent.text for c in '?"“”«»？؟') or any(
        w.dep.split(":")[0]
        in {"conj", "ccomp", "xcomp", "advcl", "aux", "mark", "parataxis"}
        or (w.dep in {"advmod", "neg"} and form(w) not in negatives)
        or "Mood=Cnd" in w.feats
        or "Mood=Sub" in w.feats
        for w in sent.words
    ):
        return
    nominals = {"PROPN", "NOUN", "X"} if lang in {"ar", "ms"} else {"PROPN", "NOUN"}
    roles = {}
    neg = None
    objects = {w.id for w in children if w.dep == "obj"}
    linked = [
        w
        for w in sent.words
        if w.head in objects
        and (
            w.dep == "nmod"
            and (w.pos == "PROPN" or lang in {"ms", "ar"} and w.pos == "X")
            or form(w) in negatives
        )
    ]
    for w in children + linked:
        role = None
        if form(w) in negatives or "Polarity=Neg" in w.feats:
            neg = w
        elif w.dep == "nsubj":
            role = "LENDER"
        elif w.dep == "obj":
            role = "ITEM"
        elif w.dep == "iobj" or (
            w.dep.split(":")[0] in {"obl", "nmod"}
            and any(
                t.head == w.id and t.dep == "case" and form(t) in markers
                for t in sent.words
            )
        ):
            role = "BORROWER"
        if role:
            if role in roles or w.pos not in nominals:
                return
            roles[role] = w
    if len(roles) == 3:
        return roles, neg
