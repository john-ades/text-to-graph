"""Document-local graph records. All evidence offsets index the original Python string."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256


@dataclass
class DocumentGraph:
    revision: str
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Builder:
    def __init__(self, text):
        self.text = text
        self.graph = DocumentGraph(sha256(text.encode()).hexdigest())
        self.nodes = {}
        self.spans = {}
        self.edges = set()

    def span(self, start, end, lang):
        key = f"s:{start}:{end}"
        if not 0 <= start < end <= len(self.text):
            raise ValueError(f"Invalid evidence span {key}")
        if key not in self.spans:
            record = {
                "id": key,
                "start": start,
                "end": end,
                "text": self.text[start:end],
                "language": lang,
            }
            self.spans[key] = record
            self.graph.evidence.append(record)
        return key

    def node(self, kind, span, canonical=None, suffix="", **fields):
        key = f"{kind}:{span[2:]}" + (f":{suffix}" if suffix else "")
        if key not in self.nodes:
            record = dict(
                id=key,
                kind=kind,
                label=self.spans[span]["text"],
                canonical_id=canonical,
                evidence=[span],
                **fields,
            )
            self.nodes[key] = record
            self.graph.nodes.append(record)
        return key

    def edge(self, source, predicate, target, evidence, **fields):
        if source is None or target is None:
            return
        key = (source, predicate, target, fields.get("source_event"))
        if key not in self.edges:
            self.edges.add(key)
            self.graph.edges.append(
                dict(
                    source=source,
                    predicate=predicate,
                    target=target,
                    evidence=list(dict.fromkeys(evidence)),
                    **fields,
                )
            )
