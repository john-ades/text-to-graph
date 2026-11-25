"""Bind clause-local arguments, then project accepted roles into directed edges."""

import re
from collections import defaultdict

from .vocabulary import CONCEPTS, FRAMES, PROJECTIONS, WEEKDAYS


def child(token, *deps):
    return next((c for c in token.children if c.dep_ in deps), None)


def preps(token):
    for c in token.children:
        if c.pos_ == "ADP":
            obj = child(c, "pobj", "pcomp")
            if obj is not None:
                yield c.lower_, obj
        else:
            case = child(c, "case")
            if case is not None:
                yield case.lower_, c


def prep(token, key):
    return next((obj for word, obj in preps(token) if word == key), None)


def phrase(token):
    """The nominal core, excluding attached clauses and prepositional modifiers."""
    parts = [token]
    for c in token.children:
        if (
            c.dep_ in {"compound", "amod", "flat", "flat:name", "nummod", "nmod"}
            and c.pos_ != "PRON"
        ):
            parts.extend(c.subtree)
    if token.lemma_.lower() in {"city", "university", "unión", "department"}:
        complement = prep(token, "of")
        if complement is not None:
            parts.extend(phrase(complement))
    return token.doc[min(t.i for t in parts) : max(t.i for t in parts) + 1]


class Interpretation:
    def __init__(self, builder, doc, offset, lang, clusters, candidates):
        self.b, self.doc, self.offset, self.lang = builder, doc, offset, lang
        self.clusters = clusters
        self.candidates = candidates
        self.names = defaultdict(set)
        self.entities = {}
        self.anchors = {}
        self.events = {}
        self.roles = defaultdict(dict)
        self.ambiguous = {}
        self.projected = []
        self.role_evidence = defaultdict(dict)
        self.coordinated = {}

    def span(self, token):
        if hasattr(token, "start_char"):
            start, end = token.start_char, token.end_char
        else:
            start, end = token.idx, token.idx + len(token.text)
        return self.b.span(self.offset + start, self.offset + end, self.lang)

    def remember(self, key, name):
        self.names[name.casefold()].add(key)

    def entity(self, t):
        if t is None:
            return None
        if t.i in self.entities:
            return self.entities[t.i]
        p = phrase(t)
        span = self.span(p)
        name = p.text
        key = None
        ambiguous = t.i in self.ambiguous
        pronoun = t.pos_ == "PRON" or t.lower_ in {"she", "he", "it", "lo", "l’", "l'"}
        determiner = child(t, "det")
        definite = determiner is not None and determiner.lower_ in {
            "the",
            "this",
            "that",
            "le",
            "la",
            "el",
        }
        candidates = self.names[name.casefold()]
        if not candidates and definite and len(p) == 1:
            candidates = self.names[t.lemma_.casefold()]
        if (
            not ambiguous
            and (definite or t.pos_ == "PROPN")
            and len(candidates) == 1
            and not (
                determiner is not None and determiner.lower_ in {"another", "other"}
            )
        ):
            key = next(iter(candidates))
        antecedents = []
        if key is None and not ambiguous and (pronoun or (definite and len(p) == 1)):
            prior = {
                self.anchors[(a, z)]
                for cluster in self.clusters
                if any(a <= t.idx < z for a, z in cluster)
                for a, z in cluster
                if z <= t.idx and (a, z) in self.anchors
            }
            if len(prior) == 1:
                key = prior.pop()
                antecedents = [
                    self.b.span(self.offset + a, self.offset + z, self.lang)
                    for cluster in self.clusters
                    if any(a <= t.idx < z for a, z in cluster)
                    for a, z in cluster
                    if z <= t.idx and self.anchors.get((a, z)) == key
                ]
            elif prior:
                self.ambiguous[t.i] = sorted(prior)
        if key is None and pronoun:
            self.b.graph.unresolved.append(
                {
                    "kind": "reference",
                    "evidence": [span],
                    "candidates": sorted(k for k in self.ambiguous.get(t.i, []) if k),
                }
            )
            self.entities[t.i] = None
            return None
        concept = WEEKDAYS.get(t.lower_) or CONCEPTS.get(t.lemma_.lower())
        prediction = self.candidates.get((p.start_char, p.end_char), {})
        if key is None:
            kind = "value" if concept in WEEKDAYS.values() or t.like_num else "entity"
            key = self.b.node(
                kind,
                span,
                concept
                or prediction.get("type")
                or {
                    "PERSON": "PERSON",
                    "ORG": "ORGANIZATION",
                    "GPE": "PLACE",
                    "LOC": "PLACE",
                    "WORK_OF_ART": "WORK",
                    "DATE": "TIME",
                }.get(t.ent_type_),
            )
        rec = self.b.nodes[key]
        if (
            len(candidates) > 1
            and key not in candidates
            and (definite or t.pos_ == "PROPN")
        ):
            self.b.graph.unresolved.append(
                {
                    "kind": "identity",
                    "mention": key,
                    "evidence": [span],
                    "candidates": sorted(candidates),
                }
            )
        if antecedents:
            self.b.spans[span]["identity_link"] = {
                "entity": key,
                "method": "fastcoref",
                "antecedents": sorted(set(antecedents)),
            }
        if span not in rec["evidence"]:
            rec["evidence"].append(span)
        self.entities[t.i] = key
        self.anchors[(p.start_char, p.end_char)] = key
        for cluster in self.clusters:
            for a, z in cluster:
                if a <= t.idx < z:
                    mention = self.doc.char_span(a, z)
                    if mention is not None and mention.root == t:
                        self.anchors[(a, z)] = key
        if not pronoun:
            self.remember(key, name)
            self.remember(key, t.lemma_)
        for c in t.children:
            if c.dep_ == "amod" and c.pos_ == "ADJ":
                ev = self.span(c)
                value = self.b.node("value", ev, c.lemma_.upper())
                self.b.edge(key, "HAS_PROPERTY", value, [span, ev])
        return key

    def alias(self, original, title):
        key = self.entity(original)
        if key and title is not None:
            p = phrase(title)
            self.entities[title.i] = key
            self.anchors[(p.start_char, p.end_char)] = key
            self.remember(key, p.text)
            self.b.nodes[key]["label"] = p.text
            self.b.nodes[key]["evidence"].append(self.span(p))

    def role(self, event, role, token, key=None):
        if token is None:
            return
        target = key or self.entity(token)
        ev = self.span(phrase(token))
        if key is not None and self.b.nodes[key]["kind"] == "value":
            ev = self.b.nodes[key]["evidence"][0]
        if target is None:
            self.b.graph.unresolved.append(
                {"kind": "role", "event": event, "role": role, "evidence": [ev]}
            )
            return
        self.roles[event][role] = target
        self.role_evidence[event][role] = ev
        self.b.edge(event, role, target, [self.b.nodes[event]["evidence"][0], ev])

    def event(self, verb, sense, group=""):
        anchor = self.span(verb)
        scope = {"polarity": "positive", "modality": "asserted"}
        if any(
            c.dep_ == "neg" or c.lower_ in {"pas", "no", "never"} for c in verb.children
        ):
            scope["polarity"] = "negative"
        if any(
            c.lower_ in {"would", "will", "may", "might", "could"}
            for c in verb.children
        ):
            scope["modality"] = "prospective"
        event = self.b.node(
            "event", anchor, sense, suffix=group, scope=scope, lemma=verb.lemma_
        )
        for cue in verb.children:
            if cue.dep_ == "neg" or cue.lower_ in {
                "pas",
                "no",
                "never",
                "would",
                "will",
                "may",
                "might",
                "could",
            }:
                self.b.nodes[event]["evidence"].append(self.span(cue))
        if group:
            self.coordinated[event] = self.events[verb.i]
        self.events.setdefault(verb.i, event)
        self.projected.append(event)
        if sense is None:
            self.b.graph.unresolved.append(
                {"kind": "predicate", "event": event, "evidence": [anchor]}
            )
        return event

    def qualifiers(self, event, verb):
        for marker, obj in preps(verb):
            if (
                obj.lower_ in WEEKDAYS
                or obj.ent_type_ in {"DATE", "TIME"}
                or obj.lower_ == "noon"
            ):
                self.role(event, "TIME", obj)
            elif (
                marker in {"at", "in", "beside", "under", "into"}
                and obj.lemma_ not in {"title", "depth"}
                and not (verb.lemma_ == "translate" and marker == "into")
            ):
                self.role(event, "LOCATION", obj)
        for c in verb.children:
            if c.lower_ in WEEKDAYS:
                self.role(event, "TIME", c)

    def bind(self, verb):
        lemma = verb.lemma_.lower()
        subj = child(verb, "nsubj", "nsubjpass", "nsubj:pass")
        if subj is None and verb.dep_ in {"conj", "xcomp"}:
            subj = child(verb.head, "nsubj", "nsubjpass")
        objs = [c for c in verb.children if c.dep_ in {"dobj", "obj", "attr", "acomp"}]
        obj = next(
            (c for c in objs if c.pos_ != "PRON" and not child(c, "case")),
            next(iter(objs), None),
        )
        sense, srole, orole = FRAMES.get(lemma, (None, "SUBJECT", "OBJECT"))
        to = prep(verb, "to")
        if lemma == "write" and to is not None:
            sense, srole, orole, obj = "WROTE_TO", "SENDER", "ADDRESSEE", to
        if lemma == "be" and obj is not None and obj.lemma_ == "author":
            sense, srole, orole, obj = "AUTHORED", "AUTHOR", "WORK", prep(obj, "of")
        if lemma in {"call", "name"} and verb.dep_ == "acl":
            self.alias(verb.head, child(verb, "oprd", "attr", "dobj"))
            return
        event = self.event(verb, sense)
        if lemma == "tell" and subj is not None and obj is not None:
            for clause in (c for c in verb.children if c.dep_ == "ccomp"):
                ref = child(clause, "nsubj")
                if ref is not None and ref.lower_ in {"she", "he"}:
                    self.ambiguous[ref.i] = [self.entity(subj), self.entity(obj)]
        if subj is not None and subj.dep_ == "nsubjpass":
            obj, subj = subj, prep(verb, "by")
        self.role(event, srole, subj)
        if lemma in {"acheter", "comprar"}:
            obj = next(
                (t for t in verb.sent if t.lower_ in {"lo", "la", "l’", "l'"}), obj
            )
        if lemma == "record" and obj is not None:
            obj = self.reading(event, verb, obj)
        self.role(event, orole, obj)
        if sense == "MEASURE" and obj is not None:
            self.role(event, "SITE", prep(obj, "of"))
        if sense == "FOUND" and obj is not None:
            for marker, place in preps(obj):
                if marker == "in" and place.ent_type_ in {"LOC", "GPE"}:
                    self.role(event, "LOCATION", place)
        self.qualifiers(event, verb)
        if sense in {"LOAN", "GIFT", "SALE"}:
            self.transfer(event, verb, sense, srole, orole, subj, obj, to)
        if lemma == "belong":
            self.role(event, "OWNER", to)
        if lemma == "translate":
            title = prep(verb, "under")
            if title is not None and title.lemma_ == "title":
                work = child(title, "appos")
                if work is not None:
                    self.role(event, "TRANSLATION", work)
                    self.remember(self.entity(work), "translation")
            self.role(event, "LANGUAGE", prep(verb, "into"))
        if lemma == "name":
            author = prep(verb, "as")
            if author is not None and author.lemma_ == "author":
                claim = self.event(author, "AUTHORED")
                self.role(claim, "AUTHOR", obj)
                self.role(claim, "WORK", prep(author, "of"))
                self.attribute(event, claim)
        if lemma == "stop":
            activity = child(verb, "xcomp")
            if activity is not None and activity.lemma_ == "transmit":
                self.b.nodes[event]["canonical_id"] = "STOP_TRANSMITTING"
        for marker, earlier in preps(verb):
            if marker in {"after", "before"}:
                other = self.entity(earlier)
                a, b = (other, event) if marker == "after" else (event, other)
                self.b.edge(
                    a,
                    "BEFORE",
                    b,
                    [self.span(earlier.head), self.span(earlier), self.span(verb)],
                )

    def reading(self, event, verb, obj):
        orole = "VALUE"
        match = re.match(
            r"[-+]?\d+(?:\.\d+)?\s*°\s*([CF])\b",
            self.doc.text[verb.idx + len(verb.text) : verb.sent.end_char].lstrip(),
        )
        if match:
            tail = self.doc.text[verb.idx + len(verb.text) : verb.sent.end_char]
            start = (
                self.offset + verb.idx + len(verb.text) + len(tail) - len(tail.lstrip())
            )
            span = self.b.span(start, start + len(match[0]), self.lang)
            value = self.b.node(
                "value",
                span,
                "TEMPERATURE",
                value=float(match[0].split("°")[0]),
                unit="°" + match[1],
            )
            self.role(event, orole, obj, value)
            obj = None
        depth = prep(verb, "at")
        if depth is not None and depth.lemma_ == "depth":
            unit = prep(depth, "of")
            if unit is not None:
                number = child(unit, "nummod")
                if number is not None and number.text.replace(".", "", 1).isdigit():
                    span = self.span(phrase(unit))
                    value = self.b.node(
                        "value",
                        span,
                        "LENGTH",
                        value=float(number.text),
                        unit=unit.lemma_,
                    )
                    self.role(event, "DEPTH", unit, value)
        return obj

    def transfer(self, event, verb, sense, srole, orole, subj, obj, to):
        lemma = verb.lemma_
        if lemma in {"borrow", "buy", "acheter", "comprar"}:
            recipient = prep(verb, "from")
        else:
            recipient = to or prep(verb, "à") or prep(verb, "a")
            if recipient is None and obj is not None:
                recipient = prep(obj, "to")
        dat = child(verb, "dative", "iobj")
        if recipient is None and dat is not None and dat.pos_ != "ADP":
            recipient = dat
        role = {
            "LOAN": "LENDER" if lemma == "borrow" else "BORROWER",
            "GIFT": "RECIPIENT",
            "SALE": "SELLER" if srole == "BUYER" else "BUYER",
        }[sense]
        if (
            obj is not None
            and obj.lemma_ in {"ticket", "flight", "route"}
            and recipient is not None
            and recipient.ent_type_ in {"GPE", "LOC"}
        ):
            self.b.edge(
                self.entity(obj),
                "DESTINATION",
                self.entity(recipient),
                [self.span(obj), self.span(phrase(recipient))],
            )
        else:
            self.role(event, role, recipient)
        if obj is not None:
            for part in (*verb.children, *obj.children):
                if (
                    part.dep_ == "conj"
                    and part.pos_ in {"NOUN", "PROPN"}
                    and prep(part, "to") is not None
                ):
                    second = self.event(verb, sense, str(part.idx))
                    self.role(second, srole, subj)
                    self.role(second, orole, part)
                    self.role(second, role, prep(part, "to"))
                    self.qualifiers(second, verb)

    def attribute(self, parent, nested):
        role = self.roles[parent]
        speaker = role.get("SPEAKER") or role.get("SOURCE")
        self.b.nodes[nested]["scope"].update(attributed_to=speaker, reported_by=parent)
        self.b.edge(
            parent,
            "CONTENT",
            nested,
            self.b.nodes[parent]["evidence"] + self.b.nodes[nested]["evidence"],
        )

    def run(self):
        for v in self.doc:
            if v.pos_ == "VERB" or (v.pos_ == "AUX" and v.dep_ == "ROOT"):
                self.bind(v)
        for i, event in self.events.items():
            v = self.doc[i]
            parent = self.events.get(v.head.i)
            if parent and parent != event:
                scope = self.b.nodes[event]["scope"]
                ev = [self.span(v.head), self.span(v)]
                if (
                    v.dep_ == "ccomp"
                    and self.b.nodes[parent]["canonical_id"] == "REPORT"
                ):
                    self.attribute(parent, event)
                elif v.dep_ == "xcomp" and v.head.lemma_ == "offer":
                    scope["modality"] = "prospective"
                    self.b.edge(parent, "CONTENT", event, ev)
                elif v.dep_ == "xcomp" and v.head.lemma_ != "cause":
                    scope.update(modality="embedded", governed_by=parent)
                    self.b.edge(parent, "ACTIVITY", event, ev)
                elif v.head.lemma_ == "cause" and v.dep_ in {"ccomp", "xcomp"}:
                    self.role(parent, "EFFECT", v, event)
                for qualifier in ("reported_by", "attributed_to"):
                    if qualifier in self.b.nodes[parent]["scope"]:
                        scope.setdefault(
                            qualifier, self.b.nodes[parent]["scope"][qualifier]
                        )
                mark = child(v, "mark")
                if mark is not None and mark.lower_ in {"if", "unless"}:
                    self.b.nodes[parent]["scope"]["condition"] = event
                    self.b.nodes[parent]["evidence"].append(self.span(mark))
                    scope["modality"] = "hypothetical"
                    self.b.edge(parent, "CONDITION", event, ev + [self.span(mark)])
        for group, first in self.coordinated.items():
            self.b.nodes[group]["scope"].update(self.b.nodes[first]["scope"])
        for event in self.projected:
            node = self.b.nodes[event]
            roles = self.roles[event]
            for a, p, b in PROJECTIONS.get(node["canonical_id"], []):
                if a in roles and b in roles:
                    support = [
                        *node["evidence"],
                        self.role_evidence[event][a],
                        self.role_evidence[event][b],
                    ]
                    self.b.edge(
                        roles[a],
                        p,
                        roles[b],
                        support,
                        source_event=event,
                        rule=f"{a}.{b}",
                        scope=node["scope"].copy(),
                    )
        for chunk in self.doc.noun_chunks:
            self.entity(chunk.root)
