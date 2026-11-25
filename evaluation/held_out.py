"""Small hand-labeled evaluation. Run from a fresh process with the models staged."""

import json
import resource
import sys
from collections import Counter
from statistics import median
from time import perf_counter

from text_to_graph import extract_graph

# Each record lists all expected canonical event roles and derived triples.
# Triples include polarity, modality, and reporting scope in their comparison.
CASES = [
    (
        "publishing",
        "Asha Mehta edited River Almanac. Leon Park did not write River Almanac.",
        [
            ("EDITED", "EDITOR", "Asha Mehta"),
            ("EDITED", "WORK", "River Almanac"),
            ("AUTHORED", "AUTHOR", "Leon Park"),
            ("AUTHORED", "WORK", "River Almanac"),
        ],
        [
            ("Asha Mehta", "EDITED", "River Almanac", "positive", "asserted", False),
            ("Leon Park", "AUTHORED", "River Almanac", "negative", "asserted", False),
        ],
    ),
    (
        "commerce",
        "The art gallery sold a bronze sculpture to Anika Rao.",
        [
            ("SALE", "SELLER", "art gallery"),
            ("SALE", "GOODS", "bronze sculpture"),
            ("SALE", "BUYER", "Anika Rao"),
        ],
        [
            ("Anika Rao", "BOUGHT", "bronze sculpture", "positive", "asserted", False),
            ("art gallery", "SOLD", "bronze sculpture", "positive", "asserted", False),
        ],
    ),
    (
        "operations",
        "The eclipse preceded the outage. The monitor flickered.",
        [("BEFORE", "EARLIER", "eclipse"), ("BEFORE", "LATER", "outage")],
        [("eclipse", "BEFORE", "outage", "positive", "asserted", False)],
    ),
    (
        "reporting",
        "The broker said that the museum would buy the painting.",
        [
            ("REPORT", "SPEAKER", "broker"),
            ("REPORT", "CONTENT", "buy"),
            ("SALE", "BUYER", "museum"),
            ("SALE", "GOODS", "painting"),
        ],
        [("museum", "BOUGHT", "painting", "positive", "prospective", True)],
    ),
    (
        "museum-fr",
        "Le conservateur a prêté un télescope à Marie.",
        [
            ("LOAN", "LENDER", "conservateur"),
            ("LOAN", "BORROWER", "Marie"),
            ("LOAN", "ITEM", "télescope"),
        ],
        [
            ("conservateur", "LENT", "télescope", "positive", "asserted", False),
            ("Marie", "BORROWED", "télescope", "positive", "asserted", False),
        ],
    ),
    (
        "museum-es",
        "El museo prestó un telescopio a Lucía.",
        [
            ("LOAN", "LENDER", "museo"),
            ("LOAN", "BORROWER", "Lucía"),
            ("LOAN", "ITEM", "telescopio"),
        ],
        [
            ("museo", "LENT", "telescopio", "positive", "asserted", False),
            ("Lucía", "BORROWED", "telescopio", "positive", "asserted", False),
        ],
    ),
]


def compare():
    counts = Counter()
    durations = []
    results = []
    for domain, text, roles, triples in CASES:
        begin = perf_counter()
        graph = extract_graph(text)
        durations.append(perf_counter() - begin)
        nodes = {n["id"]: n for n in graph.nodes}
        actual_roles, actual_triples = Counter(), Counter()
        for edge in graph.edges:
            source, target = nodes[edge["source"]], nodes[edge["target"]]
            if "rule" in edge:
                scope = edge["scope"]
                actual_triples[
                    (
                        source["label"],
                        edge["predicate"],
                        target["label"],
                        scope["polarity"],
                        scope["modality"],
                        "reported_by" in scope,
                    )
                ] += 1
            elif source["kind"] == "event" and source["canonical_id"]:
                actual_roles[
                    (source["canonical_id"], edge["predicate"], target["label"])
                ] += 1
        correct = sum((Counter(roles) & actual_roles).values())
        extra = sum((actual_triples - Counter(triples)).values())
        counts.update(
            correct_roles=correct,
            expected_roles=len(roles),
            emitted_roles=actual_roles.total(),
            unsupported_projections=extra,
            emitted_projections=actual_triples.total(),
            missing_projections=(Counter(triples) - actual_triples).total(),
            unknown_events=sum(
                n["kind"] == "event" and n["canonical_id"] is None
                for n in nodes.values()
            ),
            events=sum(n["kind"] == "event" for n in nodes.values()),
        )
        results.append(
            {
                "domain": domain,
                "language": graph.profile["language"]["lang"],
                "status": graph.profile["status"],
                "missing_roles": list((Counter(roles) - actual_roles).elements()),
                "extra_roles": list((actual_roles - Counter(roles)).elements()),
            }
        )
    warm = []
    for _, text, _, _ in CASES:
        begin = perf_counter()
        extract_graph(text)
        warm.append(perf_counter() - begin)
    identity_text = "Ada Stone founded Linden Press. She sold a book to Nora Bell."
    identity = extract_graph(identity_text)
    evidence = {row["id"]: row["text"] for row in identity.evidence}
    clusters = [
        {evidence[s] for s in n["evidence"]}
        for n in identity.nodes
        if n["kind"] == "entity"
    ]
    pairs = [
        ("Ada Stone", "She", True),
        ("Ada Stone", "Nora Bell", False),
        ("She", "Nora Bell", False),
    ]
    for x, y, same in pairs:
        linked = any({x, y} <= cluster for cluster in clusters)
        counts["false_merges"] += int(linked and not same)
        counts["missed_links"] += int(not linked and same)
    counts["identity_pairs"] = len(pairs)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "counts": counts,
        "cases": results,
        "cached_cold_first_document_seconds": round(durations[0], 3),
        "warm_median_seconds": round(median(warm), 3),
        "peak_rss_mib": round(rss / (1024**2 if sys.platform == "darwin" else 1024)),
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2, ensure_ascii=False))
