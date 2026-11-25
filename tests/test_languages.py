import json
import unittest
from pathlib import Path

from text_to_graph import extract_graph
from text_to_graph.native import QUALIFIED


class Languages(unittest.TestCase):
    def test_qualified_loans(self):
        cases = json.loads(Path(__file__).with_name("loans.json").read_text())
        self.assertEqual(set(cases), QUALIFIED)
        for code, readings in cases.items():
            for text, lender, borrower, item, polarity in readings:
                with self.subTest(language=code, text=text):
                    graph = extract_graph(text, language_code=code)
                    nodes = {n["id"]: n["label"] for n in graph.nodes}
                    edges = [
                        (
                            nodes[e["source"]],
                            e["predicate"],
                            nodes[e["target"]],
                            e["scope"]["polarity"],
                        )
                        for e in graph.edges
                        if e["predicate"] in ("LENT", "BORROWED")
                    ]
                    self.assertCountEqual(
                        edges,
                        [
                            (lender, "LENT", item, polarity),
                            (borrower, "BORROWED", item, polarity),
                        ],
                    )
                    self.assertTrue(
                        any(
                            n["label"] == item and n["canonical_id"] == "BOOK"
                            for n in graph.nodes
                        )
                    )
                    for span in graph.evidence:
                        self.assertEqual(
                            text[span["start"] : span["end"]], span["text"]
                        )

    def test_chinese_scripts_share_one_slot(self):
        from text_to_graph.cache import LanguageCache
        from text_to_graph.models import parser

        cache = LanguageCache(1)
        for text, model in [
            ("这家书店卖书。", "gsdsimp"),
            ("這家書店賣書。", "chinese-gsd-"),
        ]:
            with cache.acquire("zh", parser) as entry:
                self.assertIn(model, entry.parser(text).parser_name)
            self.assertEqual(cache.resident(), ["zh"])
