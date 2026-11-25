import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from text_to_graph import extract_graph


def run_cli(text):
    graph = extract_graph(text).to_dict()
    if graph["profile"]["status"] != "complete":
        raise AssertionError(graph["profile"])
    return View(graph, text)


class View:
    def __init__(self, g, text):
        self.g = g
        self.nodes = {n["id"]: n for n in g["nodes"]}
        self.spans = {s["id"]: s for s in g["evidence"]}
        for s in self.spans.values():
            assert text[s["start"] : s["end"]] == s["text"]
        for e in g["edges"]:
            assert e["source"] in self.nodes and e["target"] in self.nodes
            assert e["evidence"] and all(s in self.spans for s in e["evidence"])

    def facts(self, p):
        return [
            (self.nodes[e["source"]]["label"], self.nodes[e["target"]]["label"])
            for e in self.g["edges"]
            if e["predicate"] == p
        ]

    def frames(self, sense):
        return [
            (
                node,
                {
                    e["predicate"]: self.nodes[e["target"]]["label"]
                    for e in self.g["edges"]
                    if e["source"] == node["id"]
                },
            )
            for node in self.nodes.values()
            if node["kind"] == "event" and node["canonical_id"] == sense
        ]


PARAGRAPHS = (
    (Path(__file__).resolve().parents[1] / "evaluation/example_paragraphs.txt")
    .read_text()
    .splitlines()
)


class Acceptance(unittest.TestCase):
    def test_gifts(self):
        v = run_cli(PARAGRAPHS[0])
        self.assertCountEqual(
            v.facts("RECEIVED"), [("Omar", "brass key"), ("Lin", "red map")]
        )
        self.assertEqual(v.frames("PLACE")[0][1]["ITEM"], "red map")

    def test_measurements(self):
        v = run_cli(PARAGRAPHS[1])
        roles = [r for _, r in v.frames("READING")]
        self.assertEqual(
            [(r["INSTRUMENT"], r["VALUE"]) for r in roles],
            [("Probe A", "8 °C"), ("Probe B", "5 °C")],
        )
        self.assertEqual(roles[1]["DEPTH"], "12 meters")
        self.assertTrue(v.facts("BEFORE"))
        self.assertFalse(v.facts("CAUSES"))
        v = run_cli("Storm damage to its antenna caused Probe A to stop transmitting.")
        self.assertTrue(v.facts("CAUSES"))

    def test_conditions(self):
        v = run_cli(PARAGRAPHS[2])
        node, roles = v.frames("REOPEN")[0]
        self.assertEqual(roles["TIME"], "Friday")
        self.assertIn("condition", node["scope"])
        self.assertEqual(node["scope"]["modality"], "prospective")
        self.assertEqual(v.frames("REPORT")[0][1]["TIME"], "Tuesday")
        neg = next(
            n for n, _ in v.frames("PASS") if n["scope"]["polarity"] == "negative"
        )
        self.assertEqual(v.nodes[neg["scope"]["attributed_to"]]["label"], "inspector")
        state = v.frames("STATE")[0][1]
        self.assertEqual((state["SUBJECT"], state["VALUE"]), ("lower trail", "open"))

    def test_works(self):
        v = run_cli(PARAGRAPHS[3])
        self.assertEqual(v.facts("AUTHORED"), [("Ravi Shah", "Night Ferry")])
        self.assertEqual(v.facts("TRANSLATION_OF"), [("Le Bac de nuit", "Night Ferry")])
        self.assertEqual(v.frames("TRANSLATE")[0][1]["TRANSLATOR"], "Inez Costa")
        self.assertEqual(v.facts("WON"), [("Le Bac de nuit", "Oriole Prize")])
        self.assertEqual(v.frames("AWARD")[0][1]["TIME"], "2023")

    def test_finder(self):
        v = run_cli(PARAGRAPHS[4])
        node, roles = v.frames("FIND")[0]
        self.assertNotIn("FINDER", roles)
        self.assertIn("attributed_to", node["scope"])
        self.assertEqual(v.facts("OWNS"), [("conductor", "red scarf")])
        v = run_cli(PARAGRAPHS[4].replace("she had", "Inez had"))
        self.assertEqual(v.frames("FIND")[0][1]["FINDER"], "Inez")

    def test_authorship(self):
        v = run_cli(PARAGRAPHS[5])
        self.assertCountEqual(
            v.facts("AUTHORED"),
            [("Nia Voss", "Blue Hour")] * 2 + [("Omar Reed", "Blue Hour")],
        )
        self.assertEqual(v.facts("WROTE_TO"), [("Omar Reed", "Nia Voss")])
        self.assertEqual(v.facts("EDITED"), [("Omar Reed", "Blue Hour")])
        scopes = [n["scope"] for n, _ in v.frames("AUTHORED")]
        self.assertTrue(any("attributed_to" in s for s in scopes))
        self.assertEqual(sum(s["polarity"] == "negative" for s in scopes), 1)
        v = run_cli("Blue Hour was written by Nia Voss.")
        self.assertEqual(v.facts("AUTHORED"), [("Nia Voss", "Blue Hour")])

    def test_beacon(self):
        v = run_cli(PARAGRAPHS[6])
        self.assertEqual(v.facts("BOUGHT"), [("Northstar Labs", "Beacon")])
        self.assertEqual(v.frames("DESIGN")[0][1]["DESIGNER"], "Maya Chen")
        node, roles = v.frames("TEST")[0]
        self.assertEqual(
            (roles["TESTER"], roles["ITEM"], roles["LOCATION"]),
            ("Northstar Labs", "Beacon", "Denver"),
        )
        self.assertEqual(node["scope"]["modality"], "prospective")
        self.assertIn("attributed_to", node["scope"])
        self.assertEqual(v.frames("SALE")[0][1]["TIME"], "Monday")
        v = run_cli(PARAGRAPHS[6].replace("sold", "offered to sell"))
        self.assertEqual(v.frames("SALE")[0][0]["scope"]["modality"], "prospective")
        self.assertTrue(v.frames("OFFER"))

    def test_loans(self):
        for lang, text in [
            ("en", "On Monday, Jo lent a telescope to Kim. Kim did not buy it."),
            ("fr", "Lundi, Jo a prêté un télescope à Kim. Kim ne l’a pas acheté."),
            ("es", "El lunes, Jo le prestó un telescopio a Kim. Kim no lo compró."),
        ]:
            with self.subTest(lang=lang):
                v = run_cli(text)
                self.assertEqual(v.g["profile"]["routes"][0]["language"], lang)
                _, r = v.frames("LOAN")[0]
                self.assertEqual((r["LENDER"], r["BORROWER"]), ("Jo", "Kim"))
                self.assertTrue(
                    any(
                        n["canonical_id"] == "TELESCOPE" and n["label"] == r["ITEM"]
                        for n in v.nodes.values()
                    )
                )
                self.assertTrue(
                    any(
                        n["canonical_id"] == "MONDAY" and n["label"] == r["TIME"]
                        for n in v.nodes.values()
                    )
                )
                self.assertEqual(
                    v.frames("SALE")[0][0]["scope"]["polarity"], "negative"
                )
        v = run_cli("Kim borrowed the telescope from Jo.")
        self.assertEqual(v.facts("LENT"), [("Jo", "telescope")])

    def test_destinations(self):
        v = run_cli("The vendor sold a ticket to Denver.")
        self.assertFalse(v.facts("BOUGHT"))
        self.assertTrue(v.facts("DESTINATION"))
        v = run_cli("The vendor sold Beacon to the City of Denver.")
        self.assertEqual(v.facts("BOUGHT"), [("City of Denver", "Beacon")])

    def test_blank(self):
        self.assertEqual(extract_graph(" \r\n").nodes, [])

    def test_perspectives_and_partial_frames(self):
        for sentence in (
            "The outage followed the storm.",
            "The storm preceded the outage.",
        ):
            self.assertEqual(run_cli(sentence).facts("BEFORE"), [("storm", "outage")])
        for sentence in (
            "Rina is the author of Green Sky.",
            "Green Sky was written by Rina.",
        ):
            self.assertEqual(
                run_cli(sentence).facts("AUTHORED"), [("Rina", "Green Sky")]
            )
        v = run_cli("Maria sold the book yesterday.")
        self.assertFalse(v.facts("BOUGHT"))
        self.assertTrue(v.facts("SOLD"))

    def test_uncertain_mapping(self):
        v = run_cli("The technician recalibrated the instrument on Tuesday.")
        self.assertTrue(any(u["kind"] == "predicate" for u in v.g["unresolved"]))

    def test_repeat_and_unicode(self):
        text = "On Monday, Zoë lent a telescope to Kim.\r\nOn Tuesday, Kim lent a telescope to Jo."
        a, b = run_cli(text), run_cli(text)
        self.assertEqual(a.g["nodes"], b.g["nodes"])
        self.assertEqual(a.g["edges"], b.g["edges"])
        self.assertEqual(len(a.frames("LOAN")), 2)
        self.assertEqual(
            len([n for n in a.nodes.values() if n["canonical_id"] == "TELESCOPE"]), 2
        )

    def test_routes(self):
        text = "On Monday, Jo lent a telescope to Kim. Kim did not buy it.\nLundi, Jo a prêté un télescope à Kim. Kim ne l’a pas acheté.\nEl lunes, Jo le prestó un telescopio a Kim. Kim no lo compró."
        v = run_cli(text)
        self.assertEqual(
            [r["language"] for r in v.g["profile"]["routes"]], ["en", "fr", "es"]
        )
        self.assertEqual(len(v.frames("LOAN")), 3)
        g = extract_graph("Some text.", language_code="vo")
        self.assertEqual(g.profile["status"], "partial")
        self.assertEqual(
            g.profile["routes"][0]["stages"]["parser"]["status"], "unavailable"
        )

    def test_missing_models(self):
        from text_to_graph.cache import LanguageCache

        with (
            patch("text_to_graph.models.language_cache", return_value=LanguageCache(1)),
            patch("text_to_graph.models.parser", side_effect=OSError("missing")),
            patch("text_to_graph.models.resolver", side_effect=OSError("missing")),
            patch("text_to_graph.models.predict", side_effect=OSError("missing")),
        ):
            g = extract_graph("The library opened on Monday morning.")
        self.assertEqual(g.profile["status"], "partial")
        self.assertTrue(
            all(
                s["status"] == "unavailable"
                for s in g.profile["routes"][0]["stages"].values()
            )
        )

    def test_long_and_overlapping_windows(self):
        # One sentence exceeds the GLiNER word window. The sale occurs after it.
        text = (
            "The checklist included "
            + ", ".join(["a notebook"] * 205)
            + ". The office sold a telescope to Dana."
        )
        v = run_cli(text)
        self.assertEqual(v.facts("BOUGHT"), [("Dana", "telescope")])
        self.assertEqual(len(v.frames("SALE")), 1)

    def test_installed_cli(self):
        import subprocess

        exe = str(Path(sys.executable).with_name("text-to-graph"))
        text = "On Monday, Harbor Robotics sold Beacon to Northstar Labs.\r\n"
        from urllib.request import urlopen

        from test_cache_api import api

        from text_to_graph import models
        from text_to_graph.cache import LanguageCache

        cache = LanguageCache(4)
        with (
            api() as port,
            patch.object(models, "language_cache", return_value=cache),
            patch.object(models, "parser", wraps=models.parser) as loads,
        ):
            cmd = [exe, "--port", str(port)]
            proc = subprocess.run(
                cmd, input=text.encode(), capture_output=True, check=True
            )
            v = View(json.loads(proc.stdout), text)
            self.assertEqual(v.facts("BOUGHT"), [("Northstar Labs", "Beacon")])
            subprocess.run([*cmd, text], capture_output=True, check=True)
            loads.assert_called_once_with("en")
            for _ in range(2):
                blank = subprocess.run([*cmd, ""], capture_output=True, check=True)
                self.assertEqual(json.loads(blank.stdout)["nodes"], [])
            with urlopen(f"http://127.0.0.1:{port}/health") as res:
                self.assertIn("en", json.load(res)["loaded_languages"])
        stopped = subprocess.run([*cmd, ""], capture_output=True, check=False)
        self.assertNotEqual(stopped.returncode, 0)
        self.assertIn(b"text-to-graph-server", stopped.stderr)
        self.assertEqual(stopped.stdout, b"")
