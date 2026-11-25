import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from test_cache_api import api

from text_to_graph import models
from text_to_graph.batching import isolated
from text_to_graph.cache import LanguageCache
from text_to_graph.native import NativeParser, Sentence


def post(port, path, body):
    request = Request(
        f"http://127.0.0.1:{port}/v1/{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def graph_fields(value):
    if isinstance(value, dict):
        return {
            k: None
            if k in ("score", "confidence", "model_score") and isinstance(v, float)
            else graph_fields(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [graph_fields(v) for v in value]
    return value


class ModelBatching(unittest.TestCase):
    def test_short_provider_result_is_never_assigned_to_another_input(self):
        def infer(items):
            return [item for item in items if item != "missing"]

        first, second, third = isolated(["left", "missing", "right"], infer)
        self.assertEqual((first, third), ("left", "right"))
        self.assertIsInstance(second, RuntimeError)

    def test_coreference_windows_stay_in_passage_coordinates(self):
        class Sentence(list):
            def __init__(self, start, end, pronoun=False):
                super().__init__(
                    [SimpleNamespace(pos_="PRON" if pronoun else "NOUN", idx=start)]
                )
                self.start_char, self.end_char = start, end

        sources = ["Intro. Jo. She.", "A much longer intro. Jo. She."]
        docs = []
        for source in sources:
            base = source.index("Jo")
            docs.append(
                SimpleNamespace(
                    sents=[
                        Sentence(0, base - 1),
                        Sentence(base, base + 3),
                        Sentence(base + 4, len(source), True),
                    ]
                )
            )

        def predict(texts, **kwargs):
            return [
                SimpleNamespace(get_clusters=lambda as_strings: [[(0, 2), (4, 7)]])
                if t == "Jo. She."
                else SimpleNamespace(get_clusters=lambda as_strings: [])
                for t in texts
            ]

        bundle = SimpleNamespace(coref=SimpleNamespace(predict=predict))
        rows = models.coreference_batch(sources, docs, bundle)
        for text, clusters in zip(sources, rows, strict=True):
            self.assertEqual(
                [[text[a:z] for a, z in c] for c in clusters], [["Jo", "She"]]
            )


class BatchAPI(unittest.TestCase):
    def test_ordered_item_validation_and_limits(self):
        with api() as port:
            for invalid in ({}, {"documents": []}, {"documents": [{}] * 33}):
                with self.assertRaises(HTTPError) as ctx:
                    post(port, "graph/batch", invalid)
                self.assertEqual(ctx.exception.code, 422)
            items = [
                None,
                {"text": ""},
                {"language": "en"},
                {"text": 7},
                {"text": "", "language": "xx"},
                {"text": ""},
            ]
            results = post(port, "graph/batch", {"documents": items})["results"]
            self.assertEqual([r["index"] for r in results], list(range(len(items))))
            for i in (0, 2, 3, 4):
                self.assertEqual(results[i]["error"]["code"], "invalid_document")
            self.assertEqual(results[1]["graph"], results[5]["graph"])
            self.assertEqual(results[1]["graph"]["profile"]["status"], "complete")

    def test_bad_model_input_isolated_and_healthy_rows_batched(self):
        def predict(items):
            if any(text == "bad" for text, _ in items):
                raise ValueError("unreadable")
            return [{} for _ in items]

        with (
            patch.object(models, "predict_many", side_effect=predict) as inference,
            api() as port,
        ):
            docs = [{"language": "vo", "text": t} for t in ("good", "bad", "good")]
            rows = post(port, "graph/batch", {"documents": docs})["results"]
            self.assertTrue(any(len(c.args[0]) > 1 for c in inference.call_args_list))
            states = [r["graph"]["profile"]["routes"][0]["stages"] for r in rows]
            self.assertEqual(
                [s["gliner"]["status"] for s in states],
                ["complete", "unavailable", "complete"],
            )
            self.assertTrue(all(s["parser"]["status"] == "unavailable" for s in states))

    def test_concurrent_requests_with_one_language_slot(self):
        cache = LanguageCache(1)

        def load(_):
            return NativeParser(lambda text: [Sentence(0, text, [])], "test")

        docs = [{"text": "Text", "language": lang} for lang in ("de", "it") * 3]
        with (
            patch.object(models, "_language_cache", cache),
            patch.object(models, "parser", side_effect=load),
            patch.object(
                models, "predict_many", side_effect=lambda items: [{} for _ in items]
            ),
            api() as port,
            ThreadPoolExecutor(3) as pool,
        ):
            batch = pool.submit(post, port, "graph/batch", {"documents": docs})
            single = pool.submit(post, port, "graph", docs[0])
            preload = pool.submit(
                post, port, "languages/preload", {"languages": ["it"]}
            )
            self.assertEqual(
                batch.result(timeout=10)["results"][0]["graph"],
                single.result(timeout=10),
            )
            self.assertEqual(len(preload.result(timeout=10)["loaded_languages"]), 1)

    def test_real_batch_matches_single_graphs(self):
        paragraph = (
            Path(__file__)
            .parents[1]
            .joinpath("evaluation/example_paragraphs.txt")
            .read_text()
            .splitlines()[-1]
        )
        arabic = json.loads(Path(__file__).with_name("loans.json").read_text())["ar"]
        mixed = "On Monday, Jo lent a telescope to Kim. Kim did not buy it.\r\nLundi, Jo a prêté un télescope à Kim. Kim ne l’a pas acheté."
        docs = [
            {"text": paragraph, "language": "en"},
            {"text": mixed},
            {"text": arabic[0][0], "language": "ar"},
            {"text": arabic[1][0], "language": "ar"},
            {"text": paragraph, "language": "en"},
        ]
        with api() as port:
            singles = [post(port, "graph", d) for d in docs]
            rows = post(port, "graph/batch", {"documents": docs})["results"]
        for doc, single, row in zip(docs, singles, rows, strict=True):
            graph = row["graph"]
            self.assertEqual(graph_fields(single), graph_fields(graph))
            for span in graph["evidence"]:
                self.assertEqual(doc["text"][span["start"] : span["end"]], span["text"])
