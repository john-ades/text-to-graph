"""Smoke-test each installed parser on source text (UDHR translations and native examples)."""

import argparse
import gc
import json
from pathlib import Path

from text_to_graph.catalog import ROUTES
from text_to_graph.models import parser


def check(code, text):
    pipeline = parser(code)
    doc = pipeline(text)
    if ROUTES[code]["backend"] == "installed":
        assert doc.has_annotation("DEP") and doc.has_annotation("SENT_START")
        assert all(text[t.idx : t.idx + len(t)] == t.text for t in doc)
    else:
        assert doc.sents
        for s in doc.sents:
            assert text[s.start_char : s.start_char + len(s.text)] == s.text
            ids = {w.id for w in s.words}
            assert any(w.head == 0 for w in s.words)
            assert all(w.dep and (w.head == 0 or w.head in ids) for w in s.words)
            assert all(0 <= w.start < w.end <= len(text) for w in s.words)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("codes", nargs="*")
    args = cli.parse_args()
    samples = json.loads(Path(__file__).with_name("language_samples.json").read_text())
    assert set(samples) == set(ROUTES)
    errors = 0
    for code in args.codes or sorted(ROUTES):
        try:
            check(code, samples[code])
            print(f"{code} passed", flush=True)
        except Exception as exc:  # noqa: BLE001 - record each provider failure and continue the sweep
            print(f"{code} failed: {exc}", flush=True)
            errors += 1
        finally:
            gc.collect()
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
