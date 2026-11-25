import contextlib
import json
import socket
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import uvicorn

from text_to_graph.cache import LanguageCache
from text_to_graph.server import app


@contextlib.contextmanager
def api():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if time.monotonic() > deadline:
                    raise TimeoutError("API did not start")
                time.sleep(0.02)
            yield port
        finally:
            server.should_exit = True
            thread.join(10)


class CacheTests(unittest.TestCase):
    def test_preload_cli_uses_server_cache(self):
        cache = LanguageCache(2)
        exe = Path(sys.executable).with_name("text-to-graph")
        with (
            patch("text_to_graph.models._language_cache", cache),
            patch("text_to_graph.models.parser") as load,
            patch("text_to_graph.models.resolver") as resolve,
            api() as port,
        ):

            def preload(*codes):
                return subprocess.run(
                    [exe, "--port", str(port), "--preload", *codes],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=15,
                )

            self.assertEqual(
                json.loads(preload("en", "fr", "en").stdout)["loaded_languages"],
                ["fr", "en"],
            )
            for invalid in [("ar", "unknown"), ("en", "fr", "ar")]:
                self.assertNotEqual(preload(*invalid).returncode, 0)
                self.assertEqual(cache.resident(), ["fr", "en"])
            self.assertEqual(
                json.loads(preload("en", "ar").stdout)["loaded_languages"], ["ar", "en"]
            )
            self.assertEqual(
                [call.args[0] for call in load.call_args_list], ["en", "fr", "ar"]
            )
            resolve.assert_called_once_with(load.return_value)
            load.side_effect = OSError("missing model")
            failed = preload("fr")
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Failed to preload fr", failed.stderr)
            self.assertEqual(cache.resident(), ["ar"])

    def test_lru(self):
        c = LanguageCache(4)
        calls = []

        def load(k):
            calls.append(k)
            return object()

        for lang in ("en", "fr", "es", "ar", "en", "id"):
            with c.acquire(lang, load):
                pass
        self.assertEqual(c.resident(), ["id", "en", "ar", "es"])
        self.assertEqual(calls, ["en", "fr", "es", "ar", "id"])
        with c.acquire("fr", load):
            pass
        self.assertEqual(calls[-1], "fr")

    def test_capacity_and_failure(self):
        c = LanguageCache(1)
        release = threading.Event()
        started = threading.Event()
        entered = threading.Event()
        calls = []

        def load(k):
            calls.append(k)
            return k

        def held():
            with c.acquire("en", load):
                started.set()
                release.wait(3)

        def next_language():
            with c.acquire("de", load):
                entered.set()

        with ThreadPoolExecutor(3) as pool:
            a = pool.submit(held)
            self.assertTrue(started.wait(2))
            b = pool.submit(next_language)
            d = pool.submit(next_language)
            self.assertFalse(entered.wait(0.1))
            release.set()
            for f in (a, b, d):
                f.result()
        self.assertEqual(calls, ["en", "de"])

        def broken(_):
            raise OSError("missing model")

        with self.assertRaises(OSError), c.acquire("fr", broken):
            pass
        self.assertEqual(c.resident(), [])

    def test_english_bundle_releases_its_resolver(self):
        import gc
        import weakref

        class Component:
            pass

        cache = LanguageCache(1)
        with cache.acquire("en", lambda _: Component()) as english:
            english.coref = Component()
            english.coref.nlp = english.parser
            resolver = weakref.ref(english.coref)
            parser = weakref.ref(english.parser)
        with cache.acquire("fr", lambda _: Component()):
            gc.collect()
            self.assertIsNone(resolver())
            self.assertIsNone(parser())
