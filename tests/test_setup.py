import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from text_to_graph import catalog, models, setup


class ModelDirectoryTests(unittest.TestCase):
    def test_checkout_default_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            os.environ.pop("TEXT_TO_GRAPH_MODEL_DIR", None)
            root = Path(tmp) / "checkout"
            inventory = root / "src/text_to_graph/catalog.json"
            with patch.object(catalog, "DATA", inventory):
                cache = root / ".cache/text-to-graph"
                self.assertEqual(catalog.model_dir(), cache)
                self.assertEqual(catalog.asset_path("hf/gliner"), cache / "hf/gliner")
                os.environ["TEXT_TO_GRAPH_MODEL_DIR"] = tmp
                self.assertEqual(catalog.model_dir(), Path(tmp))

    def test_unstaged_checkpoint_download_uses_project_cache(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, TEXT_TO_GRAPH_MODEL_DIR=tmp),
            patch("huggingface_hub.snapshot_download") as download,
        ):
            snapshot = str(Path(tmp) / "snapshot")
            download.side_effect = [LocalEntryNotFoundError("empty cache"), snapshot]
            self.assertEqual(models.snapshot(models.GLINER), snapshot)
            for call in download.call_args_list:
                self.assertEqual(call.kwargs["cache_dir"], Path(tmp) / "hf/hub")


class DownloadTests(unittest.TestCase):
    def test_resume_verify_repair_and_offline_reuse(self):
        data = b"local model file"

        class Response:
            def __init__(self, start):
                self.start = start
                self.status_code = 206 if start else 200
                self.headers = {
                    "Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"
                }

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            def iter_content(self, _):
                yield data[self.start :]

        offsets = []

        def get(url, headers, **kwargs):
            start = int(headers["Range"][6:-1]) if headers else 0
            offsets.append(start)
            return Response(start)

        spec = {
            "size": len(data),
            "digest": hashlib.sha256(data).hexdigest(),
            "url": "https://example.invalid/model",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights"
            path.with_suffix(".partial").write_bytes(data[:4])
            with (
                patch.object(setup, "asset_path", return_value=path),
                patch.dict(setup.ASSETS, {"test": spec}),
                patch.object(setup.requests, "get", side_effect=get),
            ):
                setup.transfer("test")
                setup.transfer("test")
                self.assertEqual(offsets, [4])
                path.write_bytes(b"x" * len(data))
                setup.transfer("test")
                self.assertEqual(offsets, [4, 0])
                self.assertEqual(path.read_bytes(), data)
