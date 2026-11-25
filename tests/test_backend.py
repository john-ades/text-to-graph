import os
import unittest
from unittest.mock import patch

from text_to_graph import backend, models


class BackendSelection(unittest.TestCase):
    def setUp(self):
        backend.selected.cache_clear()
        self.addCleanup(backend.selected.cache_clear)

    def test_forced_cpu_and_invalid_choice(self):
        with patch.dict(os.environ, TEXT_TO_GRAPH_BACKEND="cpu"):
            self.assertEqual(backend.devices()["gliner_encoder"], "cpu")
        backend.selected.cache_clear()
        with (
            patch.dict(os.environ, TEXT_TO_GRAPH_BACKEND="unknown"),
            self.assertRaises(ValueError),
        ):
            backend.selected()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_unavailable_devices(self, _metal, cuda):
        with patch.dict(os.environ, TEXT_TO_GRAPH_BACKEND="auto"):
            self.assertEqual(backend.selected(), "cpu")
            backend.selected.cache_clear()
            cuda.return_value = True
            self.assertEqual(backend.devices()["english_coreference"], "cuda")
        backend.selected.cache_clear()
        with (
            patch.dict(os.environ, TEXT_TO_GRAPH_BACKEND="metal"),
            self.assertRaises(RuntimeError),
        ):
            backend.selected()

    def test_cuda_placement(self):
        with (
            patch.object(backend, "selected", return_value="cuda"),
            patch.object(models, "snapshot", return_value="/checkpoint"),
            patch("gliner2.AutoExtractor.from_pretrained") as load,
            patch("fastcoref.FCoref") as resolver,
        ):
            # Exercise the loader without replacing a resident extractor in other tests.
            models.extractor.__wrapped__()
            models.resolver(object())
            self.assertEqual(load.call_args.kwargs["map_location"], "cuda")
            self.assertEqual(resolver.call_args.kwargs["device"], "cuda")
