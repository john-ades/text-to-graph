"""Run the multilingual encoder on MLX and the upstream GLiNER heads on CPU."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch
from gliner2.configuration import ExtractorConfig
from gliner2.models.base import load_extractor_tokenizer
from gliner2.models.boundary.engine import BoundaryExtractor
from safetensors import safe_open
from transformers import AutoConfig

from .mlx_deberta import DebertaModel


class MetalEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        states = self.encoder(
            mx.asarray(input_ids.numpy(), dtype=mx.int32),
            mx.asarray(attention_mask.numpy(), dtype=mx.int32),
        )
        mx.eval(states)
        return SimpleNamespace(last_hidden_state=torch.from_numpy(np.array(states)))


def load(folder):
    mx.set_cache_limit(64 * 2**20)
    folder = Path(folder)
    config = AutoConfig.from_pretrained(
        folder / "encoder_config", local_files_only=True
    )
    model = BoundaryExtractor(
        ExtractorConfig.from_pretrained(folder, local_files_only=True),
        encoder_config=config,
        tokenizer=load_extractor_tokenizer(str(folder)),
        load_encoder=False,
    )
    path = str(folder / "model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as weights:
        model.load_state_dict(
            {
                key: weights.get_tensor(key)
                for key in weights.keys()  # noqa: SIM118 - safe_open exposes keys()
                if not key.startswith("encoder.")
            }
        )
    weights = mx.load(path)
    config.vocab_size = weights["encoder.embeddings.word_embeddings.weight"].shape[0]
    if config.max_relative_positions < 1:
        config.max_relative_positions = config.max_position_embeddings
    encoder = DebertaModel(config)
    encoder.load_weights(
        [
            (
                key.removeprefix("encoder.")
                .replace("encoder.layer.", "encoder.layers.")
                .replace(".attention.self.", ".attention.self_attn.")
                .replace(".LayerNorm.", ".layer_norm."),
                value,
            )
            for key, value in weights.items()
            if key.startswith("encoder.")
        ],
        strict=True,
    )
    encoder.eval()
    mx.eval(encoder.parameters())
    model.encoder = MetalEncoder(encoder)
    return model.eval()
