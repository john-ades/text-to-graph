"""Inference adapter for Mesolitica's T5 dependency checkpoint and word-piece scores."""

import re

from .catalog import asset_path
from .native import Word, sentence

RELATIONS = [
    "PAD",
    "X",
    "nsubj",
    "cop",
    "det",
    "root",
    "nsubj:pass",
    "acl",
    "case",
    "obl",
    "flat",
    "punct",
    "appos",
    "amod",
    "compound",
    "advmod",
    "cc",
    "obj",
    "conj",
    "mark",
    "advcl",
    "nmod",
    "nummod",
    "dep",
    "xcomp",
    "ccomp",
    "parataxis",
    "compound:plur",
    "fixed",
    "aux",
    "csubj",
    "iobj",
    "csubj:pass",
]


def reader(route):
    import torch
    from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
    from torch.nn import functional as F
    from transformers import AutoTokenizer, T5Config, T5Model

    folder = asset_path(route["model"])
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    weights = torch.load(
        folder / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    model = T5Model(T5Config.from_pretrained(folder))
    model.load_state_dict({k: weights.pop(k) for k in model.state_dict()})
    model.eval()

    def scores(hidden, name):
        vectors = []
        for side in ["d", "h"]:
            key = f"mlp_{name}_{side}.linear"
            projected = F.leaky_relu(
                F.linear(hidden, weights[key + ".weight"], weights[key + ".bias"]), 0.1
            )
            if side == "d" or name == "rel":
                projected = F.pad(projected, (0, 1), value=1)
            vectors.append(projected)
        return torch.einsum(
            "xi,oij,yj->xyo", vectors[0], weights[name + "_attn.weight"], vectors[1]
        )

    @torch.inference_mode()
    def parse(text, tokens):
        pieces = [1]
        groups = [0]
        for i, tok in enumerate(tokens, 1):
            ids = tokenizer.encode(tok.group(), add_special_tokens=False)
            pieces.extend(ids)
            groups.extend([i] * len(ids))
        ids = torch.tensor([pieces])
        hidden = model(
            input_ids=ids, decoder_input_ids=ids, use_cache=False
        ).last_hidden_state[0]
        # Pool scores over all pieces belonging to each original word.
        pool = F.one_hot(torch.tensor(groups), num_classes=len(tokens) + 1).T.float()
        pool = F.normalize(pool, p=1, dim=1)
        arcs = (pool @ scores(hidden, "arc").squeeze(-1) @ pool.T).numpy()
        heads = chuliu_edmonds_one_root(arcs)
        labels = torch.einsum(
            "ix,xyr,jy->ijr", pool, scores(hidden, "rel"), pool
        ).argmax(-1)
        return sentence(
            text,
            [
                Word(
                    i,
                    int(heads[i]),
                    RELATIONS[labels[i, heads[i]]],
                    "X",
                    tok.group(),
                    tok.start(),
                    tok.end(),
                )
                for i, tok in enumerate(tokens, 1)
            ],
        )

    def read(text):
        batch = []
        for token in re.finditer(r"\w+(?:[-’']\w+)*|[^\w\s]", text):
            batch.append(token)
            if token.group() in ".!?" or len(batch) >= 96:
                yield parse(text, batch)
                batch = []
        if batch:
            yield parse(text, batch)

    return read
