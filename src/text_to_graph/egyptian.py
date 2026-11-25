"""Egyptian Arabic CATiB parsing with the CAMeL dialect checkpoint."""

import pickle
import types
import unicodedata

from .catalog import ASSETS, ROUTES, asset_path
from .native import Word, sentence, stanza_pipeline


def reader(route):
    import supar
    import torch
    from transformers import AutoTokenizer, BertTokenizerFast

    from .setup import valid

    # The upstream checkpoint serializes its training tokenizer with the
    # Transformers 4 module name. The local runtime uses Transformers 5.
    class TokenizerCompat(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == (
                "transformers.models.bert.tokenization_bert_fast",
                "BertTokenizerFast",
            ):
                return BertTokenizerFast
            return super().find_class(module, name)

    model_file = route["model"]
    path = asset_path(model_file)
    if not valid(path, ASSETS[model_file]):
        raise OSError(
            "Missing or damaged Egyptian Arabic checkpoint. Run text-to-graph-setup."
        )
    legacy = types.ModuleType("camel_checkpoint")
    legacy.Unpickler = TokenizerCompat
    state = torch.load(
        path, pickle_module=legacy, weights_only=False, map_location="cpu"
    )
    folder = str(asset_path("arz/bert"))
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    args = state["args"]
    args.bert = folder
    model = supar.PARSER[state["name"]].MODEL(**args)
    model.load_pretrained(state["pretrained"])
    # Transformers 5 regenerates this deterministic position buffer.
    state["state_dict"].pop("encoder.model.embeddings.position_ids", None)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    relations = state["transform"].DEPREL.vocab
    field = state["transform"].FORM[0]
    del state
    arabic = ROUTES["ar"]
    segment = stanza_pipeline(
        {
            **arabic,
            "processors": {p: arabic["processors"][p] for p in ("tokenize", "mwt")},
        }
    )

    @torch.no_grad()
    def read(text):
        for sent in segment(text).sentences:
            tokens = [(w, t) for t in sent.tokens for w in t.words]
            pieces = [[tokenizer.cls_token_id]]
            for w, _ in tokens:
                form = "".join(
                    c for c in w.text if c != "ـ" and not unicodedata.combining(c)
                )
                ids = tokenizer.encode(form, add_special_tokens=False) or [
                    tokenizer.unk_token_id
                ]
                pieces.append(ids[: field.fix_len] if field.fix_len else ids)
            width = max(map(len, pieces))
            words = torch.tensor(
                [
                    [
                        row + [tokenizer.pad_token_id] * (width - len(row))
                        for row in pieces
                    ]
                ]
            )
            arcs, rels = model(words, [])
            mask = torch.ones(words.shape[:2], dtype=torch.bool)
            mask[:, 0] = False
            heads, labels = model.decode(arcs, rels, mask, tree=True, proj=True)
            parsed = []
            for i, (w, t) in enumerate(tokens, 1):
                dep = relations[int(labels[0, i])]
                parsed.append(
                    Word(
                        i,
                        int(heads[0, i]),
                        dep,
                        "X",
                        w.text,
                        w.start_char if w.start_char is not None else t.start_char,
                        w.end_char if w.end_char is not None else t.end_char,
                    )
                )
            yield sentence(text, parsed)

    return read
