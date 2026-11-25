"""CPU inference for the published Amharic arc-eager parser (Keras 2.2 weights)."""

import gzip
import json
import re
from pathlib import Path

import h5py
import numpy as np

from .catalog import asset_path
from .native import Word, sentence


def weights(path):
    result = {}
    with h5py.File(path) as f:

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                result[name.split("/", 1)[1].removesuffix(":0")] = obj[()]

        f.visititems(collect)
    return result


def dense(x, params, layer):
    return x @ params[layer + "/kernel"] + params[layer + "/bias"]


def recurrent(x, params, layer):
    # Keras 2.2 uses hard_sigmoid for the input, forget and output gates.
    wx = dense(x, params, layer)
    wh = params[layer + "/recurrent_kernel"]
    hidden = np.zeros(wh.shape[0], dtype=np.float32)
    cell = hidden.copy()
    seq = []
    for row in wx:
        i, f, g, o = np.split(row + hidden @ wh, 4)
        gates = [np.clip(0.2 * v + 0.5, 0, 1) for v in (i, f, o)]
        cell = gates[1] * cell + gates[0] * np.tanh(g)
        hidden = gates[2] * np.tanh(cell)
        seq.append(hidden)
    return np.stack(seq)


def pos_tags(words, features, tags, params):
    vectors = np.zeros((len(words), len(features)), dtype=np.float32)
    for i, term in enumerate(words):
        raw = {
            "term": term,
            "prev_word": words[i - 1] if i else "",
            "next_word": words[i + 1] if i + 1 < len(words) else "",
            "is_first": int(i == 0),
            "is_last": int(i + 1 == len(words)),
            "nb_terms": len(words),
        }
        for n in (1, 2, 3):
            raw[f"prefix-{n}"] = term[:n]
            raw[f"suffix-{n}"] = term[-n:]
        for key, value in raw.items():
            name = f"{key}={value}" if isinstance(value, str) else key
            if name in features:
                vectors[i, features[name]] = 1 if isinstance(value, str) else value
    h = np.maximum(dense(vectors, params, "dense_21"), 0)
    h = np.maximum(dense(h, params, "dense_22"), 0)
    return [tags[i] for i in dense(h, params, "dense_23").argmax(axis=1)]


def padded(values):
    return np.pad(values, (0, 25 - len(values))).astype(int)


def transitions(stack, buffer, word_ids, tag_ids, p):
    def encode(indices, embedding, layer):
        words = padded([word_ids[i] for i in indices])
        tags = padded([tag_ids[i] for i in indices])
        w = p[f"embedding_{embedding}/embeddings"][words]
        t = p[f"embedding_{embedding + 1}/embeddings"][tags]
        return recurrent(w @ t.T, p, layer)

    left = encode(stack, 5, "lstm_4")
    right = encode(buffer, 7, "lstm_5")
    hidden = recurrent(np.concatenate((left, right), axis=1), p, "lstm_6")
    prefix = "seq_self_attention_2/seq_self_attention_2_Add_"
    # The trained classifier reads position 22 of the attended sequence.
    z = hidden[22] @ p[prefix + "Wt"] + hidden @ p[prefix + "Wx"] + p[prefix + "bh"]
    e = 1 / (1 + np.exp(-(np.tanh(z) @ p[prefix + "Wa"] + p[prefix + "ba"])))
    attention = np.exp(e - e.max())
    return dense((attention.T @ hidden / attention.sum())[0], p, "time_distributed_2")


def parse_heads(words, tags, word_ids, tag_ids, params):
    def encode(items, lookup):
        return [lookup["-root-"]] + [lookup.get(x, lookup["-OOV-"]) for x in items]

    w, t = encode(words, word_ids), encode(tags, tag_ids)
    stack, buffer, heads = [0], list(range(1, len(w))), [0] * len(w)
    while buffer:
        actions = transitions(stack, buffer, w, t, params)
        if len(stack) == 1:
            actions[[1, 3]] = -np.inf
        action = actions.argmax()
        if action in (0, 2):
            child = buffer.pop(0)
            if action == 2:
                heads[child] = stack[-1]
            stack.append(child)
        else:
            child = stack.pop()
            if action == 1:
                heads[child] = buffer[0]
    return heads[1:]


def reader(route):
    root = asset_path(route["model"])
    folder = root / "dependencyparser_module"
    pos = weights(root / "POS_tag_module/tagClassifier2.h5")
    actions = weights(folder / "TBNADP.h5")
    relations = weights(folder / "TBNADPrel.h5")
    word_ids, tag_ids, rel_ids = [
        json.loads((folder / (name + "2index.json")).read_text())
        for name in ("word", "tag", "rel")
    ]
    labels = {i: k for k, i in rel_ids.items()}
    with gzip.open(Path(__file__).with_name("amharic_features.json.gz"), "rt") as f:
        vocab = json.load(f)
    features = {s: i for i, s in enumerate(vocab["features"])}

    def parse(text, tokens):
        words = [t.group() for t in tokens]
        tags = pos_tags(words, features, vocab["tags"], pos)
        heads = parse_heads(words, tags, word_ids, tag_ids, actions)
        parent = [tags[h - 1] if h else "root" for h in heads]
        encoded = [
            relations[f"embedding_{i}/embeddings"][
                padded([tag_ids.get(t, tag_ids["-OOV-"]) for t in seq])
            ]
            for i, seq in [(7, tags), (8, parent)]
        ]
        hidden = np.concatenate(encoded, axis=1)
        for n in (7, 8):
            left = recurrent(hidden, relations, f"bidirectional_{n}/forward_lstm_{n}")
            right = recurrent(
                hidden[::-1], relations, f"bidirectional_{n}/backward_lstm_{n}"
            )[::-1]
            hidden = np.concatenate((left, right), axis=1)
        predicted = dense(hidden, relations, "time_distributed_4").argmax(axis=1)
        return sentence(
            text,
            [
                Word(
                    i + 1,
                    heads[i],
                    labels[predicted[i]],
                    tags[i],
                    t.group(),
                    t.start(),
                    t.end(),
                )
                for i, t in enumerate(tokens)
            ],
        )

    def read(text):
        tokens = []
        for token in re.finditer(r"[^\s።!?፧]+|[።!?፧]", text):
            tokens.append(token)
            if token.group() in {"።", "!", "?", "፧"} or len(tokens) == 24:
                yield parse(text, tokens)
                tokens = []
        if tokens:
            yield parse(text, tokens)

    return read
