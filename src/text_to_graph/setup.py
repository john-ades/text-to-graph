"""Install the pinned model files, resuming incomplete transfers."""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from filelock import FileLock

from .catalog import ASSETS, LANGUAGES, ROUTES, asset_path, model_dir


def valid(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    expected = spec["digest"]
    digest = hashlib.sha256()
    if expected.startswith("git:"):
        expected = expected[4:]
        digest = hashlib.sha1(f"blob {spec['size']}\0".encode())
    with path.open("rb") as stream:
        while data := stream.read(2**20):
            digest.update(data)
    return digest.hexdigest() == expected


def existing_copy(name, spec):
    if name.startswith("hf/"):
        from huggingface_hub import try_to_load_from_cache

        repo, tail = (
            spec["url"].removeprefix("https://huggingface.co/").split("/resolve/")
        )
        rev, filename = tail.split("/", 1)
        for cache_dir in (asset_path("hf/hub"), None):
            cached = try_to_load_from_cache(
                repo, filename, revision=rev, cache_dir=cache_dir
            )
            if isinstance(cached, str):
                return Path(cached).resolve()
    if name.startswith("detector/"):
        return Path(tempfile.gettempdir()) / "fasttext-langdetect/lid.176.ftz"


def transfer(name):
    spec = ASSETS[name]
    path = asset_path(name)
    if valid(path, spec):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    cached = existing_copy(name, spec)
    if cached and valid(cached, spec):
        partial.unlink(missing_ok=True)
        try:
            os.link(cached, partial)
        except OSError:
            shutil.copyfile(cached, partial)
    else:
        for attempt in range(3):
            start = partial.stat().st_size if partial.exists() else 0
            if start >= spec["size"]:
                if valid(partial, spec):
                    break
                partial.unlink()
                start = 0
            try:
                with requests.get(
                    spec["url"],
                    headers={"Range": f"bytes={start}-"} if start else {},
                    stream=True,
                    timeout=(15, 90),
                ) as response:
                    response.raise_for_status()
                    if response.status_code == 206 and not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {start}-"):
                        raise OSError("invalid resumed range")
                    with partial.open(
                        "ab" if start and response.status_code == 206 else "wb"
                    ) as stream:
                        for chunk in response.iter_content(2**20):
                            stream.write(chunk)
                if valid(partial, spec):
                    break
                if partial.stat().st_size >= spec["size"]:
                    partial.unlink()
                raise OSError("incomplete download or checksum mismatch")
            except (requests.RequestException, OSError):
                if attempt == 2:
                    raise
    partial.replace(path)


def embedding_parts():
    return [("stanza/az/models/pretrain/native.pt." + str(i)) for i in range(2)]


def joined_ok(path):
    if not path.is_file():
        return False
    with path.open("rb") as f:
        for name in embedding_parts():
            spec = ASSETS[name]
            digest = hashlib.sha256()
            remaining = spec["size"]
            while remaining:
                data = f.read(min(remaining, 2**20))
                if not data:
                    return False
                remaining -= len(data)
                digest.update(data)
            if digest.hexdigest() != spec["digest"]:
                return False
        return not f.read(1)


def assemble(dest):
    first, second = map(asset_path, embedding_parts())
    tmp = dest.with_suffix(".joining")
    first.replace(tmp)
    with tmp.open("ab") as out, second.open("rb") as inp:
        shutil.copyfileobj(inp, out, 2**20)
    tmp.replace(dest)
    second.unlink()


def install():
    root = model_dir()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".setup.lock"):
        parts = embedding_parts()
        dest = asset_path(parts[0]).with_suffix("")
        joined = joined_ok(dest)
        if joined:
            for name in parts:
                asset_path(name).unlink(missing_ok=True)
        tmp = dest.with_suffix(".joining")
        if tmp.is_file() and not joined:
            with tmp.open("r+b") as f:
                f.truncate(ASSETS[parts[0]]["size"])
            tmp.replace(asset_path(parts[0]))
        pending = []
        volumes = defaultdict(int)
        disks = {}
        for name, spec in ASSETS.items():
            if joined and name in parts:
                continue
            path = asset_path(name)
            if valid(path, spec):
                continue
            pending.append(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            device = path.parent.stat().st_dev
            disks[device] = path.parent
            cached = existing_copy(name, spec)
            if not (
                cached
                and cached.is_file()
                and cached.stat().st_dev == device
                and valid(cached, spec)
            ):
                partial = path.with_suffix(path.suffix + ".partial")
                reusable = (
                    partial.stat().st_size
                    if partial.is_file() and partial.stat().st_size < spec["size"]
                    else 0
                )
                volumes[device] += spec["size"] - reusable
        if not joined:
            device = root.stat().st_dev
            disks[device] = root
            volumes[device] += ASSETS[parts[1]]["size"]
        for device, path in disks.items():
            free = shutil.disk_usage(path).free
            print(
                f"{volumes[device] / 2**30:.2f} GiB required, {free / 2**30:.2f} GiB free on {path.anchor}",
                flush=True,
            )
            if volumes[device] > free:
                raise OSError(
                    "Insufficient space. Set TEXT_TO_GRAPH_MODEL_DIR to a larger volume."
                )

        def fetch(name):
            try:
                transfer(name)
                print(f"Verified {name}", flush=True)
            except (OSError, ValueError) as exc:
                return f"{name}: {exc}"

        with ThreadPoolExecutor(max_workers=4) as pool:
            errors = [reason for reason in pool.map(fetch, pending) if reason]
        if errors:
            raise OSError("\n".join(errors))
        if not joined:
            assemble(dest)
        print(f"Ready. {len(ROUTES)} language routes staged.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--list", action="store_true")
    if cli.parse_args().list:
        from .native import QUALIFIED

        print("code\tparser\tmodel\tcanonical constructions")
        for code in LANGUAGES:
            route = ROUTES.get(code, {})
            print(
                code,
                route.get("backend", "detection-only"),
                route.get("model", "-"),
                "proposal cases"
                if code == "en"
                else "loan"
                if code in {"fr", "es"} | QUALIFIED
                else "unqualified",
                sep="\t",
            )
        return
    try:
        install()
    except OSError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
