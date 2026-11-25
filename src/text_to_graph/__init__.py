from .graph import DocumentGraph
from .pipeline import extract_graph

__all__ = ["DocumentGraph", "extract_graph"]


def main():
    import argparse
    import json
    import sys
    import urllib.error
    import urllib.request

    cli = argparse.ArgumentParser(
        description="Convert a document to an evidence-backed JSON graph."
    )
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument(
        "text", nargs="?", help="quoted document text (default: read UTF-8 from stdin)"
    )
    mode.add_argument(
        "--preload",
        nargs="+",
        metavar="CODE",
        help="load language models on the running server",
    )
    cli.add_argument("--language", help="override the detector for this document")
    cli.add_argument("--port", type=int, default=8000)
    args = cli.parse_args()
    if args.preload and args.language is not None:
        cli.error("--language applies only to graph requests")
    base = f"http://127.0.0.1:{args.port}"
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with client.open(base + "/health", timeout=3) as response:
            status = json.load(response)
        if (
            not isinstance(status, dict)
            or status.get("service") != "text-to-graph"
            or status.get("api_version") != 1
        ):
            raise ValueError("incompatible service")
    except (OSError, ValueError) as exc:
        suffix = f" --port {args.port}" if args.port != 8000 else ""
        cli.exit(
            1,
            f"Graph API unavailable ({exc}). Start it with uv run --locked text-to-graph-server{suffix}\n",
        )
    if args.preload:
        path = "/v1/languages/preload"
        body = {"languages": args.preload}
    else:
        path = "/v1/graph"
        text = args.text
        if text is None:
            if sys.stdin.isatty():
                cli.error("provide document text or pipe it to stdin")
            try:
                text = sys.stdin.buffer.read().decode("utf-8")
            except UnicodeDecodeError:
                cli.error("stdin must contain UTF-8 text")
        body = {"text": text, "language": args.language}
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with client.open(req) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        cli.exit(
            1, f"Graph API returned {exc.code}: {exc.read().decode(errors='replace')}\n"
        )
    except (OSError, ValueError) as exc:
        cli.exit(1, f"Graph API request failed: {exc}\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
