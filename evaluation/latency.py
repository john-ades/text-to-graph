"""Benchmark a fresh localhost server, including JSON serialization and transport."""

import argparse
import json
import math
import os
import resource
import socket
import sys
import threading
from statistics import median
from time import monotonic, sleep
from urllib.request import Request, urlopen

import uvicorn

from text_to_graph.server import app


def documents(count):
    templates = (
        ("en", "On Monday, Jo lent a telescope to Kim at station {}."),
        ("en", "Maya founded Beacon Labs in Boston. She sold sensor {} to Northstar."),
        ("fr", "Le conservateur a prêté un télescope à Marie au musée {}."),
        ("es", "El museo prestó un telescopio a Lucía el día {}."),
    )
    docs = []
    for i in range(count):
        lang, template = templates[i % len(templates)]
        docs.append({"language": lang, "text": template.format(i + 1)})
    return docs


def run(repeats, single_only):
    docs = documents(32)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
        worker.start()
        peak_gpu = 0
        stop = threading.Event()

        def memory():
            nonlocal peak_gpu

            import torch

            while not stop.wait(0.05):
                if os.environ.get("TEXT_TO_GRAPH_BACKEND") == "metal":
                    peak_gpu = max(peak_gpu, torch.mps.driver_allocated_memory())
                elif os.environ.get("TEXT_TO_GRAPH_BACKEND") == "cuda":
                    peak_gpu = torch.cuda.max_memory_reserved()

        def request(group, batch):
            body = {"documents": group} if batch else group[0]
            path = "/v1/graph/batch" if batch else "/v1/graph"
            req = Request(
                url + path,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            start = monotonic()
            with urlopen(req, timeout=600) as response:
                data = json.load(response)
            elapsed = monotonic() - start
            graphs = [item["graph"] for item in data["results"]] if batch else [data]
            if len(graphs) != len(group) or any(
                g["profile"]["status"] != "complete" for g in graphs
            ):
                raise RuntimeError(f"Incomplete benchmark response at {path}")
            return elapsed

        sampler = threading.Thread(target=memory)
        try:
            while not server.started:
                if not worker.is_alive():
                    raise RuntimeError("Server failed to start")
                sleep(0.01)
            sampler.start()
            print(
                "cold_first_request_s", round(request(docs[:1], False), 3), flush=True
            )
            for doc in docs[:4]:
                request([doc], False)
            for size in (1, 8, 32):
                timings = {"single": [], "batch": []}
                calls = []
                if not single_only:
                    request(docs[:size], True)
                for repeat in range(repeats):
                    modes = (
                        ["single"]
                        if single_only
                        else (
                            ["single", "batch"]
                            if repeat % 2 == 0
                            else ["batch", "single"]
                        )
                    )
                    for mode in modes:
                        if mode == "single":
                            durations = [request([doc], False) for doc in docs[:size]]
                            calls.extend(durations)
                            timings[mode].append(sum(durations))
                        else:
                            timings[mode].append(request(docs[:size], True))
                for mode, samples in timings.items():
                    if samples:
                        latency = sorted(calls if mode == "single" else samples)
                        print(
                            size,
                            mode,
                            "docs/s",
                            round(size / median(samples), 2),
                            "request_p50/p95_s",
                            round(median(latency), 3),
                            round(latency[math.ceil(0.95 * len(latency)) - 1], 3),
                            flush=True,
                        )
        finally:
            stop.set()
            if sampler.ident:
                sampler.join()
            server.should_exit = True
            worker.join()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        "peak_rss_MiB",
        round(rss / (2**20 if sys.platform == "darwin" else 1024)),
        "sampled_gpu_MiB",
        round(peak_gpu / 2**20),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "metal", "cuda"], default="cpu")
    cli.add_argument("--repeats", type=int, default=5)
    cli.add_argument("--single-only", action="store_true")
    args = cli.parse_args()
    if args.repeats < 1:
        cli.error("--repeats must be positive")
    os.environ["TEXT_TO_GRAPH_BACKEND"] = args.backend
    run(args.repeats, args.single_only)
