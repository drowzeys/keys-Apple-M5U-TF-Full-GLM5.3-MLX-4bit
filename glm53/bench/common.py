"""Shared client code for the checks and benchmarks (OpenAI-style HTTP API).

URL defaults to the new server (http://127.0.0.1:8020); set URL and MODEL for
another one, e.g. a stock oMLX server:  URL=http://127.0.0.1:8000 MODEL=GLM-5.3-4bit-MTP
"""

import json
import os
import random
import time
import urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8020").rstrip("/")
MODEL = os.environ.get("MODEL", "default_model")
KEY = os.environ.get("API_KEY", "")
ENDPOINT = URL + "/v1/chat/completions"


def _request(body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    return urllib.request.Request(ENDPOINT, json.dumps(body).encode(), headers)


def chat(messages, max_tokens=300, temperature=0.0, timeout=1800, **kw):
    """One non-streaming request. Returns (message dict, finish_reason, usage, seconds)."""
    body = dict(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature, **kw)
    t = time.time()
    with urllib.request.urlopen(_request(body), timeout=timeout) as r:
        d = json.load(r)
    c = d["choices"][0]
    return c["message"], c.get("finish_reason"), d.get("usage", {}), time.time() - t


def text_of(message):
    return (message.get("content") or "")


def stream(messages, max_tokens=300, temperature=0.0, timeout=1800, **kw):
    """One streaming request. Returns a dict with the text, time to the first
    token, decode rate between the first and the last token, and usage."""
    body = dict(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature,
                stream=True, stream_options={"include_usage": True}, **kw)
    t0 = time.time()
    first = last = None
    pieces, chunks, usage, reasoning, thoughts = [], 0, {}, 0, []
    with urllib.request.urlopen(_request(body), timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except ValueError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                piece = delta.get("content") or ""
                think = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if piece or think:
                    now = time.time()
                    first = first or now
                    last = now
                    chunks += 1
                    reasoning += 1 if think else 0
                    pieces.append(piece)
                    thoughts.append(think)
    t1 = time.time()
    n = usage.get("completion_tokens") or chunks
    span = (last - first) if first and last and last > first else None
    return dict(text="".join(pieces), reasoning="".join(thoughts), ttft=(first - t0) if first else None, total=t1 - t0,
                completion_tokens=n, prompt_tokens=usage.get("prompt_tokens"),
                decode_tps=((n - 1) / span) if span and n > 1 else None, chunks=chunks,
                reasoning_chunks=reasoning, usage=usage)


_WORDS = ("river stone amber winter signal copper garden lantern harbor meadow orbit thread "
          "velvet canyon ember timber frost marble echo saddle willow anchor bridge cedar "
          "delta falcon granite hollow island jasper kettle ledger mirror nectar onyx pebble "
          "quartz ribbon summit tunnel umber valley walnut yonder zephyr").split()


def filler(tokens, seed):
    """About `tokens` tokens of varied prose that no cache has seen (the seed
    changes the text from the first word on)."""
    rnd = random.Random(seed)
    out, n = [], 0
    while n < tokens:
        k = rnd.randrange(8, 18)
        out.append(" ".join(rnd.choice(_WORDS) for _ in range(k)).capitalize() + ".")
        n += k + 1
    return f"Notebook {seed}. " + " ".join(out)


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def fmt(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"
