"""Smoke test for the TP=2 server.

    python tp2/smoke.py [--url http://127.0.0.1:8000/v1] [--needle-tokens 20000]

Checks: Canberra; streaming against a non-streamed request in the same cache state; a needle in a ~20k-token
prompt; three requests back to back; two at once (the second queues); a prompt over the context limit (HTTP 400);
the completions endpoint. Exit code 0 when every check passes.

Note: the server reuses the cached prefix of the previous request. The first (cold) answer to a prompt can differ
from later (warm) ones at near-ties, so equality checks compare requests made in the same cache state.
"""
import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request


def post(url, body, stream=False):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=3600)
    if not stream:
        return json.loads(r.read())
    chunks = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def answer(r):
    m = r["choices"][0]["message"]
    return m.get("reasoning_content", ""), m.get("content", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--needle-tokens", type=int, default=20000)
    a = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name, detail, flush=True)
        ok &= bool(cond)

    q = [{"role": "user", "content": "What is the capital of Australia? Answer in one word."}]
    body = {"messages": q, "max_tokens": 400, "temperature": 0}
    t = time.time()
    r = post(a.url + "/chat/completions", body)
    _, ans = answer(r)
    check("Canberra", "Canberra" in ans, f"(answer {ans!r}, {time.time() - t:.1f} s, {r.get('timings')})")

    s = post(a.url + "/chat/completions", dict(body, stream=True), stream=True)
    sr = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in s if c.get("choices"))
    sa = "".join(c["choices"][0]["delta"].get("content", "") for c in s if c.get("choices"))
    r2 = post(a.url + "/chat/completions", body)
    rr, ra = answer(r2)
    check("streaming equals a non-streamed request in the same cache state",
          (sr.strip(), sa.strip()) == (rr.strip(), ra.strip()), f"({len(s)} chunks; answer {sa!r})")

    rnd = random.Random(7)
    words = "river stone lantern orchard meadow copper violet harbor willow ember quartz falcon".split()
    filler = [" ".join(rnd.choice(words) for _ in range(12)) + "." for _ in range(max(1, a.needle_tokens // 15))]
    filler.insert(len(filler) * 2 // 3, "The secret code for the vault is 4817-ZEBRA.")
    m = [{"role": "user", "content": "\n".join(filler) + "\n\nWhat is the secret code for the vault? Reply with the code only."}]
    t = time.time()
    # short thinking (GLM-5.3's template: reasoning_effort low/high, default max) and room for the answer
    r = post(a.url + "/chat/completions", {"messages": m, "max_tokens": 3000, "temperature": 0,
                                           "reasoning_effort": "low"})
    _, ans = answer(r)
    check("needle", "4817-ZEBRA" in ans, f"(prompt {r['usage']['prompt_tokens']} tokens, answer {ans!r}, "
          f"{time.time() - t:.0f} s, {r.get('timings')})")

    outs = [answer(post(a.url + "/chat/completions", body)) for _ in range(3)]
    check("three requests back to back", all("Canberra" in o[1] for o in outs) and outs[1] == outs[2],
          "(warm answers identical)" if outs[1] == outs[2] else repr(outs))

    res = [None, None]

    def one(k):
        try:
            res[k] = post(a.url + "/chat/completions", body)
        except Exception as e:
            res[k] = e
    th = [threading.Thread(target=one, args=(k,)) for k in range(2)]
    t = time.time()
    [x.start() for x in th]
    [x.join() for x in th]
    good = [isinstance(x, dict) and "Canberra" in answer(x)[1] for x in res]
    check("two concurrent requests (the second queues)", all(good), f"({time.time() - t:.1f} s)")

    limit = json.loads(urllib.request.urlopen(a.url + "/models").read())["data"][0].get("context_length")
    if limit:
        try:
            post(a.url + "/completions", {"prompt": [1] * (limit + 1), "max_tokens": 1})
            check("over-long prompt refused", False, "(no error)")
        except urllib.error.HTTPError as e:
            check("over-long prompt refused", e.code == 400, f"(HTTP {e.code})")
    r = post(a.url + "/completions", {"prompt": "The three primary colours are", "max_tokens": 24, "temperature": 0})
    check("completions endpoint", len(r["choices"][0]["text"]) > 0, repr(r["choices"][0]["text"][:60]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
