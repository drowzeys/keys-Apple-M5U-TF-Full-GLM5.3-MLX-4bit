"""Decode rate as a streaming client sees it: prose, code, and a whole-file edit.

    python bench/decode.py [prose|code|edit ...]      (default: all three)
    RUNS=3 TEMP=0.6 TOKENS=400 python bench/decode.py prose

tok/s = (completion tokens - 1) / (time of last token - time of first token).
Prose is fresh essays (little repetition: the MTP head's case), code is a new
program, edit returns a given file with small changes (context copies: the
lookup drafts' case).
"""

import os
import sys

from common import filler, fmt, median, stream

RUNS = int(os.environ.get("RUNS", "3"))
TEMP = float(os.environ.get("TEMP", "0.6"))
TOKENS = int(os.environ.get("TOKENS", "400"))
CTX = int(os.environ.get("CTX", "0"))      # about this many tokens of unrelated notes before the task (long-context decode)

PROSE = ["Write an essay on how tides shaped coastal settlements.",
         "Write an essay on the history of the bicycle and what it changed.",
         "Write an essay on why bread rises, for a curious adult.",
         "Write an essay comparing rivers and roads as ways to move goods."]
CODE = ["Write a complete Python module implementing an LRU cache class with tests.",
        "Write a complete Python command-line tool that counts word frequencies in files.",
        "Write a complete JavaScript module implementing a debounce and a throttle function with comments."]
FILE = '''import math

def mean(values):
    return sum(values) / len(values)

def variance(values):
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)

def stddev(values):
    return math.sqrt(variance(values))

def median(values):
    s = sorted(values)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def percentile(values, p):
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)

def zscores(values):
    m = mean(values)
    sd = stddev(values)
    return [(v - m) / sd for v in values]

def summary(values):
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "stddev": stddev(values),
        "p90": percentile(values, 90),
    }
'''
EDITS = ["Add a one-line docstring to every function. Return the whole file, nothing else.",
         "Rename the parameter `values` to `xs` everywhere. Return the whole file, nothing else.",
         "Add type hints to every function. Return the whole file, nothing else."]


def bench(name, prompts, temp, tokens):
    rates, ttfts = [], []
    for i in range(RUNS):
        task = prompts[i % len(prompts)]
        if CTX:
            task = filler(CTX, 7000 + i) + "\n\nIgnore the notes above. " + task
        r = stream([{"role": "user", "content": task}], max_tokens=tokens, temperature=temp)
        rates.append(r["decode_tps"])
        ttfts.append(r["ttft"])
        print(f"  {name} run {i + 1}: {fmt(r['decode_tps'])} tok/s, {r['completion_tokens']} tokens, "
              f"first token {fmt(r['ttft'], 2)} s", flush=True)
    print(f"{name}: median {fmt(median(rates))} tok/s (T={temp}), first token {fmt(median(ttfts), 2)} s")
    return median(rates)


def main():
    which = sys.argv[1:] or ["prose", "code", "edit"]
    out = {}
    if "prose" in which:
        out["prose"] = bench("prose", PROSE, TEMP, TOKENS)
        out["prose T=0"] = bench("prose T=0", PROSE, 0.0, TOKENS)
    if "code" in which:
        out["code"] = bench("code", CODE, 0.0, TOKENS)
    if "edit" in which:
        out["edit"] = bench("edit", [e + "\n\n```python\n" + FILE + "```" for e in EDITS], 0.0, 700)
    print("SUMMARY " + " | ".join(f"{k} {fmt(v)}" for k, v in out.items()))


if __name__ == "__main__":
    main()
