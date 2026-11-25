"""Prompt processing: tokens per second for fresh prompts of about 1k, 4k and
16k tokens, and the time to the first token of a follow-up turn on each.

    python bench/prompt.py [1000 4000 16000]
    SEED=7 python bench/prompt.py 32000

Every prompt starts with a different word (SEED and the size), so nothing is
served from a prompt cache on the first request. The rate is
prompt_tokens / time to the first token, which includes tokenizing and one
decode pass: a slight under-estimate.
"""

import os
import sys

from common import filler, fmt, stream

SEED = int(os.environ.get("SEED", "1"))


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [1000, 4000, 16000]
    rows = []
    for n in sizes:
        text = filler(n, SEED * 100003 + n)
        conv = [{"role": "user", "content": text + "\n\nHow many sentences above mention a river? Answer briefly."}]
        r = stream(conv, max_tokens=48, temperature=0.0)
        pt = r["prompt_tokens"]
        rate = (pt / r["ttft"]) if pt and r["ttft"] else None
        conv += [{"role": "assistant", "content": r["text"]},
                 {"role": "user", "content": "And how many mention a garden? Answer briefly."}]
        r2 = stream(conv, max_tokens=48, temperature=0.0)
        rows.append((n, pt, r["ttft"], rate, r2["ttft"], r["decode_tps"]))
        print(f"~{n}: {pt} prompt tokens, first token {fmt(r['ttft'], 2)} s = {fmt(rate, 0)} tok/s; "
              f"follow-up turn first token {fmt(r2['ttft'], 2)} s; decode at this context {fmt(r['decode_tps'])} tok/s",
              flush=True)
    print("SUMMARY " + " | ".join(f"{pt or n} tok: {fmt(rate, 0)} tok/s, next turn {fmt(t2, 2)} s"
                                  for n, pt, t1, rate, t2, d in rows))


if __name__ == "__main__":
    main()
