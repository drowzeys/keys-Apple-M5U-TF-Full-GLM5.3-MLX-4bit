"""OpenAI-compatible server for GLM-5.3 tensor-parallel over two Macs (MLX, jaccl over Thunderbolt 5 RDMA).

Start it on both Macs at once with mlx.launch (oneshot-tp2.sh does this). Rank 0 serves HTTP:
    GET  /v1/models, /health
    POST /v1/chat/completions, /v1/completions   (stream: true gives server-sent events)
Rank 1 follows: it connects to rank 0's control port, receives each request's token ids and settings, and runs
the same forward passes; the two ranks meet in the model's all_sums.

One request at a time (requests queue). The KV cache of the last request is kept, and a new request reuses its
longest common prefix. MTP depth 1 runs for greedy requests (temperature 0) when the rank folders carry the
MTP layer; sampled requests use plain decoding.

    mlx.launch --backend jaccl --hostfile tp2-hostfile.json -- tp2/server.py \
        --rank0 PATH --rank1 PATH [--port 8000] [--mtp] [--kv8] [--max-context 131072]
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlx.core as mx  # noqa: E402

from glm_tp import decode as D  # noqa: E402
from glm_tp import model as M  # noqa: E402


def log(*a):
    print("[tp2]", *a, flush=True)


# ---------------------------------------------------------------- control channel (rank 0 -> rank 1)

def ctrl_host():
    """Rank 0's Thunderbolt link-local address (mlx.launch exports it as the jaccl coordinator)."""
    return os.environ.get("MLX_JACCL_COORDINATOR", "127.0.0.1:0").rsplit(":", 1)[0]


def read_secret(path):
    if not path:
        return None
    st = os.stat(path)
    if st.st_mode & 0o077:
        raise SystemExit(f"{path} must be readable by its owner only (chmod 600)")
    s = Path(path).read_bytes().strip()
    if len(s) < 32:
        raise SystemExit(f"{path}: the shared secret must be at least 32 characters")
    return s


def answer_challenge(secret, nonce):
    import hashlib
    import hmac

    return hmac.new(secret, b"tp2-ctrl:" + nonce, hashlib.sha256).hexdigest()


def send_json(sock, obj):
    data = json.dumps(obj).encode() + b"\n"
    sock.sendall(data)


def recv_json(f):
    line = f.readline()
    if not line:
        raise ConnectionError("control channel closed")
    return json.loads(line)


# ---------------------------------------------------------------- engine (runs on every rank)

class Engine:
    def __init__(self, args, group):
        self.args, self.group = args, group
        self.rank = group.rank()
        self.leader = self.rank == 0
        from mlx_lm.utils import load_tokenizer, maybe_set_recommended_wired_limit

        maybe_set_recommended_wired_limit()
        mx.set_cache_limit(int(args.cache_gb * 2**30))
        t = time.time()
        path = Path(args.rank0 if self.leader else args.rank1)
        meta = path / "tp_shard.json"
        if meta.exists():
            self.model, _ = M.load_rank(path, group, mtp=args.mtp)
        else:  # rank 0 straight from the full checkpoint
            info = json.loads((Path(args.rank1) / "tp_shard.json").read_text()) if Path(args.rank1).exists() else {}
            fr = args.fractions or info.get("fractions")
            hfr = args.head_fractions or info.get("head_fractions")
            self.model, _ = M.load_full_sharded(path, group, fr, hfr, mtp=args.mtp)
        heads = mx.distributed.all_sum(mx.array([self.model.model.layers[0].self_attn.num_heads]), group=group)
        if int(heads.item()) != 64:
            raise SystemExit(f"rank {self.rank}: the two rank folders do not belong together (heads {int(heads.item())})")
        log(f"rank {self.rank}: loaded in {time.time() - t:.0f} s, {mx.get_active_memory() / 2**30:.1f} GiB")
        if args.kv8:
            from glm_tp import kv8

            kv8.install()
        if not args.no_prefill_fixes:
            from glm_tp import lc

            log(f"rank {self.rank}: prefill fixes {lc.install()}")
        self.tok = None
        if self.leader:
            # GLM-5.3 ends a turn with <|user|> / <|observation|> as well as <|endoftext|>: take every end id from
            # config.json and generation_config.json (mlx-lm's loader alone keeps only the tokenizer's own)
            eos = set()
            for f in ("config.json", "generation_config.json"):
                fp = path / f
                if fp.exists():
                    v = json.loads(fp.read_text()).get("eos_token_id")
                    eos.update(v if isinstance(v, list) else [v] if v is not None else [])
            self.tok = load_tokenizer(path, eos_token_ids=sorted(eos) or None)
        self.cache = self.model.make_cache()
        self.mcache = self.model.make_mtp_cache() if args.mtp else None
        self.cached = []          # the token ids whose KV is in self.cache

    # a request, as both ranks see it: ids, settings, keep (cached prefix to reuse)
    def run(self, req, emit=None, should_stop=None):
        ids, keep = req["ids"], req["keep"]
        D.trim_all(self.cache, D.offset(self.cache) - keep)
        use_mtp = self.args.mtp and req["temperature"] <= 0
        if self.mcache is not None:
            if use_mtp:
                D.trim_all(self.mcache, self.mcache[0].offset - keep)
            else:  # plain decoding leaves the head behind; rebuild it on the next MTP request
                D.trim_all(self.mcache, self.mcache[0].offset)
        sync = D.StopSync(self.group, self.leader)
        t0 = time.perf_counter()
        new = ids[keep:]
        if use_mtp:
            lg, hp = D.prefill(self.model, self.cache, new, self.args.chunk, hidden=True)
        else:
            lg = D.prefill(self.model, self.cache, new, self.args.chunk)
        key = mx.random.key(req["seed"])
        key, sub = mx.random.split(key)
        y = int(D.pick(lg, req["temperature"], req["top_p"], sub).item())
        t1 = time.perf_counter()
        stop = should_stop or (lambda tok: False)
        if use_mtp:
            gen = D.mtp_greedy(self.model, self.cache, self.mcache, hp, new, y, sync, stop)
        else:
            gen = D.plain(self.model, self.cache, y, sync, stop, req["temperature"], req["top_p"], req["seed"] + 1)
        out = []
        for t_ in gen:
            out.append(t_)
            if emit:
                emit(t_)
        t2 = time.perf_counter()
        n_cached = D.offset(self.cache)
        self.cached = (ids + out)[:n_cached]
        return out, {"prompt_tokens": len(ids), "cached_tokens": keep, "prefill_s": t1 - t0, "decode_s": t2 - t1,
                     "mtp": use_mtp}

    def plan_keep(self, ids):
        n = 0
        for a, b in zip(self.cached, ids):
            if a != b:
                break
            n += 1
        n = min(n, len(ids) - 1)
        if self.args.mtp:
            n = min(max(n - 1, 0), self.mcache[0].offset)
        return max(n, 0)


def follower(engine, args):
    host = ctrl_host()
    secret = read_secret(args.secret_file)
    for i in range(600):
        try:
            sock = socket.create_connection((host, args.ctrl_port), timeout=10)
            break
        except OSError:
            time.sleep(1)
    else:
        raise SystemExit(f"rank 1: cannot reach rank 0's control port {host}:{args.ctrl_port}")
    sock.settimeout(30)
    f0 = sock.makefile("r")
    hello = recv_json(f0)                     # {"nonce": hex} when rank 0 asks for the shared secret
    if hello.get("nonce"):
        if secret is None:
            raise SystemExit("rank 0 asks for the shared secret: start rank 1 with --secret-file too")
        send_json(sock, {"mac": answer_challenge(secret, bytes.fromhex(hello["nonce"]))})
        if not recv_json(f0).get("ok"):
            raise SystemExit("rank 0 refused rank 1's shared secret")
    else:
        send_json(sock, {"mac": None})
        recv_json(f0)
    sock.settimeout(None)
    f = f0
    log(f"rank 1: following rank 0 at {host}:{args.ctrl_port}")
    while True:
        req = recv_json(f)
        if req.get("op") == "quit":
            log("rank 1: quit")
            return
        engine.run(req)
        send_json(sock, {"done": True})


# ---------------------------------------------------------------- HTTP (rank 0)

class ContextTooLong(ValueError):
    def __init__(self, n, limit):
        super().__init__(f"the prompt has {n} tokens; this server's context is {limit} (--max-context)")


def split_reasoning(text, thinking=True):
    """GLM-5.3 thinks first: '<think>...</think>answer' (the chat template opens the tag in the prompt). Returns
    (reasoning, answer). Without a closing tag the whole text is reasoning when the prompt opened thinking
    (the answer never came, e.g. max_tokens), otherwise it is the answer."""
    j = text.find("</think>")
    if j < 0:
        return (text.replace("<think>", "").strip(), "") if thinking else ("", text)
    r = text[:j]
    r = r[len("<think>"):] if r.startswith("<think>") else r
    return r.strip(), text[j + len("</think>"):].lstrip()

class Leader:
    def __init__(self, engine, args):
        self.e, self.args = engine, args
        import queue

        self.lock = threading.Lock()
        self.jobs = queue.Queue()
        # The control port listens on the Thunderbolt link-local address only, and with --secret-file a peer
        # must answer an HMAC-SHA256 challenge before it gets any request.
        import secrets as _secrets
        import hmac

        secret = read_secret(args.secret_file)
        if secret is None:
            log("warning: no --secret-file: anything that reaches the Thunderbolt address can take rank 1's place")
        bind = ctrl_host() if args.ctrl_bind == "auto" else args.ctrl_bind
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((bind, args.ctrl_port))
        srv.listen(4)
        log(f"rank 0: waiting for rank 1 on {bind}:{args.ctrl_port}")
        while True:
            peer, addr = srv.accept()
            peer.settimeout(30)
            pf = peer.makefile("r")
            try:
                if secret is None:
                    send_json(peer, {"nonce": None})
                    recv_json(pf)
                    send_json(peer, {"ok": True})
                    break
                nonce = _secrets.token_bytes(32)
                send_json(peer, {"nonce": nonce.hex()})
                mac = recv_json(pf).get("mac") or ""
                if hmac.compare_digest(mac, answer_challenge(secret, nonce)):
                    send_json(peer, {"ok": True})
                    break
                send_json(peer, {"ok": False})
                log(f"rank 0: refused a control connection from {addr[0]} (wrong shared secret)")
            except (OSError, ValueError, ConnectionError) as e:
                log(f"rank 0: dropped a control connection from {addr[0]}: {e!r}")
            peer.close()
        peer.settimeout(None)
        self.peer, self.peer_f = peer, pf
        srv.close()
        log(f"rank 0: rank 1 connected from {addr[0]}")

    # MLX streams belong to the thread that created them, so every model call runs on the main thread (loop());
    # HTTP threads hand their requests over through a queue and receive text chunks back.
    def generate(self, ids, max_tokens, temperature, top_p, seed, stops, emit_text):
        import queue

        q = queue.Queue()
        proxy = None
        if emit_text is not None:
            def proxy(s):
                q.put(("text", s))
            proxy.closed = emit_text.closed
        self.jobs.put((q, (ids, max_tokens, temperature, top_p, seed, stops, proxy)))
        while True:
            kind, val = q.get()
            if kind == "text":
                emit_text(val)
            elif kind == "error":
                raise val
            else:
                return val

    def loop(self):
        while True:
            q, a = self.jobs.get()
            try:
                q.put(("done", self._generate(*a)))
            except Exception as e:  # a failed request must not stop the loop (both ranks stay in step)
                log(f"request failed: {e!r}")
                q.put(("error", e))

    def _generate(self, ids, max_tokens, temperature, top_p, seed, stops, emit_text):
        e, tok = self.e, self.e.tok
        if len(ids) >= self.args.max_context:
            raise ContextTooLong(len(ids), self.args.max_context)
        if len(ids) + max_tokens > self.args.max_context:
            max_tokens = self.args.max_context - len(ids)
        eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
        state = {"n": 0, "text": "", "finish": None, "sent": 0}
        detok = tok.detokenizer
        detok.reset()
        hold = max((len(s) for s in stops), default=0)

        def cut_stops():
            for s in stops:
                j = state["text"].find(s)
                if j >= 0:
                    state["text"] = state["text"][:j]
                    state["finish"] = "stop"

        def flush(final=False):
            if not emit_text:
                return
            upto = len(state["text"]) if (final or state["finish"] == "stop") else len(state["text"]) - hold
            if upto > state["sent"]:
                emit_text(state["text"][state["sent"]:upto])
                state["sent"] = upto

        def should_stop(t):
            if state["finish"] is not None:
                return True
            if t in eos:
                state["finish"] = "stop"
                return True
            if state["n"] >= max_tokens:
                state["finish"] = "length"
                return True
            if emit_text and emit_text.closed():
                state["finish"] = "disconnected"
                return True
            return False

        def emit(t):
            state["n"] += 1
            detok.add_token(t)
            state["text"] = detok.text
            cut_stops()
            flush()

        with self.lock:
            keep = e.plan_keep(ids)
            req = {"op": "gen", "ids": ids, "keep": keep, "temperature": float(temperature), "top_p": float(top_p),
                   "seed": int(seed)}
            send_json(self.peer, req)
            out, stats = e.run(req, emit=emit, should_stop=should_stop)
            ack = recv_json(self.peer_f)
            assert ack.get("done")
        detok.finalize()
        if state["finish"] != "stop" or not stops:
            state["text"] = detok.text
            cut_stops()
        flush(final=True)
        stats.update(completion_tokens=len(out), finish_reason=state["finish"] or "length", token_ids=out)
        return state["text"], stats


def make_handler(leader, args):
    tok = leader.e.tok
    model_name = args.model_name

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local",
                                                                 "context_length": args.max_context}]})
            elif self.path == "/health":
                self._json(200, {"status": "ok"})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": {"message": "invalid JSON"}})
            chat = self.path.endswith("/chat/completions")
            if not (chat or self.path.endswith("/completions")):
                return self._json(404, {"error": {"message": "not found"}})
            if chat:
                kw = dict(body.get("chat_template_kwargs") or {})
                if body.get("reasoning_effort") and "reasoning_effort" not in kw:
                    kw["reasoning_effort"] = body["reasoning_effort"]  # GLM-5.3's template: "low" | "high" | max
                ids = tok.apply_chat_template(body.get("messages", []), add_generation_prompt=True, tokenize=True, **kw)
                if hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
                ids = list(ids)
            else:
                p = body.get("prompt", "")
                if isinstance(p, list) and p and all(isinstance(v, int) for v in p):
                    ids = list(p)                        # token ids (OpenAI allows a token array)
                else:
                    ids = list(tok.encode(p if isinstance(p, str) else "".join(p)))
            tail = tok.decode(ids[-6:]) if chat else ""
            thinking = chat and "<think>" in tail and "</think>" not in tail.split("<think>")[-1]
            if len(ids) >= args.max_context:
                return self._json(400, {"error": {"message": str(ContextTooLong(len(ids), args.max_context)),
                                                  "type": "invalid_request_error", "code": "context_length_exceeded"}})
            max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or args.default_max_tokens)
            temperature = float(body.get("temperature", 0.0) if body.get("temperature") is not None else 0.0)
            top_p = float(body.get("top_p", 1.0) or 1.0)
            seed = int(body.get("seed", int(time.time() * 1000) % 2**31))
            stops = body.get("stop") or []
            stops = [stops] if isinstance(stops, str) else list(stops)
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            created = int(time.time())
            stream = bool(body.get("stream"))
            if not stream:
                text, st = leader.generate(ids, max_tokens, temperature, top_p, seed, stops, None)
                usage = {"prompt_tokens": st["prompt_tokens"], "completion_tokens": st["completion_tokens"],
                         "total_tokens": st["prompt_tokens"] + st["completion_tokens"],
                         "prompt_tokens_details": {"cached_tokens": st["cached_tokens"]}}
                timings = {"prefill_s": round(st["prefill_s"], 3), "decode_s": round(st["decode_s"], 3),
                           "decode_tok_s": round(st["completion_tokens"] / st["decode_s"], 2) if st["decode_s"] else None,
                           "mtp": st["mtp"]}
                if chat:
                    reasoning, answer = split_reasoning(text, thinking)
                    msg = {"role": "assistant", "content": answer}
                    if reasoning:
                        msg["reasoning_content"] = reasoning
                    choice = {"index": 0, "message": msg, "finish_reason": st["finish_reason"]}
                    obj = "chat.completion"
                else:
                    choice = {"index": 0, "text": text, "finish_reason": st["finish_reason"]}
                    obj = "text_completion"
                if body.get("return_token_ids"):
                    choice["token_ids"] = st["token_ids"]
                return self._json(200, {"id": rid, "object": obj, "created": created, "model": model_name,
                                        "choices": [choice], "usage": usage, "timings": timings})
            # streaming
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            wf = self.wfile
            closed = {"v": False}

            def sse(obj):
                try:
                    wf.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    wf.flush()
                except OSError:
                    closed["v"] = True

            def chunk(delta=None, finish=None, text=None):
                if chat:
                    return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
                return {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                        "choices": [{"index": 0, "text": text or "", "finish_reason": finish}]}

            if chat:
                sse(chunk({"role": "assistant", "content": ""}))

            split = {"buf": "", "answer": not thinking}
            TAG = "</think>"

            def emit_text(s):
                if not chat:
                    return sse(chunk(text=s))
                if split["answer"]:
                    return sse(chunk({"content": s}))
                split["buf"] += s
                j = split["buf"].find(TAG)
                if j >= 0:
                    r, rest = split["buf"][:j], split["buf"][j + len(TAG):].lstrip()
                    if r:
                        sse(chunk({"reasoning_content": r}))
                    split["answer"], split["buf"] = True, ""
                    if rest:
                        sse(chunk({"content": rest}))
                elif len(split["buf"]) > len(TAG):
                    out = split["buf"][:-len(TAG)]
                    split["buf"] = split["buf"][-len(TAG):]
                    sse(chunk({"reasoning_content": out.replace("<think>", "")}))
            emit_text.closed = lambda: closed["v"]
            text, st = leader.generate(ids, max_tokens, temperature, top_p, seed, stops, emit_text)
            if chat and split["buf"]:  # reasoning never closed (as split_reasoning): the rest is reasoning too
                sse(chunk({"reasoning_content": split["buf"].replace("<think>", "")}))
            sse(chunk(finish=st["finish_reason"]))
            sse({"id": rid, "object": "chat.completion.chunk" if chat else "text_completion", "created": created,
                 "model": model_name, "choices": [],
                 "usage": {"prompt_tokens": st["prompt_tokens"], "completion_tokens": st["completion_tokens"],
                           "total_tokens": st["prompt_tokens"] + st["completion_tokens"]}})
            try:
                wf.write(b"data: [DONE]\n\n")
                wf.flush()
            except OSError:
                pass
            self.close_connection = True

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank0", required=True, help="rank 0's folder: a rank folder (preshard.py) or the full checkpoint")
    ap.add_argument("--rank1", required=True, help="rank 1's folder (on the second Mac)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctrl-port", type=int, default=5470)
    ap.add_argument("--ctrl-bind", default="auto", help="control port address (auto = rank 0's Thunderbolt address)")
    ap.add_argument("--secret-file", help="shared secret (same file on both Macs, chmod 600) for the control channel")
    ap.add_argument("--mtp", action="store_true", help="MTP depth 1 for greedy requests (rank folders written with --mtp)")
    ap.add_argument("--kv8", action="store_true", help="8-bit latent KV cache (57.7 instead of 95.2 KB per token)")
    ap.add_argument("--max-context", type=int, default=131072)
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk rows")
    ap.add_argument("--cache-gb", type=float, default=4.0, help="MLX free-buffer cache bound")
    ap.add_argument("--default-max-tokens", type=int, default=2048)
    ap.add_argument("--model-name", default="glm-5.3-tp2")
    ap.add_argument("--fractions", type=lambda s: [float(v) for v in s.split(",")])
    ap.add_argument("--head-fractions", type=lambda s: [float(v) for v in s.split(",")])
    ap.add_argument("--no-prefill-fixes", action="store_true")
    args = ap.parse_args()
    if os.environ.get("MLX_METAL_FAST_SYNCH") != "1":
        log("warning: MLX_METAL_FAST_SYNCH=1 is not set; each all_sum then costs about 0.2 ms and decode slows by ~40%")
    group = mx.distributed.init(strict=True, backend=os.environ.get("TP2_BACKEND", "jaccl"))
    if group.size() != 2:
        raise SystemExit("run on exactly two ranks")
    engine = Engine(args, group)
    if group.rank() != 0:
        return follower(engine, args)
    leader = Leader(engine, args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(leader, args))
    log(f"rank 0: serving http://{args.host}:{args.port}/v1 (model {args.model_name}, mtp {args.mtp}, kv8 {args.kv8})")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        leader.loop()
    finally:
        try:
            send_json(leader.peer, {"op": "quit"})
        except OSError:
            pass


if __name__ == "__main__":
    main()
