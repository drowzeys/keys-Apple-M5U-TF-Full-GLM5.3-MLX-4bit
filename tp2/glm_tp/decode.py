"""Prefill, sampling and the two decode loops (plain, and MTP depth 1) shared by every rank.

Every decision comes from logits that are identical on all ranks (the hidden state is all-summed before the
replicated lm_head), and sampling uses the same seeded key on every rank, so all ranks pick the same tokens
without exchanging them. The only per-step message is the leader's stop flag (one all_sum of one integer).
"""
import mlx.core as mx


def prefill(model, cache, ids, chunk=2048, hidden=False):
    """Run `ids` (list of ints) through the model in chunks. Returns the last position's logits [V] and,
    with hidden=True, the post-norm hidden of every position [1, L, D] (for the MTP head)."""
    x = mx.array(ids)[None]
    hs, lg = [], None
    for i in range(0, x.shape[1], chunk):
        if hidden:
            out, h_raw = model(x[:, i:i + chunk], cache=cache, return_hidden=True)
            hp = model.model.norm(h_raw)
            mx.eval(out, hp)
            hs.append(hp)
        else:
            out = model(x[:, i:i + chunk], cache=cache)
            mx.eval(out)
        lg = out[0, -1]
        mx.clear_cache()
    return (lg, mx.concatenate(hs, axis=1)) if hidden else lg


def pick(logits, temperature, top_p, key):
    """Greedy when temperature == 0, otherwise temperature + nucleus sampling with `key`."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    lg = logits.astype(mx.float32) / temperature
    if 0 < top_p < 1:
        order = mx.argsort(-lg, axis=-1)
        sorted_lg = mx.take_along_axis(lg, order, axis=-1)
        probs = mx.softmax(sorted_lg, axis=-1)
        cum = mx.cumsum(probs, axis=-1)
        keep = (cum - probs) < top_p
        sorted_lg = mx.where(keep, sorted_lg, -float("inf"))
        choice = mx.random.categorical(sorted_lg, key=key)
        return mx.take_along_axis(order, choice[..., None], axis=-1)[..., 0]
    return mx.random.categorical(lg, key=key)


class StopSync:
    """The leader's stop decision, shared once per step: all_sum of [flag] (followers add 0)."""

    def __init__(self, group, leader):
        self.group, self.leader = group, leader

    def __call__(self, stop_leader):
        if self.group is None:
            return stop_leader
        v = mx.distributed.all_sum(mx.array([1 if (self.leader and stop_leader) else 0], mx.int32), group=self.group)
        return int(v.item()) > 0


def plain(model, cache, first, sync, should_stop, temperature=0.0, top_p=1.0, seed=0):
    """Yields tokens, starting with `first` (already chosen from the prefill logits). The step after a token
    is queued before the token is handed out (as mlx-lm's generate_step does)."""
    key = mx.random.key(seed)
    y = first
    while True:
        key, sub = mx.random.split(key)
        nxt = pick(model(mx.array([[y]]), cache=cache)[0, -1], temperature, top_p, sub)
        mx.async_eval(nxt)
        stop = sync(should_stop(y))
        if stop:
            mx.eval(nxt)
            return
        yield y
        y = int(nxt.item())


def mtp_greedy(model, cache, mcache, h_post, ids_from, y, sync, should_stop):
    """Depth-1 MTP speculative greedy decode, exact by construction (a draft is kept only when it equals the
    verify pass's argmax). h_post: post-norm trunk hidden of the positions just prefilled; ids_from: the
    prompt tokens of those positions; y: the first token (argmax of the prefill logits).
    Yields tokens one at a time; rolls the trunk cache back by one on a rejected draft."""
    norm = model.model.norm
    d = mx.argmax(model.mtp_forward(h_post, mx.array([list(ids_from[1:]) + [y]]), cache=mcache, logits_keep=1)[:, -1],
                  axis=-1)
    while True:
        if sync(should_stop(y)):
            return
        yield y
        inp = mx.concatenate([mx.array([[y]]), d[:, None]], axis=1)
        logits, h_raw = model(inp, cache=cache, return_hidden=True)
        pred = mx.argmax(logits, axis=-1)[0]
        mx.eval(pred, d)
        p0, p1 = pred.tolist()
        dv = int(d.item())
        hp = norm(h_raw)
        if p0 == dv:
            # the draft is the next token; the bonus token p1 follows it
            if sync(should_stop(dv)):
                return
            yield dv
            y = p1
            d = mx.argmax(model.mtp_forward(hp, mx.array([[dv, p1]]), cache=mcache, logits_keep=1)[:, -1], axis=-1)
        else:
            model.mtp_partial_rollback(cache, 0, 1)
            y = p0
            d = mx.argmax(model.mtp_forward(hp[:, :1], mx.array([[p0]]), cache=mcache, logits_keep=1)[:, -1], axis=-1)


def trim_all(caches, n):
    """Drop the last n tokens from every cache (trunk CacheLists or the MTP head's flat list)."""
    if n <= 0:
        return
    for c in caches:
        c.trim(n)


def offset(cache):
    c = cache[0]
    c = c.caches[0] if hasattr(c, "caches") else c
    return c.offset
