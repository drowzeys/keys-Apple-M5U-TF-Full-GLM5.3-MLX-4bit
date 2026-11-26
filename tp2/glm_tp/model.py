"""GLM-5.3 tensor parallelism over two Macs on MLX: sharding, rank folders, loading.

Model code: oMLX's glm_moe_dsa (Apache-2.0, installed as a dependency; stock mlx-lm cannot load GLM-5.3's
shared-indexer layers). The split follows oMLX's Model.shard() (attention heads, dense-MLP and expert
intermediate columns split; indexer, latent projections, router, norms, embeddings and lm_head replicated),
with the fused expert gate/up weight split as two segments (oMLX's own shard() splits it as one block, which
gives wrong numbers) and optional uneven shares.

    write(src, dst, rank=1, ...)      a rank folder (plain MLX safetensors) from the full checkpoint
    load_rank(dst, group)             build, shard, load strictly, one file at a time, wired
    load_full_sharded(src, group)     the full checkpoint, lazily, keeping only this rank's share
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def use_omlx_glm(mtp=False):
    """GLM-5.3 needs oMLX's glm_moe_dsa (shared indexer layers; stock mlx-lm 0.32 misses 285 tensors).
    mtp=True also applies oMLX's MTP patch, so the model carries `mtp[0]` (layer 78) and mtp_forward."""
    from omlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch

    apply_glm_moe_dsa_patch()
    # the MTP patch is applied either way: with MTP off its sanitize drops a merged checkpoint's mtp.* tensors
    # (stock sanitize only drops model.layers.78.*, and the merged folder carries six remapped mtp.0.* tensors)
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active

    apply_mlx_lm_mtp_patch()
    set_mtp_active(bool(mtp))


class SumQuantizedLinear(nn.QuantizedLinear):
    """A QuantizedLinear holding a slice of the input dimension; sums the partial outputs over the group.
    Same parameter names as QuantizedLinear, so checkpoints load unchanged."""

    def __call__(self, x):
        return mx.distributed.all_sum(super().__call__(x), group=self._g53_group)


def _bounds(n, unit, fractions, r):
    cum, acc = [0], 0.0
    for f in fractions:
        acc += f
        cum.append(min(n, int(round(n * acc / unit)) * unit))
    cum[-1] = n
    return cum[r], cum[r + 1]


def _sl(w, axis, lo, hi):
    idx = [slice(None)] * w.ndim
    idx[axis] = slice(lo, hi)
    return w[tuple(idx)]


def _rows(mod, lo, hi, segments=1):
    """Keep output rows [lo, hi) of each of `segments` stacked blocks (axis -2 of weight/scales/biases)."""
    for name in ("weight", "scales", "biases"):
        if name not in mod:
            continue
        w = mod[name]
        ax = w.ndim - 2
        seg = w.shape[ax] // segments
        parts = [_sl(w, ax, k * seg + lo, k * seg + hi) for k in range(segments)]
        mod[name] = mx.contiguous(parts[0] if segments == 1 else mx.concatenate(parts, axis=ax))
    assert "bias" not in mod


def _cols(mod, lo, hi):
    """Keep input columns [lo, hi) (in unpacked units; multiples of the group size)."""
    gs, bits = mod.group_size, mod.bits
    assert lo % gs == 0 and hi % gs == 0, (lo, hi, gs)
    for name in ("weight", "scales", "biases"):
        if name not in mod:
            continue
        per = 32 // bits if name == "weight" else gs
        mod[name] = mx.contiguous(_sl(mod[name], mod[name].ndim - 1, lo // per, hi // per))
    assert "bias" not in mod


def _in_dims(mod):
    return mod.weight.shape[-1] * 32 // mod.bits


def shard_glm(model, group, fractions=None, head_fractions=None):
    """Tensor-parallel split of oMLX's GLM-5.3 (same layout as its Model.shard(), with the fused
    gate/up split as two segments), with optional uneven shares per rank, e.g. (0.6, 0.4):
    attention heads, dense-MLP and expert intermediate columns are split by `fractions`
    (heads in whole heads, intermediate in multiples of the quantization group); `head_fractions`
    splits the heads separately (oMLX's native sparse-MLA prefill and the tensor-unit prefill
    kernel need a multiple of 32 heads per rank: 0.5/0.5 keeps 32/32 while the experts go 0.6/0.4);
    indexer, latent projections, router, norms, embeddings and lm_head are replicated."""
    N, r = group.size(), group.rank()
    fr = list(fractions) if fractions else [1.0 / N] * N
    hfr = list(head_fractions) if head_fractions else fr
    assert len(fr) == N and abs(sum(fr) - 1) < 1e-6 and len(hfr) == N and abs(sum(hfr) - 1) < 1e-6
    layers = list(model.model.layers) + [b.block for b in (getattr(model, "mtp", None) or [])]
    for layer in layers:
        at = layer.self_attn
        H = at.num_heads
        h0, h1 = _bounds(H, 1, hfr, r)
        rph = at.q_b_proj.weight.shape[0] // H
        _rows(at.q_b_proj, h0 * rph, h1 * rph)
        cph = _in_dims(at.o_proj) // H
        _cols(at.o_proj, h0 * cph, h1 * cph)
        at.o_proj.__class__ = SumQuantizedLinear
        at.o_proj._g53_group = group
        at.num_heads = h1 - h0
        at.embed_q.apply(lambda w: w[h0:h1])
        at.unembed_out.apply(lambda w: w[h0:h1])
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            mlp.sharding_group = group
            for owner in (mlp.switch_mlp, mlp.shared_experts):
                if hasattr(owner, "gate_up_proj"):
                    I = owner.gate_up_proj.weight.shape[-2] // 2
                    i0, i1 = _bounds(I, 64, fr, r)
                    _rows(owner.gate_up_proj, i0, i1, segments=2)
                else:
                    I = owner.gate_proj.weight.shape[-2]
                    i0, i1 = _bounds(I, 64, fr, r)
                    _rows(owner.gate_proj, i0, i1)
                    _rows(owner.up_proj, i0, i1)
                _cols(owner.down_proj, i0, i1)
        else:
            I = mlp.gate_proj.weight.shape[0]
            i0, i1 = _bounds(I, 64, fr, r)
            _rows(mlp.gate_proj, i0, i1)
            _rows(mlp.up_proj, i0, i1)
            _cols(mlp.down_proj, i0, i1)
            mlp.down_proj.__class__ = SumQuantizedLinear
            mlp.down_proj._g53_group = group


def _sink(files):
    """TP2_SINK = a command line; __FILES__ (or the end) receives the file paths. Used to move each finished
    file to the other Mac right away."""
    import shlex
    import subprocess

    cmd = shlex.split(os.environ["TP2_SINK"])
    cmd = [x for c in cmd for x in (files if c == "__FILES__" else [c])] if "__FILES__" in cmd else cmd + files
    subprocess.run(cmd, check=True)


class StandInGroup:
    """Enough of mx.distributed.Group for shard(): size and rank. Never used to communicate."""

    def __init__(self, size, rank):
        self._s, self._r = size, rank

    def size(self):
        return self._s

    def rank(self):
        return self._r


def _quant_map(model):
    out = {}
    for name, m in model.named_modules():
        if hasattr(m, "bits") and hasattr(m, "group_size") and "scales" in m:
            out[name] = {"group_size": m.group_size, "bits": m.bits, "mode": getattr(m, "mode", "affine")}
    return out


def write(src, dst, rank, size, chunk_gb, limit_layers=None, fractions=None, mtp_only=False, head_fractions=None):
    from mlx_lm.utils import load_model

    mx.set_default_device(mx.cpu)
    use_omlx_glm(mtp=mtp_only)
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, config = load_model(src, lazy=True, strict=True)
    qmap = _quant_map(model)
    shard_glm(model, StandInGroup(size, rank), fractions, head_fractions)
    flat = tree_flatten(model.parameters())
    del model
    if mtp_only:  # add the MTP layer to an existing rank folder
        flat = [(k, v) for k, v in flat if k.startswith("mtp.")]
        qmap = {k: v for k, v in qmap.items() if k.startswith("mtp.")}
    if limit_layers is not None:  # for a quick test: keep the first N layers and the non-layer tensors
        flat = [(k, v) for k, v in flat if not k.startswith("model.layers.") or int(k.split(".")[2]) < limit_layers]
    total = sum(v.nbytes for _, v in flat)
    print(f"rank {rank}/{size}: {len(flat)} arrays, {total / 2**30:.1f} GiB to write", flush=True)
    if os.environ.get("PRESHARD_DRY"):
        return
    files, cur, cur_b, index, done = [], {}, 0, {}, 0
    lim = int(chunk_gb * 2**30)

    def flush():
        nonlocal cur, cur_b
        if not cur:
            return
        name = f"model-{'mtp-' if mtp_only else ''}{len(files) + 1:05d}.safetensors"
        mx.eval(list(cur.values()))
        tmp = dst / (name + ".part.safetensors")
        mx.save_safetensors(str(tmp), cur, metadata={"format": "mlx"})
        os.replace(tmp, dst / name)
        if os.environ.get("TP2_SINK"):  # e.g. rsync to the other Mac and remove here: local disk holds one chunk
            import subprocess
            _sink([str(dst / name)])
        for k in cur:
            index[k] = name
        files.append(name)
        cur, cur_b = {}, 0

    while flat:
        k, v = flat.pop(0)
        cur[k] = v
        cur_b += v.nbytes
        done += v.nbytes
        if cur_b >= lim:
            flush()
            print(f"  {done / 2**30:7.1f} / {total / 2**30:.1f} GiB, {time.time() - t0:6.0f} s", flush=True)
    flush()
    if mtp_only:
        idx = json.loads((dst / "model.safetensors.index.json").read_text())
        idx["weight_map"].update(index)
        idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + total
        (dst / "model.safetensors.index.json").write_text(json.dumps(idx))
        qm = json.loads((dst / "quant_map.json").read_text())
        qm.update(qmap)
        (dst / "quant_map.json").write_text(json.dumps(qm))
        meta = json.loads((dst / "tp_shard.json").read_text())
        if meta.get("fractions") != fractions or meta.get("head_fractions") != head_fractions or (meta["size"], meta["rank"]) != (size, rank):
            raise SystemExit(f"{dst} was written with {meta}, not fractions {fractions} rank {rank}/{size}")
        meta["mtp"] = True
        (dst / "tp_shard.json").write_text(json.dumps(meta))
        if os.environ.get("TP2_SINK"):
            import subprocess
            _sink([str(dst / n) for n in ("model.safetensors.index.json", "quant_map.json", "tp_shard.json")])
        print(f"done: {len(files)} MTP files, {total / 2**30:.2f} GiB, {time.time() - t0:.0f} s", flush=True)
        return
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": index}, f)
    with open(dst / "quant_map.json", "w") as f:
        json.dump(qmap, f)
    with open(dst / "tp_shard.json", "w") as f:
        json.dump({"size": size, "rank": rank, "fractions": fractions, "head_fractions": head_fractions, "source": str(src), "limit_layers": limit_layers,
                   "tool": "tp2/preshard.py"}, f)
    for p in src.iterdir():
        if p.suffix != ".safetensors" and p.name != "model.safetensors.index.json" and p.is_file():
            shutil.copy(p, dst / p.name)
    if os.environ.get("TP2_SINK"):
        import subprocess
        extra = [str(p) for p in dst.iterdir() if p.suffix != ".safetensors"]
        _sink(extra)
    print(f"done: {len(files)} files, {total / 2**30:.1f} GiB, {time.time() - t0:.0f} s", flush=True)


def load_rank(path, group, lazy=False, limit_layers=None, mtp=False):
    """Build the model, shard it on `group` (arrays stay lazy), then load this rank's folder strictly:
    every shape must match what shard() produced, so a wrong folder fails here."""
    from mlx_lm.utils import _get_classes, load_config

    path = Path(path)
    meta = json.loads((path / "tp_shard.json").read_text())
    if mtp and not meta.get("mtp"):
        raise ValueError(f"{path} has no MTP layer (write it with --mtp)")
    use_omlx_glm(mtp=mtp)
    if (meta["size"], meta["rank"]) != (group.size(), group.rank()):
        raise ValueError(f"{path} holds rank {meta['rank']} of {meta['size']}, this is rank {group.rank()} of {group.size()}")
    config = load_config(path)
    if limit_layers is not None:
        config["num_hidden_layers"] = limit_layers
    from omlx.patches.mlx_lm_mtp import is_mtp_active

    config["num_nextn_predict_layers"] = 1 if is_mtp_active() else 0
    model_class, args_class = _get_classes(config=config)
    model = model_class(args_class.from_dict(config))
    qmap = json.loads((path / "quant_map.json").read_text())
    if not is_mtp_active():
        qmap = {k: v for k, v in qmap.items() if not k.startswith("mtp.")}
    nn.quantize(model, class_predicate=lambda p, m: qmap.get(p, False))
    shard_glm(model, group, meta.get("fractions"), meta.get("head_fractions"))
    from mlx.utils import tree_flatten as _tf

    want = {k: v.shape for k, v in _tf(model.parameters())}
    got = set()
    files = [f for f in sorted(glob.glob(str(path / "model-*.safetensors")))
             if is_mtp_active() or "model-mtp-" not in f]
    for f in files:
        # one file at a time, evaluated and touched on the GPU right away: loading everything in one
        # eval left ~40 GiB of freshly read, not yet resident buffers for macOS to compress
        # (256 Mac, even split, 11-26 13:38: compressor 38.6 GiB, guard kill)
        w = mx.load(f)
        for k, v in w.items():
            if k not in want:
                raise ValueError(f"{f}: {k} is not a parameter of the sharded model")
            if tuple(v.shape) != tuple(want[k]):
                raise ValueError(f"{f}: {k} has shape {v.shape}, the sharded model wants {want[k]}")
        model.load_weights(list(w.items()), strict=False)
        got.update(w)
        if not lazy:
            mx.eval(list(w.values()))
            mx.eval(sum(a.reshape(-1)[:1].astype(mx.float32).sum() for a in w.values()))
        del w
    missing = set(want) - got
    if missing:
        raise ValueError(f"{len(missing)} parameters missing from {path}: {sorted(missing)[:5]}")
    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    return model, config


def load_full_sharded(src, group, fractions=None, head_fractions=None, mtp=False, limit_layers=None):
    """Rank 0 straight from the full checkpoint: built lazily, sharded, then evaluated one layer at a time
    (only this rank's share is read into memory). No rank folder needed on this Mac."""
    from mlx_lm.utils import load_model

    use_omlx_glm(mtp=mtp)
    cfg = {"num_nextn_predict_layers": 1 if mtp else 0}
    if limit_layers:
        cfg["num_hidden_layers"] = limit_layers
    model, config = load_model(Path(src), lazy=True, strict=True, model_config=cfg)
    if group is not None:
        shard_glm(model, group, fractions, head_fractions)
    for layer in model.model.layers:
        mx.eval(layer.parameters())
    mx.eval(model.parameters())
    model.eval()
    return model, config
