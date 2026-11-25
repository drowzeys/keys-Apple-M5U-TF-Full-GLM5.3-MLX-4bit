"""Put the MTP head next to a 4-bit GLM-5.3 checkpoint, without copying the checkpoint.

    python tools/merge_mtp_head.py inspect --head DIR
    python tools/merge_mtp_head.py merge --base DIR --head DIR --out DIR [--copy-head]

`inspect` reads only safetensors headers and prints what the head contains:
key prefix, tensor count, shapes, and the bit width of quantized tensors. Run
it first: the key naming of the head repository was not known when this was
written.

`merge` builds --out as a directory of symlinks to every file of --base, plus
the head's weight file(s) under names mlx-lm picks up (model-mtp-*.safetensors),
an index that lists the head's tensors, and a config with
num_nextn_predict_layers = 1. Nothing in --base or --head is changed.

Key names oMLX's loader understands for the head (omlx/patches/mlx_lm_mtp/
glm_moe_dsa_model.py, `sanitize`):
    model.layers.78.*   the raw layout (layer index = num_hidden_layers), or
    mtp.0.*             the already remapped layout.
If the head uses another prefix, pass --rename OLD=NEW: the head file is then
rewritten with renamed keys (same bytes, new header) instead of linked.
Pure Python, no MLX: safe to run while the GPU is busy.
"""

import argparse
import json
import os
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 512 << 20:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        return json.loads(f.read(n)), 8 + n


def weight_files(d):
    if os.path.isfile(d):
        return [d]
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".safetensors"))


def inspect(head):
    files = weight_files(head)
    if not files:
        raise SystemExit(f"no .safetensors in {head}")
    total = 0
    for path in files:
        hdr, off = read_header(path)
        meta = hdr.pop("__metadata__", None)
        keys = sorted(hdr)
        size = os.path.getsize(path)
        total += size
        print(f"{os.path.basename(path)}: {len(keys)} tensors, {size / 1e9:.2f} GB, metadata {meta}")
        prefixes = {}
        for k in keys:
            parts = k.split(".")
            p = ".".join(parts[:3]) if parts[0] == "model" else ".".join(parts[:2])
            prefixes[p] = prefixes.get(p, 0) + 1
        for p, n in sorted(prefixes.items()):
            print(f"   prefix {p}.  {n} tensors")
        shown = 0
        for k in keys:
            if k.endswith(".scales") or shown < 12:
                info = hdr[k]
                extra = ""
                if k.endswith(".scales"):
                    w = hdr.get(k[:-7] + ".weight")
                    if w:
                        # weight [..., in * bits / 32] uint32, scales [..., in / group]
                        extra = f"  (packed weight {w['shape']} {w['dtype']})"
                if shown < 40:
                    print(f"   {k}: {info['dtype']} {info['shape']}{extra}")
                shown += 1
        style = ("mtp.* (already remapped)" if any(k.startswith("mtp.") for k in keys) else
                 "model.layers.N.* (raw extra layer)" if any(k.startswith("model.layers.") for k in keys)
                 else "UNKNOWN: use --rename")
        print(f"   layout: {style}")
    print(f"total {total / 1e9:.2f} GB")


def rewrite(src, dst, renames):
    hdr, off = read_header(src)
    meta = hdr.pop("__metadata__", None)
    out = {}
    for k, v in hdr.items():
        nk = k
        for old, new in renames:
            if k.startswith(old):
                nk = new + k[len(old):]
                break
        out[nk] = v
    if meta is not None:
        out["__metadata__"] = meta
    blob = json.dumps(out, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        fi.seek(off)
        while True:
            buf = fi.read(64 << 20)
            if not buf:
                break
            fo.write(buf)
    return [k for k in out if k != "__metadata__"]


def merge(base, head, out, renames, copy_head):
    base, head, out = (os.path.abspath(p) for p in (base, head, out))
    cfg = json.load(open(os.path.join(base, "config.json")))
    if cfg.get("model_type") != "glm_moe_dsa":
        raise SystemExit(f"--base is model_type {cfg.get('model_type')!r}, expected glm_moe_dsa")
    n_main = int(cfg["num_hidden_layers"])
    os.makedirs(out, exist_ok=False)
    for name in sorted(os.listdir(base)):
        if name in ("config.json", "model.safetensors.index.json") or name.startswith("."):
            continue
        os.symlink(os.path.join(base, name), os.path.join(out, name))
    index_path = os.path.join(base, "model.safetensors.index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {"metadata": {}, "weight_map": {}}
    wm = index.setdefault("weight_map", {})
    if any(k.startswith(("mtp.", f"model.layers.{n_main}.")) for k in wm):
        raise SystemExit("--base already lists MTP head tensors")
    files = weight_files(head)
    added = 0
    for i, src in enumerate(files):
        name = f"model-mtp-{i + 1:05d}-of-{len(files):05d}.safetensors"
        dst = os.path.join(out, name)
        if renames:
            keys = rewrite(src, dst, renames)
        else:
            if copy_head:
                import shutil

                shutil.copyfile(src, dst)
            else:
                os.symlink(src, dst)
            keys = [k for k in read_header(src)[0] if k != "__metadata__"]
        ok = [k for k in keys if k.startswith(("mtp.", f"model.layers.{n_main}."))]
        if len(ok) != len(keys):
            bad = [k for k in keys if k not in ok][:5]
            raise SystemExit(f"{name}: {len(keys) - len(ok)} tensors are neither mtp.* nor "
                             f"model.layers.{n_main}.* (for example {bad}); use --rename OLD=NEW")
        for k in keys:
            wm[k] = name
        added += len(keys)
    cfg["num_nextn_predict_layers"] = max(1, int(cfg.get("num_nextn_predict_layers", 0) or 0))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    json.dump(index, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    print(f"merged: {out}\n  {added} head tensors in {len(files)} file(s); base files are symlinks; "
          f"num_nextn_predict_layers = {cfg['num_nextn_predict_layers']}")
    q = cfg.get("quantization") or {}
    print(f"  base quantization: bits={q.get('bits')} group_size={q.get('group_size')} mode={q.get('mode', 'affine')}; "
          f"a head quantized differently is handled by oMLX's loader from the tensor shapes")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inspect")
    a.add_argument("--head", required=True)
    b = sub.add_parser("merge")
    b.add_argument("--base", required=True)
    b.add_argument("--head", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--rename", action="append", default=[], help="OLD=NEW key prefix")
    b.add_argument("--copy-head", action="store_true", help="copy the head file instead of linking it")
    args = ap.parse_args()
    if args.cmd == "inspect":
        inspect(args.head)
    else:
        renames = [tuple(r.split("=", 1)) for r in args.rename]
        merge(args.base, args.head, args.out, renames, args.copy_head)


if __name__ == "__main__":
    sys.exit(main())
