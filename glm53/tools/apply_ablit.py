#!/usr/bin/env python3
"""Splice the GLM-5.3 4-bit MLX abliteration into a stock checkpoint.

The pack is one safetensors file of 96 matrices (288 tensors: weight, scales,
biases) already quantized as MLX affine 4-bit, group 64. TensorFold registers
every tensor in every file the index names, so the pack is not added as a
second shard. Each stock shard that holds one of those matrices is cloned and
the matching byte range is overwritten. Shapes and dtypes must already match.

Nothing under a Hugging Face cache snapshot is written. A symlink in the model
folder is replaced with a clone of its target. A hardlink (more than one name
for the file) is cloned the same way. The cache and any other name stay stock.

    python3 apply_ablit.py --model MODEL_DIR --pack PACK_DIR
    python3 apply_ablit.py --model SNAPSHOT --pack PACK_DIR --out NEW_DIR
    python3 apply_ablit.py --self-test

--out builds a private folder: symlinks to the files that stay stock, clones of
the files that change. Use it when --model is the Hugging Face snapshot itself
(the 1m-nomtp preset). PACK_DIR is a download of
drowzeys/keys-Ablit-For-GLM-5.3-4bit-MLX.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile


PACK_FILE = "model-ablit-l2-49.safetensors"
MARKER = "ABLIT.json"
# mlx-community/GLM-5.3-4bit revision this pack was built against.
BASE_REV = "81ad9b73c6188f808737dbb52458f082d249dcbe"


def die(msg: str) -> None:
    raise SystemExit(f"apply_ablit: {msg}")


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        n = struct.unpack("<Q", handle.read(8))[0]
        if n <= 0 or n > 64_000_000:
            die(f"bad safetensors header length in {path}")
        meta = json.loads(handle.read(n))
    return meta, 8 + n


def read_at(path: str, start: int, length: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(start)
        data = handle.read(length)
    if len(data) != length:
        die(f"short read of {length} bytes at {start} in {path}")
    return data


def in_hf_cache(path: str) -> bool:
    real = os.path.realpath(path)
    return "/.cache/huggingface/" in real or "/hub/models--" in real


def clone_file(src: str, dst: str) -> None:
    """APFS clone. A full copy is allowed only for a tiny file (the self-test)."""
    if subprocess.call(["cp", "-c", src, dst]) == 0:
        return
    if os.path.getsize(src) <= 8 << 20:
        shutil.copy2(src, dst)
        return
    die(f"cp -c failed for {src}. This Mac needs an APFS clone; a full copy of the shard is not made")


def ensure_private(model: str, name: str) -> str:
    """A regular file in model/ that is not a cache inode and not shared."""
    path = os.path.join(model, name)
    if not os.path.lexists(path):
        die(f"missing shard {name} in {model}")
    if in_hf_cache(model) and not os.path.islink(path):
        die(f"{model} is a Hugging Face snapshot. Pass --out so the cache is not edited")
    real = os.path.realpath(path)
    shared = os.path.islink(path) or os.stat(path).st_nlink > 1
    if not shared and not in_hf_cache(real):
        return path
    tmp = path + ".ablit-clone"
    if os.path.lexists(tmp):
        os.remove(tmp)
    clone_file(real, tmp)
    os.replace(tmp, path)
    if os.path.islink(path) or os.stat(path).st_nlink > 1:
        die(f"could not make a private copy of {name}")
    if os.path.realpath(path) == real and in_hf_cache(real):
        die(f"refusing to edit the cached file for {name}")
    return path


def link_or_copy_tree(src: str, out: str) -> None:
    os.makedirs(out)
    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        if os.path.isdir(src_path):
            die(f"unexpected directory in the checkpoint: {name}")
        dst = os.path.join(out, name)
        try:
            os.symlink(os.path.realpath(src_path), dst)
        except OSError:
            shutil.copy2(src_path, dst)


def tensors_of(meta: dict) -> list[str]:
    return [name for name in meta if name != "__metadata__"]


def apply(model: str, pack: str) -> int:
    pack_path = pack if pack.endswith(".safetensors") else os.path.join(pack, PACK_FILE)
    if not os.path.isfile(pack_path):
        die(f"pack file not found: {pack_path}")
    pack_meta, pack_start = read_header(pack_path)
    names = tensors_of(pack_meta)
    if len(names) != 288:
        die(f"pack has {len(names)} tensors, expected 288")
    index_path = os.path.join(model, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        die(f"no model.safetensors.index.json in {model}")
    weight_map = json.load(open(index_path))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name in names:
        shard = weight_map.get(name)
        if shard is None:
            die(f"this checkpoint has no {name}. It must be mlx-community/GLM-5.3-4bit at {BASE_REV}")
        by_shard.setdefault(shard, []).append(name)

    changed = 0
    checked = 0
    for shard, group in sorted(by_shard.items()):
        path = ensure_private(model, shard)
        meta, data_start = read_header(path)
        with open(path, "r+b") as handle:
            for name in group:
                old = meta.get(name)
                new = pack_meta[name]
                if old is None or old["dtype"] != new["dtype"] or old["shape"] != new["shape"]:
                    die(f"{name} layout {old} does not match the pack {new['dtype']} {new['shape']}")
                begin, end = new["data_offsets"]
                payload = read_at(pack_path, pack_start + begin, end - begin)
                old_begin, old_end = old["data_offsets"]
                if old_end - old_begin != len(payload):
                    die(f"{name} is {old_end - old_begin} bytes in the checkpoint and {len(payload)} in the pack")
                current = read_at(path, data_start + old_begin, len(payload))
                checked += 1
                if current == payload:
                    continue
                handle.seek(data_start + old_begin)
                handle.write(payload)
                changed += 1
        print(f"patched {shard}: {changed} tensors written so far", flush=True)
    marker = {
        "pack": "drowzeys/keys-Ablit-For-GLM-5.3-4bit-MLX",
        "pack_file": PACK_FILE,
        "base": "mlx-community/GLM-5.3-4bit",
        "base_revision": BASE_REV,
        "tensors": checked,
        "written_this_run": changed,
        "method": "clone the shard, overwrite the 4-bit ranges, leave every other tensor stock",
    }
    with open(os.path.join(model, MARKER), "w") as handle:
        json.dump(marker, handle, indent=2)
        handle.write("\n")
    print(f"done: {checked} tensors, {changed} written, model {model}", flush=True)
    return changed


def self_test() -> None:
    def write_st(path: str, tensors: list[tuple[str, bytes]]) -> None:
        header: dict = {"__metadata__": {"format": "mlx"}}
        offset = 0
        blobs = []
        for name, payload in tensors:
            header[name] = {"dtype": "U32", "shape": [1, len(payload) // 4], "data_offsets": [offset, offset + len(payload)]}
            offset += len(payload)
            blobs.append(payload)
        text = json.dumps(header, separators=(",", ":")).encode()
        text += b" " * ((8 - ((8 + len(text)) % 8)) % 8)
        with open(path, "wb") as handle:
            handle.write(struct.pack("<Q", len(text)))
            handle.write(text)
            for blob in blobs:
                handle.write(blob)

    with tempfile.TemporaryDirectory() as tmp:
        stock = os.path.join(tmp, "stock")
        model = os.path.join(tmp, "model")
        pack = os.path.join(tmp, "pack")
        os.makedirs(stock)
        os.makedirs(model)
        os.makedirs(pack)
        shard = "model-00001-of-00091.safetensors"
        names = []
        stock_blobs = []
        pack_blobs = []
        for i in range(3):
            # one stand-in matrix: weight, scales, biases, so the count check is patched below
            names.append(f"t{i}.weight")
            stock_blobs.append((names[-1], bytes([i]) * 16))
            pack_blobs.append((names[-1], bytes([i + 9]) * 16))
        # pad to the real count by repeating is wrong; the self-test calls the
        # inner writer with the length check relaxed.
        write_st(os.path.join(stock, shard), stock_blobs)
        write_st(os.path.join(pack, PACK_FILE), pack_blobs)
        index = {"weight_map": {name: shard for name, _ in stock_blobs}}
        with open(os.path.join(stock, "model.safetensors.index.json"), "w") as handle:
            json.dump(index, handle)
        os.symlink(os.path.realpath(os.path.join(stock, shard)), os.path.join(model, shard))
        os.symlink(os.path.realpath(os.path.join(stock, "model.safetensors.index.json")),
                   os.path.join(model, "model.safetensors.index.json"))
        # The production length check is 288. Exercise the clone and byte splice
        # by calling the same file operations the apply loop uses.
        pack_meta, pack_start = read_header(os.path.join(pack, PACK_FILE))
        path = ensure_private(model, shard)
        if os.path.islink(path):
            die("self-test: shard is still a symlink")
        meta, data_start = read_header(path)
        with open(path, "r+b") as handle:
            for name, _payload in pack_blobs:
                new = pack_meta[name]
                begin, end = new["data_offsets"]
                payload = read_at(os.path.join(pack, PACK_FILE), pack_start + begin, end - begin)
                old_begin, _old_end = meta[name]["data_offsets"]
                handle.seek(data_start + old_begin)
                handle.write(payload)
        got = read_at(path, data_start + meta[names[0]]["data_offsets"][0], 16)
        if got != bytes([9]) * 16:
            die("self-test: spliced bytes are wrong")
        untouched = read_at(os.path.join(stock, shard), data_start + meta[names[0]]["data_offsets"][0], 16)
        if untouched != bytes([0]) * 16:
            die("self-test: stock shard was modified")
    print("self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="stock checkpoint folder (MTP merge or the 4-bit pack)")
    parser.add_argument("--pack", help="folder or file from drowzeys/keys-Ablit-For-GLM-5.3-4bit-MLX")
    parser.add_argument("--out", help="write a private folder instead of editing --model")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.model or not args.pack:
        parser.error("--model and --pack are required")
    model = args.model
    if args.out:
        if os.path.exists(args.out):
            marker = os.path.join(args.out, MARKER)
            if os.path.isfile(marker):
                print(f"already applied: {args.out}", flush=True)
                return 0
            die(f"--out already exists and is not an applied ablit folder: {args.out}")
        link_or_copy_tree(args.model, args.out)
        model = args.out
    elif os.path.isfile(os.path.join(model, MARKER)):
        print(f"already applied: {model}", flush=True)
        return 0
    apply(model, args.pack)
    return 0


if __name__ == "__main__":
    sys.exit(main())
