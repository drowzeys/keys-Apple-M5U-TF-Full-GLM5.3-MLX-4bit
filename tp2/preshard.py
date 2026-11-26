"""Write the two rank folders for GLM-5.3 TP=2 from the MLX 4-bit pack, and copy rank 1 to the second Mac.

Input: a model folder holding mlx-community/GLM-5.3-4bit (revision 81ad9b73) and, for MTP, GLM-5.3's MTP layer
(inferencerlabs/GLM-5.3-MTP-MLX, merged as model.layers.78.* by oneshot-tp2.sh). Output: plain MLX safetensors
per rank (even split: 0.5/0.5 experts, 32/32 attention heads; ~198 GiB each with the MTP layer).

    python tp2/preshard.py --model MODEL_DIR --rank 1 --out RANK1_DIR [--mtp] [--to user@second-mac:RANK1_DIR]
    python tp2/preshard.py --model MODEL_DIR --rank 0 --out RANK0_DIR [--mtp]   # optional: rank 0 can also
                                                                                 # load the model folder directly
    python tp2/preshard.py --check RANK_DIR [--mtp]

With --to, every finished file is sent with rsync (over the Thunderbolt link when the address is the peer's
link-local address) and deleted here, so this Mac needs room for one 4 GiB file only.
Runs on the CPU (reads the checkpoint once, about 8 min for one rank over TB5).
"""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="the model folder (4-bit pack, plus the MTP layer for --mtp)")
    ap.add_argument("--rank", type=int, choices=[0, 1])
    ap.add_argument("--out", help="the rank folder to write")
    ap.add_argument("--mtp", action="store_true", help="include the MTP layer (needed for --mtp in the server)")
    ap.add_argument("--fractions", default="0.5,0.5", help="expert shares per rank (default even)")
    ap.add_argument("--head-fractions", default="0.5,0.5", help="head shares (32/32 keeps the fast prefill path)")
    ap.add_argument("--chunk-gb", type=float, default=4.0)
    ap.add_argument("--to", help="rsync destination host:folder; each file is moved there as soon as it is written")
    ap.add_argument("--check", help="load a rank folder strictly on the CPU and report")
    a = ap.parse_args()

    import mlx.core as mx

    from glm_tp import model as M

    if a.check:
        mx.set_default_device(mx.cpu)
        meta = json.loads((Path(a.check) / "tp_shard.json").read_text())
        model, _ = M.load_rank(a.check, M.StandInGroup(meta["size"], meta["rank"]), lazy=True, mtp=a.mtp)
        from mlx.utils import tree_flatten

        n = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
        print(f"check ok: rank {meta['rank']}/{meta['size']} (experts {meta.get('fractions')}, heads "
              f"{meta.get('head_fractions')}, MTP {bool(meta.get('mtp'))}) loads strictly, {n / 2**30:.1f} GiB")
        return
    if a.model is None or a.rank is None or a.out is None:
        ap.error("--model, --rank and --out are required")
    if a.to:
        host, _, folder = a.to.partition(":")
        os.environ["TP2_SINK"] = " ".join(["rsync", "-a", "--remove-source-files",
                                           "-e", "ssh -o BatchMode=yes"]) + f" __FILES__ {shlex.quote(a.to.rstrip('/') + '/')}"
        os.system(f"ssh -o BatchMode=yes {shlex.quote(host)} mkdir -p {shlex.quote(folder)}")
    fr = [float(v) for v in a.fractions.split(",")]
    hfr = [float(v) for v in a.head_fractions.split(",")]
    M.write(a.model, a.out, a.rank, 2, a.chunk_gb, None, fr, False, hfr)
    if a.mtp:
        if a.to:  # the main write moved everything; bring the three small metadata files back for the MTP pass
            for f in ("model.safetensors.index.json", "quant_map.json", "tp_shard.json"):
                os.system(f"rsync -a -e 'ssh -o BatchMode=yes' {shlex.quote(a.to.rstrip('/') + '/' + f)} "
                          f"{shlex.quote(str(Path(a.out) / f))}")
        M.write(a.model, a.out, a.rank, 2, a.chunk_gb, None, fr, True, hfr)


if __name__ == "__main__":
    main()
