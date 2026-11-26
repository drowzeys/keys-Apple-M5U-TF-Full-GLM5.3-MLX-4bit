#!/bin/bash
# build_bundle.sh VERSION OUTDIR: build the TP=2 release bundle (macOS arm64) from this repository checkout.
#
# Needs: the oMLX 0.8.0rc1 wheel at $TP2_OMLX_WHEEL, network access to PyPI and GitHub. CPython 3.11.14 comes from
# python-build-standalone release 20260211 (pinned SHA-256). Produces OUTDIR/keys-glm53-tp2-macos-arm64-
# VERSION.tar.gz and its .sha256. Every wheel is pinned (requirements.lock); the install is offline.
set -eu
VER="$1"; OUT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
NAME="keys-glm53-tp2-macos-arm64-${VER}"
WORK="$(mktemp -d -t tp2bundle)"; B="$WORK/$NAME"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.11.14%2B20260211-aarch64-apple-darwin-install_only.tar.gz"
PBS_SHA256="508e8cbf83f542cd94b77b6604ceef4bb008c4aefc13fb6fdfae9a7a80e36faf"
OMLX="${TP2_OMLX_WHEEL:?set TP2_OMLX_WHEEL to omlx-0.8.0rc1-cp311-cp311-macosx_15_0_universal2.whl}"
MLXLM_COMMIT="0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d"
mkdir -p "$B/wheels" "$B/tools" "$B/LICENSES/components" "$OUT"

echo "== Python"
curl -fsSL -o "$WORK/py.tgz" "$PBS_URL"
echo "$PBS_SHA256  $WORK/py.tgz" | shasum -a 256 -c -
tar -xzf "$WORK/py.tgz" -C "$B"          # -> $B/python
find "$B/python" -name "__pycache__" -prune -exec rm -rf {} +
"$B/python/bin/python3.11" -V

echo "== wheels"
"$B/python/bin/python3.11" -m venv "$WORK/v"
"$WORK/v/bin/python" -m pip -q install --upgrade "pip==26.2.1"
cp "$OMLX" "$B/wheels/"
"$WORK/v/bin/pip" -q wheel --no-deps -w "$B/wheels" "git+https://github.com/ml-explore/mlx-lm@${MLXLM_COMMIT}"
grep -v -e '^mlx-lm==' -e '^omlx==' "$HERE/requirements.lock" > "$WORK/pypi.txt"
# MLX's macOS 26 wheels (macosx_26_0_arm64) carry the RDMA-capable jaccl and the M5 kernels; the macOS 15 wheels
# of the same version cannot start jaccl over Thunderbolt ("Cannot initialize jaccl distributed backend")
"$WORK/v/bin/pip" -q download --only-binary=:all: --no-deps --platform macosx_26_0_arm64 --python-version 3.11 \
  --implementation cp -d "$B/wheels" mlx==0.32.4 mlx-metal==0.32.4
grep -v -e '^mlx==' -e '^mlx-metal==' "$WORK/pypi.txt" > "$WORK/pypi-rest.txt"
"$WORK/v/bin/pip" -q download --only-binary=:all: --no-deps --platform macosx_15_0_arm64 --python-version 3.11 \
  --implementation cp -d "$B/wheels" -r "$WORK/pypi-rest.txt"
ls "$B/wheels" | grep -E '^mlx(_metal)?-0\.32\.4-.*macosx_26_0_arm64\.whl$' | wc -l | grep -q 2 || { echo "MLX macOS 26 wheels missing"; exit 1; }
cp "$HERE/requirements.lock" "$B/requirements.lock"
# the lock must resolve offline from the wheelhouse alone
"$WORK/v/bin/pip" -q install --no-index --no-deps --find-links "$B/wheels" -r "$B/requirements.lock"
"$WORK/v/bin/python" -c "import mlx.core, mlx_lm, omlx"

echo "== code"
mkdir -p "$B/tp2"
cp -R "$HERE/glm_tp" "$B/tp2/"
cp "$HERE/server.py" "$HERE/preshard.py" "$HERE/smoke.py" "$B/tp2/"   # oneshot-tp2.sh is its own release asset
cp "$HERE/install.sh" "$HERE/RUN.txt" "$B/"
cp "$ROOT/glm53/tools/merge_mtp_head.py" "$B/tools/"
find "$B/tp2" -name "__pycache__" -prune -exec rm -rf {} +

echo "== licences"
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$B/"
cp "$HERE/THIRD_PARTY_NOTICES-tp2.md" "$B/THIRD_PARTY_NOTICES.md"
cp "$ROOT/LICENSES/Apache-2.0.txt" "$ROOT/LICENSES/MIT.txt" "$B/LICENSES/"
cp "$B/python/lib/python3.11/LICENSE.txt" "$B/LICENSES/components/CPython-3.11.14-LICENSE.txt"
"$WORK/v/bin/python" - "$B/wheels" "$B/LICENSES/components" <<'PY'
import sys, zipfile, pathlib, re
wheels, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for w in sorted(wheels.glob("*.whl")):
    z = zipfile.ZipFile(w)
    names = [n for n in z.namelist() if ".dist-info/" in n and re.search(r"(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", n.split("/")[-1], re.I)
             or re.search(r"\.dist-info/licenses/", n)]
    pkg = w.name.split("-")[0]
    for n in names:
        if n.endswith("/"):
            continue
        dst = out / pkg / n.split(".dist-info/")[-1].replace("licenses/", "")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(n))
    meta = z.read([n for n in z.namelist() if n.endswith(".dist-info/METADATA")][0]).decode("utf-8", "replace")
    lic = re.search(r"^License(?:-Expression)?: (.+)$", meta, re.M)
    print(f"{w.name}: {lic.group(1).strip() if lic else 'see classifiers'}; files: {len(names)}")
PY

echo "== privacy check"
# our own identifiers anywhere (text and inside wheels), and private paths, addresses or e-mails in our own files
bad=0
IDS="${TP2_PRIVATE_IDS:-}"      # extra words to refuse, '|'-separated (e.g. your user and host names)
PAT='/Users/(?!Shared\b|runner\b)[A-Za-z][A-Za-z0-9_.-]*|/home/[a-z][a-z0-9_-]*/|\b192\.168\.[0-9]+\.[0-9]+\b|\b169\.254\.[0-9]+\.[0-9]+\b|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}'
"$WORK/v/bin/python" - "$B" "$PAT" "$IDS" <<'PY' || bad=1
import sys, re, pathlib, zipfile
root, pat, ids = pathlib.Path(sys.argv[1]), re.compile(sys.argv[2]), sys.argv[3]
idre = re.compile(ids, re.I) if ids else None
ours = [root / "tp2", root / "tools", root / "RUN.txt", root / "install.sh", root / "THIRD_PARTY_NOTICES.md",
        root / "requirements.lock", root / "python" / "lib" / "python3.11" / "_sysconfigdata__darwin_darwin.py"]
bad = 0
for p in ours:
    for f in ([p] if p.is_file() else [x for x in p.rglob("*") if x.is_file()]):
        t = f.read_text("utf-8", "replace")
        for m in pat.finditer(t):
            s = m.group(0)
            if s.endswith(("@example.com",)) or s in ("git@github.com",):
                continue
            print("private-looking text in", f.relative_to(root), ":", s); bad = 1
if idre:
    for f in root.rglob("*"):
        if f.is_file() and f.suffix != ".whl":
            if idre.search(f.read_bytes().decode("latin-1")):
                print("identifier in", f.relative_to(root)); bad = 1
    for w in (root / "wheels").glob("*.whl"):
        z = zipfile.ZipFile(w)
        for n in z.namelist():
            if idre.search(z.read(n).decode("latin-1")):
                print("identifier in", w.name, n); bad = 1
sys.exit(bad)
PY
[ "$bad" -eq 0 ] || { echo "privacy check failed"; exit 1; }

echo "== tar"
xattr -cr "$B" 2>/dev/null || true
COPYFILE_DISABLE=1 tar --uid 0 --gid 0 --uname root --gname wheel -czf "$OUT/$NAME.tar.gz" -C "$WORK" "$NAME"
( cd "$OUT" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256" && cat "$NAME.tar.gz.sha256" )
ls -l "$OUT/$NAME.tar.gz"
rm -rf "$WORK"
