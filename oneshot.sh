#!/usr/bin/env bash
# One-shot setup and start for full GLM-5.3 (MLX 4-bit) on TensorFold + glm_ax, Apple M5-class Mac.
#
#   ./oneshot.sh --preset prose            # check the Mac, fetch binary + weights, start, smoke-test
#   ./oneshot.sh --preset code --dir /Volumes/big/glm53-tf
#   ./oneshot.sh --preset 1m --dry-run     # print every step and command, change nothing
#   ./oneshot.sh --status | --stop
#
# Presets (numbers measured on an Apple M5 Ultra, 512 GB; see README.md):
#   prose  MTP depth 1, no confidence cut, 32k context
#   code   built-in MTP default (depth 2 + 0.7 confidence cut), 32k context
#   long   bf16 cache, MTP depth 1, 393,216-token context
#   1m     8-bit cache, MTP depth 1, 1,048,576-token context
#   1m-nomtp  8-bit cache, no MTP head, 1,048,576-token context
#
# Nothing here uses sudo except the optional wired-limit step, which is only run after you answer yes.
# Bash 3.2 compatible (the macOS system bash).

set -euo pipefail

REPO="drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit"
PACK_REPO="mlx-community/GLM-5.3-4bit"
PACK_REV="81ad9b73c6188f808737dbb52458f082d249dcbe"
HEAD_REPO="inferencerlabs/GLM-5.3-MTP-MLX"
HEAD_REV="f1187efee8cf60319a7d782c3488cfd96017c5ce"
MIN_MACOS="27.2.1"
MIN_MEM_GB=512
PACK_DISK_GB=420
HEAD_DISK_GB=6
WIRED_MB=500000
PORT=8020
HOST=127.0.0.1

PRESET=""
DIR="${HOME}/glm53-tf"
DEFAULT_RELEASE="v0.3.1-glm53-reuse"
DEFAULT_SHA="3f6a3aea32ad6cca81ea12229b4ebdd06d5769691b7a98afaf77056cb60d0624"
RELEASE="$DEFAULT_RELEASE"
DRY=0
YES=0
FORCE=0
ACTION="run"

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --preset NAME     prose | code | long | 1m | 1m-nomtp   (required for a run)
  --dir PATH        where the binary, logs, merged model folder and (if needed) the weights go
                    (default ~/glm53-tf)
  --release TAG     release tag to fetch the binary from (default: v0.3.1-glm53-reuse, SHA-256 pinned here;
                    "latest" takes the newest release and checks its published .sha256 only)
  --port N          server port (default 8020)
  --dry-run         print every step and command; download, write and start nothing
  --yes             answer yes to the wired-limit question (still runs sudo, which asks for your password)
  --force           continue when a hardware check fails (not recommended)
  --status          show whether the server started by this script is running
  --stop            stop that server
  -h, --help        this text
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --preset) PRESET="${2:-}"; shift 2 ;;
    --dir) DIR="${2:-}"; shift 2 ;;
    --release) RELEASE="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --yes) YES=1; shift ;;
    --force) FORCE=1; shift ;;
    --status) ACTION="status"; shift ;;
    --stop) ACTION="stop"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

case "$DIR" in "~"*) DIR="${HOME}${DIR#\~}" ;; esac
PIDFILE="${DIR}/server.pid"
LOGDIR="${DIR}/logs"

say() { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '   WARNING: %s\n' "$*" >&2; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
# run CMD...: print it; run it unless --dry-run
run() {
  printf '   $ %s\n' "$*"
  if [ "$DRY" -eq 0 ]; then "$@"; fi
}

# ---------------------------------------------------------------------------------------------------------------
# presets: env, flags, model folder kind (mtp | plain), and what was measured
preset_setup() {
  case "$PRESET" in
    prose)
      ENVS="GLM_AX_LOAD_LIMIT_GB=440 GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off"
      FLAGS="--context 32768 --parallel 1 --reasoning-effort low --no-update-check"
      MODEL_KIND=mtp
      NEED="about 420 GB free unified memory (peak wired about 415 GiB at 78k; less at 32k)"
      MEASURED="v0.3.1: prose 36.2 tok/s (T=0.6), greedy 37.5, code 39.3, edit 40.9"
      ;;
    code)
      ENVS="GLM_AX_LOAD_LIMIT_GB=440"
      FLAGS="--context 32768 --parallel 1 --reasoning-effort low --no-update-check"
      MODEL_KIND=mtp
      NEED="about 420 GB free unified memory (peak wired about 415 GiB at 78k; less at 32k)"
      MEASURED="v0.3.1: code 41.8 tok/s, edit 42.7, prose 36.2 (T=0.6), greedy 36.8"
      ;;
    long)
      ENVS="GLM_AX_LOAD_LIMIT_GB=488 GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off"
      FLAGS="--context 393216 --parallel 1 --reasoning-effort low --prompt-cache-gib 0 --no-update-check"
      MODEL_KIND=mtp
      NEED="about 455 GB of the machine: peak wired 442.6 GiB at a 375k-token prompt; nothing else large may run"
      MEASURED="375,314-token needle found; first token 784 s (about 479 tok/s); decode 20.3 tok/s with MTP depth 1"
      ;;
    1m)
      ENVS="GLM_AX_K8=1 GLM_AX_PROMPT_ROWS=1024 GLM_AX_LOAD_LIMIT_GB=495 GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off"
      FLAGS="--context 1048576 --parallel 1 --reasoning-effort low --prompt-cache-gib 0 --no-update-check"
      MODEL_KIND=mtp
      NEED="peak wired 453.4 GiB at a 1,026,813-token prompt; headroom is thin (lowest reclaimable 43-54 GiB across runs): close every other app"
      MEASURED="1,026,813-token needle found; first token 3,802 s (63 min); decode 9.6 tok/s with MTP depth 1"
      ;;
    1m-nomtp)
      ENVS="GLM_AX_K8=1 GLM_AX_PROMPT_ROWS=2048 GLM_AX_LOAD_LIMIT_GB=495 GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off"
      FLAGS="--context 1048576 --parallel 1 --reasoning-effort low --prompt-cache-gib 0 --no-drafts --no-update-check"
      MODEL_KIND=plain
      NEED="peak wired 449.1 GiB at a 1,026,813-token prompt (lowest reclaimable about 50 GiB); nothing else large may run"
      MEASURED="1,026,813-token needle found; first token 3,495 s (58 min, about 294 tok/s); decode 7.4 tok/s (no MTP head)"
      ;;
    "") die "--preset is required: prose | code | long | 1m | 1m-nomtp" ;;
    *) die "unknown preset '$PRESET': prose | code | long | 1m | 1m-nomtp" ;;
  esac
}

# ---------------------------------------------------------------------------------------------------------------
version_ge() { # version_ge A B: A >= B (dotted numbers)
  local a b i x y
  IFS=. read -r -a a <<<"$1"
  IFS=. read -r -a b <<<"$2"
  for i in 0 1 2; do
    x="${a[$i]:-0}"; y="${b[$i]:-0}"
    if [ "$x" -gt "$y" ]; then return 0; fi
    if [ "$x" -lt "$y" ]; then return 1; fi
  done
  return 0
}

check_fail() { if [ "$FORCE" -eq 1 ]; then warn "$1 (continuing: --force)"; else die "$1 (use --force to continue anyway)"; fi; }

check_mac() {
  say "1. Checking this Mac"
  if [ "$(uname -s)" != "Darwin" ]; then
    if [ "$DRY" -eq 1 ]; then warn "not macOS: the checks below are skipped in --dry-run"; return; fi
    die "macOS only (Metal); this is $(uname -s)"
  fi
  [ "$(uname -m)" = "arm64" ] || check_fail "Apple Silicon (arm64) required"
  local chip os mem_gb
  chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  os="$(sw_vers -productVersion)"
  mem_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
  info "chip: $chip; macOS $os; unified memory ${mem_gb} GB"
  case "$chip" in *"Apple M5"*) ;; *) check_fail "an M5-generation GPU (tensor units) is required; found '$chip'" ;; esac
  version_ge "$os" "$MIN_MACOS" || check_fail "macOS $MIN_MACOS or later is required (the binary's minimum); found $os"
  [ "$mem_gb" -ge "$MIN_MEM_GB" ] || check_fail "${MIN_MEM_GB} GB of unified memory is required for the full model; found ${mem_gb} GB"
  info "memory need for preset '$PRESET': $NEED"
  command -v curl >/dev/null || die "curl not found"
  command -v shasum >/dev/null || die "shasum not found"
  command -v python3 >/dev/null || die "python3 not found (install the Xcode Command Line Tools: xcode-select --install)"
}

# ---------------------------------------------------------------------------------------------------------------
fetch_binary() {
  say "2. Prebuilt binary (release: $RELEASE)"
  local api url sha name
  if [ "$RELEASE" = "latest" ]; then api="https://api.github.com/repos/$REPO/releases/latest"
  else api="https://api.github.com/repos/$REPO/releases/tags/$RELEASE"; fi
  info "release information: $api"
  # reading the release information is the one network call --dry-run makes (a few KB)
  url="$(curl -fsSL "$api" 2>/dev/null | grep -o '"browser_download_url": *"[^"]*macos-arm64\.tar\.gz"' | head -1 | sed 's/.*"\(https[^"]*\)"/\1/' || true)"
  if [ -z "$url" ]; then
    [ "$DRY" -eq 1 ] || die "no macos-arm64 .tar.gz asset found in $api"
    url="https://github.com/$REPO/releases/download/<tag>/<asset>.tar.gz"
  fi
  info "asset: $url"
  name="$(basename "$url" .tar.gz)"
  BIN="${DIR}/${name}/tensorfold-native"
  if [ "$DRY" -eq 0 ] && [ -x "$BIN" ]; then info "already unpacked: $BIN"; return; fi
  run mkdir -p "$DIR"
  run curl -fL --retry 3 -C - -o "${DIR}/${name}.tar.gz" "$url"
  run curl -fsSL -o "${DIR}/${name}.tar.gz.sha256" "${url}.sha256"
  info "verifying SHA-256"
  if [ "$DRY" -eq 0 ]; then
    sha="$(awk '{print $1}' "${DIR}/${name}.tar.gz.sha256")"
    if [ "$RELEASE" = "$DEFAULT_RELEASE" ] && [ "$sha" != "$DEFAULT_SHA" ]; then
      die "the release's .sha256 ($sha) is not the one pinned in this script ($DEFAULT_SHA)"
    fi
    echo "${sha}  ${DIR}/${name}.tar.gz" | shasum -a 256 -c - || die "SHA-256 mismatch for ${name}.tar.gz"
  else
    printf '   $ shasum -a 256 -c  (expected hash from %s.tar.gz.sha256)\n' "$name"
  fi
  run tar -xzf "${DIR}/${name}.tar.gz" -C "$DIR"
  info "the binary is ad-hoc signed, not notarized: clearing the download quarantine flag"
  run xattr -dr com.apple.quarantine "${DIR}/${name}"
}

# ---------------------------------------------------------------------------------------------------------------
hf_home() { echo "${HF_HOME:-${HOME}/.cache/huggingface}"; }

# snapshot_complete DIR: every file the index lists is present (following symlinks)
snapshot_complete() {
  [ -f "$1/model.safetensors.index.json" ] || [ -f "$1/config.json" ] || return 1
  python3 -I - "$1" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    files = set(json.load(open(idx))["weight_map"].values())
else:
    files = {f for f in os.listdir(d) if f.endswith(".safetensors")}
    if not files:
        sys.exit(1)
missing = [f for f in files if not os.path.isfile(os.path.join(d, f))]
sys.exit(1 if missing else 0)
PY
}

# find_snapshot REPO REV: print the cached snapshot folder if complete, in the default HF cache or under --dir
find_snapshot() {
  local sub base
  sub="models--${1//\//--}/snapshots/$2"
  for base in "$(hf_home)/hub" "${DIR}/hf/hub"; do
    if [ -d "${base}/${sub}" ] && snapshot_complete "${base}/${sub}"; then echo "${base}/${sub}"; return 0; fi
  done
  return 1
}

# hf_cli: print the command that runs the Hugging Face CLI (it may sit outside PATH, e.g. after pip install --user)
hf_cli() {
  local d ub
  if command -v hf >/dev/null; then echo hf; return; fi
  if command -v huggingface-cli >/dev/null; then echo huggingface-cli; return; fi
  ub="$(python3 -m site --user-base 2>/dev/null || true)"
  for d in "${HOME}/.local/bin" "${ub:+${ub}/bin}"; do
    [ -n "$d" ] || continue
    if [ -x "${d}/hf" ]; then echo "${d}/hf"; return; fi
    if [ -x "${d}/huggingface-cli" ]; then echo "${d}/huggingface-cli"; return; fi
  done
  if python3 -c 'import huggingface_hub.cli.hf' >/dev/null 2>&1; then
    echo "python3 -m huggingface_hub.cli.hf"; return
  fi
  if python3 -c 'import huggingface_hub.commands.huggingface_cli' >/dev/null 2>&1; then
    echo "python3 -m huggingface_hub.commands.huggingface_cli"; return
  fi
  echo ""
}

# need_hf_cli: find the CLI, or offer to install huggingface_hub for this user (no sudo)
need_hf_cli() {
  local cli ans="n"
  cli="$(hf_cli)"
  if [ -n "$cli" ]; then echo "$cli"; return; fi
  if [ "$DRY" -eq 1 ]; then
    printf '   $ python3 -m pip install --user -U huggingface_hub   (offered when no hf CLI is found)\n' >&2
    echo hf; return
  fi
  printf '   No Hugging Face CLI found (hf, huggingface-cli, ~/.local/bin, the python user bin).\n' >&2
  if [ "$YES" -eq 1 ]; then ans="y"; elif [ -t 0 ]; then read -r -p "   Install it now with 'python3 -m pip install --user -U huggingface_hub'? [y/N] " ans; fi
  case "$ans" in
    y|Y|yes) python3 -m pip install --user -U huggingface_hub >&2 ;;
    *) die "install it with: python3 -m pip install --user -U huggingface_hub" ;;
  esac
  cli="$(hf_cli)"
  [ -n "$cli" ] || die "huggingface_hub installed but no CLI found; add $(python3 -m site --user-base)/bin to PATH"
  echo "$cli"
}

# get_snapshot REPO REV DISK_GB: echo the snapshot folder, downloading it (resumable) when it is not cached
get_snapshot() {
  local repo="$1" rev="$2" gb="$3" found cli free
  if found="$(find_snapshot "$repo" "$rev")"; then info "found in the Hugging Face cache: $found" >&2; echo "$found"; return; fi
  cli="$(need_hf_cli)"
  if [ "$DRY" -eq 0 ]; then
    mkdir -p "$DIR"
    free=$(df -g "$DIR" | awk 'NR==2{print $4}')
    [ "$free" -ge "$gb" ] || check_fail "about ${gb} GB free disk needed under $DIR for $repo; ${free} GB free"
  fi
  info "downloading $repo at $rev into ${DIR}/hf (resumable: rerun to continue)" >&2
  printf '   $ HF_HOME=%s %s download %s --revision %s\n' "${DIR}/hf" "$cli" "$repo" "$rev" >&2
  # shellcheck disable=SC2086 # cli may be "python3 -m ..." (a word list)
  if [ "$DRY" -eq 0 ]; then HF_HOME="${DIR}/hf" $cli download "$repo" --revision "$rev" >&2; fi
  echo "${DIR}/hf/hub/models--${repo//\//--}/snapshots/${rev}"
}

merge_script() {
  local here
  here="$(cd "$(dirname "$0")" && pwd)"
  if [ -f "${here}/glm53/tools/merge_mtp_head.py" ]; then echo "${here}/glm53/tools/merge_mtp_head.py"; return; fi
  local tag="$RELEASE"; [ "$tag" = "latest" ] && tag=main
  run curl -fsSL -o "${DIR}/merge_mtp_head.py" "https://raw.githubusercontent.com/${REPO}/${tag}/glm53/tools/merge_mtp_head.py" >&2
  echo "${DIR}/merge_mtp_head.py"
}

fetch_weights() {
  say "3. Weights"
  PACK="$(get_snapshot "$PACK_REPO" "$PACK_REV" "$PACK_DISK_GB")"
  if [ "$MODEL_KIND" = plain ]; then MODEL="$PACK"; info "model folder: $MODEL (no MTP head)"; return; fi
  HEADDIR="$(get_snapshot "$HEAD_REPO" "$HEAD_REV" "$HEAD_DISK_GB")"
  MODEL="${DIR}/GLM-5.3-4bit-MTP"
  if [ "$DRY" -eq 0 ] && [ -f "${MODEL}/config.json" ] && grep -q '"num_nextn_predict_layers": 1' "${MODEL}/config.json"; then
    info "merged MTP folder already there: $MODEL"; return
  fi
  local ms
  ms="$(merge_script)"
  info "building the merged MTP folder (symlinks to the pack, no second copy of the weights)"
  run python3 -I "$ms" merge --base "$PACK" --head "$HEADDIR" --out "$MODEL"
}

# ---------------------------------------------------------------------------------------------------------------
wired_limit() {
  say "4. GPU wired-memory limit"
  local cur
  cur="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
  info "current iogpu.wired_limit_mb = $cur (0 means the macOS default, far too low for this model)"
  if [ "$cur" -ge "$WIRED_MB" ]; then info "high enough"; return; fi
  cat <<EOF
   The model wires about 410-450 GiB of GPU memory. macOS must be told to allow that:

       sudo sysctl iogpu.wired_limit_mb=${WIRED_MB}

   Risk: this lets the GPU wire almost all memory. If other large programs run at the same time, macOS can stall
   or reboot. Close them first; a memory watchdog is recommended. The setting resets at reboot. Your own risk.
EOF
  if [ "$DRY" -eq 1 ]; then printf '   $ sudo sysctl iogpu.wired_limit_mb=%s   (only after you answer yes)\n' "$WIRED_MB"; return; fi
  local ans="n"
  if [ "$YES" -eq 1 ]; then ans="y"; elif [ -t 0 ]; then read -r -p "   Run it now with sudo? [y/N] " ans; fi
  case "$ans" in
    y|Y|yes) run sudo sysctl "iogpu.wired_limit_mb=${WIRED_MB}" ;;
    *) check_fail "wired limit not raised; the model will not load. Run the sudo command above, then rerun" ;;
  esac
}

# ---------------------------------------------------------------------------------------------------------------
server_alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

start_server() {
  say "5. Starting the server (preset '$PRESET')"
  info "measured: $MEASURED"
  if [ "$DRY" -eq 0 ] && server_alive; then die "a server started by this script is already running (pid $(cat "$PIDFILE")): ./oneshot.sh --stop first"; fi
  local log="${LOGDIR}/server-${PRESET}.log"
  run mkdir -p "$LOGDIR"
  # shellcheck disable=SC2086 # ENVS and FLAGS are word lists on purpose
  printf '   $ env %s %s serve %s --name glm53 --host %s --port %s %s > %s 2>&1 &\n' "$ENVS" "$BIN" "$MODEL" "$HOST" "$PORT" "$FLAGS" "$log"
  if [ "$DRY" -eq 1 ]; then return; fi
  # shellcheck disable=SC2086
  nohup env $ENVS "$BIN" serve "$MODEL" --name glm53 --host "$HOST" --port "$PORT" $FLAGS > "$log" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  info "pid $(cat "$PIDFILE"), log $log; loading takes about 45 s from a warm file cache, longer from disk"
  local n=0
  until curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null; do
    server_alive || { tail -5 "$log" >&2; rm -f "$PIDFILE"; die "the server exited during load (log above: $log)"; }
    sleep 5; n=$((n + 1))
    [ "$n" -le 360 ] || die "no answer after 30 minutes; see $log"
  done
  info "up after about $((n * 5)) s"
}

smoke() {
  say "6. Smoke test"
  local body='{"model":"glm53","messages":[{"role":"user","content":"What is the capital city of Australia? Reply with the city name only."}],"max_tokens":40,"temperature":0}'
  printf "   $ curl -s http://%s:%s/v1/chat/completions -d '%s'\n" "$HOST" "$PORT" "$body"
  if [ "$DRY" -eq 1 ]; then return; fi
  local out
  out="$(curl -s "http://${HOST}:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")"
  if printf '%s' "$out" | grep -q Canberra; then
    info "answer contains 'Canberra': OK"
    cat <<EOF

Ready: OpenAI-compatible API at http://${HOST}:${PORT}/v1, model name "glm53".
Stop it with: $0 --stop --dir "$DIR"
EOF
  else
    die "smoke test failed; reply: $(printf '%s' "$out" | cut -c1-300)"
  fi
}

status() {
  if server_alive; then
    info "running: pid $(cat "$PIDFILE")"
    if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null; then info "API answering at http://${HOST}:${PORT}/v1"
    else info "API not answering yet (loading?)"; fi
  else
    info "not running (no live pid in $PIDFILE)"
  fi
  if command -v vm_stat >/dev/null; then
    vm_stat | awk '/Pages wired down/{w=$4} /Pages free/{f=$3} END{g=16384/1073741824; printf "   wired %.1f GiB, free %.1f GiB\n", w*g, f*g}'
  fi
}

stop() {
  if server_alive; then
    local p; p="$(cat "$PIDFILE")"
    run kill "$p"
    if [ "$DRY" -eq 0 ]; then
      local n=0
      while kill -0 "$p" 2>/dev/null && [ "$n" -lt 60 ]; do sleep 1; n=$((n + 1)); done
      kill -0 "$p" 2>/dev/null && run kill -9 "$p"
      rm -f "$PIDFILE"
    fi
    info "stopped"
  else
    info "not running"; rm -f "$PIDFILE"
  fi
}

case "$ACTION" in
  status) status; exit 0 ;;
  stop) stop; exit 0 ;;
esac

preset_setup
[ "$DRY" -eq 1 ] && say "DRY RUN: nothing is downloaded, written or started"
info "working folder: $DIR"
check_mac
fetch_binary
fetch_weights
wired_limit
start_server
smoke
