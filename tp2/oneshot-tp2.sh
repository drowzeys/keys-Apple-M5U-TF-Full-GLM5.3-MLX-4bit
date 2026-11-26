#!/bin/bash
# oneshot-tp2.sh: GLM-5.3 on two 256 GB Apple M5 Ultra Macs, tensor-parallel (TP=2) over Thunderbolt 5 RDMA.
#
# Run it on the first Mac (rank 0, serves HTTP); it drives the second Mac (rank 1) over ssh.
#
#   ./oneshot-tp2.sh --peer USER@SECOND_MAC --dry-run     # print every step and command, change nothing
#   ./oneshot-tp2.sh --peer USER@SECOND_MAC               # check, install, download, pre-shard, start, smoke test
#   ./oneshot-tp2.sh --peer USER@SECOND_MAC --status      # / --stop
#
# Options:
#   --dir DIR        install and data folder, the same absolute path on both Macs (default /Users/Shared/glm53-tp2)
#   --bundle FILE    use a local release bundle instead of downloading it
#   --release TAG    release to download (default: the pinned one below)
#   --port N         HTTP port on the first Mac (default 8000; bound to 127.0.0.1 unless --host is given)
#   --host ADDR      HTTP bind address (default 127.0.0.1)
#   --no-mtp         plain decoding only (no MTP layer download, no MTP slice)
#   --kv8            8-bit latent KV cache (57.7 instead of 95.2 KB per token: more context and lanes)
#   --max-context N  (default 131072)
#   --yes            do not ask before starting the server
#
# What it does (no sudo anywhere):
#   1. checks both Macs: macOS 26.2 or later, Apple GPU, at least 256 GB of memory, RDMA enabled (rdma_ctl), ssh to the second
#      Mac, free disk;
#   2. finds the Thunderbolt 5 RDMA links (ibv_devices, link-local addresses, neighbour table) and writes the
#      jaccl hostfile with every link;
#   3. installs the release bundle (Python 3.11.14 + exact pinned wheels, offline) on both Macs, copying it to
#      the second Mac over Thunderbolt;
#   4. downloads the pinned weights on the first Mac (Hugging Face), merges the MTP layer (symlinks, renamed keys);
#   5. writes rank 1's folder and moves it to the second Mac file by file over Thunderbolt (~198 GiB);
#   6. starts rank 1 and rank 0 with mlx.launch (MLX_METAL_FAST_SYNCH=1) and waits for a "Canberra" answer.
set -u

RELEASE="tp2-v0.1.0"
REPO="drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit"
BUNDLE_NAME="keys-glm53-tp2-macos-arm64-0.1.0.tar.gz"
BUNDLE_SHA256="89dca1cfc73dd930d10dbdb0b1cf6a8a0478e69f29c25232084e5a15f9b374ba"
PACK_REPO="mlx-community/GLM-5.3-4bit"
PACK_REV="81ad9b73c6188f808737dbb52458f082d249dcbe"
HEAD_REPO="inferencerlabs/GLM-5.3-MTP-MLX"
HEAD_REV="f1187efee8cf60319a7d782c3488cfd96017c5ce"
MIN_MACOS="26.2"   # Thunderbolt RDMA (rdma_ctl, jaccl) and the macOS 26 MLX wheels in the bundle
MIN_RAM_GB=256

PEER=""; DIR="/Users/Shared/glm53-tp2"; BUNDLE=""; PORT=8000; HOST="127.0.0.1"; MTP=1; KV8=0; MAXCTX=131072
DRY=0; YES=0; ACTION=run
while [ $# -gt 0 ]; do
  case "$1" in
    --peer) PEER="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    --bundle) BUNDLE="$2"; shift ;;
    --release) RELEASE="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --host) HOST="$2"; shift ;;
    --no-mtp) MTP=0 ;;
    --kv8) KV8=1 ;;
    --max-context) MAXCTX="$2"; shift ;;
    --dry-run) DRY=1 ;;
    --yes|-y) YES=1 ;;
    --status) ACTION=status ;;
    --stop) ACTION=stop ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
[ -n "$PEER" ] || { echo "--peer USER@SECOND_MAC is required (the second Mac, reachable over ssh without a password)"; exit 2; }

say() { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
fail() { printf '   FAILED: %s\n' "$*"; exit 1; }
run() { printf '   $ %s\n' "$*"; [ "$DRY" -eq 1 ] || eval "$@"; }
peer() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" "$@"; }
LOG="$DIR/logs"; APP="$DIR/app"; VENV="$DIR/venv"; HF="$DIR/hf"; MODEL="$DIR/GLM-5.3-4bit-MTP"; R1="$DIR/rank1"
HOSTFILE="$DIR/tp2-hostfile.json"

# ---------------------------------------------------------------------------------------------------------------
status() {
  say "Status"
  if curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then info "server answers on port ${PORT}"; else info "server not answering on port ${PORT}"; fi
  pgrep -fl "tp2/server.py" | sed 's/^/   first Mac: /'
  peer 'pgrep -fl "tp2/server.py"' | sed 's/^/   second Mac: /'
}
stop() {
  say "Stopping the server on both Macs"
  run "pkill -f 'mlx.launch.*tp2/server.py' ; pkill -f 'tp2/server.py' ; true"
  run "peer \"pkill -f 'tp2/server.py' ; true\""
}
[ "$ACTION" = status ] && { status; exit 0; }
[ "$ACTION" = stop ] && { stop; exit 0; }

# ---------------------------------------------------------------------------------------------------------------
vercmp_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }
check_mac() {  # check_mac LABEL "command runner"
  local label="$1" r="$2" ver mem gpu rdma
  ver="$($r sw_vers -productVersion)"; mem="$($r sysctl -n hw.memsize)"; gpu="$($r sysctl -n machdep.cpu.brand_string)"
  rdma="$($r rdma_ctl status 2>&1 | head -1)"
  info "$label: macOS $ver, $((mem / 1024 / 1024 / 1024)) GB, $gpu, RDMA: $rdma"
  vercmp_ge "$ver" "$MIN_MACOS" || fail "$label runs macOS $ver; macOS $MIN_MACOS or later is required on both Macs (RDMA over Thunderbolt and the bundle's MLX build for macOS 26 need it). Update macOS, then run again"
  [ $((mem / 1024 / 1024 / 1024)) -ge $MIN_RAM_GB ] || fail "$label: at least ${MIN_RAM_GB} GB of memory is needed"
  case "$gpu" in *Apple*) ;; *) fail "$label: an Apple silicon Mac is needed";; esac
  if [ "$rdma" != "enabled" ]; then
    cat <<EOF
   $label: RDMA over Thunderbolt is not enabled. It can only be switched on from macOS Recovery:
     1. shut down, then hold the power button until "Loading startup options" appears; choose Options;
     2. in Recovery open Utilities > Terminal and run:   rdma_ctl enable
     3. restart. (This script never uses sudo and cannot do this for you.)
EOF
    fail "$label: RDMA disabled"
  fi
}
checks() {
  say "1. Checking both Macs"
  peer true || fail "cannot ssh to $PEER without a password (set up an ssh key first)"
  check_mac "first Mac" ""
  check_mac "second Mac" "peer"
  local free0 free1
  free0=$(df -g /Users/Shared | tail -1 | awk '{print $4}'); free1=$(peer "df -g /Users/Shared | tail -1" | awk '{print $4}')
  info "free disk: first Mac ${free0} GB (needs ~420 for the weights), second Mac ${free1} GB (needs ~215)"
  [ "$free0" -ge 420 ] || info "warning: the first Mac may run out of disk"
  [ "$free1" -ge 215 ] || info "warning: the second Mac may run out of disk"
  cat <<EOF
   Memory: each Mac holds ~198 GiB of weights. Quit other GPU and memory-heavy apps on BOTH Macs first (other model
   servers, image/video apps, browsers with many tabs). Measured headroom with nothing else running: the second Mac
   keeps ~42 GiB reclaimable after loading; long contexts and many lanes use that up (see the README).
EOF
}

# ---------------------------------------------------------------------------------------------------------------
normmac() { echo "$1" | tr 'A-F' 'a-f' | awk -F: '{for (k=1;k<=NF;k++) printf "%s%02s", (k>1?":":""), $k}' | tr ' ' 0; }
# links: for every active Thunderbolt RDMA port here, find the port it is cabled to on the second Mac
links() {
  say "2. Thunderbolt 5 RDMA links"
  local devs pdevs
  devs=$(ibv_devices | awk '/rdma_/ {print $1}')
  LINKS=$(mktemp -t tp2links)
  local pinfo
  pinfo=$(peer 'for d in $(ibv_devices | awk "/rdma_/ {print \$1}"); do i=${d#rdma_}; s=$(ibv_devinfo -d $d | awk "/state:/ {print \$2}"); ip=$(ifconfig $i 2>/dev/null | awk "/inet 169.254/ {print \$2}"); mac=$(ifconfig $i 2>/dev/null | awk "/ether/ {print \$2}"); echo "$d $i $s $ip $mac"; done')
  info "second Mac ports: $(echo "$pinfo" | awk '$3=="PORT_ACTIVE" && $4!="" {printf "%s(%s) ", $1, $4}')"
  local d i s ip peerip mac pd
  for d in $devs; do
    i=${d#rdma_}
    s=$(ibv_devinfo -d "$d" | awk '/state:/ {print $2}')
    ip=$(ifconfig "$i" 2>/dev/null | awk '/inet 169.254/ {print $2}')
    [ "$s" = PORT_ACTIVE ] && [ -n "$ip" ] || continue
    # the cable's far end: ask the link's IPv6 all-nodes group, then read this port's own neighbour table
    # (IPv4 cannot do it: every 169.254 address is routed through one port)
    ping6 -c 2 -i 0.2 "ff02::1%$i" >/dev/null 2>&1
    own=$(ifconfig "$i" | awk '/ether/ {print $2}')
    mac=""
    for m in $(ndp -an -i "$i" 2>/dev/null | awk -v i="$i" '$3==i {print $2}'); do
      m=$(normmac "$m"); [ "$m" = "$(normmac "$own")" ] && continue
      echo "$pinfo" | awk '{print $5}' | while read -r pm; do [ "$(normmac "$pm")" = "$m" ] && echo "$pm"; done | head -1 | grep -q . && { mac=$m; break; }
    done
    pd=""
    [ -n "$mac" ] && pd=$(echo "$pinfo" | while read -r a b c e f; do [ "$(normmac "$f")" = "$mac" ] && [ "$c" = PORT_ACTIVE ] && echo "$a $e"; done | head -1)
    [ -n "$pd" ] || continue
    echo "$d $ip $pd" >> "$LINKS"
    info "link: here $d ($ip)  <->  second Mac ${pd% *} (${pd#* })"
  done
  local n; n=$(wc -l < "$LINKS" | tr -d ' ')
  [ "$n" -ge 1 ] || fail "no Thunderbolt RDMA link found between the Macs (cables in, both ports active?)"
  info "$n link(s); jaccl stripes every collective across all of them"
  local l0 p0 here there
  l0=$(head -1 "$LINKS" | awk '{print $2}'); p0=$(head -1 "$LINKS" | awk '{print $4}')
  here=$(awk '{printf "%s\"%s\"", (NR>1?", ":""), $1}' "$LINKS")
  there=$(awk '{printf "%s\"%s\"", (NR>1?", ":""), $3}' "$LINKS")
  PEER_TB="${PEER%@*}@$p0"; [ "$PEER_TB" = "@$p0" ] && PEER_TB="$p0"
  [ "$DRY" -eq 1 ] && return
  cat > "$HOSTFILE" <<EOF
[
 {"ssh": "$l0", "ips": ["$l0"], "rdma": [null, [$here]]},
 {"ssh": "$p0", "ips": ["$p0"], "rdma": [[$there], null]}
]
EOF
  info "hostfile: $HOSTFILE"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$PEER_TB" true || fail "cannot ssh to the second Mac over Thunderbolt ($PEER_TB)"
  # mlx.launch starts rank 0 over ssh as well (to this Mac's Thunderbolt address)
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$l0" true 2>/dev/null || \
    fail "mlx.launch must reach this Mac over ssh at $l0: turn on Remote Login (System Settings > General > Sharing) and add your own public key to ~/.ssh/authorized_keys"
}

# ---------------------------------------------------------------------------------------------------------------
install_bundle() {
  say "3. Install the release bundle on both Macs"
  run "mkdir -p '$DIR' '$LOG'"
  run "peer \"mkdir -p '$DIR' '$LOG'\""
  if [ -z "$BUNDLE" ]; then
    BUNDLE="$DIR/$BUNDLE_NAME"
    if [ ! -f "$BUNDLE" ]; then
      run "curl -fL -o '$BUNDLE' 'https://github.com/${REPO}/releases/download/${RELEASE}/${BUNDLE_NAME}'"
    fi
  fi
  if [ "$DRY" -eq 0 ]; then
    local got; got=$(shasum -a 256 "$BUNDLE" | awk '{print $1}')
    if [ "$BUNDLE_SHA256" != "__BUNDLE_SHA256__" ] && [ "$got" != "$BUNDLE_SHA256" ]; then
      fail "bundle SHA-256 $got does not match the pinned $BUNDLE_SHA256"
    fi
    info "bundle SHA-256 $got"
  fi
  local inst="rm -rf '$APP' && mkdir -p '$APP' && tar -xzf '$DIR/bundle.tar.gz' -C '$APP' --strip-components 1 && bash '$APP/install.sh' '$DIR'"
  run "cp '$BUNDLE' '$DIR/bundle.tar.gz' 2>/dev/null || true; $inst"
  run "rsync -a '$DIR/bundle.tar.gz' '$PEER_TB:$DIR/bundle.tar.gz'"
  run "ssh -o BatchMode=yes '$PEER_TB' \"$inst\""
}

# ---------------------------------------------------------------------------------------------------------------
weights() {
  say "4. Weights (first Mac only)"
  local hf="$VENV/bin/hf"
  run "HF_HOME='$HF' '$hf' download $PACK_REPO --revision $PACK_REV"
  PACK="$HF/hub/models--mlx-community--GLM-5.3-4bit/snapshots/$PACK_REV"
  if [ "$MTP" -eq 1 ]; then
    run "HF_HOME='$HF' '$hf' download $HEAD_REPO --revision $HEAD_REV"
    local head="$HF/hub/models--inferencerlabs--GLM-5.3-MTP-MLX/snapshots/$HEAD_REV"
    if [ -f "$MODEL/config.json" ] && grep -q '"num_nextn_predict_layers": 1' "$MODEL/config.json"; then
      info "merged MTP folder already there: $MODEL"
    else
      # oMLX's loader takes the head in the raw layout (model.layers.78.*); the merge links the pack's files and
      # rewrites only the head's file with renamed keys (~5.6 GB)
      run "'$VENV/bin/python' -I '$APP/tools/merge_mtp_head.py' merge --base '$PACK' --head '$head' --out '$MODEL' \
        --rename mtp.shared_head_norm.=mtp.0.norm. --rename mtp.eh_proj.=mtp.0.eh_proj. \
        --rename mtp.enorm.=mtp.0.enorm. --rename mtp.hnorm.=mtp.0.hnorm. --rename mtp.=model.layers.78."
    fi
  else
    MODEL="$PACK"
  fi
}

preshard() {
  say "5. Rank 1's folder, written here and moved to the second Mac over Thunderbolt"
  if [ "$DRY" -eq 0 ] && ssh -o BatchMode=yes "$PEER_TB" "test -f '$R1/tp_shard.json'" && \
     { [ "$MTP" -eq 0 ] || ssh -o BatchMode=yes "$PEER_TB" "grep -q '\"mtp\": true' '$R1/tp_shard.json'"; }; then
    info "rank 1 folder already on the second Mac"; return
  fi
  local m=""; [ "$MTP" -eq 1 ] && m="--mtp"
  run "'$VENV/bin/python' '$APP/tp2/preshard.py' --model '$MODEL' --rank 1 --out '$DIR/rank1-stage' $m --to '$PEER_TB:$R1' 2>&1 | tee '$LOG/preshard.log'"
  run "ssh -o BatchMode=yes '$PEER_TB' \"'$VENV/bin/python' '$APP/tp2/preshard.py' --check '$R1' $m\""
}

# ---------------------------------------------------------------------------------------------------------------
serve() {
  say "6. Start (rank 1 on the second Mac, rank 0 here)"
  local m="" k=""; [ "$MTP" -eq 1 ] && m="--mtp"; [ "$KV8" -eq 1 ] && k="--kv8"
  cat <<EOF
   macOS may ask on each Mac whether Python may "find devices on your local network": allow it (System Settings >
   Privacy & Security > Local Network), or the two ranks cannot connect.
EOF
  if [ "$YES" -eq 0 ] && [ "$DRY" -eq 0 ] && [ -t 0 ]; then
    local ans; read -r -p "   Start the server now? [Y/n] " ans; case "$ans" in n|N|no) exit 0 ;; esac
  fi
  run "nohup '$VENV/bin/mlx.launch' --backend jaccl --hostfile '$HOSTFILE' --cwd '$DIR' --python '$VENV/bin/python' \
    --env MLX_METAL_FAST_SYNCH=1 --env PYTHONUNBUFFERED=1 -- '$APP/tp2/server.py' --rank0 '$MODEL' --rank1 '$R1' \
    --host $HOST --port $PORT --max-context $MAXCTX $m $k > '$LOG/server.log' 2>&1 &"
  [ "$DRY" -eq 1 ] && return
  info "waiting for the server (loading ~200 GiB on each Mac, about a minute)"
  for i in $(seq 1 240); do
    grep -q "serving http" "$LOG/server.log" 2>/dev/null && break
    grep -qiE "Traceback|Error" "$LOG/server.log" 2>/dev/null && { tail -20 "$LOG/server.log"; fail "the server did not start"; }
    sleep 2
  done
  local ans
  ans=$(curl -fsS -m 300 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of Australia? Answer in one word."}],"max_tokens":200,"temperature":0}')
  echo "$ans" | grep -q "Canberra" && info "smoke test: Canberra" || { echo "$ans"; fail "smoke test did not answer Canberra"; }
  info "serving http://${HOST}:${PORT}/v1 (OpenAI-compatible; log $LOG/server.log)"
}

checks
[ "$DRY" -eq 1 ] || mkdir -p "$DIR" "$LOG"
links
install_bundle
weights
preshard
serve
