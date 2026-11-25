# keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit

Full GLM-5.3 (`glm_moe_dsa`) served on a single Apple M5 Ultra Mac by
[TensorFold](https://github.com/ashhart/TensorFold), Ash Hart's Zig inference engine, straight from the
`mlx-community/GLM-5.3-4bit` weights, with MTP speculative decoding through TensorFold's native server.

This is a modified copy of TensorFold 1.3.4 (upstream commit `d30325dc`) with one added model family, `glm_ax`.
Upstream's own README is kept as [README-TensorFold.md](README-TensorFold.md); what was changed is listed in
[MODIFICATIONS.md](MODIFICATIONS.md).

- 78 layers, MLA attention with the DSA sparse indexer (top-2048 keys), 256 routed experts (top-8) plus one shared
  expert, and GLM-5.3's one MTP layer as the draft head.
- Metal kernels using the M5 GPU's tensor units for prompt processing and decode attention.
- MTP drafts plus context-copy drafts, verified in one pass, through TensorFold's OpenAI-compatible server.
- One stream at a time (`--parallel 1`); measured up to about 20k tokens of context.

It is an experiment, not a release of upstream TensorFold. No container image is provided: Metal does not run
inside Docker on macOS.

## Measured on

- Apple M5 Ultra, 512 GB unified memory, macOS 27.2.1, Xcode 27.2 (Metal toolchain), Zig 0.17.0.
- Weights: `mlx-community/GLM-5.3-4bit` at revision `81ad9b73c6188f808737dbb52458f082d249dcbe`, plus the MTP head
  from `inferencerlabs/GLM-5.3-MTP-MLX` at revision `f1187efee8cf60319a7d782c3488cfd96017c5ce`.

## Results

All numbers are from the native server at `--context 32768 --parallel 1 --reasoning-effort low`, streaming,
measured by the scripts in [glm53/bench](glm53/bench) (decode rate = (completion tokens - 1) / time from the first
to the last token; medians of 3 runs; 400 new tokens for prose and code, up to 700 for edit). The prompt cache was
off for these runs (`--prompt-cache-gib 0`) so every request is processed from scratch.

### Decode, tokens per second

| Build | Drafts | Prose, T=0.6 | Prose, greedy | Code, greedy | Edit, greedy |
|---|---|---:|---:|---:|---:|
| this source (tf16) | MTP, depth 1 | **35.8** | **37.1** | **39.3** | **34.0** |
| previous build (tf15) | MTP, depth 1 | 35.4 | 37.3 | 39.5 | 33.7 |
| previous build (tf15) | none | 29.4 | 30.0 | 29.9 | - |

- Prose is fresh essays, code is a new program, edit returns a given 40-line file with small changes.
- With MTP, 67-96% of drafts were accepted depending on the task. A verify pass of 2 rows takes about 46 ms on the
  GPU against about 33 ms for a plain 1-row step; the MTP draft pass about 1.2 ms.
- MTP at about 16k tokens of context (tf15): prose 33.3 tok/s at T=0.6, 33.5 greedy.
- First token on a short prompt: 0.30-0.35 s (0.85 s for the edit prompt).
- Earlier builds measured depth 1 a little ahead of depths 2 and 3 for prose (32.5 / 31.8 / 31.3 tok/s on one
  older build), so depth 1 is the recommended setting; deeper drafts helped edits on older builds.

### Prompt processing (tf16, no drafts)

| Prompt | First token | Prompt rate | Decode at that context |
|---:|---:|---:|---:|
| 1,284 tokens | 1.93 s | 667 tok/s | 28.4 tok/s |
| 4,941 tokens | 8.33 s | 593 tok/s | 26.7 tok/s |
| 19,611 tokens | 34.43 s | 570 tok/s | 25.8 tok/s |

With the prompt cache on (the server's default), a follow-up turn of the same conversation started in about 0.34 s
on an earlier build instead of reprocessing the whole conversation.

Whole decode step without the server (`tf-glm-ax-run`, tf16): 33.9 ms on the GPU at 422 tokens of context
(22.25 GB read a step, 657 GB/s), 38.5 ms at 19.6k tokens.

### Correctness checks

- Known answer: "What is the capital city of Australia?" gives "Canberra" (whole model, greedy, before every
  benchmark run).
- Needle retrieval: a code hidden in about 6.2k tokens of filler text, and two codes hidden at 10% and 90% of about
  19.6k tokens, were all returned exactly.
- Layer probes against MLX (mlx-lm on the same weights), run on an earlier build of this family: after 8 layers
  of a 3,226-token prompt, the per-row cosine of the fp32 residual was mean 0.9996, min 0.992 over 1,178 rows; on
  a 24-token prompt, min 0.99998. The published build's decode row (8 layers, position 3,226) matches that checked
  build at cosine 0.999995.
- The fused decode paths (absorb + rope, unembed) are bit-identical to the unfused ones.

The output is not bit-identical to MLX: the kernels sum in a different order (fp32 sums of exact bf16 products).

### Which build is this

This repository is the `tf16` source tree. It was benchmarked with the fused unembed path switched on; afterwards
that path was put behind `GLM_AX_FUSE_UNEMBED=1` (off by default) because it was about 0.5 ms a step slower and
bit-identical. The default build published here (absorb fusion on, fused unembed off, fused MoE off, residency set
off) was not re-benchmarked as a whole; to run exactly the measured configuration set `GLM_AX_FUSE_UNEMBED=1`.

## Memory

- Weights: 389.6 GiB without the MTP head, 394.8 GiB with it. Caches for 32k context: 11-17 GiB.
- The Mac's GPU wired-memory limit has to be raised above the default, for example:

  ```sh
  sudo sysctl iogpu.wired_limit_mb=500000
  ```

  This lets the GPU wire almost all of the machine's memory. Do it at your own risk: with too little left for
  macOS, the machine can stall or reboot. Close other large programs first, and a memory watchdog (a script that
  stops the server when free memory drops below a few GB) is recommended. The setting resets at reboot.
- `GLM_AX_LOAD_LIMIT_GB=440` raises the engine's own load limit (default 75% of RAM, too low for this model).
- A 512 GB Mac is required in practice. Loading takes about 40 s from a warm file cache.

## Get the weights

Download the 4-bit pack and the MTP head (about 418 GB together):

```sh
hf download mlx-community/GLM-5.3-4bit --revision 81ad9b73c6188f808737dbb52458f082d249dcbe --local-dir GLM-5.3-4bit
hf download inferencerlabs/GLM-5.3-MTP-MLX --revision f1187efee8cf60319a7d782c3488cfd96017c5ce --local-dir GLM-5.3-MTP-MLX
```

The 4-bit pack does not contain GLM-5.3's MTP layer (`num_nextn_predict_layers` is 0 in its config). Build a
model folder that has both, without copying the large files:

```sh
python3 glm53/tools/merge_mtp_head.py inspect --head GLM-5.3-MTP-MLX
python3 glm53/tools/merge_mtp_head.py merge --base GLM-5.3-4bit --head GLM-5.3-MTP-MLX --out GLM-5.3-4bit-MTP
```

`GLM-5.3-4bit-MTP` then holds symlinks to every file of the pack, the head's two weight files as
`model-mtp-00001-of-00002.safetensors` and `model-mtp-00002-of-00002.safetensors` (59 tensors under `mtp.*`), an
index listing them, and a config with `num_nextn_predict_layers` set to 1. Nothing in the source folders is
changed. The script reads only safetensors headers and needs no Python packages. Serve the plain pack folder to run
without MTP.

## Build

```sh
# Zig 0.17.0, and full Xcode with its Metal toolchain (xcodebuild -downloadComponent MetalToolchain)
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer zig build native
```

The server is `zig-out/native/bin/tensorfold-native`. `zig build` defaults to the fast (ReleaseFast) mode. The
Metal kernels are compiled from embedded sources when the model loads. `zig build tf-glm-ax-run` builds the
server-less runner.

### Prebuilt binary

The [v0.1.0-glm53-mtp release](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.1.0-glm53-mtp)
has `tensorfold-native` built from this source (`zig build native`, fast mode, Xcode 27.2 Metal toolchain) as a
.tar.gz with its SHA-256, plus the licence files. It needs macOS 27.2.1 or later (the SDK it was built with). It is
ad-hoc signed, not notarized: after downloading, run `xattr -d com.apple.quarantine tensorfold-native`. No weights
are included. The binary is a rebuild of the measured source with the defaults described above, not the exact file
that produced the benchmark numbers.

## Serve

```sh
GLM_AX_LOAD_LIMIT_GB=440 GLM_AX_DRAFT_DEPTH=1 \
  ./zig-out/native/bin/tensorfold-native serve /path/to/GLM-5.3-4bit-MTP \
  --name glm53 --host 127.0.0.1 --port 8020 \
  --context 32768 --parallel 1 --reasoning-effort low --no-update-check
```

The benchmarks added `--prompt-cache-gib 0`; leave it out for normal use so follow-up turns reuse the cache. Pass
`--no-drafts` (and the plain pack folder) to serve without speculative decoding.

Then, for example:

```sh
curl -s 127.0.0.1:8020/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm53","messages":[{"role":"user","content":"What is the capital city of Australia?"}],"max_tokens":40,"temperature":0}'
```

### Settings (environment variables)

| Variable | Effect |
|---|---|
| `GLM_AX_LOAD_LIMIT_GB` | load limit in GB (default 75% of RAM); 440 was used |
| `GLM_AX_DRAFT_DEPTH` | MTP drafts a window holds, 1-3 (default 3); 1 was used and is recommended |
| `GLM_AX_FUSE_UNEMBED=1` | fused unembed for decode rows (bit-identical, measured slightly slower) |
| `GLM_AX_MOE_FUSED=1` | fused decode MoE (bit-identical; slower on the measured runs) |
| `GLM_AX_RESIDENCY=1` | hold the weights in a Metal residency set (measured no decode gain, slower prompt after idle) |
| `GLM_AX_UNFUSED=1` | switch off the decode fusions |

## Reproduce the benchmarks

With the server running on port 8020:

```sh
cd glm53/bench
URL=http://127.0.0.1:8020 MODEL=glm53 RUNS=3 python3 decode.py prose code edit
URL=http://127.0.0.1:8020 MODEL=glm53 python3 prompt.py 1000 4000 16000
URL=http://127.0.0.1:8020 MODEL=glm53 RUNS=2 CTX=16000 python3 decode.py prose
```

The scripts use only the Python standard library.

## Limitations

- Single stream only (`--parallel 1`).
- Measured up to about 20k tokens of context; longer contexts (up to the 32k the server was started with, or
  beyond) were not measured.
- Not bit-identical to MLX (summation order), see above.
- DFlash2 drafting is not in this build; drafts are GLM-5.3's MTP head plus context copies.
- Only the MLX affine 4-bit (group 64) layout of `mlx-community/GLM-5.3-4bit` was tested.
- Built and tested on an M5 Ultra only. The prompt kernels need the M5 generation's GPU tensor units.

## Credits and licences

- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and the TensorFold contributors, Apache-2.0. This
  repository keeps its [LICENSE](LICENSE), [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
  [LICENSES](LICENSES). The glm_ax kernels are derived from TensorFold's GLM-5.3-Flash kernels. The changes in this
  repository are Apache-2.0 as well.
- [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) by Z.ai, under the GLM-5.3 License (a permissive licence that
  requires keeping its copyright and permission notice, and asks Model-as-a-Service operators with more than
  US$10 billion in yearly revenue to pass Z.ai's security review before commercial use). Read the licence on the
  model page before using the weights. No weights are included here.
- [mlx-community/GLM-5.3-4bit](https://huggingface.co/mlx-community/GLM-5.3-4bit): the MLX 4-bit quantisation
  (same GLM-5.3 License).
- [inferencerlabs/GLM-5.3-MTP-MLX](https://huggingface.co/inferencerlabs/GLM-5.3-MTP-MLX): GLM-5.3's MTP layer
  extracted and quantised for MLX (ships the GLM-5.3 License).
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple (MIT): the
  weight format, and the reference the layer probes were compared against.
- Multi-token prediction: Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"
  (2024), and the MTP module design of DeepSeek-V3 (2024) that GLM-5.3's MTP layer follows; speculative decoding:
  Leviathan et al. (2023) and Chen et al. (2023).
- The `merge_mtp_head.py` key names follow the `mtp.*` layout that oMLX's GLM loader also uses.
