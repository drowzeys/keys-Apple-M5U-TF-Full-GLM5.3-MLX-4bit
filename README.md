# keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit

Full GLM-5.3 (`glm_moe_dsa`) served on a single Apple M5 Ultra Mac by
[TensorFold](https://github.com/ashhart/TensorFold), Ash Hart's Zig inference engine, straight from the
`mlx-community/GLM-5.3-4bit` weights, with MTP speculative decoding through TensorFold's native server.

This is a modified copy of TensorFold v1.3.5 (upstream commit `02b324ae`) with one added model family, `glm_ax`.
Upstream's own README is kept as [README-TensorFold.md](README-TensorFold.md); what was changed, and the code it
was adapted from, is listed in [MODIFICATIONS.md](MODIFICATIONS.md).

- 78 layers, MLA attention with the DSA sparse indexer (top-2048 keys), 256 routed experts (top-8) plus one shared
  expert, and GLM-5.3's one MTP layer as the draft head.
- Metal kernels using the M5 GPU's tensor units for prompt processing and decode attention.
- MTP drafts plus context-copy drafts, verified in one pass, through TensorFold's OpenAI-compatible server. The
  default is two MTP drafts with a confidence cut (best for code and edits); prose is best at one draft. Opt-in:
  the DFlash2 and DSpark drafters.
- One stream at a time (`--parallel 1`); served and checked up to about 21k tokens of context, prompt rate measured
  to 78k.

It is an experiment, not a release of upstream TensorFold. No container image is provided: Metal does not run
inside Docker on macOS.

## Releases

| Release | Source | What changed |
|---|---|---|
| [v0.2.1-glm53-prefill](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.2.1-glm53-prefill) (this source) | tf39 on TensorFold v1.3.5 | prefill update: 8192-row prompt chunks and fused gate/up/SwiGLU for the prompt's experts (TensorFold core `affine_mm.gatherGlu`): 735-784 tok/s, +4/+9/+7% over v0.2.0 at 1.3k/4.9k/19.6k. Decode level with v0.2.0. Built-in MTP default is now depth 2 with the 0.7 confidence cut |
| [v0.2.0-glm53-prefill](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.2.0-glm53-prefill) | tf33 on TensorFold v1.3.5 | prefill: 4096-row prompt chunks, two-part probabilities, MetalPerformancePrimitives prompt attention; prompt processing is now flat at about 707-718 tok/s from 1.3k to 19.6k tokens. DFlash2 / DSpark / MTP confidence cut added as options |
| [v0.1.0-glm53-mtp](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.1.0-glm53-mtp) | tf16 on TensorFold `d30325dc` (1.3.4) | first release: full model + MTP |

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

| Build | Drafts (settings) | Prose, T=0.6 | Prose, greedy | Code, greedy | Edit, greedy | Prose at ~16k context |
|---|---|---:|---:|---:|---:|---:|
| **v0.2.1 (tf39)** | MTP, depth 1 (`GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off`) | **36.0** | **36.9** | **39.3** | **38.3** | - |
| **v0.2.1 (tf39)** | MTP, built-in default (depth 2 + confidence cut 0.7) | **35.5** | **36.7** | **40.2** | **41.4** | **31.8** |
| v0.2.0 release binary | MTP, depth 1 (`GLM_AX_DRAFT_DEPTH=1`) | 36.0 | 37.9 | 39.8 | 40.0 | - |
| v0.2.0 release binary, option | MTP, depth 2 + confidence cut (`GLM_AX_DRAFT_DEPTH=2 GLM_AX_MTP_FLOOR=0.7`) | 35.7 | 38.2 | 42.4 | 42.7 | - |
| v0.2.0 source (tf33) as first benchmarked | MTP, depth 1 | 36.5 | 36.6 | 38.6 | 38.5 | 33.0 |
| v0.2.0 source (tf33) as first benchmarked, option | MTP, depth 2 + confidence cut | 35.1 | 36.4 | 40.1 | 42.0 | 32.6 |
| earlier build (tf32), option | DSpark (`GLM_AX_DSPARK=<dir>`, `GLM_AX_DSPARK_FLOOR=0.5`) | 30.9 | 32.4 | 43.0 | 49.8 | - |
| v0.1.0 binary (tf16) | MTP, depth 1 | 36.3 | 37.4 | 40.2 | 35.1 | - |
| tf16 as first benchmarked (fused unembed on) | MTP, depth 1 | 35.8 | 37.1 | 39.3 | 34.0 | - |
| tf15 | MTP, depth 1 | 35.4 | 37.3 | 39.5 | 33.7 | 33.3 |
| tf15 | none | 29.4 | 30.0 | 29.9 | - | - |

- The v0.2.0 release-binary rows are medians of 2 runs (`RUNS=2`), the others of 3.
- v0.2.1 decodes level with v0.2.0 within run-to-run noise; its gain is prompt processing.
- At default settings v0.2.0 decodes prose level with v0.1.0 (36.5 / 36.0 against 36.3 tok/s at T=0.6) and code
  slightly lower (38.6 / 39.8 against 40.2, up to 4%); edits are faster (38.5 / 40.0 against 35.1). The gain of
  this release is prompt processing.
- Depth 2 with the 0.7 cut is the built-in default from v0.2.1 (in v0.2.0 it was opt-in, and unset meant depth 3
  with no cut): a second MTP draft is offered only while the head's running confidence stays at or above 0.7. It
  is best for code and edits; for prose, `GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off` is slightly faster.
- DSpark (RedHat AI's GLM-5.3 drafter) helps code and edits most but costs prose; its row was measured on an
  earlier build (tf32) of the same code, with the drafter from `RedHatAI/GLM-5.3-speculator.dspark`.
- Prose is fresh essays, code is a new program, edit returns a given 40-line file with small changes.
- v0.1.0's run: verify pass 44.8-47.1 ms on the GPU; first-position draft acceptance 76% for prose at T=0.6 and
  93% greedy.

### Prompt processing (first-token rate, prompt cache off)

| Prompt | v0.2.1 (tf39), no drafts | v0.2.1 (tf39), MTP server | v0.2.0 (tf33), no drafts | v0.2.0 release binary, MTP server | same family on v1.3.5 without the prefill changes | v0.1.0 (tf16), no drafts |
|---:|---:|---:|---:|---:|---:|---:|
| 1,284 tokens | **735 tok/s** (+4% / +15%) | 639 tok/s | 706 tok/s | 682 tok/s | 640 tok/s | 667 tok/s |
| 4,941 tokens | **784 tok/s** (+9% / +39%) | 732 tok/s | 718 tok/s | 700 tok/s | 566 tok/s | 593 tok/s |
| 19,611 tokens | **754 tok/s** (+7% / +46%) | 717 tok/s | 707 tok/s | 681 tok/s | 516 tok/s | 570 tok/s |

Percentages are against v0.2.0 (no drafts) and against the v1.3.5 base column. In v0.2.1 the prompt's routed
experts take 0.565 ms a token at 19.6k tokens, against 0.656 ms in v0.2.0 (about 80 against 69 TFLOPS, estimated). The rate is prompt tokens / time to the first token (it includes
tokenizing and one decode pass). With the prompt cache on (the server's default), a follow-up turn of the same
conversation started in about 0.34 s on an earlier build instead of reprocessing the whole conversation.

Prompt rate against context length, without the server (`tf-glm-ax-run`, one fresh prompt each):

| Prompt | 1.3k | 4.9k | 9.9k | 19.6k | 39k | 78k |
|---|---:|---:|---:|---:|---:|---:|
| v0.2.1 (tf39), tok/s | 767 | 793 | 768 | 757 | 732 | 677 |
| v0.2.0 (tf33), tok/s | 734 | 727 | 721 | 711 | 684 | 624 |

Whole decode step without the server (`tf-glm-ax-run`, tf16): 33.9 ms on the GPU at 422 tokens of context
(22.25 GB read a step, 657 GB/s), 38.5 ms at 19.6k tokens.

### Correctness checks

v0.2.1 source (tf39):
- Canberra; needles at about 6k and at 10% and 90% of about 20k tokens returned exactly; through the server, a
  needle at 90% of a 21,125-token prompt was found (29.7 s).
- Drafted greedy output equals undrafted, at depth 1 and at the built-in default.
- 8-layer decode-row probe at position 3,226: cosine 0.999994 against the MLX-checked milestone build.
- The v0.2.1 release binary's own run on the model is pending.

v0.2.0 release binary (the published asset, run on the model after publishing):
- SHA-256 matched; Canberra passed; a needle at 90% of a 21,125-token prompt was found (first answer after 32.8 s).
- A greedy essay was byte-identical to tf33's undrafted output, both at the default settings and with
  `GLM_AX_DRAFT_DEPTH=2 GLM_AX_MTP_FLOOR=0.7`.

v0.2.0 source (tf33):
- Known answer: "What is the capital city of Australia?" gives "Canberra".
- Needle retrieval: codes hidden in about 6k tokens, and at 10% and 90% of about 20k tokens, were all returned
  exactly; through the server, a needle at 90% of a 21,125-token prompt was found.
- With MTP drafts, greedy output equals the undrafted output.
- 8-layer decode-row probe at position 3,226: cosine 0.999994 against the milestone build that was checked
  against MLX.

Earlier:
- Layer probes against MLX (mlx-lm on the same weights), on an earlier build of this family: after 8 layers of a
  3,226-token prompt, the per-row cosine of the fp32 residual was mean 0.9996, min 0.992 over 1,178 rows; on a
  24-token prompt, min 0.99998.
- The v0.1.0 binary was run on the model after publishing: SHA-256 matched, Canberra passed, a greedy essay with
  MTP drafts was byte-identical to the undrafted one, and a needle at 90% of a 21,125-token prompt was found.
- The fused decode paths (absorb + rope, unembed) are bit-identical to the unfused ones.

The output is not bit-identical to MLX: the kernels sum in a different order (fp32 sums of exact bf16 products).

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

The [v0.2.1-glm53-prefill release](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.2.1-glm53-prefill)
has `tensorfold-native` built from this source (`zig build native`, fast mode, Zig 0.17.0, Xcode 27.2 Metal
toolchain) as a .tar.gz with its SHA-256, plus the licence files. It needs macOS 27.2.1 or later (the SDK it was
built with). It is ad-hoc signed, not notarized: after downloading, run
`xattr -d com.apple.quarantine tensorfold-native`. No weights are included. The
[v0.1.0 release](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.1.0-glm53-mtp)
and the [v0.2.0 release](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.2.0-glm53-prefill) stay available, with their sources at their tags; both of those binaries were run on the model after
publishing.

## Serve

Prose (one MTP draft, no cut):

```sh
GLM_AX_LOAD_LIMIT_GB=440 GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off \
  ./zig-out/native/bin/tensorfold-native serve /path/to/GLM-5.3-4bit-MTP \
  --name glm53 --host 127.0.0.1 --port 8020 \
  --context 32768 --parallel 1 --reasoning-effort low --no-update-check
```

Code and edits (the built-in default: two MTP drafts with the 0.7 confidence cut):

```sh
GLM_AX_LOAD_LIMIT_GB=440 \
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
| `GLM_AX_DRAFT_DEPTH` | MTP drafts a window holds, 1-3 (default 2 since v0.2.1); 1 is best for prose |
| `GLM_AX_MTP_FLOOR` | 0-1: verify a draft past the first only while the MTP head's running confidence (product of its drafts' top probabilities) stays at or above it, and stop drafting there; default 0.7 since v0.2.1, `off` verifies every requested draft |
| `GLM_AX_DSPARK=<dir>` | use RedHat AI's DSpark drafter (`RedHatAI/GLM-5.3-speculator.dspark`) instead of MTP; `GLM_AX_DSPARK_FLOOR` sets its confidence cut (0.5 measured), `GLM_AX_DSPARK_DEPTH` its most drafts (1-8) |
| `GLM_AX_DFLASH2=<dir>` | use a DFlash2 drafter for GLM-5.3 (for example `incoai/GLM-5.3-DFlash2`) instead of MTP; `GLM_AX_DFLASH2_DEPTH` 1-7. Experimental: no numbers are published for it in this README |
| `GLM_AX_NO_GLU=1` | prompt experts without the fused gate/up/SwiGLU product (the v0.2.0 path) |
| `GLM_AX_GROUPS=1` | **experimental, do not use**: routes prompt products through the GLM-5.3-Flash family's column-group kernels; with glm_ax this currently gives wrong output and is slower |
| `GLM_AX_NO_MPP=1` | prompt attention by the older tensor-unit kernel instead of the MetalPerformancePrimitives one |
| `GLM_AX_METAL4=1` | run glm_ax on a Metal 4 queue (experimental; Metal 3 by default) |
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
- Decode and correctness were measured through the server up to about 21k tokens of context; the prompt rate was
  measured to 78k tokens without the server. Decode and answers at longer contexts were not measured.
- Not bit-identical to MLX (summation order), see above.
- The DFlash2 drafter path is in the code but not benchmarked for this README; the DSpark row comes from an
  earlier build (tf32).
- `GLM_AX_GROUPS=1` is known to give wrong output; leave it unset.
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
- [Splash](https://github.com/incoai/splash) (Apache-2.0): `glm_ax_mpp.metal` adapts its
  `listed_attention.metal` kernel, which ports the tensor-unit sparse MLA method of
  [oMLX](https://github.com/jundot/omlx) 0.8.0rc1 (`sparse_mla_nax.py`, Apache-2.0).
- DSpark: [RedHatAI/GLM-5.3-speculator.dspark](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) (under the
  GLM-5.3 License; not included), with semantics from RedHat AI's
  [speculators](https://github.com/vllm-project/speculators) and [vLLM](https://github.com/vllm-project/vllm)
  (Apache-2.0). DFlash2: [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) (Apache-2.0; not
  included), semantics from [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0).
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple (MIT): the
  weight format, and the reference the layer probes were compared against.
- Multi-token prediction: Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"
  (2024), and the MTP module design of DeepSeek-V3 (2024) that GLM-5.3's MTP layer follows; speculative decoding:
  Leviathan et al. (2023) and Chen et al. (2023).
- The `merge_mtp_head.py` key names follow the `mtp.*` layout that oMLX's GLM loader also uses.
