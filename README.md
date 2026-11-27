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
- Up to 1,048,576 tokens of context (GLM-5.3's full window) with the opt-in 8-bit KV cache and the MTP head: a
  needle was found in a 1,026,813-token prompt, decode 9.6 tok/s. With the bf16 cache, up to 393,216 tokens.
- One stream at a time (`--parallel 1`) by default.

**Two ways to run it:** one 512 GB Mac → [`oneshot.sh`](#one-shot) (TensorFold); two 256 GB Macs →
[`tp2/oneshot-tp2.sh`](#two-256-gb-macs-tp2) (MLX, tensor parallel over Thunderbolt 5).

It is an experiment, not a release of upstream TensorFold. No container image is provided: Metal does not run
inside Docker on macOS.

## One-shot

macOS only (Apple M5-class GPU, macOS 27.2.1 or later, 512 GB unified memory). `oneshot.sh` checks the Mac,
downloads the v0.3.1 release binary and verifies its SHA-256 (pinned in the script; `--release latest` takes the
newest release instead), downloads the pinned weights with `hf download`
(or reuses them from your Hugging Face cache), builds the MTP model folder from symlinks, offers to raise the GPU
wired limit, starts the server and waits for a "Canberra" smoke answer.

```sh
git clone https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit.git
cd keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit
./oneshot.sh --preset prose --dry-run     # print every step and command, change nothing
./oneshot.sh --preset prose               # or: code | long | 1m | 1m-nomtp
./oneshot.sh --status                     # / --stop
```

Or without cloning: download `oneshot.sh` from the [latest release](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/latest)
and run it the same way (it then fetches the merge script from the repository). Large files go under `--dir`
(default `~/glm53-tf`); about 420 GB of disk is needed unless the 4-bit pack is already in your Hugging Face cache.

| Preset | What it runs | Measured | Memory |
|---|---|---|---|
| `prose` | MTP depth 1, no cut, 32k context | v0.3.1: prose 36.2 tok/s (T=0.6), greedy 37.5 | peak wired about 415 GiB at 78k (less at 32k) |
| `code` | built-in MTP default (depth 2 + 0.7 cut), 32k | v0.3.1: code 41.8 tok/s, edit 42.7 | as `prose` |
| `long` | bf16 cache, MTP depth 1, 393,216 context, load limit 488 | 375k-token needle found, first token 784 s, decode 20.3 tok/s | peak wired 442.6 GiB |
| `1m` | 8-bit cache, MTP depth 1, 1,048,576 context, 1024-row prompt chunks, load limit 495 | 1.03M-token needle found, first token 63 min, decode 9.6 tok/s | peak wired 453.4 GiB; headroom thin (lowest reclaimable 43-54 GiB across runs): close other apps |
| `1m-nomtp` | 8-bit cache, no MTP head, 1,048,576 context, 2048-row chunks | 1.03M-token needle found, first token 58 min, decode 7.4 tok/s | peak wired 449.1 GiB |

Safety:
- The model wires 410-450 GiB of GPU memory. The script asks before running
  `sudo sysctl iogpu.wired_limit_mb=500000` and never runs it silently; raising the limit is at your own risk (with
  other large programs running, macOS can stall or reboot). Close other large programs; a memory watchdog is
  recommended. The setting resets at reboot.
- The binary is ad-hoc signed, not notarized; the script clears the download quarantine flag
  (`xattr -dr com.apple.quarantine`) on the unpacked folder only.
- The script uses no sudo except that one optional step.

## Two 256 GB Macs (TP=2)

> **Known issue in 0.1.0:** step 5 of `oneshot-tp2.sh` (pre-shard) fails with an rsync error
> (`BatchMode=yes: (l)stat: No such file or directory`), and the server then cannot start. A fixed 0.1.1 is being
> tested. Workaround: in `tp2/preshard.py` (line 57), the `TP2_SINK` command is built with `" ".join([...])`, and
> `glm_tp/model.py` splits it again with `shlex.split`, so the `-e` value `ssh -o BatchMode=yes` falls apart into
> three arguments. Build it with `shlex.join([...])` instead (same list) so that value stays one argument, then rerun
> the one-shot.

Instead of one 512 GB Mac, GLM-5.3 can run on **two 256 GB M5 Ultra Macs** joined by Thunderbolt 5 cables, split
tensor-parallel (TP=2) on [MLX](https://github.com/ml-explore/mlx): each Mac holds half of every layer (half the
attention heads and half of every expert's columns, ~198 GiB), the two halves are summed after each attention and
MoE block with MLX's `jaccl` RDMA backend striped over every Thunderbolt link. The model code is
[oMLX](https://github.com/jundot/omlx)'s GLM-5.3 implementation; this is a separate engine from the TensorFold
recipe above.

```sh
# on the first Mac (it serves HTTP and drives the second Mac over ssh); oneshot-tp2.sh is a release asset
# (also in tp2/ of this repository)
./oneshot-tp2.sh --peer USER@SECOND_MAC --dry-run     # print every step, change nothing
./oneshot-tp2.sh --peer USER@SECOND_MAC               # check, install, download, pre-shard, start, smoke test
./oneshot-tp2.sh --peer USER@SECOND_MAC --status      # / --stop
```

The script checks both Macs (macOS 26.2 or later on both, at least 256 GB, RDMA enabled; it refuses older macOS), finds the Thunderbolt RDMA links and
writes the hostfile, installs the release bundle `keys-glm53-tp2-macos-arm64-<ver>.tar.gz` (CPython 3.11.14 and every
pinned wheel, offline, identical on both Macs; SHA-256 pinned in the script), downloads the pinned weights on the
first Mac, writes rank 1's half and moves it to the second Mac file by file over Thunderbolt, then starts both ranks
with `MLX_METAL_FAST_SYNCH=1` and waits for a "Canberra" answer. Server: OpenAI-compatible
`http://127.0.0.1:8000/v1` (`/chat/completions`, `/completions`, streaming), one request at a time, prefix reuse
between requests.

Before you start:
- **RDMA over Thunderbolt** must be enabled on both Macs, and that is only possible from macOS Recovery (Terminal:
  `rdma_ctl enable`, then restart). The script never uses sudo.
- Password-less ssh from the first Mac to the second, and to itself (`mlx.launch` starts rank 0 over ssh too):
  System Settings > General > Sharing > Remote Login.
- macOS asks whether Python may find devices on the local network: allow it on both Macs.
- **Memory**: quit other GPU and memory-heavy apps on both Macs. Measured with nothing else running, the second Mac
  keeps ~42 GiB reclaimable after loading its half; long contexts and many streams use that up.
- Disk: ~420 GB on the first Mac (weights), ~215 GB on the second (its half).

### Measured (2026-11-26, two M5 Ultra Macs, four Thunderbolt 5 cables)

Even split (experts 0.5/0.5, 32/32 heads), greedy, 1k-token prompts for decode. "One Mac" is the same MLX code on
one 512 GB Mac (not TensorFold).

| | Two 256 GB Macs (TP=2) | One Mac, same code |
|---|---|---|
| One stream, plain | 34.5 tok/s | 26.9 |
| One stream, MTP depth 1: prose / code | 38.0 / 45.5 tok/s (acceptance 0.65 / 0.85) | 29.6 / 31.7 |
| 4 / 8 / 16 / 32 streams (total) | 76.0 / 105.1 / 119.3 / 152.1 tok/s | 43.9 / 66.6 / 77.5 / 101.2 |
| Prompt processing at 4k / 32k tokens | 968 / 961 tok/s | 736 / 700 |
| 128k context, one stream (8-bit KV cache) | prefill 625 tok/s, decode 25.9 tok/s | prefill 503, decode 21.5 (bf16 cache) |
| 256k context, one stream (8-bit KV cache) | prefill 530 tok/s, decode 22.1 tok/s | |

The stream counts above were measured with the benchmark harness (batched decode); the server itself runs one
request at a time.

### Method and exactness

- Each layer's attention heads and expert columns are split between the Macs; two all_sums per layer (156 per token)
  cost 2-4 ms per token over four links. Without `MLX_METAL_FAST_SYNCH=1` each one costs ~0.25 ms (39 ms per
  token): that setting is required (finding credited to [machd](https://github.com/machddev/machd)).
- Logits against one Mac (4 layers): max difference 0.047 (bf16 rounding of the summed halves), same top-10.
- Plain server (no `--mtp`): its greedy tokens equal the development runner's at the same settings, 128/128,
  tested from a clean install of the release bundle on both Macs.
- **MTP output is not bit-identical to plain greedy on TP=2** (unlike the one-Mac TensorFold build). A draft is
  kept only when it equals the verify pass's argmax, but the 2-row verify pass runs a different kernel path than
  1-row decode, so about 2-6 % of positions flip at near-ties (largest logit gap 0.5). Teacher-forced along plain
  greedy's own text, verify agrees with 1-row decode at 157/160 positions on code and 150/160 on prose; no
  disagreement is tied to a rejected draft's rollback (2 of 13 fall on rollback cycles, all within the same
  rounding gap). For exact greedy output, start the server without `--mtp` (`--no-mtp` in the one-shot).
- 8-bit latent KV cache (`--kv8`, 57.7 instead of 95.2 KB per token): at 32k, top-1 agreement along the bf16 text
  97/97; at 1k 92/97, every miss within 0.25 logits (a rounding tie). Decode 2-7 % slower.

### Known limits

- One request at a time in the server; MTP only for greedy requests (sampled requests decode plainly).
- The server reuses the cached prefix of the previous request. A repeated greedy request then runs a shorter
  prefill than the first one did, and its answer can differ from the first ("cold") answer at near-ties;
  requests made in the same cache state give identical answers.
- Chat answers put GLM's thinking in `reasoning_content` and the reply in `content` (also when streaming).
- KV cache is held in full on both Macs (not split), so context × streams is bounded by the second Mac's ~42 GiB:
  about 300k tokens in total with the bf16 cache, ~1.65x that with `--kv8`.
- Both Macs must be the same macOS/Python setup (the bundle takes care of Python) and stay cabled; if one rank
  exits, restart both (`--stop`, then run again).
- Rank 0's control port (5470, rank 1 connects to it) listens on all network interfaces and has no
  authentication: run it on a trusted network or block that port in the firewall for anything but the
  Thunderbolt link.

## Releases

| Release | Source | What changed |
|---|---|---|
| [v0.3.1-glm53-reuse](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.3.1-glm53-reuse) (this source) | tf48 on TensorFold v1.3.5 | follow-up turns: live prompt reuse with no copy (128k follow-up first token 1.14 s instead of 203 s), hybrid prompt chunks for prompts of 64k or more, a short request after a long one reuses the parked allocation (0.97 s -> 0.24-0.29 s), whole-context cache only when 48 GiB stays reclaimable |
| [v0.3.0-glm53-1m](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.3.0-glm53-1m) | tf43 on TensorFold v1.3.5 | 1M context: `max_context` 1,048,576; opt-in 8-bit KV cache (`GLM_AX_K8=1`); `GLM_AX_PROMPT_ROWS`; exact parallel-scan top-k (17% faster top-k at 78k, bit-identical); experimental batched verify and DSpark/MTP choice; `oneshot.sh` |
| [v0.2.1-glm53-prefill](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.2.1-glm53-prefill) | tf39 on TensorFold v1.3.5 | prefill update: 8192-row prompt chunks and fused gate/up/SwiGLU for the prompt's experts (TensorFold core `affine_mm.gatherGlu`): 735-784 tok/s, +4/+9/+7% over v0.2.0 at 1.3k/4.9k/19.6k. Decode level with v0.2.0. Built-in MTP default is now depth 2 with the 0.7 confidence cut |
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
| **v0.3.1** | `prose` preset: MTP depth 1, floor off | **36.2** | **37.5** | **39.3** | **40.9** | - |
| **v0.3.1** | `code` preset: built-in default (depth 2 + 0.7 cut) | **36.2** | **36.8** | **41.8** | **42.7** | - |
| **v0.3.1 release binary** (one-shot run, 2 runs) | `prose` preset | **35.7** | **37.5** | **39.2** | **40.9** | - |
| v0.3.0 release binary (one-shot run) | `prose` preset | 35.2 | 36.7 | 39.2 | 41.0 | - |
| v0.2.1 release binary | MTP, depth 1 (`GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off`) | 35.7 | 37.6 | 39.3 | 41.0 | - |
| v0.2.1 release binary | MTP, built-in default (depth 2 + confidence cut 0.7) | 35.2 | 36.7 | 41.7 | 42.7 | - |
| v0.2.1 source (tf39) as first benchmarked | MTP, depth 1, floor off | 36.0 | 36.9 | 39.3 | 38.3 | - |
| v0.2.1 source (tf39) as first benchmarked | MTP, built-in default | 35.5 | 36.7 | 40.2 | 41.4 | 31.8 |
| v0.2.0 release binary | MTP, depth 1 (`GLM_AX_DRAFT_DEPTH=1`) | 36.0 | 37.9 | 39.8 | 40.0 | - |
| v0.2.0 release binary, option | MTP, depth 2 + confidence cut (`GLM_AX_DRAFT_DEPTH=2 GLM_AX_MTP_FLOOR=0.7`) | 35.7 | 38.2 | 42.4 | 42.7 | - |
| v0.2.0 source (tf33) as first benchmarked | MTP, depth 1 | 36.5 | 36.6 | 38.6 | 38.5 | 33.0 |
| v0.2.0 source (tf33) as first benchmarked, option | MTP, depth 2 + confidence cut | 35.1 | 36.4 | 40.1 | 42.0 | 32.6 |
| earlier build (tf32), option | DSpark (`GLM_AX_DSPARK=<dir>`, `GLM_AX_DSPARK_FLOOR=0.5`) | 30.9 | 32.4 | 43.0 | 49.8 | - |
| v0.1.0 binary (tf16) | MTP, depth 1 | 36.3 | 37.4 | 40.2 | 35.1 | - |
| tf16 as first benchmarked (fused unembed on) | MTP, depth 1 | 35.8 | 37.1 | 39.3 | 34.0 | - |
| tf15 | MTP, depth 1 | 35.4 | 37.3 | 39.5 | 33.7 | 33.3 |
| tf15 | none | 29.4 | 30.0 | 29.9 | - | - |

- The release-binary rows are medians of 2 runs (`RUNS=2`), the others of 3.
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

| Prompt | v0.2.1 release binary, no drafts | v0.2.1 (tf39), no drafts | v0.2.1 (tf39), MTP server | v0.2.0 (tf33), no drafts | v0.2.0 release binary, MTP server | same family on v1.3.5 without the prefill changes | v0.1.0 (tf16), no drafts |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,284 tokens | **736 tok/s** | 735 tok/s (+4% / +15%) | 639 tok/s | 706 tok/s | 682 tok/s | 640 tok/s | 667 tok/s |
| 4,941 tokens | **786 tok/s** | 784 tok/s (+9% / +39%) | 732 tok/s | 718 tok/s | 700 tok/s | 566 tok/s | 593 tok/s |
| 19,611 tokens | **755 tok/s** | 754 tok/s (+7% / +46%) | 717 tok/s | 707 tok/s | 681 tok/s | 516 tok/s | 570 tok/s |

Percentages in the tf39 column are against v0.2.0 (no drafts) and against the v1.3.5 base column. In v0.2.1 the prompt's routed
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

v0.3.1 (tf48) gate:
- A greedy essay is byte-identical to the undrafted output with both the `prose` and the `code` preset; Canberra.
- Needles at 10%, 50% and 90% of about 18k tokens found; with the `long` preset, a needle in a 127,800-token prompt
  found (first token 210.3 s, decode 16.5 tok/s).

v0.3.1 release binary, run through the released `oneshot.sh` (`prose` preset): tarball and binary SHA-256 matched
the published ones, server up after about 55 s, Canberra, greedy essay byte-identical to undrafted, first token
0.36 s, decode 35.7 / 37.5 / 39.2 / 40.9 tok/s (prose T=0.6 / T=0 / code / edit, 2 runs).

v0.3.0 release binary: the one-shot `prose` run on the model (Canberra, decode 35.2 / 36.7 / 39.2 / 41.0 tok/s).

v0.2.1 release binary (the published asset, run on the model after publishing):
- SHA-256 matched; Canberra passed in all three server runs; a needle at 90% of a 21,125-token prompt was found
  (29.1 s).
- A greedy essay was byte-identical to tf39's undrafted output, at depth 1 and at the built-in default.

v0.2.1 source (tf39):
- Canberra; needles at about 6k and at 10% and 90% of about 20k tokens returned exactly; through the server, a
  needle at 90% of a 21,125-token prompt was found (29.7 s).
- Drafted greedy output equals undrafted, at depth 1 and at the built-in default.
- 8-layer decode-row probe at position 3,226: cosine 0.999994 against the MLX-checked milestone build.

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

## Follow-up turns and short prompts (v0.3.1)

A follow-up turn of a conversation no longer reprocesses the whole conversation:

| Case | Fresh first token | Follow-up first token | How |
|---|---:|---:|---|
| 32k prompt, `--prompt-cache-gib 24` | 49.5 s | 0.69 s | prompt-cache snapshot |
| 128k prompt, `long` preset | 202.6 s | 1.14 s | live reuse: the finished stream's caches stay allocated, no copy; output bit-identical to a fresh run |

- Live reuse is on by default (`GLM_AX_LIVE_REUSE=0` turns it off). Prompts of 65,536 tokens or more run in
  8192-row chunks with the last partial chunk in 1024-row pieces (`GLM_AX_TAIL_ROWS`, `GLM_AX_TAIL_FROM`), the same
  way on every run, so a follow-up reuses its prefix and stays bit-identical to the same prompt run fresh.
- A short request after a long stream takes over the parked allocation: first token 0.24-0.29 s instead of 0.97 s.
- A stream gets a whole-context cache only when at least 48 GiB stays reclaimable after it; otherwise prompt plus
  reply. The choice is logged at startup and for each stream.
- `GLM_AX_STEPS=1` (one prompt chunk a call, shortest-remaining-first) is opt-in and not measured.

Fresh short prompts (no cache, p50 first token):

| Prompt tokens | about 15 | 33 | 86 | 162 | 300 | 567 | 1,465 |
|---|---:|---:|---:|---:|---:|---:|---:|
| First token | 0.29 s | 375 ms | 451 ms | 534 ms | 734 ms | 1,007 ms | 2,061 ms |

Every prompt pass reads nearly all of the 256 experts' weights, because each prompt token routes to its own 8. So
about 0.3-0.37 s is the floor for a fresh prompt on one Mac, however short it is; beyond that, each further
prompt token adds about 1.2 ms up to 1.5k tokens.

## Concurrency and context sweep

### Method

v0.3.0 build (tf43) on the MTP model folder, built-in MTP default (depth 2 + 0.7 cut), `GLM_AX_LOAD_LIMIT_GB=475`,
`--parallel 4 --prompt-cache-gib 0`, `--context` = the per-stream context shown, `GLM_AX_K8=1` for the 8-bit rows
and `GLM_AX_BATCH=1` (batched verify) or `0`. Each stream gets a fresh prompt of "notes" prose and a greedy answer
(1,200 new tokens at 32k, 3,000 at 90k, 3,500 at 128k). Decode rates are measured only in the window where all
streams decode at once. The prompts are prefilled one after another, so a later stream's first token includes the
earlier streams' prompts. Each stream was then also run alone; "greedy = alone" means the greedy output under load
was byte-identical to the same prompt alone. `GLM_AX_BATCH=1` is experimental.

### (a) Decode with several streams, tok/s

| Context a stream | Cache | Streams | Batched: total (per stream) | Unbatched: total (per stream) | Greedy = alone |
|---:|---|---:|---|---|---|
| 32k | bf16 | 1 | 32.0 | 32.0 | - |
| 32k | bf16 | 2 | **39.6** (18.1 / 21.5) | 33.3 (15.2 / 18.1) | 2/2 |
| 32k | bf16 | 4 | **43.0** (10.0-11.9) | 33.0 (7.7-9.1) | 4/4 |
| 32k | 8-bit | 1 | 31.1 | 31.1 | - |
| 32k | 8-bit | 2 | **38.6** (17.9 / 20.7) | 32.3 (15.0 / 17.3) | 2/2 |
| 32k | 8-bit | 4 | **42.6** (9.8-11.7) | 32.6 (7.5-9.0) | 4/4 |
| 90k (prompts 85.6-86.0k) | bf16 | 4 | **39.9** (9.0-11.6) | 29.4 (6.6-8.6) | 4/4 |
| 128k (prompts 127.1-127.6k) | 8-bit | 4 | **37.0** (8.3-10.7) | 26.6 (6.0-7.7) | 4/4 |
| 180k | 8-bit | 4 | pending | pending | pending |

90k is the largest bf16 context for 4 streams, 180k the largest 8-bit one, under this memory rule. Batched verify
gives +30-31% total at 4 streams at 32k and +36-39% at 90-128k; the greedy check passed in every batched run.

### (b) First token with several streams

| Context a stream | Cache | First token under load, 4 streams (s) | Alone (s) | Prefill alone | Decode alone (tok/s) |
|---:|---|---|---:|---:|---:|
| 32k (31.8-32.1k) | bf16 | 43.7 / 87.6 / 131.2 / 174.8 | 43.6-44.0 | about 730 tok/s | 32.0-34.7 |
| 32k | 8-bit | 44.6 / 89.2 / 133.9 / 179.6 | 44.4-45.3 | about 710 tok/s | 31.1-34.4 |
| 90k (85.6-86.0k) | bf16 | 133 / 261 / 390 / 521 | 127.5-128.1 | about 671 tok/s | 28.1-30.5 |
| 128k (127.1-127.6k) | 8-bit | 209 / 417 / 625 / 834 | 202.4-203.3 | about 627 tok/s | 25.3-28.3 |
| 180k | 8-bit | pending | pending | pending | pending |

Prefill alone = prompt tokens / first-token time alone. Under load, the n-th stream waits for the n-1 prompts
before it.

### (c) Memory

| Run | Peak wired | Lowest reclaimable |
|---|---:|---:|
| 90k bf16, 4 streams, batched | 439.5 GiB | 67.4 GiB |
| 90k bf16, 4 streams, unbatched | 440.7 GiB | 66.4 GiB |
| 180k 8-bit, 4 streams | pending | pending |

The memory sampler did not record in the 32k and 128k runs.

### Single stream against context

MTP depth 1 (`GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off`) unless noted; the "default" column is the built-in depth 2
+ 0.7 cut.

| Context | Prefill (first token) | Prose decode, depth 1 | Prose decode, default | Code decode, depth 1 / default | Build |
|---:|---|---:|---:|---:|---|
| 1.3k | 736 tok/s (undrafted) | 35.7 | 35.2 | 39.3 / 41.7 | v0.2.1 binary |
| 4.9k | 786 tok/s (undrafted) | - | - | - | v0.2.1 binary |
| 16-20k | 755 tok/s at 19.6k (undrafted) | 33.0 at 16k (tf33) | 31.8 at 16k (tf39) | - | v0.2.1 binary / tf33 / tf39 |
| 39k | 732 tok/s (in-process) | - | - | - | tf39 |
| 63k | 647 tok/s (98 s) | - | - | - | tf39 |
| 78k | 677 tok/s (in-process) | - | - | - | tf39 |
| 128k | 617 tok/s (207 s); 622 tok/s (205 s) in the decode run | 26.0 | 24.4 | 28.0 / 30.9 | v0.2.1 binary |
| 256k | 534 tok/s (478 s) | 21.3 | 20.3 | - | v0.2.1 binary |
| 375k (bf16) | 479 tok/s (784 s) | 20.3 | - | - | tf42 |
| 1M (8-bit) | 270 tok/s (3,802 s) | 9.6 (7.4 without the MTP head) | - | - | tf43 |

First tokens of fresh long prompts are compute-bound: about 650-780 tok/s up to 80k, falling to about 530 tok/s at
256k and 270 tok/s at 1M, so a 128k prompt takes about 3.5 minutes. Several fresh long prompts queue behind one
another (the tables above). A follow-up turn in the same conversation does not need to reprocess the whole prompt:
that is the prompt cache's job (work in progress; it gave about 0.34 s follow-up first tokens on an earlier build,
but all runs here were measured with it off).

## Memory

**Peak wired memory 415 GiB at a 77k-token prompt (78k context, `GLM_AX_LOAD_LIMIT_GB=440`); needs about 420 GB of
free unified memory and a raised GPU wired limit.**

Measured on v0.2.1 (tf39, built-in MTP default, `--context 78000 --prompt-cache-gib 0`, 8192-row prompt chunks),
sampling `vm_stat` every second:

| Point | Wired |
|---|---:|
| after load (394.8 GiB weights incl. the MTP head + 7.1 GiB caches for 78k context) | 408.4 GiB |
| after a 71,144-token prompt | 414.5 GiB |
| after a 76,687-token prompt | 409.6 GiB |
| peak | 415.3 GiB |

- Prompt processing adds about 7 GiB of scratch on top of weights and caches. The compressor stayed flat at
  0.72 GiB and free memory never went below 76.3 GiB on the 512 GB machine.
- A needle at 50% depth was found at both lengths: 110.7 s for 71,144 tokens (about 643 tok/s) and 117.0 s for
  76,687 tokens (about 655 tok/s), including the MTP layer's pass over the prompt.
- The KV and index caches take about 96.6 KB a token. With `GLM_AX_LOAD_LIMIT_GB=440` the largest `--context` the
  server accepts is 78,601 tokens; longer contexts need a higher limit (470 for 256k, see below).
- Weights: 389.6 GiB without the MTP head, 394.8 GiB with it. At the 32k context used for the benchmarks, the
  caches took 11-17 GiB on earlier builds.
- The Mac's GPU wired-memory limit has to be raised above the default, for example:

  ```sh
  sudo sysctl iogpu.wired_limit_mb=500000
  ```

  This lets the GPU wire almost all of the machine's memory. Do it at your own risk: with too little left for
  macOS, the machine can stall or reboot. Close other large programs first, and a memory watchdog (a script that
  stops the server when free memory drops below a few GB) is recommended. The setting resets at reboot.
- `GLM_AX_LOAD_LIMIT_GB=440` raises the engine's own load limit (default 75% of RAM, too low for this model).
- A 512 GB Mac is required in practice: about 420 GB free for a 78k context, about 440 GB for 256k. Loading
  takes about 40 s from a warm file cache.

### Long context (up to 256k tokens)

**Up to 256k tokens of context with `GLM_AX_LOAD_LIMIT_GB=470` and `--context 262144`; peak wired 432 GiB at a
255k prompt (about 440 GB free needed); prefill 617 tok/s at 128k, 534 at 255k.**

Same v0.2.1 build and binary, only the flags change:

```sh
GLM_AX_LOAD_LIMIT_GB=470 GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off \
  ./zig-out/native/bin/tensorfold-native serve /path/to/GLM-5.3-4bit-MTP \
  --name glm53 --host 127.0.0.1 --port 8020 \
  --context 262144 --parallel 1 --reasoning-effort low --no-update-check
```

Measured with `--prompt-cache-gib 0`. After load: 394.8 GiB of weights plus 35.0 GiB of caches. A needle at 50%
depth was found at each length:

| Prompt | First token | Prompt rate | Wired after the prompt |
|---:|---:|---:|---:|
| 63,355 tokens | 98.0 s | 647 tok/s | 414.2 GiB |
| 127,800 tokens | 207.1 s | 617 tok/s | 420.0 GiB |
| 255,439 tokens | 478.4 s | 534 tok/s | 431.5 GiB |

Peak wired memory was 431.7 GiB; the compressor stayed flat at 0.71 GiB, free memory never went below 60.9 GiB,
and the whole machine stayed under 500 GB.

#### Decode at long context

Same binary and flags (`GLM_AX_LOAD_LIMIT_GB=470`, `--context 262144`, `--parallel 1`), prompt cache 16 GiB,
400-token answers, tok/s:

| Context | Drafts | Prose, T=0.6 | Prose, greedy | Code, greedy |
|---:|---|---:|---:|---:|
| 127.5k | MTP depth 1 (`GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off`) | 26.0 | 25.2 | 28.0 |
| 127.5k | built-in default (depth 2 + cut 0.7) | 24.4 | 25.2 | 30.9 |
| 254.7k | MTP depth 1 | 21.3 | 20.9 | not run |
| 254.7k | built-in default | 20.3 | 20.8 | not run |

- First token: 205 s at 127.5k (622 tok/s), 476 s at 254.7k (535 tok/s). Peak wired memory during this run
  431.7 GiB, compressor 0.71 GiB.
- The 2-token verify pass grows from about 46 ms at short context to about 65 ms at 128k and 79 ms at 256k, about
  0.13 ms per 1,000 tokens of context, from the DSA indexer's per-row scoring and top-k.
- At long context, use depth 1 for prose and the built-in default for code.
- Code decode at 256k has not been measured.

### bf16 cache up to 393k tokens (with MTP)

`GLM_AX_LOAD_LIMIT_GB=488 GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off`, `--context 393216 --prompt-cache-gib 0`
(the largest bf16 context that keeps the machine under the memory rule used here), on the MTP model folder: a
needle at 50% depth of a 375,314-token prompt was found; first token 784 s (about 479 tok/s); decode 20.3 tok/s
with MTP depth 1; peak wired 442.6 GiB. Measured on tf42 (v0.2.1 plus the parallel top-k and the 1M
`max_context` of v0.3.0).

## 1M context (8-bit KV cache)

GLM-5.3's full 1,048,576-token window fits on a 512 GB M5 Ultra with the 8-bit KV cache, with the MTP head on.
The exact command of the measured run (paths replaced):

```sh
GLM_AX_K8=1 GLM_AX_PROMPT_ROWS=1024 GLM_AX_LOAD_LIMIT_GB=495 GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off \
  ./tensorfold-native serve /path/to/GLM-5.3-4bit-MTP \
  --name glm53 --host 127.0.0.1 --port 8020 \
  --context 1048576 --parallel 1 --reasoning-effort low --prompt-cache-gib 0 --no-update-check
```

`/path/to/GLM-5.3-4bit-MTP` is the merged 4-bit + MTP folder (see [Get the weights](#get-the-weights)).
`./oneshot.sh --preset 1m` runs exactly this.

Measured (v0.3.0 source, tf43), with and without the MTP head:

| | `1m`: MTP depth 1, 1024-row prompt chunks | `1m-nomtp`: no MTP head, 2048-row prompt chunks |
|---|---|---|
| Prompt | 1,026,813 tokens, needle at 50% depth found ("The secret code is 4417-COBALT-29", plus an accurate description of the text) | the same prompt, needle found |
| First token | 3,802 s (63 min) | 3,495 s (58.3 min, about 294 tok/s) |
| Decode at 1M | **9.6 tok/s** (verify pass 172 ms, first-draft acceptance 118/174) | 7.4 tok/s (134.6 ms a step) |
| After load | 394.8 GiB weights + 63.0 GiB caches | 389.6 GiB weights + 67.5 GiB caches |
| Peak wired | 453.4 GiB | 449.1 GiB |
| Compressor / lowest reclaimable | 0.11 GiB / 54.1 GiB (the memory guard never fired) | 0.49 GiB / about 50 GiB |

The no-MTP variant (`./oneshot.sh --preset 1m-nomtp`) serves the plain 4-bit pack folder:

```sh
GLM_AX_K8=1 GLM_AX_PROMPT_ROWS=2048 GLM_AX_LOAD_LIMIT_GB=495 GLM_AX_DRAFT_DEPTH=1 GLM_AX_MTP_FLOOR=off \
  ./tensorfold-native serve /path/to/GLM-5.3-4bit \
  --name glm53 --host 127.0.0.1 --port 8020 \
  --context 1048576 --parallel 1 --reasoning-effort low --prompt-cache-gib 0 --no-drafts --no-update-check
```

(`--no-drafts` is required with the plain pack; the two draft settings have no effect there and are kept because
that run used them.)

Requirements: a 512 GB Mac, the GPU wired limit raised (`sudo sysctl iogpu.wired_limit_mb=500000`, at your own
risk), and nothing else large running.

**Memory headroom at 1M with the MTP head is thin.** The lowest reclaimable memory was 54.1 GiB in the run above,
but two later runs fell to 43-44.5 GiB, depending on what else was in memory. Close other apps before starting it.
A safer option is coming.

`GLM_AX_PROMPT_ROWS` shrinks the prompt-time scratch (default 8192 rows). With the MTP head, 1024-row chunks keep
the 1M prompt inside memory; an earlier estimate that the head would not fit at 1M was too conservative, and it
was measured above. Smaller chunks slow prompt processing somewhat (63 against 58 minutes here).

### 8-bit cache quality

`GLM_AX_K8=1` stores the MLA latent cache as 576 int8 codes plus 9 half-precision scales a row (8 groups of 64
latent values, 1 group for the 64 roped dimensions, quantised after the rope) and the indexer keys as 128 int8 codes
plus 2 half scales: 49.8 KB a token against 96.6 KB in bf16. `GLM_AX_K8=lat` or `=idx` quantises only one of the
two caches.

| Check | 8-bit | bf16 |
|---|---:|---:|
| Teacher-forced next-token matches with real text, 3k-token prompt | 774 | 781 |
| Same, 20k-token prompt | 4,812 | 4,786 |
| Top-1 agreement with bf16 (random-word prompt) | about 80% | 80.19% between two exact bf16 attention kernels |
| Prefill at 39k tokens | 688.9 tok/s | 699.9 tok/s |
| Short-context decode: prose / greedy / code (tok/s) | 34.3 / 35.8 / 41.4 | 34.5 / 36.6 / 41.6 |

- Per-layer attention error about 0.5% (2.2% after 8 layers); the indexer selects identical keys.
- Known answers identical; needles 5/5 at about 18k and 71k tokens.
- The top-1 agreement on the random-word prompt is the same as swapping between two exact bf16 attention kernels,
  i.e. at noise level.

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

The [v0.3.1-glm53-reuse release](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.3.1-glm53-reuse)
has `tensorfold-native` built from this source (`zig build native`, fast mode, Zig 0.17.0, Xcode 27.2 Metal
toolchain) as a .tar.gz with its SHA-256, plus the licence files. It needs macOS 27.2.1 or later (the SDK it was
built with). It is ad-hoc signed, not notarized: after downloading, run
`xattr -d com.apple.quarantine tensorfold-native`. No weights are included. `oneshot.sh` is attached to the same
release. **The published v0.3.1 binary was verified on the model** through the released `oneshot.sh` (`prose`
preset; see Results). The
[v0.1.0](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.1.0-glm53-mtp), [v0.2.0](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.2.0-glm53-prefill), [v0.2.1](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.2.1-glm53-prefill) and
[v0.3.0](https://github.com/drowzeys/keys-Apple-M5U-TF-Full-GLM5.3-MLX-4bit/releases/tag/v0.3.0-glm53-1m) releases stay available, with their sources at their tags; those binaries were run
on the model after publishing.

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
| `GLM_AX_LOAD_LIMIT_GB` | load limit in GB (default 75% of RAM); 440 for the 32k benchmarks, 470 for 256k, 488 for 393k (bf16), 495 for 1M (8-bit) |
| `GLM_AX_K8` | `1`: 8-bit latent and index caches (49.8 KB a token instead of 96.6 KB); `lat` or `idx`: only one of them; unset: bf16 |
| `GLM_AX_LIVE_REUSE=0` | turn off live reuse of a finished stream's caches for a follow-up (on by default since v0.3.1) |
| `GLM_AX_TAIL_ROWS`, `GLM_AX_TAIL_FROM` | hybrid prompt chunks: piece size for the last partial chunk (1024) and the prompt length from which it applies (65,536) |
| `GLM_AX_STEPS=1` | one prompt chunk a call with shortest-remaining-first ordering; opt-in, not measured |
| `GLM_AX_TP_RANKS=2` | **experimental**: the two-rank tensor-parallel harness (`tf-glm-ax-tp`); not used by the server presets |
| `GLM_AX_PROMPT_ROWS` | prompt chunk rows, 1024-8192 in steps of 64 (default 8192); smaller means less prompt scratch memory (2048 for 1M) |
| `GLM_AX_BATCH=1` | **experimental**: batched verify across streams for `--parallel` > 1 (+30-39% total throughput at 4 streams in the [concurrency sweep](#concurrency-and-context-sweep); greedy output under load equal to running alone in every batched run) |
| `GLM_AX_DUAL=1` | **experimental**, with `GLM_AX_DSPARK`: each stream's rounds use DSpark or the MTP head, whichever measures faster; exact, but slower than MTP alone on prose |
| `GLM_AX_DECODE_NAX=1` | **untested, do not use**: tensor-unit attention for decode rows |
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

- One stream at a time by default (`--parallel 1`); multi-stream batching (`GLM_AX_BATCH=1`) is experimental.
- Answers (needles) and memory were checked up to 1,026,813 tokens (8-bit cache, with and without MTP) and 375k
  (bf16, MTP). Code decode at 256k and at 1M was not run. Decode slows with context (about 26 tok/s for prose at
  128k, 21 at 256k, 9.6 at 1M with MTP, 7.4 without).
- At 1M the first token takes about an hour.
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
- Two-Mac TP=2 recipe (`tp2/`): [MLX](https://github.com/ml-explore/mlx) and mlx-lm (MIT; `mx.distributed` with
  the `jaccl` Thunderbolt RDMA backend), [oMLX](https://github.com/jundot/omlx) 0.8.0rc1 (Apache-2.0; its GLM-5.3
  model code, installed as a dependency, and the sources the 8-bit cache, the prefill kernel and the two
  run-time-length helper kernels in `tp2/glm_tp/lc/` are derived from; Copyright 2026 Apple Inc. on that kernel
  file),
  [machd](https://github.com/machddev/machd) (MIT; the `MLX_METAL_FAST_SYNCH=1` finding, no code), CPython via
  python-build-standalone, tensor parallelism after Megatron-LM (Shoeybi et al., 2019). The release bundle carries
  every component's licence (`THIRD_PARTY_NOTICES.md`, `LICENSES/components/`).
- The `merge_mtp_head.py` key names follow the `mtp.*` layout that oMLX's GLM loader also uses.
