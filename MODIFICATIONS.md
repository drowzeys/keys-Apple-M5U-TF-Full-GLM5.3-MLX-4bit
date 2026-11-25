# Modifications to TensorFold

This repository is TensorFold (https://github.com/ashhart/TensorFold, Apache-2.0) at upstream tag `v1.3.5`
(commit `02b324ae9767253d9911d753d5c835a5070eed1b`), with the changes below. Everything else is upstream's,
unchanged. Upstream's history is not included: each release of this repository is a snapshot commit.

| Release here | Upstream base | glm_ax build |
|---|---|---|
| v0.2.1-glm53-prefill | v1.3.5 (`02b324ae`) | tf39: 8192-row prompt chunks, fused gate/up/SwiGLU prompt experts; MTP default depth 2 + cut 0.7 |
| v0.2.0-glm53-prefill | v1.3.5 (`02b324ae`) | tf33: prefill changes, DFlash2/DSpark/confidence cut opt-in |
| v0.1.0-glm53-mtp | `d30325dc` (1.3.4) | tf16 |

## New files (the `glm_ax` family)

| Path | What it is |
|---|---|
| `zig/src/families/glm_ax/*.zig` | Full GLM-5.3 (`glm_moe_dsa`) on MLX affine 4-bit weights, one Mac: config, weight loading, caches, the forward pass, the draft (MTP) window logic and its confidence cut, prompt-cache snapshots, the load-limit budget, the optional residency set |
| `zig/src/families/glm_ax/dflash2.zig` | Opt-in DFlash2 drafter (`GLM_AX_DFLASH2=<dir>`), ported from TensorFold's MiniMax-M3 DFlash2 drafter |
| `zig/src/families/glm_ax/dspark.zig` | Opt-in DSpark drafter (`GLM_AX_DSPARK=<dir>`) |
| `zig/src/native/glm_ax.zig` | The family's entry point for TensorFold's native server (lane host) |
| `zig/kernels/metal/glm_ax/glm_ax.metal` | Norms, MLA in latent space over each row's key list, the DSA indexer, fused absorb + rope for decode rows |
| `zig/kernels/metal/glm_ax/glm_ax_sparse.metal` | Sparse MLA and the DSA indexer on the M5 tensor units, and the split/combine decode path |
| `zig/kernels/metal/glm_ax/glm_ax_mpp.metal` | Prompt-chunk sparse MLA with MetalPerformancePrimitives cooperative tensors (default since v0.2.0; `GLM_AX_NO_MPP=1` falls back to `glm_ax_sparse.metal`) |
| `zig/kernels/metal/glm_ax/glm_ax_moe.metal` | Fused decode MoE (opt-in, `GLM_AX_MOE_FUSED=1`) |
| `zig/kernels/metal/glm_ax/glm_ax_dflash2.metal` | The DFlash2 and DSpark drafters' kernels and the target's taps for them |
| `zig/tests/glm_ax_run.zig` | `tf-glm-ax-run`: the model without the server (prompt, greedy reply, timings, layer probes) |
| `zig/tests/glm_ax_kcheck.zig` | `tf-glm-ax-kcheck`: attention kernels on synthetic keys |
| `zig/tests/glm_ax_drawcheck.zig` | `tf-glm-ax-drawcheck`: the bounded sampler against the general one, CPU only |
| `glm53/bench/` | The client-side benchmark scripts used for the numbers in README.md |
| `glm53/tools/merge_mtp_head.py` | Builds the model folder with the MTP head next to the 4-bit pack (symlinks, no copy) |
| `README.md`, `MODIFICATIONS.md` | This repository's documentation |

### Where the code comes from

- The glm_ax Metal kernels were written by this repository's author, derived from TensorFold's own GLM-5.3-Flash
  kernels: `zig/kernels/metal/glm_sparse_nax.metal` (sparse MLA on the tensor units, 512-wide keys; extended here
  with the 64-wide rope part to 576-wide keys) and `zig/kernels/metal/glm_index_nax.metal` (the DSA indexer's M5
  kernel), including their split and combine decode path. They use TensorFold's `nax.h` helpers, and the forward
  pass uses TensorFold's core `row_projection`, `affine_mm` and `moe_route` code and the step5 family's glue
  kernels.
- `glm_ax_mpp.metal` contains code adapted from **Splash** (https://github.com/incoai/splash, commit `d23cfa1f`,
  Apache-2.0): `runtime/metal/kernels/shared/listed_attention.metal`, `attention_bf16_listed_kv1_g64`. That kernel
  ports the method of **oMLX** 0.8.0rc1's tensor-unit sparse MLA attention (https://github.com/jundot/omlx,
  `omlx/patches/glm_moe_dsa/sparse_mla_nax.py`, Apache-2.0). Here it was changed to read glm_ax's 576-wide cache
  rows in place through each row's key list; the file's header names both sources.
- The DFlash2 and DSpark drafter ports reuse TensorFold's MiniMax-M3 drafter kernels and `lanes.drafter` contract.
  The DFlash2 drafter follows the semantics of SGLang's `DFlash2DraftModel` (Apache-2.0); the DSpark drafter
  follows RedHat AI's speculators (`DSparkDraftModel`) and vLLM's `qwen3_dspark` / dspark speculator code (both
  Apache-2.0). Those projects were read for semantics only; no source text from them is included.
- The prompt's routed experts use TensorFold's core `affine_mm.gatherGlu` (fused gate/up/SwiGLU), as upstream's
  GLM-5.3-Flash family does. An opt-in path (`GLM_AX_GROUPS=1`) reuses that family's `glm/mm_groups.zig`
  column-group kernels; it is experimental and currently gives wrong output with glm_ax.
- The model's math (MLA with weight absorption, the DSA indexer, sigmoid routing with the correction bias, the MTP
  block) follows the published GLM-5.3 architecture.

## Changed upstream files

| Path | Change |
|---|---|
| `zig/src/tensorfold.zig` | exports `glm_ax` |
| `zig/kernels/metal/sources.zig` | embeds the five glm_ax Metal sources |
| `zig/src/native/metal.zig` | `glm_moe_dsa` also accepts `mlx-q4g64`; an MLX-format GLM-5.3 folder opens with `glm_ax`, an exlr8 one still opens with upstream's `glm_dsa`; `metal4 = true` compiles the Metal 4 queue into the wrapper (glm_ax uses it only with `GLM_AX_METAL4=1`, other families keep Metal 3) |
| `zig/src/metal/metal4.zig` | opt-in diagnostics (`TF_METAL4_STATS=1`), and the queue's wait polls the shared event every 20 us for up to `TF_METAL4_POLL_MS` (default 200) before its blocking wait |
| `zig/build/exlr8.zig` | adds the `tf-glm-ax-run`, `tf-glm-ax-kcheck` and `tf-glm-ax-drawcheck` build steps |
| `zig/src/families/glm_dsa/kernels.zig` | the DSA indexer's blocked scan scores 256 keys a threadgroup (128 threads x 2 keys), so it now dispatches `last_pos / 256 + 1` threadgroups instead of `last_pos / 512 + 1` |
| `README.md` -> `README-TensorFold.md` | upstream's README, unchanged, under a new name |
| `zig/build/release_version.zig`, `.github/workflows/app.yml` | point at `README-TensorFold.md` instead of `README.md` |
| `NOTICE` | a section at the top describing this copy and its credits |
