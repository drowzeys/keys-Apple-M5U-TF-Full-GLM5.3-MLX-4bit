# Modifications to TensorFold

This repository is TensorFold (https://github.com/ashhart/TensorFold, Apache-2.0) at upstream commit
`d30325dcb5422d3981ab6de6633e2ae0af6ad17e` (version 1.3.4, "changelog: #3906 under Unreleased"), with the changes
below. Everything else is upstream's, unchanged. Upstream's history is not included: this repository starts from a
single snapshot commit.

## New files (the `glm_ax` family)

| Path | What it is |
|---|---|
| `zig/src/families/glm_ax/*.zig` | Full GLM-5.3 (`glm_moe_dsa`) on MLX affine 4-bit weights, one Mac: config, weight loading, caches, the forward pass, the draft (MTP) window logic, prompt-cache snapshots, the load-limit budget and the optional residency set |
| `zig/src/native/glm_ax.zig` | The family's entry point for TensorFold's native server (lane host) |
| `zig/kernels/metal/glm_ax/glm_ax.metal` | Norms, MLA in latent space over each row's key list, the DSA indexer, fused absorb + rope for decode rows |
| `zig/kernels/metal/glm_ax/glm_ax_sparse.metal` | Sparse MLA and the DSA indexer on the M5 tensor units for prompt chunks, and the split/combine decode path |
| `zig/kernels/metal/glm_ax/glm_ax_moe.metal` | Fused decode MoE (eight routed experts plus the shared one in two launches; opt-in, `GLM_AX_MOE_FUSED=1`) |
| `zig/tests/glm_ax_run.zig` | `tf-glm-ax-run`: the model without the server (prompt, greedy reply, timings, layer probes) |
| `zig/tests/glm_ax_kcheck.zig` | `tf-glm-ax-kcheck`: attention kernels on synthetic keys, tensor-unit kernel against the decode kernels |
| `zig/tests/glm_ax_drawcheck.zig` | `tf-glm-ax-drawcheck`: the bounded sampler against the general one, CPU only |
| `glm53/bench/` | The client-side benchmark scripts used for the numbers in README.md |
| `glm53/tools/merge_mtp_head.py` | Builds the model folder with the MTP head next to the 4-bit pack (symlinks, no copy) |
| `README.md`, `MODIFICATIONS.md` | This repository's documentation |

### Where the kernels come from

The glm_ax Metal kernels were written by this repository's author by deriving them from TensorFold's own
GLM-5.3-Flash kernels: `zig/kernels/metal/glm_sparse_nax.metal` (sparse MLA on the tensor units, 512-wide keys;
extended here with the 64-wide rope part to 576-wide keys) and `zig/kernels/metal/glm_index_nax.metal` (the DSA
indexer's M5 kernel), including their split and combine decode path. They use TensorFold's `nax.h` tensor-unit
helpers, and the forward pass uses TensorFold's core `row_projection`, `affine_mm` and `moe_route` code and the
step5 family's glue kernels. The model's math (MLA with weight absorption, the DSA indexer, sigmoid routing with
the correction bias, the MTP block) follows the published GLM-5.3 architecture. No code from oMLX, mlx-lm,
vllm-metal or other projects was copied into these files.

## Changed upstream files

| Path | Change |
|---|---|
| `zig/src/tensorfold.zig` | exports `glm_ax` |
| `zig/kernels/metal/sources.zig` | embeds the three glm_ax Metal sources |
| `zig/src/native/metal.zig` | `glm_moe_dsa` also accepts `mlx-q4g64`; an MLX-format GLM-5.3 folder opens with `glm_ax`, an exlr8 one still opens with upstream's `glm_dsa` |
| `zig/build/exlr8.zig` | adds the `tf-glm-ax-run`, `tf-glm-ax-kcheck` and `tf-glm-ax-drawcheck` build steps |
| `zig/src/families/glm_dsa/kernels.zig` | the DSA indexer's blocked scan scores 256 keys a threadgroup (128 threads x 2 keys), so it now dispatches `last_pos / 256 + 1` threadgroups instead of `last_pos / 512 + 1` |
| `README.md` -> `README-TensorFold.md` | upstream's README, unchanged, under a new name |
| `zig/build/release_version.zig`, `.github/workflows/app.yml` | point at `README-TensorFold.md` instead of `README.md` |
| `NOTICE` | a section at the top describing this copy |
