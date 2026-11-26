# Third-party notices: the TP=2 bundle

The TP=2 server and tools in `tp2/` are part of this repository (Apache-2.0, see `LICENSE` and `NOTICE`). The bundle
also contains, unmodified, the software below; each component's own licence files are in `LICENSES/components/`.

## Model code and runtime

- **MLX** and **mlx-metal** 0.32.4, Copyright (c) 2023 Apple Inc., MIT License. Arrays, Metal kernels, and the
  `jaccl` RDMA backend of `mx.distributed` that carries the all_sums between the two Macs.
- **mlx-lm** 0.32.3.dev8, commit `0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d` of
  https://github.com/ml-explore/mlx-lm, Copyright (c) 2023 Apple Inc., MIT License. Model loading, tokenizer
  wrapper, caches.
- **oMLX** 0.8.0rc1, https://github.com/jundot/omlx, Apache License 2.0. Its GLM-5.3 (`glm_moe_dsa`) model code,
  MTP head and native kernels are used as an installed dependency (the wheel is redistributed unmodified). At run
  time `tp2/glm_tp/kv8.py` recompiles oMLX's `GlmMoeDsaAttention.__call__` from its own source with four reads
  of the latent cache swapped (8-bit cache), and `tp2/glm_tp/lc/prefill_nax.py` builds a prefill kernel from the
  source of oMLX's `sparse_mla_nax.py` with a rope term added; both are derived works of oMLX under Apache-2.0.
  `tp2/glm_tp/model.py` follows the layout of oMLX's `Model.shard()` (with the fused expert gate/up split as two
  segments).
  `tp2/glm_tp/lc/blockmask.py` and `tp2/glm_tp/lc/reduce.py` contain Metal kernel source adapted from oMLX's
  `omlx/patches/glm_moe_dsa/sparse_mla.py` (file header "Copyright © 2026 Apple Inc.", shipped in oMLX 0.8.0rc1
  under the Apache License 2.0): `topk_indices_to_block_masks` and `fused_index_score_reduce`, changed to read the
  row and key counts from an input instead of the kernel template. Both files are derived works under Apache-2.0.
- **CPython** 3.11.14, Python Software Foundation License (`LICENSES/components/CPython-3.11.14-LICENSE.txt`,
  which also covers the libraries CPython bundles), as built by the python-build-standalone project
  (https://github.com/astral-sh/python-build-standalone, build scripts under MPL-2.0; no build scripts included).
- Python packages pulled in by the above, unmodified, each under its own licence (files in
  `LICENSES/components/<package>/`, copied from the wheels). The 39 wheels and their licences (from the wheels'
  metadata):

  | Licence | Packages |
  |---|---|
  | MIT | annotated-doc, annotated-types, anyio, filelock, h11, markdown-it-py, mdurl, mlx, mlx-lm, mlx-metal, pydantic, pydantic_core, PyYAML, rich, truststore, typer, typing-inspection |
  | BSD-3-Clause | click, fsspec, httpcore2, httpx2, idna, Jinja2, MarkupSafe, numpy (with bundled-library notices in its licence files), protobuf |
  | BSD-2-Clause | Pygments |
  | Apache-2.0 | hf-xet, huggingface_hub, omlx, safetensors, sentencepiece, tokenizers, transformers |
  | Apache-2.0 OR BSD-2-Clause | packaging |
  | Apache-2.0 AND CNRI-Python | regex |
  | MPL-2.0 AND MIT | tqdm (unmodified; source at https://github.com/tqdm/tqdm) |
  | ISC | shellingham |
  | PSF-2.0 | typing_extensions |

  The sentencepiece and tokenizers wheels carry no licence file of their own; both are Apache-2.0, whose text is
  `LICENSES/Apache-2.0.txt`.

## Methods and findings

- **machd** (https://github.com/machddev/machd, MIT): the finding that MLX's default GPU-to-CPU synchronisation
  costs ~0.2 ms per collective and that `MLX_METAL_FAST_SYNCH=1` removes it, which is what makes per-layer
  all_sums cheap on these Macs. No machd code is included.
- **TensorFold** (https://github.com/ashhart/TensorFold, Apache-2.0): this repository's base; the one-Mac recipe
  and the `glm53/tools/merge_mtp_head.py` merge tool used to attach the MTP layer.
- Tensor parallelism of transformer layers: Shoeybi et al., "Megatron-LM" (2019). Multi-token prediction:
  Gloeckle et al. (2024) and DeepSeek-V3 (2024); speculative decoding: Leviathan et al. (2023), Chen et al. (2023).

## Weights (not included)

- GLM-5.3 by Z.ai, GLM-5.3 License; mlx-community/GLM-5.3-4bit (revision 81ad9b73); the MTP layer from
  inferencerlabs/GLM-5.3-MTP-MLX (revision f1187efe). Read the licence on the model pages before use.
