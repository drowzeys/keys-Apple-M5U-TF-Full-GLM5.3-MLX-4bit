"""Switches for the prefill fixes (all on by default)."""
import os


def _flag(name, default):
    return os.environ.get(name, "1" if default else "0") == "1"


LC_REDUCE = _flag("TP2_LC_REDUCE", True)    # index-score reduce and block masks without a kernel per context length
PF_ATTN = _flag("TP2_PF_ATTN", True)        # tensor-unit sparse attention for prefill chunks
PF_INDEX = _flag("TP2_PF_INDEX", True)      # tensor-unit index scores for prefill chunks
PF_MIN_ROWS = int(os.environ.get("TP2_PF_MIN_ROWS", "16"))
