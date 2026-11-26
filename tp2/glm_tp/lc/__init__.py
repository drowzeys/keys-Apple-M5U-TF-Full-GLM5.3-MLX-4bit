"""Prefill fixes for oMLX's GLM-5.3 code: hooks on module globals of omlx.patches.glm_moe_dsa."""


def install():
    from . import blockmask, prefill_nax, reduce

    return {"reduce": reduce.install(), "blockmask": blockmask.install(), "prefill": prefill_nax.install()}
