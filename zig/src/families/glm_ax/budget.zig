//! What a GLM-5.3 (affine) load may hold: core's load budget over this family's caches.
const cfg = @import("config.zig");
const st = @import("state.zig");
const load = @import("../../core/load_budget.zig");

pub const caches = load.caches;
/// The load limit itself: 75% of RAM, or GLM_AX_LOAD_LIMIT_GB's value capped at the GPU's working set.
pub const limit = load.limit;

/// The longest context one stream's caches fit in `bytes`, at most `most`.
pub fn longest(c: *const cfg.Config, bytes: u64, most: usize) usize {
    return load.longest(c, st.cacheBytes, bytes, most);
}
