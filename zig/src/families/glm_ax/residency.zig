//! One initial inventory and a stable follow hook for every served GLM-5.3 cache and saved state.
const std = @import("std");
const mtl = @import("metal");
const Resident = mtl.resident.Resident;

/// Weights, routers and scratches in one residency set for the keepalive, inside the family's load limit (h.limit;
/// the 70%-of-RAM rule when unset): GLM-5.3's 391-397 GiB need GLM_AX_LOAD_LIMIT_GB past 70% of a 512 GB Mac.
pub fn hold(h: anytype) !void {
    // opt-in (GLM_AX_RESIDENCY=1): measured on a 512 GB M5 Ultra, holding the 391 GiB set did not change decode, and a
    // new prompt after an idle gap took 16.0 s against 8.4 s without it (4.9k tokens)
    if (std.c.getenv("GLM_AX_RESIDENCY") == null or std.c.getenv("GLM_AX_NO_RESIDENCY") != null) {
        h.resident = null;
        h.warm = .{ .queue = h.model.tickQueue() };
        return;
    }
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    const m = h.model;
    for (m.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer);
    for (m.weights.layers) |l| if (l.ffn == .moe) try list.appendSlice(h.gpa, &.{ l.ffn.moe.router, l.ffn.moe.bias });
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.scratch);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.prompt);
    h.resident = try Resident.holdUnder(h.gpa, h.io, m.device, m.queue, list.items, "glm_ax", h.limit);
    if (h.resident) |*r| h.metal.setResident(r.follow());
    h.warm = if (h.resident) |*r| r.target() else .{ .queue = m.tickQueue() };
}

pub fn drop(h: anytype) void {
    h.metal.setResident(null);
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.model.tickQueue() };
}
