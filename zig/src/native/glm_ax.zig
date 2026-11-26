//! GLM-5.3 on MLX affine weights served by the lane host on one Mac, contexts as far as its cache budget reaches.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const s5 = tf.glm_ax;
const lanes = tf.lanes;
const Resident = @import("resident.zig").Resident;

const Host = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *s5.Model,
    metal: *s5.backend.Metal,
    warm: mtl.keepalive.Target = undefined, // the residency set's, or the bare queue's, for the lane host's idle ticker
    resident: ?Resident = null, // weights and scratch kept wired between requests, inside the load limit
    limit: ?u64 = null, // GLM_AX_LOAD_LIMIT_GB's bytes, when set: the residency set's ceiling
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    /// GLM_AX_DFLASH2=<dir>: incoai's DFlash2 drafter, and the lane engine's wrapper that drafts with it.
    drafter: ?*s5.dflash2.Drafter = null,
    /// GLM_AX_DSPARK=<dir>: RedHat's DSpark drafter instead (its confidence head's chances: GLM_AX_DSPARK_FLOOR).
    dspark: ?*s5.dspark.Drafter = null,
    wrap: ?lanes.drafted.Drafted = null,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states, in what the streams' caches leave of the budget
    cache_cap: u64 = 0, // --prompt-cache-gib's bytes (the budget when unset)

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        if (h.wrap) |*w| w.deinit();
        if (h.drafter) |d| d.deinit();
        if (h.dspark) |d| d.deinit();
        h.config.deinit(h.gpa);
        s5.residency.drop(h);
        h.metal.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (try o.slideRefusal(a, "Macs")) |why| {
        problem.* = why;
        return null;
    }
    if (std.c.getenv("GLM_AX_TP_RANKS")) |v| if (!std.mem.eql(u8, std.mem.span(v), "1")) {
        // tensor parallel runs only under the two-rank harness so far (tf-glm-ax-tp)
        problem.* = "GLM_AX_TP_RANKS is set: the glm_ax engine serves one Mac; tensor parallel runs only in tf-glm-ax-tp for now (unset it)";
        return null;
    };
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = s5.Model.load(gpa, io, o.dir, .{ .mtp = o.drafts, .require_mtp = o.drafts }) catch |err| {
        problem.* = if (err == error.GlmMtpHeadMissing) try std.fmt.allocPrint(a, "drafts are on but {s} has no GLM-5.3 MTP head under mtp.* names: serve it with --no-drafts, or the folder that holds the head", .{o.dir}) else try std.fmt.allocPrint(a, "the native GLM-5.3 (MLX affine) engine cannot load this checkpoint ({s})", .{@errorName(err)});
        return null;
    };
    errdefer model.deinit();
    var ram: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) ram = 0;
    const text: ?[]const u8 = if (std.c.getenv("GLM_AX_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const c = &model.config; // the decode window's and the prompt chunk's buffers, scores sized for the context
    const want: usize = @as(usize, @intCast(@max(1, @min(o.context orelse 16384, s5.config.max_context)))) + s5.state.window_rows + 2;
    const scratch: u64 = s5.state.scratchBytes(c, want, s5.state.window_rows) + s5.state.scratchBytes(c, want, s5.state.prompt_rows);
    const budget = s5.budget.caches(ram, model.device.maxWorkingSet(), text, model.checkpoint.residentBytes() + scratch) catch {
        model.deinit();
        problem.* = "GLM_AX_LOAD_LIMIT_GB is not a load limit: give a number of GB above 0 and at most this Mac's RAM, or unset it for 75% of RAM";
        return null;
    };
    const fits = s5.budget.longest(&model.config, budget, s5.config.max_context);
    const window: i64 = o.context orelse @intCast(@min(fits, 16384));
    if (window < 1 or window > fits) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: this Mac's GLM-5.3 caches fit {d} tokens under its load limit (75% of RAM, or GLM_AX_LOAD_LIMIT_GB)", .{ window, fits });
        model.deinit(); // after the message, so a failed allocation leaves the model to the errdefer
        return null;
    }
    std.log.info("glm_ax: GLM-5.3 (MLX 4-bit) loaded in {d:.1} s, {d:.1} GiB of weights, {d:.1} GiB for caches, context up to {d} tokens, MTP head {s}", .{ model.load_seconds, @as(f64, @floatFromInt(model.checkpoint.residentBytes())) / (1 << 30), @as(f64, @floatFromInt(budget)) / (1 << 30), window, if (model.config.mtp > 0) "on" else "off" });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.io = io;
    h.model = model;
    h.resident = null;
    h.limit = if (text != null) (s5.budget.limit(ram, model.device.maxWorkingSet(), text) catch null) else null;
    h.metal = try s5.backend.Metal.init(gpa, model, .{ .capacity = want, .streams = @max(o.lanes, 1), .budget = budget });
    errdefer h.metal.deinit();
    try h.metal.warm();
    {
        const fullb = s5.state.cacheBytes(&model.config, want);
        const r = s5.backend.Metal.reclaimableBytes();
        std.log.info("glm_ax: allocation: a whole-context stream cache is {d:.1} GiB; reclaimable now {d:.1} GiB: streams get {s}", .{ @as(f64, @floatFromInt(fullb)) / (1 << 30), if (r) |x| @as(f64, @floatFromInt(x)) / (1 << 30) else -1.0, if (r) |x| (if (x >= fullb + (48 << 30)) "whole-context caches (live reuse in place)" else "prompt + reply caches (whole-context would leave under 48 GiB reclaimable)") else "prompt + reply caches" });
    }
    h.drafter = null;
    h.wrap = null;
    if (std.c.getenv("GLM_AX_DFLASH2")) |dir| if (o.drafts) {
        try h.metal.measureWindows();
        h.drafter = s5.dflash2.Drafter.init(gpa, io, model, std.mem.span(dir)) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "GLM_AX_DFLASH2={s}: the DFlash2 drafter cannot load ({s})", .{ std.mem.span(dir), @errorName(err) });
            h.metal.deinit();
            gpa.destroy(h);
            model.deinit();
            return null;
        };
        h.wrap = try lanes.drafted.Drafted.init(gpa, h.metal.backend(), h.drafter.?.drafter());
    };
    h.dspark = null;
    if (h.wrap == null) if (std.c.getenv("GLM_AX_DSPARK")) |dir| if (o.drafts) {
        try h.metal.measureWindows();
        h.dspark = s5.dspark.Drafter.init(gpa, io, model, std.mem.span(dir)) catch |err| {
            problem.* = try std.fmt.allocPrint(a, "GLM_AX_DSPARK={s}: the DSpark drafter cannot load ({s})", .{ std.mem.span(dir), @errorName(err) });
            h.metal.deinit();
            gpa.destroy(h);
            model.deinit();
            return null;
        };
        if (std.c.getenv("GLM_AX_DUAL")) |v| if (v[0] == '1') {
            h.dspark.?.mtp = h.metal;
            std.log.info("glm_ax: GLM_AX_DUAL=1: each stream's rounds drafted by DSpark or the MTP head, whichever measures more tokens a ms", .{});
        };
        h.wrap = try lanes.drafted.Drafted.init(gpa, h.metal.backend(), h.dspark.?.drafter());
    };
    errdefer if (h.wrap) |*w| w.deinit();
    errdefer if (h.drafter) |d| d.deinit();
    errdefer if (h.dspark) |d| d.deinit();
    const target_facts = h.metal.facts();
    h.config = try lanes.Config.init(gpa, if (h.wrap) |*w| w.facts(target_facts) else target_facts, s5.state.window_rows, s5.state.window_rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, if (h.wrap) |*w| w.backend() else h.metal.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = o.lanes, .context_window = @intCast(window), .prefill_step = @intCast(s5.state.prompt_rows), .decode_share = o.decode_share, .structures = true, .logprobs = true });
    // shortest-remaining-first among prompt passes and replies (GLM-5.3-Flash's default on one Mac), now that the
    // backend steps prompts a chunk a call (GLM_AX_STEPS); TENSORFOLD_PREFILL_ORDER still overrides
    if (std.c.getenv("TENSORFOLD_PREFILL_ORDER") == null and o.decode_share == null and std.c.getenv("TENSORFOLD_PREFILL_SHARE") == null and h.host.share.p >= 1)
        h.host.srpt = @TypeOf(h.host.srpt).withDefault(true);
    h.cache = null;
    h.cache_cap = if (o.prompt_cache_gib) |g| @intFromFloat(g * (1 << 30)) else budget;
    if (h.cache_cap > 0) {
        const M = s5.backend.Metal;
        // the MTP head keys row at-1 with token at; prompt chunks change arithmetic with their rows
        const rules: api.prompt_cache.Rules = .{ .lookahead = @intFromBool(model.config.mtp > 0), .planned = true, .grid = @intCast(s5.state.prompt_rows), .min_prompt = 0 };
        h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = h.metal, .vtable = &.{ .bytes = M.snapBytes, .save = M.snapSave, .restore = M.snapRestore, .drop = M.snapDrop } }, rules, @min(h.cache_cap, budget));
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        h.metal.room = .{ .ctx = h, .fit = fitCache };
    }
    errdefer if (h.cache) |*store| store.deinit();
    try s5.residency.hold(h);
    errdefer s5.residency.drop(h);
    // GLM_AX_NO_KEEPALIVE=1: no idle keepalive ticker (its probes are command buffers of their own on the GPU)
    if (std.c.getenv("GLM_AX_NO_KEEPALIVE") == null) h.host.keepalive_target = .{ .ctx = &h.warm, .tick = mtl.keepalive.Target.tick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// Metal.room's fit: the kept states' budget is what the streams leave, at most the cap; the least valuable go first.
fn fitCache(ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    const store = if (h.cache) |*owned| owned else return;
    store.budget = @min(h.cache_cap, bytes);
    const Left = struct {
        s: *const api.prompt_cache.Store,
        pub fn fits(l: @This()) bool {
            return l.s.held <= l.s.budget;
        }
    };
    _ = store.giveBack(restoring, prompt, Left{ .s = store });
}

