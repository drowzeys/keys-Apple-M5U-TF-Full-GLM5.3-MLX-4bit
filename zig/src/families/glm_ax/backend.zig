//! GLM-5.3 (affine) behind the lane engine's backend: windows of the pending token and the MTP layer's drafts, a cache a stream.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;
const snapshot = @import("snapshot.zig");
const draw = @import("draw.zig").draw;

fn now() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

fn secs(a: i128, b: i128) f64 {
    return @as(f64, @floatFromInt(b - a)) / 1e9;
}

/// A backend call's start for the round profile: the gap since this stream's last call ended.
fn profStart(c: *st.Cache) i128 {
    const t = now();
    if (c.prof.last_end != 0) c.prof.gap += secs(c.prof.last_end, t);
    return t;
}
const be = lanes.backend;

/// The longest context a stream may take, the streams at once, and the bytes every stream's caches may hold.
pub const Options = struct { capacity: usize, streams: u32 = 4, budget: u64 };

/// Drafts a window holds at most: one a MTP layer.
pub const max_drafts = 3;

/// GLM_AX_DRAFT_DEPTH (1-3): the most drafts a window holds, for depth trials; max_drafts when unset.
fn depthCap() usize {
    // default 2: the second draft offered only while the head's running chance holds (mtpFloor)
    const v = std.c.getenv("GLM_AX_DRAFT_DEPTH") orelse return 2;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return max_drafts;
    return @max(1, @min(n, max_drafts));
}

/// GLM_AX_MTP_FLOOR (0-1): the lane engine verifies a held draft past the first only while the head's running
/// chance (the product of its drafts' max probabilities) stays at or above it, and the head stops drafting there
/// (no second MTP pass for a draft that would not be verified); null when unset (every requested draft runs).
fn mtpFloor() ?f64 {
    // default 0.7 (measured: prose 36.6, greedy 38.2, code 42.4, edit 42.8 against depth 1's 35.8 / 37.1 / 38.9 /
    // 38.5); GLM_AX_MTP_FLOOR=off verifies every requested draft
    const v = std.c.getenv("GLM_AX_MTP_FLOOR") orelse return 0.7;
    if (std.mem.eql(u8, std.mem.span(v), "off")) return null;
    const f = std.fmt.parseFloat(f64, std.mem.span(v)) catch return null;
    return if (f >= 0 and f <= 1) f else null;
}

/// The probability of the max of bf16 logits `l` (softmax at T = 1).
fn topProbability(l: []const u16) f64 {
    var most: f32 = -std.math.inf(f32);
    for (l) |w| most = @max(most, logit(w));
    var sum: f64 = 0;
    for (l) |w| sum += @exp(@as(f64, logit(w) - most));
    return 1.0 / sum;
}

/// The prompt cache's room beside the streams: its kept states fit in `bytes`, never evicting `restoring`.
pub const Room = struct { ctx: *anyopaque, fit: *const fn (ctx: *anyopaque, bytes: u64, restoring: ?*anyopaque, prompt: []const u32) void };

pub const Metal = struct {
    gpa: std.mem.Allocator,
    model: *Model,
    options: Options,
    scratch: st.Scratch, // decode rounds
    prompt: st.Scratch, // prompt chunks
    caches: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Cache) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,
    held: u64 = 0, // bytes the streams' caches hold
    room: ?Room = null, // the prompt cache, which takes what the streams leave of the budget
    ahead: fwd.Pending = .{}, // the next decode row of `ahead_stream`, encoded while the GPU ran the last one
    ahead_stream: ?*lanes.Stream = null,
    /// A drafter's features: each stream's tapped rows at every position (prepare_features turns them on).
    tapping: bool = false,
    /// GLM_AX_BATCH: windows verified together, and their rows
    batched_rounds: usize = 0,
    batched_rows: usize = 0,
    taps: std.AutoHashMapUnmanaged(*st.Cache, mtl.Buffer) = .empty,
    /// GLM_AX_LIVE_REUSE (on unless "0"): the last finished stream's caches, kept live (no copy) for a follow-up whose
    /// prompt shares their prefix (the next turn of a conversation); any other request frees them first.
    parked: ?Parked = null,
    reused_rows: u64 = 0,

    const Parked = struct {
        cache: *st.Cache,
        tokens: []u32, // the token of each cache row (the prompt, then the reply's committed tokens)
        prompt_len: usize, // rows up to here came from prompt chunks on the grid (the rest from decode windows)
    };

    // the machine's reclaimable memory (free + inactive + speculative + purgeable pages), as the memory guard counts it
    const VmStats = extern struct {
        free_count: u32, active_count: u32, inactive_count: u32, wire_count: u32,
        zero_fill_count: u64, reactivations: u64, pageins: u64, pageouts: u64, faults: u64, cow_faults: u64,
        lookups: u64, hits: u64, purges: u64,
        purgeable_count: u32, speculative_count: u32,
        decompressions: u64, compressions: u64, swapins: u64, swapouts: u64,
        compressor_page_count: u32, throttled_count: u32, external_page_count: u32, internal_page_count: u32,
        total_uncompressed_pages_in_compressor: u64,
    };
    extern "c" fn mach_host_self() u32;
    extern "c" fn host_statistics64(host: u32, flavor: c_int, info: *VmStats, count: *u32) c_int;
    extern "c" var vm_page_size: usize;
    pub fn reclaimableBytes() ?u64 {
        var v: VmStats = undefined;
        var n: u32 = @sizeOf(VmStats) / 4;
        if (host_statistics64(mach_host_self(), 4, &v, &n) != 0) return null; // HOST_VM_INFO64
        const pages: u64 = @as(u64, v.free_count) + v.inactive_count + v.speculative_count + v.purgeable_count;
        return pages * vm_page_size;
    }
    /// A whole-context cache only when the machine keeps this much reclaimable after it (the guard's line is 45 GiB)
    const full_margin: u64 = 48 << 30;

    fn liveReuse() bool {
        const v = std.c.getenv("GLM_AX_LIVE_REUSE") orelse return true;
        return v[0] != '0';
    }

    fn freeParked(self: *Metal) void {
        const pk = self.parked orelse return;
        self.parked = null;
        self.held -= st.cacheBytes(&self.model.config, pk.cache.capacity);
        pk.cache.deinit();
        self.gpa.destroy(pk.cache);
        self.gpa.free(pk.tokens);
        if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
    }

    /// The parked caches' rows a new prompt can start from, or 0: its common prefix with their tokens, cut so each
    /// kept row is bit-identical to a fresh prefill's (rows from whole prompt chunks on the grid, the row after the
    /// last kept one's token matching for the MTP layer's entry, and at least one row left to feed).
    fn parkedRows(pk: Parked, ids: []const u32) usize {
        var lcp: usize = 0;
        const n = @min(pk.tokens.len, ids.len);
        while (lcp < n and pk.tokens[lcp] == ids[lcp]) lcp += 1;
        const most = @min(lcp -| 1, @min(pk.prompt_len -| 1, ids.len - 1));
        return sharedBoundary(most, pk.prompt_len, ids.len);
    }

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Metal {
        const out = try gpa.create(Metal);
        errdefer gpa.destroy(out);
        var scratch = try st.Scratch.init(model.device, &model.config, options.capacity, st.window_rows);
        errdefer scratch.deinit();
        out.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = scratch, .prompt = try st.Scratch.init(model.device, &model.config, options.capacity, st.prompt_rows) };
        return out;
    }

    pub fn deinit(self: *Metal) void {
        self.freeParked();
        self.setResident(null);
        var it = self.caches.valueIterator();
        while (it.next()) |cache| {
            cache.*.deinit();
            self.gpa.destroy(cache.*);
        }
        self.caches.deinit(self.gpa);
        var ti = self.taps.valueIterator();
        while (ti.next()) |x| x.deinit();
        self.taps.deinit(self.gpa);
        self.scratch.deinit();
        self.prompt.deinit();
        self.gpa.destroy(self);
    }

    pub fn setResident(self: *Metal, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| {
            mtl.resident.fields.apply(&self.scratch, f, false);
            mtl.resident.fields.apply(&self.prompt, f, false);
        }
        self.follow = hook;
        if (hook) |f| {
            mtl.resident.fields.apply(&self.scratch, f, true);
            mtl.resident.fields.apply(&self.prompt, f, true);
        }
        var it = self.caches.valueIterator();
        while (it.next()) |cache| cache.*.setResident(hook);
    }

    pub fn createCache(self: *Metal, capacity: usize) !*st.Cache {
        const cache = try self.gpa.create(st.Cache);
        errdefer self.gpa.destroy(cache);
        cache.* = try st.Cache.init(self.model.device, &self.model.config, capacity);
        cache.setResident(self.follow);
        return cache;
    }

    pub fn facts(self: *const Metal) lanes.Model {
        // rows are independent in every kernel of a decode window, so any width to 16 gives one-row steps' bits
        const mtp = self.model.config.mtp > 0;
        return .{ .exact_width = st.window_rows, .mtp = mtp, .speculate = mtp, .speculate_early = false, .drafts = if (mtp) @intCast(depthCap()) else 1, .max_streams = self.options.streams, .streams_exact = true, .hidden_rows = batchOn(), .plain_share = batchOn(), .batch_rows = if (batchOn()) st.window_rows else 0, .draft_confidence = if (mtp) mtpFloor() else null, .draft_probabilities = mtp and mtpFloor() != null };
    }

    pub fn backend(self: *Metal) be.Backend {
        if (stepsOn()) return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .prefill_step = prefillStepFn, .reports_fill = true, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn, .prepare_features = prepareFn, .features = featuresFn, .probabilities = probabilitiesFn } };
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn, .first_row = firstRowFn, .held = heldFn, .prepare_features = prepareFn, .features = featuresFn, .probabilities = probabilitiesFn } };
    }

    fn cast(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    fn cacheOf(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        return self.caches.get(stream) orelse error.UnknownGlmStream;
    }

    /// A tapped row's bytes: every tap's hidden row, bf16.
    pub fn tapBytes(self: *const Metal) usize {
        return @as(usize, self.model.taps) * self.model.config.hidden * 2;
    }

    /// The tapped rows' place for a pass starting at `start` (null: no drafter, or the cache has none).
    fn batchOn() bool {
        const v = std.c.getenv("GLM_AX_BATCH") orelse return false;
        return v[0] == '1';
    }

    fn tapsAt(self: *Metal, cache: *st.Cache, start: usize) ?fwd.B {
        if (!self.tapping) return null;
        const x = self.taps.get(cache) orelse return null;
        return .{ .buf = x, .off = start * self.tapBytes() };
    }

    /// The drafter's layers: every stream's passes keep their tapped rows from now on.
    fn prepareFn(ptr: *anyopaque, taps: []const u32) !void {
        const self = cast(ptr);
        try self.model.setTaps(taps);
        self.tapping = true;
    }

    /// Rows [start, start + count) of the stream's tapped rows, on the device.
    fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (taps.len != self.model.taps or !self.tapping) return error.UnpreparedGlmTaps;
        const x = self.taps.get(cache) orelse return error.UnpreparedGlmTaps;
        if (count == 0 or start + count > cache.len) return error.GlmTapsNotHeld;
        const width = self.tapBytes();
        return .{ .buffer = @intFromPtr(x.id), .offset = start * width, .rows = count, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 };
    }

    /// Each decode window's forward ms by rows (1..16) on a scratch cache, for a drafter's depth rule.
    pub fn measureWindows(self: *Metal) !void {
        const c = &self.model.config;
        var cache = try st.Cache.init(self.model.device, c, 1024 + 2 * st.window_rows + 2);
        defer cache.deinit();
        var ids: [1024]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i * 37 % 20000);
        try self.feed(&cache, &ids, null);
        var win: [st.window_rows]u32 = undefined;
        for (&win, 0..) |*x, i| x.* = @intCast(2000 + i * 53 % 20000);
        for (1..st.window_rows + 1) |w| {
            var best: f64 = 1e9;
            for (0..2) |_| {
                try fwd.run(self.model, &self.scratch, &cache, win[0..w], .{ .all = true, .hidden = .{ .buf = cache.hidden } });
                best = @min(best, self.scratch.last_gpu);
            }
            self.model.window_costs[w - 1] = .{ .width = @intCast(w), .ms = 1000.0 * best };
        }
        self.model.window_costs_n = st.window_rows;
        var line: [512]u8 = undefined;
        var at: usize = 0;
        for (self.model.window_costs[0..st.window_rows]) |x| at += (std.fmt.bufPrint(line[at..], " {d}:{d:.1}", .{ x.width, x.ms }) catch break).len;
        std.log.info("glm_ax: decode windows' ms by rows:{s}", .{line[0..at]});
    }

    fn logits(self: *Metal, cache: *st.Cache) []const u16 {
        return cache.logits.slice(u16, self.model.config.vocab);
    }

    /// Run `ids` at the cache's end, MTP absorbing rows with a known next; a stream's pass keeps at its marks.
    /// GLM_AX_TAIL_ROWS (default 1024, 0: off): a prompt's last partial grid chunk goes in pieces of this many rows on
    /// their own grid, so a follow-up whose prompt ends in the same grid chunk reuses its rows to within a piece and
    /// still matches a fresh run's bits (both cut the same chunks); whole chunks keep the faster prompt_rows width.
    fn tailRows() usize {
        const v = std.c.getenv("GLM_AX_TAIL_ROWS") orelse return @min(1024, st.prompt_rows);
        const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return @min(1024, st.prompt_rows);
        if (n == 0 or n >= st.prompt_rows or st.prompt_rows % n != 0 or n % 64 != 0) return st.prompt_rows;
        return n;
    }

    /// Where the prompt chunk from absolute row `at` ends in a prompt of `total` rows: the grid chunk when the whole
    /// chunk fits, else the next piece boundary of the tail.
    /// GLM_AX_TAIL_FROM (default 65536): prompts shorter than this keep whole-width chunks throughout (a short
    /// prompt's follow-up reuses a prompt-cache snapshot instead, and its first turn stays fastest)
    fn tailFrom() usize {
        const v = std.c.getenv("GLM_AX_TAIL_FROM") orelse return 65536;
        return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 65536;
    }

    fn chunkEnd(at: usize, total: usize) usize {
        const g = st.prompt_rows;
        const whole = (at / g + 1) * g;
        if (whole <= total or total < tailFrom()) return @min(whole, total);
        const t = tailRows();
        return @min(total, (at / t + 1) * t);
    }

    /// The last row boundary both chunkings share (a prompt of p1 rows and one of p2), at or below x.
    fn sharedBoundary(x: usize, p1: usize, p2: usize) usize {
        const g = st.prompt_rows;
        const t1 = p1 / g * g;
        const t2 = p2 / g * g;
        if (x < t1) return x / g * g;
        if (t1 == t2 and p1 >= tailFrom()) return x / tailRows() * tailRows();
        return t1;
    }

    fn feed(self: *Metal, cache: *st.Cache, ids: []const u32, stream: ?*lanes.Stream) !void {
        const c = &self.model.config;
        const base = cache.len;
        const total = base + ids.len;
        var at: usize = 0;
        while (at < ids.len) {
            var end = chunkEnd(base + at, total) - base;
            if (stream) |sm| {
                if (sm.isCancelled()) return error.Cancelled;
                for (sm.chunks) |x| if (x > base + at and x < base + end) {
                    end = x - base;
                };
                for (sm.reuse.marks) |x| if (x > base + at and x < base + end) {
                    end = x - base;
                };
            }
            const s = if (end - at > st.window_rows) &self.prompt else &self.scratch;
            const n = end - at;
            const trace = std.c.getenv("GLM_AX_TRACE") != null;
            var t0: std.c.timespec = undefined;
            if (trace) _ = std.c.clock_gettime(.MONOTONIC, &t0);
            try fwd.run(self.model, s, cache, ids[at..end], .{ .hidden = .{ .buf = s.hid }, .taps = self.tapsAt(cache, cache.len) });
            if (trace) {
                var t1: std.c.timespec = undefined;
                _ = std.c.clock_gettime(.MONOTONIC, &t1);
                std.log.info("glm_ax trace: prompt rows {d}..{d} ({d}) in {d:.1} ms", .{ cache.len, cache.len + (end - at), end - at, @as(f64, @floatFromInt((t1.sec - t0.sec) * 1_000_000_000 + (t1.nsec - t0.nsec))) / 1e6 });
            }
            const start = cache.len;
            cache.len += n;
            const last: [*]const u8 = s.hid.contents() + (n - 1) * c.hidden * 2;
            @memcpy(cache.hidden.contents()[0 .. c.hidden * 2], last[0 .. c.hidden * 2]);
            const known = if (end < ids.len) n else n - 1; // rows whose next token is in ids
            if (c.mtp > 0 and known > 0) try fwd.mtp(self.model, s, cache, .{ .buf = s.hid }, ids[at + 1 .. at + 1 + known], start);
            if (stream) |sm| lanes.prefill_widths.completed(self.gpa, sm, end - at);
            at = end;
            if (stream) |sm| if (std.mem.indexOfScalar(u32, sm.reuse.marks, @intCast(cache.len)) != null) if (sm.reuse.hook) |k| k.at(k.ptr, sm, @intCast(cache.len));
        }
    }

    fn prefillFn(ptr: *anyopaque, stream: *lanes.Stream) !void {
        const self = cast(ptr);
        const cache = try self.setupPrefill(stream);
        errdefer self.dropStream(stream);
        try self.feed(cache, stream.prompt()[cache.len..], stream);
    }

    /// A stream's caches freed after a failed prompt pass (not parked: its rows may be partial).
    fn dropStream(self: *Metal, stream: *lanes.Stream) void {
        const entry = self.caches.fetchRemove(stream) orelse return;
        if (self.taps.fetchRemove(entry.value)) |e| e.value.deinit();
        self.held -= st.cacheBytes(&self.model.config, entry.value.capacity);
        entry.value.deinit();
        self.gpa.destroy(entry.value);
        if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
    }

    /// GLM_AX_STEPS=1 (opt-in until measured): one prompt chunk a call, so the lane host runs decode rounds for the other
    /// streams between them (--decode-share, SRPT); the first call sets the stream's caches up as prefillFn does.
    fn prefillStepFn(ptr: *anyopaque, stream: *lanes.Stream, max_rows: u32) !bool {
        const self = cast(ptr);
        const cache = self.caches.get(stream) orelse try self.setupPrefill(stream);
        const ids = stream.prompt();
        if (stream.isCancelled()) {
            stream.keepInterruptedPrefill(@intCast(cache.len));
            return error.Cancelled;
        }
        if (cache.len >= ids.len) return true;
        self.dropAhead();
        const c = &self.model.config;
        const at = cache.len;
        // a cap keeps the chunk grid: the step ends at the next multiple of the cap's rows (bit-identical rows)
        const step: usize = if (max_rows > 0) @min(st.prompt_rows, @max(64, @as(usize, max_rows) / 64 * 64)) else st.prompt_rows;
        var end = if (max_rows > 0) @min(ids.len, (at / step + 1) * step) else chunkEnd(at, ids.len);
        for (stream.chunks) |x| if (x > at and x < end) {
            end = x;
        };
        for (stream.reuse.marks) |x| if (x > at and x < end) {
            end = x;
        };
        const n = end - at;
        const sc = if (n > st.window_rows) &self.prompt else &self.scratch;
        try fwd.run(self.model, sc, cache, ids[at..end], .{ .hidden = .{ .buf = sc.hid }, .taps = self.tapsAt(cache, cache.len) });
        cache.len += n;
        const last: [*]const u8 = sc.hid.contents() + (n - 1) * c.hidden * 2;
        @memcpy(cache.hidden.contents()[0 .. c.hidden * 2], last[0 .. c.hidden * 2]);
        const known = if (end < ids.len) n else n - 1;
        if (c.mtp > 0 and known > 0) try fwd.mtp(self.model, sc, cache, .{ .buf = sc.hid }, ids[at + 1 .. at + 1 + known], at);
        lanes.prefill_widths.completed(self.gpa, stream, n);
        stream.fill_at.store(@intCast(end), .release);
        if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(end)) != null) if (stream.reuse.hook) |k| k.at(k.ptr, stream, @intCast(end));
        return end == ids.len;
    }

    fn stepsOn() bool { // opt-in until measured: GLM_AX_STEPS=1
        const v = std.c.getenv("GLM_AX_STEPS") orelse return false;
        return v[0] == '1';
    }

    /// A stream's caches for its prompt pass: the live-reuse adoption or a new cache (a kept snapshot restored), their
    /// length the rows already filled.
    fn setupPrefill(self: *Metal, stream: *lanes.Stream) !*st.Cache {
        self.dropAhead();
        const ids = stream.prompt();
        if (ids.len == 0 or ids.len + stream.max_new > self.options.capacity or self.caches.contains(stream)) return error.InvalidGlmPrompt;
        const need_cap = ids.len + stream.max_new + 1;
        if (self.parked) |pk| {
            const rows = parkedRows(pk, ids);
            const snap_at: usize = if (stream.reuse.saved) |saved| @as(*const snapshot.Snap, @ptrCast(@alignCast(saved))).at else 0;
            // a prompt that shares no usable prefix still takes the parked allocation when it fits (rows 0): no free and
            // allocate of a whole-context cache between a long stream and a short request
            const take = pk.cache.capacity >= need_cap and (rows > snap_at or (rows == 0 and snap_at == 0));
            if (take) {
                // the parked caches become this stream's, cut to the shared prefix: no copy, no prefill of it
                self.parked = null;
                self.gpa.free(pk.tokens);
                const cache = pk.cache;
                errdefer {
                    self.held -= st.cacheBytes(&self.model.config, cache.capacity);
                    cache.deinit();
                    self.gpa.destroy(cache);
                }
                cache.len = rows;
                cache.nheld = 0;
                cache.chance = @splat(0);
                cache.drafted = 0;
                cache.accepted = 0;
                cache.drafted_at = @splat(0);
                cache.accepted_at = @splat(0);
                cache.prof = .{};
                try self.caches.put(self.gpa, stream, cache);
                errdefer _ = self.caches.remove(stream);
                stream.cached = @intCast(rows);
                self.reused_rows += rows;
                if (rows > 0) std.log.info("glm_ax: live reuse: {d} of {d} prompt rows from the last stream's caches (no copy)", .{ rows, ids.len });
                if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, stream.reuse.saved, ids);
                return cache;
            }
            self.freeParked();
        }
        // with live reuse, a stream's caches take the whole context when they fit: the next turn reuses them in place
        const full = st.cacheBytes(&self.model.config, self.options.capacity);
        const reclaim = reclaimableBytes();
        const roomy = if (reclaim) |r| r >= full + full_margin else false;
        const capacity = if (liveReuse() and !self.tapping and roomy and self.options.capacity >= need_cap and self.held + full <= self.options.budget) self.options.capacity else need_cap;
        std.log.info("glm_ax: caches for this stream: {d} tokens ({s}), {d:.1} GiB; reclaimable before {d:.1} GiB", .{ capacity, if (capacity == self.options.capacity) "the whole context: a follow-up reuses them in place" else "prompt + reply: a whole-context cache would leave under 48 GiB reclaimable", gib(st.cacheBytes(&self.model.config, capacity)), if (reclaim) |r| gib(r) else -1.0 });
        const need = st.cacheBytes(&self.model.config, capacity);
        if (self.held + need > self.options.budget) {
            std.log.warn("glm_ax: a {d}-token stream needs {d:.1} GiB of caches; {d:.1} of {d:.1} GiB are held", .{ capacity, gib(need), gib(self.held), gib(self.options.budget) });
            return error.GlmCachesFull;
        }
        self.held += need;
        errdefer { // as releaseFn: the bytes back and the prompt cache's room fitted to them (it was fitted short below)
            self.held -= need;
            if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
        }
        if (stream.reuse.saved) |saved| { // a state the stream cannot hold beside its caches: freed now, no restore
            const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
            if (self.held + snap.bytes > self.options.budget) snap.free();
        }
        // kept states give way first
        if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, stream.reuse.saved, ids);
        const cache = try self.createCache(capacity);
        errdefer self.gpa.destroy(cache);
        errdefer cache.deinit();
        try self.caches.put(self.gpa, stream, cache); // a mark's save finds it there
        errdefer _ = self.caches.remove(stream);
        if (self.tapping) {
            const tb = try self.model.device.buffer(capacity * self.tapBytes(), mtl.ResourceOptions.shared);
            errdefer tb.deinit();
            try self.taps.put(self.gpa, cache, tb);
        }
        errdefer if (self.taps.fetchRemove(cache)) |e| e.value.deinit();
        stream.cached = 0;
        if (stream.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const snap: *const snapshot.Snap = @ptrCast(@alignCast(saved));
            if (snap.at < ids.len) {
                if (snapshot.restore(self.model.queue, &self.model.config, snap, cache)) stream.cached = snap.at else |_| stream.reuse_failed = true;
            } else stream.reuse_failed = true;
        }
        return cache;
    }

    /// A prompt chunk, a decode row, the MTP layers and the tower once: the first request skips the GPU's first touch.
    pub fn warm(self: *Metal) !void {
        const c = &self.model.config;
        var cache = try st.Cache.init(self.model.device, c, st.prompt_rows + 2);
        defer cache.deinit();
        var ids_all: [st.prompt_rows_max]u32 = undefined;
        const ids = ids_all[0..st.prompt_rows];
        for (ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37 % 20000);
        try self.feed(&cache, ids, null);
        try fwd.run(self.model, &self.scratch, &cache, &.{ids[0]}, .{ .all = true, .hidden = .{ .buf = cache.hidden } });
    }

    fn firstFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidGlmPosition;
        return try draw(self.gpa, self.logits(cache), stream.sampling, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, stream: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len) return error.InvalidGlmPosition;
        return try draw(self.gpa, self.logits(cache), stream.sampling, position, mask);
    }

    fn queueFn(ptr: *anyopaque, stream: *lanes.Stream, f: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const cache = try self.cacheOf(stream);
        if (position != cache.len + 1) return error.InvalidGlmPosition;
        const token: u32 = switch (f) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        if (token >= self.model.config.vocab) return error.InvalidGlmToken;
        // this step: encoded ahead by the last call when it was this stream's next row, else encoded now
        var cur: fwd.Pending = .{};
        if (self.ahead.live and self.ahead_stream == stream and self.ahead.start == cache.len and !self.ahead.all) {
            cur = self.ahead;
            self.ahead = .{};
        } else {
            self.dropAhead();
            try fwd.run(self.model, &self.scratch, cache, &.{token}, .{ .hidden = .{ .buf = cache.hidden }, .pending = &cur, .taps = self.tapsAt(cache, cache.len) });
        }
        cur.commit(&.{token});
        // the next row encoded while the GPU runs this one: its own ids and rows table, committed by the next call
        if (cache.len + 2 <= cache.capacity and std.c.getenv("GLM_AX_NO_AHEAD") == null) {
            self.scratch.flip();
            fwd.run(self.model, &self.scratch, cache, &.{0}, .{ .hidden = .{ .buf = cache.hidden }, .pending = &self.ahead, .start = cache.len + 1, .taps = self.tapsAt(cache, cache.len + 1) }) catch self.dropAhead();
            self.ahead_stream = stream;
        }
        try cur.wait();
        cache.len += 1;
        return try draw(self.gpa, self.logits(cache), stream.sampling, position, &.{});
    }

    /// One plain decode row through verify: the row encoded ahead by the last round when it is this stream's next,
    /// committed with the pending token; the following row encoded while the GPU runs it.
    fn verifyOne(self: *Metal, w: be.Window, o: be.Verified) !void {
        const c = &self.model.config;
        const words = (c.vocab + 31) / 32;
        if (o.sampled.len != 1 or o.drafts.len != 0) return error.UnsupportedGlmWindow;
        const cache = try self.cacheOf(w.stream);
        if (w.positions[0] != cache.len + 1) return error.InvalidGlmPosition;
        if (w.pending >= c.vocab) return error.InvalidGlmToken;
        const t_start = profStart(cache);
        var cur: fwd.Pending = .{};
        if (self.ahead.live and self.ahead_stream == w.stream and self.ahead.start == cache.len and self.ahead.all) {
            cur = self.ahead;
            self.ahead = .{};
        } else {
            self.dropAhead();
            try fwd.run(self.model, &self.scratch, cache, &.{w.pending}, .{ .all = true, .hidden = .{ .buf = cache.hidden }, .pending = &cur, .taps = self.tapsAt(cache, cache.len) });
        }
        cur.commit(&.{w.pending});
        if (cache.len + 2 <= cache.capacity and std.c.getenv("GLM_AX_NO_AHEAD") == null) {
            self.scratch.flip();
            fwd.run(self.model, &self.scratch, cache, &.{0}, .{ .all = true, .hidden = .{ .buf = cache.hidden }, .pending = &self.ahead, .start = cache.len + 1, .taps = self.tapsAt(cache, cache.len + 1) }) catch self.dropAhead();
            self.ahead_stream = w.stream;
        }
        try cur.wait();
        cache.len += 1;
        cache.nheld = 0;
        cache.prof.verify += 1;
        cache.prof.verify_gpu += cur.gpu;
        defer {
            cache.prof.last_end = now();
            cache.prof.verify_wall += secs(t_start, cache.prof.last_end);
        }
        const l = self.scratch.logits_all.slice(u16, c.vocab);
        const mask = if (w.masks.len >= words) w.masks[0..words] else &.{};
        o.sampled[0] = try draw(self.gpa, l, w.stream.sampling, w.positions[0], mask);
        if (o.rows.len == 1) o.rows[0] = row(l, o.sampled[0], w.stream.logprobs orelse 0);
        @memcpy(cache.logits.slice(u16, c.vocab), l);
    }

    /// Give up the step encoded ahead (another call is about to use the scratch or the stream's cache).
    fn dropAhead(self: *Metal) void {
        self.ahead.drop();
        self.ahead_stream = null;
    }

    fn firstRowFn(ptr: *anyopaque, stream: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        const l = self.logits(try self.cacheOf(stream));
        return row(l, @intCast(argmax(l)), stream.logprobs orelse 0);
    }

    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        if (handle >= cast(ptr).model.config.vocab) return error.InvalidGlmToken;
        return @intCast(handle);
    }

    /// Each window's pending token and drafts in one forward, every row drawn and kept until `keep` rolls back.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        // a plain decode round (one stream, its pending token alone): the step-ahead path
        if (windows.len == 1 and out.len == 1 and windows[0].rows() == 1 and windows[0].parents == null and windows[0].held == 0) return self.verifyOne(windows[0], out[0]);
        self.dropAhead();
        const c = &self.model.config;
        const words = (c.vocab + 31) / 32;
        if (windows.len != out.len) return error.InvalidGlmWindows;
        if (windows.len > 1 and !self.tapping and batchOn()) {
            var total: usize = 0;
            for (windows) |w| total += w.rows();
            if (total <= st.window_rows) return self.verifyBatch(windows, out);
        }
        for (windows, out) |w, o| {
            const rows = w.rows();
            if (rows > st.window_rows or w.parents != null or o.sampled.len != rows or o.drafts.len != rows - 1) return error.UnsupportedGlmWindow;
            const cache = try self.cacheOf(w.stream);
            if (w.held > cache.nheld) return error.GlmDraftsNotHeld;
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidGlmPosition;
            var ids: [st.window_rows]u32 = undefined;
            ids[0] = w.pending;
            @memcpy(ids[1..][0..w.held], cache.held[0..w.held]);
            @memcpy(ids[1 + w.held ..][0..w.tokens.len], w.tokens);
            const t_start = profStart(cache);
            try fwd.run(self.model, &self.scratch, cache, ids[0..rows], .{ .all = true, .hidden = .{ .buf = cache.hidden }, .taps = self.tapsAt(cache, cache.len) });
            cache.prof.verify += 1;
            cache.prof.verify_gpu += self.scratch.last_gpu;
            defer {
                cache.prof.last_end = now();
                cache.prof.verify_wall += secs(t_start, cache.prof.last_end);
            }
            cache.len += rows;
            cache.nheld = 0;
            // every draft counts as accepted here; keep (called only when a window is cut) takes back the rejected
            cache.drafted += rows - 1;
            cache.accepted += rows - 1;
            for (0..@min(rows - 1, 8)) |j| {
                cache.drafted_at[j] += 1;
                cache.accepted_at[j] += 1;
            }
            @memcpy(o.drafts, ids[1..rows]);
            const all = self.scratch.logits_all.slice(u16, rows * c.vocab);
            for (0..rows) |r| {
                const l = all[r * c.vocab ..][0..c.vocab];
                const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
                o.sampled[r] = try draw(self.gpa, l, w.stream.sampling, w.positions[r], mask);
                if (o.rows.len == rows) o.rows[r] = row(l, o.sampled[r], w.stream.logprobs orelse 0);
            }
            // the last kept row's logits are the next first draw's (the lane core keeps every row here)
            @memcpy(cache.logits.slice(u16, c.vocab), all[(rows - 1) * c.vocab ..][0..c.vocab]);
        }
    }

    /// GLM_AX_BATCH=1: several streams' windows verified in one forward (forward.runMany), at most window_rows rows.
    fn verifyBatch(self: *Metal, windows: []const be.Window, out: []be.Verified) !void {
        const c = &self.model.config;
        const words = (c.vocab + 31) / 32;
        var ids: [st.window_rows]u32 = undefined;
        var segs: [st.window_rows]fwd.Seg = undefined;
        var caches: [st.window_rows]*st.Cache = undefined;
        var at: usize = 0;
        for (windows, out, 0..) |w, o, k| {
            const rows = w.rows();
            if (rows > st.window_rows or w.parents != null or o.sampled.len != rows or o.drafts.len != rows - 1) return error.UnsupportedGlmWindow;
            const cache = try self.cacheOf(w.stream);
            for (caches[0..k]) |other| if (other == cache) return error.InvalidGlmWindows;
            if (w.held > cache.nheld) return error.GlmDraftsNotHeld;
            for (w.positions, 0..) |p, r| if (p != cache.len + r + 1) return error.InvalidGlmPosition;
            ids[at] = w.pending;
            @memcpy(ids[at + 1 ..][0..w.held], cache.held[0..w.held]);
            @memcpy(ids[at + 1 + w.held ..][0..w.tokens.len], w.tokens);
            segs[k] = .{ .cache = cache, .start = cache.len, .n = rows };
            caches[k] = cache;
            at += rows;
        }
        const t_start = profStart(caches[0]);
        try fwd.runMany(self.model, &self.scratch, segs[0..windows.len], ids[0..at], .{ .all = true, .hidden = .{ .buf = caches[0].hidden } });
        const t_end = now();
        self.batched_rounds += 1;
        self.batched_rows += at;
        if (self.batched_rounds % 500 == 1) std.log.info("glm_ax: batched verify: {d} rounds, {d:.2} rows a round; this one {d} streams, {d} rows, {d:.1} ms", .{ self.batched_rounds, @as(f64, @floatFromInt(self.batched_rows)) / @as(f64, @floatFromInt(self.batched_rounds)), windows.len, at, secs(t_start, t_end) * 1e3 });
        const all = self.scratch.logits_all.slice(u16, at * c.vocab);
        var r0: usize = 0;
        for (windows, out, 0..) |w, o, k| {
            const rows = w.rows();
            const cache = caches[k];
            cache.prof.verify += 1;
            cache.prof.verify_gpu += self.scratch.last_gpu / @as(f64, @floatFromInt(windows.len));
            cache.prof.last_end = t_end;
            cache.prof.verify_wall += secs(t_start, t_end) / @as(f64, @floatFromInt(windows.len));
            cache.len += rows;
            cache.nheld = 0;
            cache.drafted += rows - 1;
            cache.accepted += rows - 1;
            for (0..@min(rows - 1, 8)) |j| {
                cache.drafted_at[j] += 1;
                cache.accepted_at[j] += 1;
            }
            @memcpy(o.drafts, ids[r0 + 1 ..][0 .. rows - 1]);
            for (0..rows) |r| {
                const l = all[(r0 + r) * c.vocab ..][0..c.vocab];
                const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
                o.sampled[r] = try draw(self.gpa, l, w.stream.sampling, w.positions[r], mask);
                if (o.rows.len == rows) o.rows[r] = row(l, o.sampled[r], w.stream.logprobs orelse 0);
            }
            @memcpy(cache.logits.slice(u16, c.vocab), all[(r0 + rows - 1) * c.vocab ..][0..c.vocab]);
            r0 += rows;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        self.dropAhead();
        for (windows, paths) |w, path| {
            const cache = try self.cacheOf(w.stream);
            for (path, 0..) |r, i| if (r != i) return error.UnsupportedGlmTree;
            if (path.len == 0 or path.len > w.rows() or w.rows() > cache.len) return error.UnsupportedGlmWindow;
            cache.len = cache.len - w.rows() + path.len;
            // verify counted every draft as accepted; the rows past the kept path were not
            cache.accepted -= w.rows() - path.len;
            // positions past the 8 counted are not tracked; a window longer than 9 rows (copy drafts) cut past them
            const end: usize = @min(w.rows() - 1, 8);
            for (@min(path.len - 1, end)..end) |j| cache.accepted_at[j] -= 1;
        }
    }

    /// MTP layer 0 absorbs the kept rows and drafts from the last; further drafts chain the next layers, greedy.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const self = cast(ptr);
        self.dropAhead();
        const c = &self.model.config;
        const s = &self.scratch;
        for (requests) |r| {
            const cache = try self.cacheOf(r.stream);
            cache.nheld = 0;
            if (c.mtp == 0) {
                if (r.depth != 0) return error.GlmCheckpointHasNoMtp;
                continue;
            }
            var follow: [st.window_rows]u32 = undefined;
            var kept: usize = 1;
            if (r.rows) |rows| {
                kept = rows.len;
                for (rows, 0..) |x, i| if (x != i) return error.UnsupportedGlmTree;
                if (kept == 0 or kept > st.window_rows or r.follow.len < kept) return error.UnsupportedGlmWindow;
                @memcpy(follow[0..kept], r.follow[0..kept]);
            } else if (r.follow.len > 0) {
                follow[0] = r.follow[r.follow.len - 1];
            } else {
                const f = r.first orelse return error.NoFirstToken;
                follow[0] = switch (f) {
                    .value => |v| v,
                    .handle => |h| @intCast(h),
                };
            }
            // kept row i sits at position (r.position - 1 - kept) + i, its residual in cache.hidden's row i
            const first: usize = @intCast(r.position - 1 - kept);
            const t_start = profStart(cache);
            defer {
                cache.prof.draft += 1;
                cache.prof.last_end = now();
                cache.prof.draft_wall += secs(t_start, cache.prof.last_end);
            }
            try fwd.mtp(self.model, s, cache, .{ .buf = cache.hidden }, follow[0..kept], first);
            cache.prof.draft_gpu += s.last_gpu;
            const depth = @min(r.depth, depthCap());
            var pos = first + kept - 1;
            var k: usize = 0;
            const floor = mtpFloor();
            var run: f64 = 1;
            while (k < depth) : (k += 1) {
                const l = s.logits_all.slice(u16, c.vocab);
                const d: u32 = @intCast(argmax(l));
                cache.held[k] = d;
                cache.nheld = k + 1;
                if (floor != null) {
                    run *= topProbability(l);
                    cache.chance[k] = run;
                }
                if (k + 1 == depth) break;
                // a draft past the first is verified only at a running chance of the floor or more: stop drafting
                if (floor) |f| if (run < f) break;
                pos += 1;
                @memcpy(s.chain.contents()[0 .. c.hidden * 2], (s.hid.contents() + (if (k == 0) kept - 1 else 0) * c.hidden * 2)[0 .. c.hidden * 2]);
                try fwd.mtp(self.model, s, cache, .{ .buf = s.chain }, &.{d}, pos);
                cache.prof.draft_gpu += s.last_gpu;
            }
        }
    }

    /// Each held draft's running chance (GLM_AX_MTP_FLOOR); drafts the head did not run have none.
    fn probabilitiesFn(ptr: *anyopaque, stream: *lanes.Stream, out: []f64) !bool {
        const cache = try cast(ptr).cacheOf(stream);
        if (mtpFloor() == null) return false;
        for (out, 0..) |*o, i| o.* = if (i < cache.nheld) cache.chance[i] else 0;
        if (cache.nheld < out.len) stream.draft_chances_cut = cache.nheld;
        return true;
    }

    /// For a drafter that hands some streams' rounds to the MTP head (dspark.zig's per-request choice): the head's
    /// drafts for `r` (depth 0: absorb the kept rows only), its held drafts and their running chances.
    pub fn mtpDraft(self: *Metal, r: be.DraftRequest) !void {
        return draftFn(self, &.{r});
    }
    pub fn mtpHeld(self: *Metal, stream: *lanes.Stream, out: []u32, chances: []f64) !usize {
        const cache = try self.cacheOf(stream);
        const n = @min(out.len, cache.nheld);
        @memcpy(out[0..n], cache.held[0..n]);
        for (chances[0..n], 0..) |*o, i| o.* = cache.chance[i];
        return n;
    }

    /// The held drafts in window order, for a grammar to walk.
    fn heldFn(ptr: *anyopaque, stream: *lanes.Stream, out: []u32) !void {
        const cache = try cast(ptr).cacheOf(stream);
        if (out.len > cache.nheld) return error.GlmDraftsNotHeld;
        @memcpy(out, cache.held[0..out.len]);
    }

    fn releaseFn(ptr: *anyopaque, stream: *lanes.Stream) void {
        const self = cast(ptr);
        self.dropAhead();
        if (self.caches.fetchRemove(stream)) |entry| {
            const cc = entry.value;
            if (cc.drafted > 0) std.log.info("glm_ax: request done at {d} tokens: drafts accepted {d} of {d} ({d:.1}%)", .{ cc.len, cc.accepted, cc.drafted, 100.0 * @as(f64, @floatFromInt(cc.accepted)) / @as(f64, @floatFromInt(cc.drafted)) });
            if (cc.drafted > 0) std.log.info("glm_ax: by position: 1st {d}/{d}, 2nd {d}/{d}, 3rd {d}/{d}", .{ cc.accepted_at[0], cc.drafted_at[0], cc.accepted_at[1], cc.drafted_at[1], cc.accepted_at[2], cc.drafted_at[2] });
            const p = cc.prof;
            if (p.verify > 0) std.log.info("glm_ax: rounds: {d} verify passes {d:.2} ms wall / {d:.2} ms GPU each, {d} draft passes {d:.2} ms wall / {d:.2} ms GPU each, {d:.2} ms a pass between calls", .{ p.verify, 1000 * p.verify_wall / @as(f64, @floatFromInt(p.verify)), 1000 * p.verify_gpu / @as(f64, @floatFromInt(p.verify)), p.draft, if (p.draft > 0) 1000 * p.draft_wall / @as(f64, @floatFromInt(p.draft)) else 0, if (p.draft > 0) 1000 * p.draft_gpu / @as(f64, @floatFromInt(p.draft)) else 0, 1000 * p.gap / @as(f64, @floatFromInt(p.verify)) });
            if (self.taps.fetchRemove(cc)) |e| e.value.deinit();
            {
                const sp = fwd.split;
                if (sp.runs > 0) std.log.info("glm_ax: finish split over {d} passes, ms each: encode {d:.2}, commit+wait {d:.2}, GPU times read {d:.2}, failure check {d:.2}", .{ sp.runs, @as(f64, @floatFromInt(sp.encode)) / 1e6 / @as(f64, @floatFromInt(sp.runs)), @as(f64, @floatFromInt(sp.wait)) / 1e6 / @as(f64, @floatFromInt(sp.runs)), @as(f64, @floatFromInt(sp.times)) / 1e6 / @as(f64, @floatFromInt(sp.runs)), @as(f64, @floatFromInt(sp.failure)) / 1e6 / @as(f64, @floatFromInt(sp.runs)) });
                fwd.split = .{};
            }
            if (liveReuse() and !self.tapping and cc.len > 0) park: {
                // kept live for a follow-up: the token of each row (the prompt, then the reply's committed tokens)
                const prompt = stream.prompt();
                const out = stream.emitted();
                if (cc.len > prompt.len + out.len) break :park;
                const toks = self.gpa.alloc(u32, cc.len) catch break :park;
                const from_prompt = @min(cc.len, prompt.len);
                @memcpy(toks[0..from_prompt], prompt[0..from_prompt]);
                @memcpy(toks[from_prompt..], out[0 .. cc.len - from_prompt]);
                self.freeParked();
                self.parked = .{ .cache = cc, .tokens = toks, .prompt_len = prompt.len };
                return;
            }
            self.held -= st.cacheBytes(&self.model.config, entry.value.capacity);
            entry.value.deinit();
            self.gpa.destroy(entry.value);
            if (self.room) |r| r.fit(r.ctx, self.options.budget - self.held, null, &.{});
        }
    }

    /// The prompt cache's Snapshots functions: a stream's caches at a mark, copied on the GPU.
    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        return snapshot.bytes(&cast(ptr).model.config, at);
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = cast(ptr);
        const cache = try self.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        if (cache.len != at) return error.SnapshotOutOfStep;
        const snap = try self.gpa.create(snapshot.Snap);
        errdefer self.gpa.destroy(snap);
        var t0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &t0);
        snap.* = try snapshot.save(self.model.device, self.model.queue, &self.model.config, cache);
        if (std.c.getenv("GLM_AX_TRACE") != null) {
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t1);
            std.log.info("glm_ax trace: prompt state saved at {d} ({d:.2} GB) in {d:.1} ms", .{ at, @as(f64, @floatFromInt(snap.bytes)) / 1e9, @as(f64, @floatFromInt((t1.sec - t0.sec) * 1_000_000_000 + (t1.nsec - t0.nsec))) / 1e6 });
        }
        return snap;
    }

    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        const snap: *snapshot.Snap = @ptrCast(@alignCast(saved));
        snap.free();
        cast(ptr).gpa.destroy(snap);
    }
};

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn logit(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

fn argmax(l: []const u16) usize {
    var best: usize = 0;
    for (l, 0..) |w, i| if (logit(w) > logit(l[best])) {
        best = i;
    };
    return best;
}

/// The logprob row of bf16 logits `l` for `pick`: an fp64 normalizer, `count` best ids, ties to the lower id.
pub fn row(l: []const u16, pick: u32, count: u8) lanes.LogprobRow {
    const top = lanes.logprob.max_top;
    var most: f64 = -std.math.inf(f64);
    for (l) |w| most = @max(most, logit(w));
    var sum: f64 = 0;
    for (l) |w| sum += @exp(@as(f64, logit(w)) - most);
    const lse: f32 = @floatCast(most + @log(sum));
    var r: lanes.LogprobRow = .{ .token = pick, .logprob = logit(l[pick]) - lse, .count = top };
    var filled: usize = 0;
    for (l, 0..) |w, id| {
        const x = logit(w);
        if (filled == top and x <= r.logprobs[top - 1] + lse) continue;
        var at = @min(filled, top - 1);
        while (at > 0 and x > r.logprobs[at - 1] + lse) : (at -= 1) {
            r.ids[at] = r.ids[at - 1];
            r.logprobs[at] = r.logprobs[at - 1];
        }
        r.ids[at] = @intCast(id);
        r.logprobs[at] = x - lse;
        filled = @min(filled + 1, top);
    }
    r.count = @min(count, top);
    return r;
}

test "a logprob row: the normalizer, the best ids in order, ties to the lower id" {
    var l: [40]u16 = undefined;
    for (&l, 0..) |*w, i| w.* = @intCast(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 7)))) >> 16);
    const r = row(&l, 6, 20);
    try std.testing.expectEqual(@as(u32, 6), r.ids[0]);
    try std.testing.expectEqual(@as(u32, 13), r.ids[1]);
    try std.testing.expect(r.logprobs[0] == r.logprob and r.logprobs[19] <= r.logprobs[18]);
}
