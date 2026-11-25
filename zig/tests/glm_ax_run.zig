//! GLM-5.3 on MLX affine weights without the server: a prompt's chunks, a greedy reply, timings and dumps.
const std = @import("std");
/// The Metal wrapper compiles Metal 4 in (GLM_AX_METAL4=1 takes it at run time).
pub const tf_metal4 = true;
const platform = @import("sys");
const tf = @import("tensorfold");
const tok = @import("tokenizer");
const mtl = @import("metal");
const g = tf.glm_ax;
const a = std.heap.page_allocator;

const usage =
    \\tf-glm-ax-run MODEL_DIR PROMPT (a .txt file: a user message in GLM's chat format; else token ids)
    \\GLM_LAYERS (the first N layers, no head), GLM_SHARDS (main shard files to read), GLM_MAX (reply tokens, 64),
    \\GLM_CONTEXT (8192), GLM_CHUNK (prompt rows a forward, 2048), GLM_MTP=1 (load the MTP layer), GLM_PROFILE=1
    \\(a command buffer a stage: GPU ms by stage), GLM_DUMP=FILE (the last chunk's fp32 residual rows after the
    \\loaded layers), GLM_REPEAT (runs, 1), GLM_FORCED=1 (prompt rows 16 at a time, each row's pick against the next
    \\token), GLM_EFFORT (the system line's reasoning effort, Low)
;

fn env(name: [:0]const u8, default: u64) u64 {
    const v = platform.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u64, std.mem.span(v), 10) catch default;
}

fn seconds(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
}

fn logit(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

fn argmax(l: []const u16) u32 {
    var best: usize = 0;
    for (l, 0..) |w, i| if (logit(w) > logit(l[best])) {
        best = i;
    };
    return @intCast(best);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    const layers: ?u32 = if (platform.c.getenv("GLM_LAYERS") != null) @intCast(env("GLM_LAYERS", 78)) else null;
    const shards: ?usize = if (platform.c.getenv("GLM_SHARDS") != null) @intCast(env("GLM_SHARDS", 91)) else null;
    const m = try g.Model.load(a, io, argv[1], .{ .layers = layers, .mtp = env("GLM_MTP", 0) != 0, .shards = shards });
    defer m.deinit();
    const c = &m.config;
    std.debug.print("loaded in {d:.1} s: {d} layers, {d:.2} GiB of weights, Metal allocated {d:.2} GiB, MTP {d}\n", .{ m.load_seconds, m.weights.main, @as(f64, @floatFromInt(m.checkpoint.residentBytes())) / (1 << 30), @as(f64, @floatFromInt(m.device.allocated())) / (1 << 30), c.mtp });
    var t = try tok.loadTokenizer(io, a, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{argv[1]}));
    const max: usize = if (m.weights.head == null) 0 else @intCast(env("GLM_MAX", 64));
    var prompts: std.ArrayList([]const u32) = .empty;
    var longest: usize = 0;
    for (argv[2..]) |file| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(64 << 20));
        var prompt: std.ArrayList(u32) = .empty;
        if (std.mem.endsWith(u8, file, ".txt")) {
            const effort: []const u8 = if (platform.c.getenv("GLM_EFFORT")) |v| std.mem.span(v) else "Low";
            const chat = try std.fmt.allocPrint(a, "[gMASK]<sop><|system|>Reasoning Effort: {s}<|user|>{s}<|assistant|><think>", .{ effort, std.mem.trim(u8, text, " \n") });
            try prompt.appendSlice(a, try t.encode(a, chat));
        } else {
            var it = std.mem.tokenizeAny(u8, text, " \n\r\t,[]");
            while (it.next()) |w| try prompt.append(a, try std.fmt.parseInt(u32, w, 10));
        }
        longest = @max(longest, prompt.items.len);
        try prompts.append(a, prompt.items);
    }
    const context: usize = @max(@as(usize, @intCast(env("GLM_CONTEXT", 8192))), longest + max + 2);
    var scratch = try g.state.Scratch.init(m.device, c, context, g.state.window_rows);
    defer scratch.deinit();
    var big = try g.state.Scratch.init(m.device, c, context, g.state.prompt_rows);
    defer big.deinit();
    var cache = try g.state.Cache.init(m.device, c, context);
    defer cache.deinit();
    std.debug.print("context {d}, caches {d:.2} GiB, Metal allocated {d:.2} GiB\n", .{ context, @as(f64, @floatFromInt(g.state.cacheBytes(c, context))) / (1 << 30), @as(f64, @floatFromInt(m.device.allocated())) / (1 << 30) });
    for (prompts.items, argv[2..]) |prompt, file| {
        std.debug.print("== {s}: {d} prompt tokens\n", .{ file, prompt.len });
        for (0..env("GLM_REPEAT", 1)) |_| try once(m, io, &t, &scratch, &big, &cache, prompt, max);
    }
}

fn once(m: *g.Model, io: std.Io, t: *const tok.Tokenizer, scratch: *g.state.Scratch, big: *g.state.Scratch, cache: *g.state.Cache, prompt: []const u32, max: usize) !void {
    const c = &m.config;
    const forced = platform.c.getenv("GLM_FORCED") != null;
    const profiling = platform.c.getenv("GLM_PROFILE") != null;
    const chunk: usize = if (forced) g.state.window_rows else @intCast(env("GLM_CHUNK", g.state.prompt_rows));
    const head = m.weights.head != null;
    cache.len = 0;
    var agree: usize = 0;
    var pprof: g.forward.Profile = .{};
    const t0 = std.Io.Timestamp.now(io, .awake);
    var at: usize = 0;
    var last_n: usize = 0;
    var last_s: *g.state.Scratch = scratch;
    while (at < prompt.len) {
        const n = @min(chunk, prompt.len - at);
        const s = if (n > g.state.window_rows) big else scratch;
        const tc = std.Io.Timestamp.now(io, .awake);
        try g.forward.run(m, s, cache, prompt[at .. at + n], .{ .all = forced, .head = head, .hidden = if (head) .{ .buf = s.hid } else null, .profile = if (profiling) &pprof else null });
        if (platform.c.getenv("GLM_CHUNK_TRACE") != null) {
            const dt = seconds(io, tc);
            std.debug.print("chunk at {d}: {d} rows in {d:.1} ms = {d:.0} tok/s (GPU {d:.1} ms)\n", .{ at, n, 1000 * dt, @as(f64, @floatFromInt(n)) / dt, 1000 * s.last_gpu });
        }
        if (forced) for (0..n) |r| {
            const pick = argmax(s.logits_all.slice(u16, n * c.vocab)[r * c.vocab ..][0..c.vocab]);
            if (at + r + 1 < prompt.len and pick == prompt[at + r + 1]) agree += 1;
            if (r + 1 == n) @memcpy(cache.logits.slice(u16, c.vocab), s.logits_all.slice(u16, n * c.vocab)[r * c.vocab ..][0..c.vocab]);
        };
        cache.len += n;
        at += n;
        last_n = n;
        last_s = s;
    }
    const t_prompt = seconds(io, t0);
    std.debug.print("prompt: {d} tokens in {d:.3} s = {d:.1} tok/s", .{ prompt.len, t_prompt, @as(f64, @floatFromInt(prompt.len)) / t_prompt });
    if (forced) std.debug.print(", next-token agreement {d}/{d}", .{ agree, prompt.len - 1 });
    std.debug.print("\n", .{});
    if (profiling) report("prompt", &pprof, prompt.len);
    if (env("GLM_PBENCH", 0) > 0 and prompt.len >= g.state.prompt_rows) try pbench(m, big, cache, prompt, @intCast(env("GLM_PBENCH", 0)));
    if (platform.c.getenv("GLM_DUMP")) |path| {
        const bytes = last_s.resid.contents()[0 .. last_n * c.hidden * 4];
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(path), .data = bytes });
        std.debug.print("dumped {d} rows of fp32 residual to {s}\n", .{ last_n, std.mem.span(path) });
    }
    if (platform.c.getenv("GLM_PROBE")) |path| {
        if (platform.c.getenv("GLM_PROBE_DEBUG") != null) {
            std.debug.print("debug: scratch rows {d}; po {d} pm {d} pl {d} olat {d} logits_all {d} bytes\n", .{ scratch.rows, scratch.po.length(), scratch.pm.length(), scratch.pl.length(), scratch.olat.length(), scratch.logits_all.length() });
            @memset(scratch.po.slice(f32, 16 * c.heads * 512), 0);
            @memset(scratch.pm.slice(f32, 16 * c.heads), 0);
            @memset(scratch.pl.slice(f32, 16 * c.heads), 0);
            @memset(scratch.olat.slice(u16, c.heads * 512), 0);
        }
        try g.forward.run(m, scratch, cache, prompt[prompt.len - 1 ..], .{ .head = false });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(path), .data = scratch.resid.contents()[0 .. c.hidden * 4] });
        std.debug.print("probe: one row at position {d} through {d} layers, fp32 residual to {s}\n", .{ cache.len, m.weights.main, std.mem.span(path) });
        if (platform.c.getenv("GLM_PROBE_DEBUG") != null) {
            const nan = struct {
                fn bf(b: []const u16) usize {
                    var k: usize = 0;
                    for (b) |v| k += @intFromBool(std.math.isNan(@as(f32, @bitCast(@as(u32, v) << 16))));
                    return k;
                }
                fn f(b: []const f32) usize {
                    var k: usize = 0;
                    for (b) |v| k += @intFromBool(std.math.isNan(v));
                    return k;
                }
            };
            const H = c.heads;
            std.debug.print("debug: qlat NaN {d}/{d}, olat NaN {d}/{d}, pm NaN {d} (first {d:.3}), pl NaN {d} (first {d:.3}), po NaN {d}/{d}; row pos {d} list {d} nkeys {d}; keys[0..4] {d} {d} {d} {d}\n", .{
                nan.bf(scratch.qlat.slice(u16, H * 576)),       H * 576,
                nan.bf(scratch.olat.slice(u16, H * 512)),       H * 512,
                nan.f(scratch.pm.slice(f32, 16 * H)),           scratch.pm.slice(f32, 1)[0],
                nan.f(scratch.pl.slice(f32, 16 * H)),           scratch.pl.slice(f32, 1)[0],
                nan.f(scratch.po.slice(f32, H * 512)),          H * 512,
                scratch.rowtab.slice(g.state.Row, 1)[0].pos,    scratch.rowtab.slice(g.state.Row, 1)[0].list,
                scratch.rowtab.slice(g.state.Row, 1)[0].nkeys,  scratch.keys.slice(u32, 4)[0],
                scratch.keys.slice(u32, 4)[1],                  scratch.keys.slice(u32, 4)[2],
                scratch.keys.slice(u32, 4)[3],
            });
            {
                std.debug.print("debug: list[0..40]:", .{});
                for (scratch.keys.slice(i32, 40)) |v| std.debug.print(" {d}", .{v});
                std.debug.print("\n", .{});
                const po = scratch.po.slice(f32, H * 512);
                var shown: usize = 0;
                for (po, 0..) |v, i| if (std.math.isNan(v) and shown < 12) {
                    std.debug.print("  po NaN head {d} dim {d}\n", .{ i / 512, i % 512 });
                    shown += 1;
                };
                const ol = scratch.olat.slice(u16, H * 512);
                for (ol, 0..) |v, i| if (std.math.isNan(@as(f32, @bitCast(@as(u32, v) << 16)))) std.debug.print("  olat NaN head {d} dim {d}\n", .{ i / 512, i % 512 });
                const q = scratch.qlat.slice(u16, H * 576);
                var qmax: f32 = 0;
                for (q) |v| qmax = @max(qmax, @abs(@as(f32, @bitCast(@as(u32, v) << 16))));
                if (platform.c.getenv("GLM_PROBE_OLAT")) |op| {
                    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(op), .data = std.mem.sliceAsBytes(ol) });
                    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "po.bin", .data = std.mem.sliceAsBytes(scratch.po.slice(f32, H * 512)) });
                    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "pm.bin", .data = std.mem.sliceAsBytes(scratch.pm.slice(f32, H)) });
                }
                std.debug.print("  qlat max |v| {d:.3}; pm[0..4] {d:.3} {d:.3} {d:.3} {d:.3}\n", .{ qmax, scratch.pm.slice(f32, 4)[0], scratch.pm.slice(f32, 4)[1], scratch.pm.slice(f32, 4)[2], scratch.pm.slice(f32, 4)[3] });
            }
        }
    }
    if (max == 0) return;
    var reply: std.ArrayList(u32) = .empty;
    defer reply.deinit(a);
    var next = argmax(cache.logits.slice(u16, c.vocab));
    try reply.append(a, next);
    var dprof: g.forward.Profile = .{};
    const t1 = std.Io.Timestamp.now(io, .awake);
    while (reply.items.len < max) {
        var done = false;
        for (c.eos[0..c.eos_n]) |e| done = done or e == next;
        if (done) break;
        try g.forward.run(m, scratch, cache, &.{next}, .{ .profile = if (profiling) &dprof else null });
        cache.len += 1;
        next = argmax(cache.logits.slice(u16, c.vocab));
        try reply.append(a, next);
    }
    const t_decode = seconds(io, t1);
    const steps = reply.items.len - 1;
    std.debug.print("decode: {d} tokens in {d:.3} s = {d:.2} tok/s ({d:.2} ms a token) at context {d}\n", .{ steps, t_decode, @as(f64, @floatFromInt(steps)) / t_decode, 1000.0 * t_decode / @as(f64, @floatFromInt(@max(steps, 1))), cache.len });
    if (profiling) report("decode", &dprof, steps);
    if (env("GLM_ROWCHECK", 0) > 0) {
        // a window of the reply's first n tokens against the same tokens one row at a time: bits of every row's logits
        for ([_]usize{ 2, 3, 4, 8 }) |want| {
        const n: usize = @min(want, @min(g.state.window_rows, reply.items.len));
        const base = prompt.len;
        const keep_len = cache.len;
        cache.len = base;
        try g.forward.run(m, scratch, cache, reply.items[0..n], .{ .all = true });
        const V = c.vocab;
        const win = try a.alloc(u16, n * V);
        defer a.free(win);
        @memcpy(win, scratch.logits_all.slice(u16, n * V));
        var same: usize = 0;
        var worst: f32 = 0;
        var argmax_same: usize = 0;
        for (0..n) |r| {
            cache.len = base + r;
            try g.forward.run(m, scratch, cache, reply.items[r .. r + 1], .{ .all = true });
            const one = scratch.logits_all.slice(u16, V);
            const w = win[r * V ..][0..V];
            if (std.mem.eql(u16, one, w)) same += 1;
            for (one, w) |x, y| worst = @max(worst, @abs(@as(f32, @bitCast(@as(u32, x) << 16)) - @as(f32, @bitCast(@as(u32, y) << 16))));
            if (argmax(one) == argmax(w)) argmax_same += 1;
        }
        cache.len = keep_len;
        std.debug.print("rowcheck: a {d}-row window against one row at a time: {d}/{d} rows' logits bit-identical, same argmax {d}/{d}, max |diff| {d:.4}\n", .{ n, same, n, argmax_same, n, worst });
        }
    }
    if (env("GLM_KBENCH", 0) > 0) try kbench(m, scratch, cache, next, reply.items, @intCast(env("GLM_KBENCH", 0)), steps, t_decode);
    std.debug.print("ids:", .{});
    for (reply.items) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\ntext: {s}\n", .{try t.decode(a, reply.items, false)});
}

fn report(what: []const u8, p: *const g.forward.Profile, tokens: usize) void {
    var total: f64 = 0;
    for (p.gpu) |v| total += v;
    std.debug.print("{s} GPU time by stage, ms a token ({d} tokens, {d:.3} s of GPU):\n", .{ what, tokens, total });
    inline for (@typeInfo(g.forward.Stage).@"enum".field_names, 0..) |name, i| {
        std.debug.print("  {s:<9} {d:>9.3} ms  {d:>5.1}%\n", .{ name, 1000.0 * p.gpu[i] / @as(f64, @floatFromInt(@max(tokens, 1))), 100.0 * p.gpu[i] / @max(total, 1e-12) });
    }
}

fn linBytes(w: g.weights.Linear) f64 {
    const n: f64 = @floatFromInt(w.n);
    const k: f64 = @floatFromInt(w.k);
    return n * k * @as(f64, @floatFromInt(w.bits)) / 8.0 + n * (k / @as(f64, @floatFromInt(w.group))) * 4.0;
}

/// Bytes a one-row step of `stage` reads over the whole stack (weights, and keys at the cache's length).
/// One expert's code bytes (the 4-bit words only).
fn linBytesW(x: anytype) usize {
    return x.one.n * x.one.k / 2;
}

fn stageBytes(m: *const g.Model, stage: g.forward.Stage, len: usize, experts_per_layer: f64) f64 {
    const c = &m.config;
    var total: f64 = 0;
    const W = m.weights;
    const keys: f64 = @floatFromInt(@min(len + 1, c.i_topk));
    for (W.layers[0..W.main]) |l| {
        const at = l.attn;
        total += switch (stage) {
            .proj_x => linBytes(at.x),
            .proj_q => linBytes(at.q_b) + if (at.indexer) |ix| linBytes(ix.wq_b) else 0,
            .absorb => linBytes(at.embed_q),
            .unembed => linBytes(at.unembed),
            .o_proj => linBytes(at.o),
            .attend => keys * 1152.0,
            .index => if (at.indexer != null) @as(f64, @floatFromInt(len + 1)) * 256.0 else 0,
            .experts => if (l.ffn == .moe) experts_per_layer * (linBytes(l.ffn.moe.gate.one) + linBytes(l.ffn.moe.up.one) + linBytes(l.ffn.moe.down.one)) else 0,
            .shared => if (l.ffn == .moe) linBytes(l.ffn.moe.shared.gate_up) + linBytes(l.ffn.moe.shared.down) else 0,
            .dense => if (l.ffn == .dense) linBytes(l.ffn.dense.gate_up) + linBytes(l.ffn.dense.down) else 0,
            .route => if (l.ffn == .moe) @as(f64, @floatFromInt(c.experts * c.hidden * 2)) else 0,
            else => 0,
        };
    }
    if (stage == .head) total += if (W.head) |h| linBytes(h) else 0;
    return total;
}

/// Each stage alone for a one-row step over every layer, one command buffer a step: GPU ms, GB/s, dispatches.
fn kbench(m: *g.Model, scratch: *g.state.Scratch, cache: *g.state.Cache, token: u32, reply: []const u32, reps: usize, steps: usize, t_decode: f64) !void {
    // GLM_KBENCH_ROWS (1-16): a verify pass's rows (the pending token and drafts), every row's logits as verify takes them
    const nrows: usize = @max(1, @min(env("GLM_KBENCH_ROWS", 1), g.state.window_rows));
    var idbuf: [g.state.window_rows]u32 = @splat(token);
    // the rows' tokens: the reply's last ones (a verify window's tokens differ, so do their experts), or with
    // GLM_KBENCH_SAME=1 one token repeated
    const same = platform.c.getenv("GLM_KBENCH_SAME") != null;
    if (!same and reply.len >= nrows) @memcpy(idbuf[0..nrows], reply[reply.len - nrows ..]);
    const ids = idbuf[0..nrows];
    const all_rows = nrows > 1;
    var all: f64 = 0;
    var disp: usize = 0;
    for (0..reps) |_| try g.forward.run(m, scratch, cache, ids, .{ .gpu = &all, .dispatches = &disp, .all = all_rows });
    const step_ms = 1000.0 * all / @as(f64, @floatFromInt(reps));
    // the experts the rows actually read: distinct routed experts a MoE layer, from one profiled run
    var dp: g.forward.Profile = .{};
    try g.forward.run(m, scratch, cache, ids, .{ .profile = &dp, .all = all_rows });
    const per_layer: f64 = if (dp.moe_layers > 0) @as(f64, @floatFromInt(dp.distinct)) / @as(f64, @floatFromInt(dp.moe_layers)) else @floatFromInt(m.config.topk);
    std.debug.print("kernel bench: {d:.2} distinct routed experts a MoE layer over {d} row(s) (rows' tokens: {s})\n", .{ per_layer, nrows, if (same or reply.len < nrows) "one token repeated" else "the reply's last tokens" });
    var bytes_all: f64 = 0;
    inline for (@typeInfo(g.forward.Stage).@"enum".field_names, 0..) |_, i| bytes_all += stageBytes(m, @enumFromInt(i), cache.len, per_layer);
    if (env("GLM_KBENCH_STREAM", 1) > 0) {
        // the reference: a plain streaming read of the model's own resident routed-expert weights (every MoE layer's
        // gate, up and down codes), one launch a matrix in one command buffer, the best of 5
        const out = try m.device.buffer(1 << 20, mtl.ResourceOptions.shared);
        defer out.deinit();
        const per: usize = 4096; // uint4s a threadgroup: 64 KiB
        var best: f64 = 1e9;
        var total: usize = 0;
        for (0..6) |rep| {
            const cb = m.queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(m.kernel(.stream));
            total = 0;
            for (m.weights.layers[0..m.weights.main]) |l| if (l.ffn == .moe) for ([_]@TypeOf(l.ffn.moe.gate){ l.ffn.moe.gate, l.ffn.moe.up, l.ffn.moe.down }) |x| {
                const bytes = x.one.w.length() - x.one.w_off;
                const want = @min(bytes, x.experts * linBytesW(x));
                const groups = want / 16 / per;
                if (groups == 0) continue;
                e.setBuffer(x.one.w, x.one.w_off, 0);
                e.setValue(g.forward.GArgs{ .width = @intCast(per) }, 1);
                e.setBuffer(out, 0, 2);
                e.dispatchGroups(mtl.Size.of(groups, 1, 1), mtl.Size.of(256, 1, 1));
                total += groups * per * 16;
            };
            e.end();
            cb.commit();
            cb.wait();
            if (rep > 0) best = @min(best, cb.gpuSeconds());
        }
        std.debug.print("kernel bench: streaming-read reference, the routed experts' codes ({d:.1} GB, a launch a matrix): {d:.2} ms = {d:.0} GB/s (best of 5)\n", .{ @as(f64, @floatFromInt(total)) / 1e9, 1000.0 * best, @as(f64, @floatFromInt(total)) / 1e9 / best });
    }
    std.debug.print("kernel bench, {d} row(s), at context {d}: a whole step {d:.2} ms GPU (wall {d:.2} ms a decoded token), {d} dispatches a step, {d:.2} GB read = {d:.0} GB/s\n", .{ nrows, cache.len, step_ms, 1000.0 * t_decode / @as(f64, @floatFromInt(@max(steps, 1))), disp / reps, bytes_all / 1e9, bytes_all / 1e9 / (step_ms / 1000.0) });
    var sum: f64 = 0;
    inline for (@typeInfo(g.forward.Stage).@"enum".field_names, 0..) |name, i| {
        var gpu: f64 = 0;
        var d: usize = 0;
        const st: g.forward.Stage = @enumFromInt(i);
        for (0..reps) |_| try g.forward.run(m, scratch, cache, ids, .{ .gpu = &gpu, .only = st, .dispatches = &d, .all = all_rows });
        const ms = 1000.0 * gpu / @as(f64, @floatFromInt(reps));
        sum += ms;
        const by = stageBytes(m, st, cache.len, per_layer);
        std.debug.print("  {s:<9} {d:>7.3} ms  {d:>4} dispatches  {d:>7.3} GB  {d:>6.0} GB/s\n", .{ name, ms, d / reps, by / 1e9, if (by > 0) by / 1e9 / (ms / 1000.0) else 0 });
    }
    std.debug.print("  sum of stages {d:.2} ms\n", .{sum});
}

/// FLOPs a prompt chunk of rows at positions p0.. spends in `stage` over the whole stack.
fn stageFlops(m: *const g.Model, stage: g.forward.Stage, p0: usize, rows: usize) f64 {
    const c = &m.config;
    const R: f64 = @floatFromInt(rows);
    var total: f64 = 0;
    const W = m.weights;
    var keys: f64 = 0; // listed keys over the rows
    var visible: f64 = 0; // visible keys over the rows (the index scores past the top-k)
    for (0..rows) |r| {
        keys += @floatFromInt(@min(p0 + r + 1, c.i_topk));
        visible += @floatFromInt(p0 + r + 1);
    }
    for (W.layers[0..W.main]) |l| {
        const at = l.attn;
        const lin = struct {
            fn f(w: g.weights.Linear, n: f64) f64 {
                return 2.0 * n * @as(f64, @floatFromInt(w.n)) * @as(f64, @floatFromInt(w.k));
            }
        }.f;
        total += switch (stage) {
            .proj_x => lin(at.x, R),
            .proj_q => lin(at.q_b, R) + if (at.indexer) |ix| lin(ix.wq_b, R) else 0,
            .absorb => 2.0 * R * 64.0 * 192.0 * 512.0,
            .unembed => 2.0 * R * 64.0 * 512.0 * 256.0,
            .o_proj => lin(at.o, R),
            .attend => keys * 64.0 * (576.0 + 512.0) * 2.0,
            .index => if (at.indexer != null and p0 + rows > c.i_topk) visible * 32.0 * 128.0 * 2.0 else 0,
            .experts => if (l.ffn == .moe) 2.0 * R * @as(f64, @floatFromInt(c.topk)) * 3.0 * 2048.0 * 6144.0 else 0,
            .shared => if (l.ffn == .moe) lin(l.ffn.moe.shared.gate_up, R) + lin(l.ffn.moe.shared.down, R) else 0,
            .dense => if (l.ffn == .dense) lin(l.ffn.dense.gate_up, R) + lin(l.ffn.dense.down, R) else 0,
            .route => if (l.ffn == .moe) 2.0 * R * 256.0 * 6144.0 else 0,
            else => 0,
        };
    }
    return total;
}

/// Each stage alone over the prompt's last full chunk (its rows rerun at their positions): GPU ms and TFLOPS.
fn pbench(m: *g.Model, big: *g.state.Scratch, cache: *g.state.Cache, prompt: []const u32, reps: usize) !void {
    const rows = g.state.prompt_rows;
    const p0 = prompt.len - rows;
    const ids = prompt[p0..];
    const keep = cache.len;
    defer cache.len = keep;
    cache.len = p0;
    var all: f64 = 0;
    var disp: usize = 0;
    for (0..reps) |_| try g.forward.run(m, big, cache, ids, .{ .head = false, .gpu = &all, .dispatches = &disp });
    var total_flop: f64 = 0;
    inline for (@typeInfo(g.forward.Stage).@"enum".field_names, 0..) |_, i| total_flop += stageFlops(m, @enumFromInt(i), p0, rows);
    const chunk_ms = 1000.0 * all / @as(f64, @floatFromInt(reps));
    std.debug.print("prompt-chunk bench, {d} rows at positions {d}..{d}: whole chunk {d:.1} ms GPU = {d:.0} tok/s, {d} dispatches, {d:.1} TFLOPS overall\n", .{ rows, p0, prompt.len - 1, chunk_ms, @as(f64, @floatFromInt(rows)) / (chunk_ms / 1000.0), disp / reps, total_flop / (chunk_ms / 1000.0) / 1e12 });
    inline for (@typeInfo(g.forward.Stage).@"enum".field_names, 0..) |name, i| {
        const st: g.forward.Stage = @enumFromInt(i);
        var gpu: f64 = 0;
        for (0..reps) |_| try g.forward.run(m, big, cache, ids, .{ .head = false, .gpu = &gpu, .only = st });
        const ms = 1000.0 * gpu / @as(f64, @floatFromInt(reps));
        const fl = stageFlops(m, st, p0, rows);
        std.debug.print("  {s:<9} {d:>8.2} ms  {d:>6.1} TFLOPS\n", .{ name, ms, if (fl > 0) fl / (ms / 1000.0) / 1e12 else 0 });
    }
    // a stage run alone writes the chunk's cache rows (prep, index_key) from stale scratch: rerun the whole chunk
    if (env("GLM_PBENCH_NORESTORE", 0) == 0) try g.forward.run(m, big, cache, ids, .{ .head = false });
}
