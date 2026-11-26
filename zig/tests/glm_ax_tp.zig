//! GLM-5.3 on MLX affine weights, tensor parallel over two ranks in one process (milestone A: the FFN split, attention
//! whole on both): rank 0 and rank 1 each load their share into their own Model and Metal queue, run on their own
//! thread, and meet at every FFN's exchange through the in-process fake pair (glm_ax/tp.zig). A prompt's chunks,
//! then a greedy reply one row a step (no drafts), as tf-glm-ax-run does; both ranks must hold the same bits.
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
    \\tf-glm-ax-tp MODEL_DIR PROMPT (a .txt file: a user message in GLM's chat format; else token ids)
    \\Both ranks of GLM_AX_TP_RANKS=2 in one process (about 410 GiB of weights: the 512 GB Mac only).
    \\GLM_AX_TP_CHANNELS (rank 0's channels of each expert's 2048, a multiple of 64, 1024), GLM_LAYERS (the first N
    \\layers, no head), GLM_SHARDS, GLM_MAX (reply tokens, 64), GLM_CONTEXT (8192), GLM_CHUNK (prompt rows a forward),
    \\GLM_DUMP=FILE (rank 0's last chunk's fp32 residual rows after the loaded layers, as tf-glm-ax-run writes it),
    \\GLM_FORCED=1 (prompt rows 16 at a time, each row's pick against the next token), GLM_TOPDUMP=FILE (with
    \\GLM_FORCED: every row's top-1 as u32), GLM_EFFORT (the system line's reasoning effort, Low)
;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

fn env(name: [:0]const u8, default: u64) u64 {
    const v = platform.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u64, std.mem.span(v), 10) catch default;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn since(t0: u64) f64 {
    return @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
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

/// The two ranks' agreement on each greedy token (rank 0's is taken; a rank-1 difference is counted).
const Tokens = struct {
    fp: *g.tp.FakePair,
    slot: [2]u32 = .{ 0, 0 },
    differ: usize = 0,

    fn agree(t: *Tokens, rank: u32, id: u32) !u32 {
        t.slot[rank] = id;
        try t.fp.sync(rank);
        const out = t.slot[0];
        if (rank == 1 and t.slot[1] != out) t.differ += 1;
        try t.fp.sync(rank);
        return out;
    }
};

const Job = struct {
    prompt: []const u32,
    max: usize,
    forced: bool,
    chunk: usize,
    tokens: *Tokens,
};

const Rank = struct {
    r: u32,
    m: *g.Model,
    scratch: g.state.Scratch = undefined,
    big: g.state.Scratch = undefined,
    cache: g.state.Cache = undefined,
    part: mtl.Buffer = undefined,
    pair: g.tp.Pair = undefined,
    reply: std.ArrayList(u32) = .empty,
    picks: std.ArrayList(u32) = .empty,
    agree: usize = 0,
    last_n: usize = 0,
    last_big: bool = false,
    t_prompt: f64 = 0,
    t_decode: f64 = 0,
    err: ?anyerror = null,

    fn lastResid(rk: *Rank) []const u8 {
        const s = if (rk.last_big) &rk.big else &rk.scratch;
        return s.resid.contents()[0 .. rk.last_n * rk.m.config.hidden * 4];
    }

    fn main(rk: *Rank, job: *const Job) void {
        rk.drive(job) catch |e| {
            rk.err = e;
            rk.pair.abort();
        };
    }

    fn drive(rk: *Rank, job: *const Job) !void {
        const m = rk.m;
        const c = &m.config;
        const head = m.weights.head != null;
        const prompt = job.prompt;
        rk.cache.len = 0;
        const t0 = nowNs();
        var at: usize = 0;
        while (at < prompt.len) {
            const n = @min(job.chunk, prompt.len - at);
            const big = n > g.state.window_rows;
            const s = if (big) &rk.big else &rk.scratch;
            try g.forward.run(m, s, &rk.cache, prompt[at .. at + n], .{ .all = job.forced, .head = head, .hidden = if (head) .{ .buf = s.hid } else null });
            if (job.forced) for (0..n) |r| {
                const pick = argmax(s.logits_all.slice(u16, n * c.vocab)[r * c.vocab ..][0..c.vocab]);
                try rk.picks.append(a, pick);
                if (at + r + 1 < prompt.len and pick == prompt[at + r + 1]) rk.agree += 1;
                if (r + 1 == n) @memcpy(rk.cache.logits.slice(u16, c.vocab), s.logits_all.slice(u16, n * c.vocab)[r * c.vocab ..][0..c.vocab]);
            };
            rk.cache.len += n;
            at += n;
            rk.last_n = n;
            rk.last_big = big;
        }
        rk.t_prompt = since(t0);
        if (job.max == 0) return;
        var next = try job.tokens.agree(rk.r, argmax(rk.cache.logits.slice(u16, c.vocab)));
        try rk.reply.append(a, next);
        const t1 = nowNs();
        while (rk.reply.items.len < job.max) {
            var done = false;
            for (c.eos[0..c.eos_n]) |e| done = done or e == next;
            if (done) break;
            try g.forward.run(m, &rk.scratch, &rk.cache, &.{next}, .{});
            rk.cache.len += 1;
            next = try job.tokens.agree(rk.r, argmax(rk.cache.logits.slice(u16, c.vocab)));
            try rk.reply.append(a, next);
        }
        rk.t_decode = since(t1);
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    // this harness is the two ranks: GLM_AX_TP_RANKS=2 whatever the shell set (GLM_AX_TP_RANK is ignored)
    _ = setenv("GLM_AX_TP_RANKS", "2", 1);
    const layers: ?u32 = if (platform.c.getenv("GLM_LAYERS") != null) @intCast(env("GLM_LAYERS", 78)) else null;
    const shards: ?usize = if (platform.c.getenv("GLM_SHARDS") != null) @intCast(env("GLM_SHARDS", 91)) else null;
    var ranks: [2]Rank = undefined;
    for (&ranks, 0..) |*rk, r| {
        const t0 = nowNs();
        const m = try g.Model.load(a, io, argv[1], .{ .layers = layers, .mtp = false, .shards = shards, .tp_rank = @intCast(r) });
        rk.* = .{ .r = @intCast(r), .m = m };
        std.debug.print("rank {d} loaded in {d:.1} s: {d} layers, expert channels [{d}, {d}), dense [{d}, {d}), {d:.2} GiB of weights, Metal allocated {d:.2} GiB\n", .{ r, since(t0), m.weights.main, m.tp.c0, m.tp.c1, m.tp.d0, m.tp.d1, @as(f64, @floatFromInt(m.checkpoint.residentBytes())) / (1 << 30), @as(f64, @floatFromInt(m.device.allocated())) / (1 << 30) });
    }
    defer for (&ranks) |*rk| rk.m.deinit();
    const c0 = &ranks[0].m.config;
    var t = try tok.loadTokenizer(io, a, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{argv[1]}));
    const max: usize = if (ranks[0].m.weights.head == null) 0 else @intCast(env("GLM_MAX", 64));
    const file = argv[2];
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
    if (prompt.items.len == 0) return error.EmptyPrompt;
    const forced = platform.c.getenv("GLM_FORCED") != null;
    const chunk: usize = if (forced) g.state.window_rows else @intCast(@max(1, env("GLM_CHUNK", g.state.prompt_rows)));
    const context: usize = @max(@as(usize, @intCast(env("GLM_CONTEXT", 8192))), prompt.items.len + max + 2);
    const rows: usize = @max(g.state.window_rows, @min(chunk, g.state.prompt_rows));
    const h: usize = c0.hidden;
    var fp = try g.tp.FakePair.init(a, rows * h);
    defer fp.deinit();
    for (&ranks) |*rk| {
        const m = rk.m;
        const c = &m.config;
        rk.scratch = try g.state.Scratch.init(m.device, c, context, g.state.window_rows);
        rk.big = try g.state.Scratch.init(m.device, c, context, g.state.prompt_rows);
        rk.cache = try g.state.Cache.init(m.device, c, context);
        rk.part = try m.device.buffer(rows * h * 4, mtl.ResourceOptions.shared);
        rk.pair = fp.pair(rk.r, rk.part, rows);
        m.pair = &rk.pair;
        std.debug.print("rank {d}: context {d}, caches {d:.2} GiB, Metal allocated {d:.2} GiB\n", .{ rk.r, context, @as(f64, @floatFromInt(g.state.cacheBytes(c, context))) / (1 << 30), @as(f64, @floatFromInt(m.device.allocated())) / (1 << 30) });
    }
    defer for (&ranks) |*rk| {
        rk.m.pair = null;
        rk.part.deinit();
        rk.cache.deinit();
        rk.big.deinit();
        rk.scratch.deinit();
    };
    std.debug.print("== {s}: {d} prompt tokens, chunks of {d}\n", .{ file, prompt.items.len, chunk });
    var tokens = Tokens{ .fp = &fp };
    const job = Job{ .prompt = prompt.items, .max = max, .forced = forced, .chunk = chunk, .tokens = &tokens };
    const t0 = nowNs();
    const other = try std.Thread.spawn(.{}, Rank.main, .{ &ranks[1], &job });
    ranks[0].main(&job);
    other.join();
    const wall = since(t0);
    for (&ranks) |*rk| if (rk.err) |e| {
        std.debug.print("rank {d} failed: {s}\n", .{ rk.r, @errorName(e) });
        return e;
    };
    const r0 = &ranks[0];
    const r1 = &ranks[1];
    std.debug.print("prompt: {d} tokens in {d:.3} s = {d:.1} tok/s", .{ prompt.items.len, r0.t_prompt, @as(f64, @floatFromInt(prompt.items.len)) / r0.t_prompt });
    if (forced) std.debug.print(", next-token agreement {d}/{d}", .{ r0.agree, prompt.items.len - 1 });
    std.debug.print("\n", .{});
    std.debug.print("exchanges: {d}, waits at the pair rank 0 {d:.3} s, rank 1 {d:.3} s, wall {d:.3} s\n", .{ fp.exchanges, @as(f64, @floatFromInt(fp.wait_ns[0])) / 1e9, @as(f64, @floatFromInt(fp.wait_ns[1])) / 1e9, wall });
    const same_resid = std.mem.eql(u8, r0.lastResid(), r1.lastResid());
    const same_picks = std.mem.eql(u32, r0.picks.items, r1.picks.items);
    std.debug.print("ranks alike: last chunk's fp32 residual {s}, forced picks {s}, reply tokens differing {d}\n", .{ if (same_resid) "bit-identical" else "DIFFERENT", if (same_picks) "identical" else "DIFFERENT", tokens.differ });
    if (platform.c.getenv("GLM_TOPDUMP")) |path| {
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(path), .data = std.mem.sliceAsBytes(r0.picks.items) });
        std.debug.print("dumped {d} rows' top-1 to {s}\n", .{ r0.picks.items.len, std.mem.span(path) });
    }
    if (platform.c.getenv("GLM_DUMP")) |path| {
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(path), .data = r0.lastResid() });
        std.debug.print("dumped {d} rows of fp32 residual to {s}\n", .{ r0.last_n, std.mem.span(path) });
    }
    if (max == 0) return;
    const steps = r0.reply.items.len - 1;
    std.debug.print("decode: {d} tokens in {d:.3} s = {d:.2} tok/s ({d:.2} ms a token) at context {d}\n", .{ steps, r0.t_decode, @as(f64, @floatFromInt(steps)) / r0.t_decode, 1000.0 * r0.t_decode / @as(f64, @floatFromInt(@max(steps, 1))), r0.cache.len });
    std.debug.print("ids:", .{});
    for (r0.reply.items) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\ntext: {s}\n", .{try t.decode(a, r0.reply.items, false)});
}
