//! A stream's caches (every layer's latent and rope keys by position, full layers' index keys) and a window's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

/// Rows a decode forward takes at most: the router's launch limit; more take the prompt path (tensor-unit matmuls).
pub const window_rows = 16;
/// Rows a prompt chunk takes at most: each expert's weights are read once a chunk.
pub const prompt_rows = 2048;
/// Rows the indexer scores at once: its scores buffer is these rows by the context.
pub const score_rows = 64;
/// Rows the attention kernel takes a launch (its partial sums' buffer).
pub const attend_rows = 16;
/// Blocks a decode row's key list splits into for the tensor-unit attention (gx_sparse_split / gx_sparse_combine).
pub const attn_blocks = 16;
/// A cached key: 512 latent and 64 rope dims, bf16; an index key: 128 dims, bf16.
pub const key_bytes = 576 * 2;
pub const index_bytes = 128 * 2;
/// The x projections' stacked row on a full layer (q_a, kv_a, wk, padded weights_proj).
pub const proj_width = 2816;
const splits = 16;

/// A window row for the kernels (glm_ax.metal's gx_row).
pub const Row = extern struct { slot: u32 = 0, pos: u32, list: u32, nkeys: u32 = 0 };

fn stack(c: *const cfg.Config) usize {
    return c.layers + c.mtp;
}

fn fulls(c: *const cfg.Config) usize {
    var n: usize = 0;
    for (0..stack(c)) |i| n += @intFromBool(c.full.isSet(i));
    return n;
}

/// Bytes a stream of `capacity` positions holds in caches.
pub fn cacheBytes(c: *const cfg.Config, capacity: usize) usize {
    return @as(usize, c.vocab) * 2 + window_rows * @as(usize, c.hidden) * 4 + capacity * (stack(c) * key_bytes + fulls(c) * index_bytes);
}

pub const Cache = struct {
    kv: mtl.Buffer, // each layer's keys [capacity, 576] (the MTP layer after the main stack)
    index: mtl.Buffer, // full layers' index keys [capacity, 128]
    logits: mtl.Buffer,
    hidden: mtl.Buffer, // fp32 residual rows of the last window, before the final norm: the draft head's input
    held: [8]u32 = undefined, // the draft head's tokens for the next window
    nheld: usize = 0,
    /// Drafted tokens this stream's windows verified, and how many were kept (the acceptance line at release).
    drafted: u64 = 0,
    accepted: u64 = 0,
    /// The same by draft position (1st, 2nd, 3rd... row after the pending token).
    drafted_at: [8]u64 = @splat(0),
    accepted_at: [8]u64 = @splat(0),
    /// The round profile (GLM_AX_TRACE): verify and draft calls, their wall and GPU seconds, and the time between
    /// one backend call's end and the next's start (the engine and the sampler outside them).
    prof: struct { verify: u64 = 0, verify_wall: f64 = 0, verify_gpu: f64 = 0, draft: u64 = 0, draft_wall: f64 = 0, draft_gpu: f64 = 0, gap: f64 = 0, last_end: i128 = 0 } = .{},
    index_at: [cfg.max_layers]usize = @splat(0),
    capacity: usize,
    len: usize = 0,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn setResident(self: *Cache, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| mtl.resident.fields.apply(self, f, false);
        self.follow = hook;
        if (hook) |f| mtl.resident.fields.apply(self, f, true);
    }

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Cache {
        var out = Cache{ .kv = undefined, .index = undefined, .logits = undefined, .hidden = undefined, .capacity = capacity };
        var index: usize = 0;
        for (0..stack(c)) |i| if (c.full.isSet(i)) {
            out.index_at[i] = index;
            index += capacity * index_bytes;
        };
        out.kv = try device.buffer(stack(c) * capacity * key_bytes, mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.index = try device.buffer(@max(index, 16), mtl.ResourceOptions.private);
        errdefer out.index.deinit();
        out.logits = try device.buffer(@as(usize, c.vocab) * 2, mtl.ResourceOptions.shared);
        errdefer out.logits.deinit();
        out.hidden = try device.buffer(window_rows * @as(usize, c.hidden) * 4, mtl.ResourceOptions.shared);
        return out;
    }

    /// Layer `l`'s keys in `kv`.
    pub fn keys(self: *const Cache, l: usize) usize {
        return l * self.capacity * key_bytes;
    }

    /// Full layer `l`'s index keys in `index`.
    pub fn indexKeys(self: *const Cache, l: usize) usize {
        return self.index_at[l];
    }

    pub fn deinit(self: *Cache) void {
        self.setResident(null);
        self.kv.deinit();
        self.index.deinit();
        self.logits.deinit();
        self.hidden.deinit();
    }
};

pub const fields = [_][]const u8{ "ids", "ids2", "rowtab2", "resid", "x", "qa", "kva", "ik", "iw", "qr", "q", "iq", "iqr", "qlat", "qrope", "score", "keys", "rowtab", "partial", "olat", "o", "y", "shared", "gate", "up", "act", "eg", "eu", "ea", "ey", "logits32", "logits16", "pick", "wts", "xs", "sums", "offsets", "order", "cat", "hid", "chain", "logits_all", "po", "pm", "pl" };

/// Bytes of each Scratch buffer for `rows` rows at once, in `fields` order.
fn sizes(c: *const cfg.Config, capacity: usize, rows: usize) [fields.len]usize {
    const r: usize = rows;
    const prompt = rows > window_rows;
    const h: usize = c.hidden;
    const slots = r * c.topk;
    const nh: usize = c.heads;
    const inter: usize = @max(c.dense_inter, c.moe_inter);
    // the tensor-unit products' group sums: the widest [rows, k / 64] any product reads
    const sums = @max(@max(slots * h / 64, r * c.dense_inter / 64), @max(r * nh * c.kv_lora / 64, r * 2 * h / 64)) * 4;
    return .{
        r * 4, // ids
        r * 4, // ids2: the step encoded ahead
        r * @sizeOf(Row), // rowtab2
        r * h * 4, // resid
        r * h * 2, // x
        r * proj_width * 2, // qa: the x projections' stacked row (q_a, kv_a, wk, weights_proj)
        16, // kva (unused)
        16, // ik (unused)
        16, // iw (unused)
        r * c.q_lora * 2, // qr
        r * nh * c.qHead() * 2, // q
        r * c.i_heads * c.i_dim * 2, // iq
        r * c.i_heads * c.i_dim * 2, // iqr: roped for the tensor-unit scores
        r * nh * (c.kv_lora + c.rope) * 2, // qlat: each head's latent query and its roped 64
        16, // qrope (unused)
        @as(usize, @min(r, score_rows)) * capacity * 4, // score
        r * c.i_topk * 4, // keys
        r * @sizeOf(Row), // rowtab
        attend_rows * nh * splits * (c.kv_lora + 2) * 4, // partial
        r * nh * c.kv_lora * 2, // olat
        r * nh * c.v_dim * 2, // o
        r * h * 2, // y
        r * h * 2, // shared
        r * 2 * inter * 2, // gate: gate and up stacked
        r * inter * 2, // up
        r * inter * 2, // act
        slots * c.moe_inter * 2, // eg
        slots * c.moe_inter * 2, // eu
        r * (c.topk + 1) * c.moe_inter * 2, // ea: the routed picks' activations, the shared expert's after them (fused decode)
        slots * h * 2, // ey
        window_rows * c.experts * 4, // logits32
        r * c.experts * 2, // logits16: a prompt chunk's bf16 router logits
        slots * 4, // pick
        slots * 4, // wts
        if (prompt) slots * h * 2 else 16, // xs
        sums, // sums
        c.experts * 4, // offsets
        slots * 4, // order
        r * h * 2 * 2, // cat
        r * h * 4, // hid
        h * 4, // chain
        @as(usize, @min(r, window_rows)) * @as(usize, c.vocab) * 2, // logits_all
        @as(usize, @min(r, window_rows)) * attn_blocks * nh * c.kv_lora * 4, // po: decode attention's blocks, unnormalized
        @as(usize, @min(r, window_rows)) * attn_blocks * nh * 4, // pm
        @as(usize, @min(r, window_rows)) * attn_blocks * nh * 4, // pl
    };
}

/// Bytes a Scratch for `rows` rows holds, its router buffers included.
pub fn scratchBytes(c: *const cfg.Config, capacity: usize, rows: usize) usize {
    var total: usize = 7 * (@as(usize, c.experts) * window_rows * 4 + 4096);
    for (sizes(c, capacity, rows)) |n| total += n;
    return total;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    ids2: mtl.Buffer,
    rowtab2: mtl.Buffer,
    resid: mtl.Buffer, // fp32 [rows, hidden]
    x: mtl.Buffer, // bf16 normed rows
    qa: mtl.Buffer,
    kva: mtl.Buffer,
    ik: mtl.Buffer,
    iw: mtl.Buffer,
    qr: mtl.Buffer,
    q: mtl.Buffer,
    iq: mtl.Buffer,
    iqr: mtl.Buffer,
    qlat: mtl.Buffer,
    qrope: mtl.Buffer,
    score: mtl.Buffer,
    keys: mtl.Buffer, // each row's key list (the last full layer's; shared layers read it)
    rowtab: mtl.Buffer,
    partial: mtl.Buffer,
    olat: mtl.Buffer,
    o: mtl.Buffer,
    y: mtl.Buffer, // a sublayer's bf16 output [rows, hidden]
    shared: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    eg: mtl.Buffer, // [rows * topk, moe_inter] each
    eu: mtl.Buffer,
    ea: mtl.Buffer,
    ey: mtl.Buffer, // [rows * topk, hidden]
    logits32: mtl.Buffer,
    logits16: mtl.Buffer,
    pick: mtl.Buffer,
    wts: mtl.Buffer,
    xs: mtl.Buffer, // the prompt path: picks' input rows by expert, group sums, each expert's first pair, the pair order
    sums: mtl.Buffer,
    offsets: mtl.Buffer,
    order: mtl.Buffer,
    cat: mtl.Buffer, // the MTP head: [enorm(embed) | hnorm(hidden)] rows, kept residual rows, a chain's hidden, rows' logits
    hid: mtl.Buffer,
    chain: mtl.Buffer,
    logits_all: mtl.Buffer,
    po: mtl.Buffer,
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    route: [7]mtl.Buffer,
    rows: usize,
    capacity: usize,
    /// The last forward's command-buffer GPU seconds (the round profile).
    last_gpu: f64 = 0,

    /// Buffers for `rows` rows at once (window_rows for decode, prompt_rows for prompt chunks).
    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize, rows: usize) !Scratch {
        @setEvalBranchQuota(10000);
        const sizes_ = sizes(c, capacity, rows);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(sizes_[i], 16), mtl.ResourceOptions.shared);
            made += 1;
        }
        var routed: usize = 0;
        errdefer for (s.route[0..routed]) |b| b.deinit();
        for (&s.route) |*b| {
            b.* = try device.buffer(@as(usize, c.experts) * window_rows * 4 + 4096, mtl.ResourceOptions.shared);
            routed += 1;
        }
        s.rows = rows;
        s.capacity = capacity;
        return s;
    }

    /// The other ids and rows table: a step encoded ahead writes its own while this one's run.
    pub fn flip(s: *Scratch) void {
        std.mem.swap(mtl.Buffer, &s.ids, &s.ids2);
        std.mem.swap(mtl.Buffer, &s.rowtab, &s.rowtab2);
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
        for (s.route) |b| b.deinit();
    }
};
