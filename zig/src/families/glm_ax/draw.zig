//! GLM-5.3's host draw: the shared keyed fp64 sampler (lanes.sampling.chooseOrdered) on a bounded candidate set.
//! The general row draw sorts all 154,880 logits in fp64 before its top_p cut (about 11 ms a token). Here one pass
//! over the bf16 keys finds the best `candidates` tokens in the sampler's own order (value, then the lower id); the
//! top_p cut is taken against the whole row's normalizer; the kept prefix goes to chooseOrdered with the cut already
//! applied, so its min_p floor and keyed Gumbel race are the canonical ones and the same seed draws the same token.
//! Bit-exact with top_k 1-1,024 (the normalizer is the same ordered sum); with top_k off the normalizer is summed in
//! another order (float64 rounding only). Masks, greedy, rows whose nucleus does not close inside the candidates and
//! non-finite rows take the general draw.
const std = @import("std");
const lanes = @import("lanes");
const row = lanes.sampling_row;
const smp = lanes.sampling;

extern "c" fn exp(x: f64) f64;

/// The sampler's order (lanes' sampling_selection.byValueThenId): value, then the lower id at a tie.
fn byValueThenId(_: void, a: smp.Candidate, b: smp.Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// The most tokens the bounded path keeps.
pub const candidates = 1024;
/// Tokens this far below the best (in logit / T) add under 2e-22 each to the normalizer: skipped in the tail sum.
const negligible = -50.0;

pub fn draw(gpa: std.mem.Allocator, logits: []const u16, settings: ?lanes.Sampling, position: u64, mask: []const u32) !u32 {
    const s = settings orelse return row.draw(gpa, logits, settings, position, mask);
    // the row draw is already fast for greedy and for a top_k of 128 or less
    if (mask.len > 0 or !(s.temperature > 0) or (s.top_k > 0 and s.top_k <= 128) or s.top_k > candidates)
        return row.draw(gpa, logits, settings, position, mask);
    if (bounded(gpa, logits, s, position) catch null) |token| return token;
    return row.draw(gpa, logits, settings, position, mask);
}

/// The bounded draw, or null for the general one (a non-finite word, no top_p cut, or one past the candidates).
pub fn bounded(gpa: std.mem.Allocator, logits: []const u16, s: lanes.Sampling, position: u64) !?u32 {
    const counts = try gpa.alloc(u32, 1 << 16);
    defer gpa.free(counts);
    @memset(counts, 0);
    for (logits) |w| {
        if (w & 0x7f80 == 0x7f80) return null; // non-finite: the general draw refuses it by name
        counts[row.key(w)] += 1;
    }
    const want: usize = @min(if (s.top_k > 0) @as(usize, s.top_k) else candidates, logits.len);
    // the threshold key: every token above it, and the lowest ids at it, make `want`
    var acc: usize = 0;
    var thr: usize = 0xffff;
    while (true) : (thr -= 1) {
        if (acc + counts[thr] >= want) break;
        acc += counts[thr];
        if (thr == 0) return null;
    }
    var at_thr = want - acc; // how many of the tokens at the threshold key enter (the lowest ids)
    const cands = try gpa.alloc(smp.Candidate, want);
    defer gpa.free(cands);
    var n: usize = 0;
    for (logits, 0..) |w, id| {
        const k = row.key(w);
        if (k > thr or (k == thr and at_thr > 0)) {
            if (k == thr) at_thr -= 1;
            cands[n] = .{ .value = row.value(w), .id = id };
            n += 1;
        }
    }
    std.debug.assert(n == want);
    std.mem.sort(smp.Candidate, cands, {}, byValueThenId);
    // with top_k on, chooseOrdered's own cut and normalizer run over exactly these tokens
    if (s.top_k > 0) return @intCast(try smp.chooseOrdered(gpa, cands, position, s));
    if (!(s.top_p > 0.0 and s.top_p < 1.0)) return null; // no rank cut: the general draw's one-pass race
    const t = @max(s.temperature, 1e-6);
    const top = cands[0].value / t;
    const probs = try gpa.alloc(f64, want);
    defer gpa.free(probs);
    for (probs, cands) |*p, c| p.* = exp(c.value / t - top);
    // the rest of the row's mass: every token outside the candidates
    var tail: f64 = 0;
    var left_at_thr = want - acc;
    for (logits) |w| {
        const k = row.key(w);
        if (k > thr) continue;
        if (k == thr and left_at_thr > 0) {
            left_at_thr -= 1;
            continue;
        }
        const x = row.value(w) / t - top;
        if (x > negligible) tail += exp(x);
    }
    const total = smp.pairwiseSum(probs) + tail;
    var cum: f64 = 0;
    var below: usize = 0;
    for (probs) |p| {
        cum += p / total;
        if (cum < s.top_p) below += 1 else break;
    }
    if (below + 1 > want) return null; // the nucleus is wider than the candidates
    var cut = s;
    cut.top_p = 1.0; // applied here
    cut.top_k = 0;
    return @intCast(try smp.chooseOrdered(gpa, cands[0 .. below + 1], position, cut));
}
