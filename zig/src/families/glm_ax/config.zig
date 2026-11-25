//! GLM-5.3's shape for the affine engine: glm_dsa's config.json reader, one Mac.
const std = @import("std");
const dsa = @import("../glm_dsa/config.zig");

pub const Config = dsa.Config;
pub const max_layers = dsa.max_layers;
/// The longest context the engine serves.
pub const max_context = 1048576; // the checkpoint's max_position_embeddings (was 262144)

pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    return dsa.parse(gpa, bytes);
}

/// True when `dir` holds MLX shards (model*.safetensors) and no EXLR8 manifest.
pub fn isMlx(io: std.Io, dir: []const u8) bool {
    var buf: [1024]u8 = undefined;
    const m = std.fmt.bufPrint(&buf, "{s}/manifests", .{dir}) catch return false;
    if (std.Io.Dir.cwd().access(io, m, .{})) |_| return false else |_| {}
    const idx = std.fmt.bufPrint(&buf, "{s}/model.safetensors.index.json", .{dir}) catch return false;
    if (std.Io.Dir.cwd().access(io, idx, .{})) |_| return true else |_| {}
    const one = std.fmt.bufPrint(&buf, "{s}/model.safetensors", .{dir}) catch return false;
    if (std.Io.Dir.cwd().access(io, one, .{})) |_| return true else |_| {}
    return false;
}
