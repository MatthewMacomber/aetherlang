//! Aether Distributed Computing & Cluster Node System
//! Provides multi-node actor message passing, remote actor spawning,
//! and cluster grid execution over TCP binary RPC.

const std = @import("std");
const runtime = @import("runtime.zig");
const Value = runtime.Value;

pub const MessageType = enum(u8) {
    ping = 0,
    pong = 1,
    spawn_actor = 2,
    spawn_reply = 3,
    send_actor = 4,
    send_reply = 5,
};

pub const HEADER_SIZE: usize = 25;

pub const Packet = struct {
    magic: [4]u8 = "AETH".*,
    msg_type: MessageType,
    sender_id: u64,
    recipient_id: u64,
    payload_len: u32,

    pub fn toBytes(self: Packet, out: *[HEADER_SIZE]u8) void {
        @memcpy(out[0..4], &self.magic);
        out[4] = @intFromEnum(self.msg_type);
        std.mem.writeInt(u64, out[5..13], self.sender_id, .little);
        std.mem.writeInt(u64, out[13..21], self.recipient_id, .little);
        std.mem.writeInt(u32, out[21..25], self.payload_len, .little);
    }

    pub fn fromBytes(bytes: *const [HEADER_SIZE]u8) !Packet {
        if (!std.mem.eql(u8, bytes[0..4], "AETH")) return error.InvalidMagic;
        const msg_type: MessageType = @enumFromInt(bytes[4]);
        const sender_id = std.mem.readInt(u64, bytes[5..13], .little);
        const recipient_id = std.mem.readInt(u64, bytes[13..21], .little);
        const payload_len = std.mem.readInt(u32, bytes[21..25], .little);
        return .{
            .msg_type = msg_type,
            .sender_id = sender_id,
            .recipient_id = recipient_id,
            .payload_len = payload_len,
        };
    }
};

pub const ClusterNode = struct {
    allocator: std.mem.Allocator,
    port: u16,
    is_running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    listen_thread: ?std.Thread = null,

    pub fn init(allocator: std.mem.Allocator, port: u16) ClusterNode {
        return .{
            .allocator = allocator,
            .port = port,
        };
    }

    pub fn start(self: *ClusterNode) !void {
        self.is_running.store(true, .release);
        self.listen_thread = try std.Thread.spawn(.{}, listenerLoop, .{self});
    }

    pub fn stop(self: *ClusterNode) void {
        self.is_running.store(false, .release);
        if (self.listen_thread) |th| {
            th.detach();
            self.listen_thread = null;
        }
    }

    fn listenerLoop(self: *ClusterNode) void {
        const address = std.net.Address.parseIp4("127.0.0.1", self.port) catch return;
        var server = address.listen(.{ .reuse_port = true }) catch return;
        defer server.deinit();

        while (self.is_running.load(.acquire)) {
            const conn = server.accept() catch continue;
            handleConnection(self.allocator, conn) catch {};
        }
    }

    fn handleConnection(allocator: std.mem.Allocator, conn: std.net.Server.Connection) !void {
        defer conn.stream.close();

        var header_buf: [HEADER_SIZE]u8 = undefined;
        var read_bytes: usize = 0;
        while (read_bytes < HEADER_SIZE) {
            const n = try conn.stream.read(header_buf[read_bytes..]);
            if (n == 0) return;
            read_bytes += n;
        }

        const packet = try Packet.fromBytes(&header_buf);
        const payload = try allocator.alloc(u8, packet.payload_len);
        defer allocator.free(payload);

        var p_read: usize = 0;
        while (p_read < packet.payload_len) {
            const n = try conn.stream.read(payload[p_read..]);
            if (n == 0) return;
            p_read += n;
        }

        switch (packet.msg_type) {
            .ping => {
                const reply = Packet{
                    .msg_type = .pong,
                    .sender_id = 0,
                    .recipient_id = packet.sender_id,
                    .payload_len = 0,
                };
                var rep_buf: [HEADER_SIZE]u8 = undefined;
                reply.toBytes(&rep_buf);
                _ = try conn.stream.write(&rep_buf);
            },
            .spawn_actor => {
                const init_val = runtime.string(payload);
                const actor_id_val = runtime.spawnActor(init_val);
                const actor_id: u64 = @intFromFloat(runtime.toF64(actor_id_val));

                const reply = Packet{
                    .msg_type = .spawn_reply,
                    .sender_id = actor_id,
                    .recipient_id = packet.sender_id,
                    .payload_len = 0,
                };
                var rep_buf: [HEADER_SIZE]u8 = undefined;
                reply.toBytes(&rep_buf);
                _ = try conn.stream.write(&rep_buf);
            },
            .send_actor => {
                const payload_val = runtime.string(payload);
                _ = runtime.sendActor(runtime.number(@floatFromInt(packet.recipient_id)), runtime.number(@floatFromInt(packet.sender_id)), payload_val);

                const reply = Packet{
                    .msg_type = .send_reply,
                    .sender_id = 0,
                    .recipient_id = packet.sender_id,
                    .payload_len = 0,
                };
                var rep_buf: [HEADER_SIZE]u8 = undefined;
                reply.toBytes(&rep_buf);
                _ = try conn.stream.write(&rep_buf);
            },
            else => {},
        }
    }
};

pub fn remoteSpawn(allocator: std.mem.Allocator, host: []const u8, port: u16, state_str: []const u8) !u64 {
    const address = try std.net.Address.parseIp4(host, port);
    const stream = try std.net.tcpConnectToAddress(address);
    defer stream.close();

    const req = Packet{
        .msg_type = .spawn_actor,
        .sender_id = 1,
        .recipient_id = 0,
        .payload_len = @intCast(state_str.len),
    };
    var req_buf: [HEADER_SIZE]u8 = undefined;
    req.toBytes(&req_buf);
    _ = try stream.write(&req_buf);
    if (state_str.len > 0) {
        _ = try stream.write(state_str);
    }

    var rep_buf: [HEADER_SIZE]u8 = undefined;
    var read_bytes: usize = 0;
    while (read_bytes < HEADER_SIZE) {
        const n = try stream.read(rep_buf[read_bytes..]);
        if (n == 0) return error.ConnectionClosed;
        read_bytes += n;
    }

    const reply = try Packet.fromBytes(&rep_buf);
    _ = allocator;
    return reply.sender_id;
}

pub fn remoteSend(allocator: std.mem.Allocator, host: []const u8, port: u16, recipient: u64, sender: u64, payload_str: []const u8) !bool {
    const address = try std.net.Address.parseIp4(host, port);
    const stream = try std.net.tcpConnectToAddress(address);
    defer stream.close();

    const req = Packet{
        .msg_type = .send_actor,
        .sender_id = sender,
        .recipient_id = recipient,
        .payload_len = @intCast(payload_str.len),
    };
    var req_buf: [HEADER_SIZE]u8 = undefined;
    req.toBytes(&req_buf);
    _ = try stream.write(&req_buf);
    if (payload_str.len > 0) {
        _ = try stream.write(payload_str);
    }

    var rep_buf: [HEADER_SIZE]u8 = undefined;
    var read_bytes: usize = 0;
    while (read_bytes < HEADER_SIZE) {
        const n = try stream.read(rep_buf[read_bytes..]);
        if (n == 0) return error.ConnectionClosed;
        read_bytes += n;
    }

    const reply = try Packet.fromBytes(&rep_buf);
    _ = allocator;
    return reply.msg_type == .send_reply;
}

pub const ClusterCipher = struct {
    pub const key_length = std.crypto.aead.chacha_poly.ChaCha20Poly1305.key_length;
    pub const nonce_length = std.crypto.aead.chacha_poly.ChaCha20Poly1305.nonce_length;
    pub const tag_length = std.crypto.aead.chacha_poly.ChaCha20Poly1305.tag_length;

    pub fn encrypt(c: []u8, tag: *[tag_length]u8, m: []const u8, npub: [nonce_length]u8, k: [key_length]u8) void {
        std.crypto.aead.chacha_poly.ChaCha20Poly1305.encrypt(c, tag, m, "", npub, k);
    }

    pub fn decrypt(m: []u8, c: []const u8, tag: [tag_length]u8, npub: [nonce_length]u8, k: [key_length]u8) !void {
        try std.crypto.aead.chacha_poly.ChaCha20Poly1305.decrypt(m, c, tag, "", npub, k);
    }
};

test "cluster packet serialization and deserialization" {
    var buf: [HEADER_SIZE]u8 = undefined;
    const p = Packet{
        .msg_type = .spawn_actor,
        .sender_id = 10,
        .recipient_id = 20,
        .payload_len = 5,
    };
    p.toBytes(&buf);

    const decoded = try Packet.fromBytes(&buf);
    try std.testing.expectEqual(MessageType.spawn_actor, decoded.msg_type);
    try std.testing.expectEqual(@as(u64, 10), decoded.sender_id);
    try std.testing.expectEqual(@as(u64, 20), decoded.recipient_id);
    try std.testing.expectEqual(@as(u32, 5), decoded.payload_len);
}

test "cluster authenticated chacha20-poly1305 packet encryption" {
    const key: [ClusterCipher.key_length]u8 = @splat(0x42);
    const nonce: [ClusterCipher.nonce_length]u8 = @splat(0x07);
    const plaintext = "Hello authenticated cluster mesh";

    var ciphertext: [plaintext.len]u8 = undefined;
    var tag: [ClusterCipher.tag_length]u8 = undefined;

    ClusterCipher.encrypt(&ciphertext, &tag, plaintext, nonce, key);

    // Verify ciphertext differs from plaintext
    try std.testing.expect(!std.mem.eql(u8, &ciphertext, plaintext));

    var decrypted: [plaintext.len]u8 = undefined;
    try ClusterCipher.decrypt(&decrypted, &ciphertext, tag, nonce, key);
    try std.testing.expectEqualStrings(plaintext, &decrypted);
}
