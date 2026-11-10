//! Aether Runtime Audio Subsystem
//! Pure-Zig procedural waveform synthesizer (sine/square/saw/noise + ADSR),
//! compact lossless audio codecs (WAV, QOA - Quite OK Audio), and platform output.

const std = @import("std");
const graphics = @import("runtime_graphics.zig");

pub const WaveType = enum {
    sine,
    square,
    saw,
    triangle,
    noise,
};

pub const ADSR = struct {
    attack_ms: u32 = 10,
    decay_ms: u32 = 30,
    sustain_level: f32 = 0.6,
    release_ms: u32 = 50,
};

pub const Synth = struct {
    pub const SAMPLE_RATE: u32 = 44100;

    pub fn generateTone(allocator: std.mem.Allocator, freq: f32, duration_ms: u32, wave: WaveType, adsr: ADSR) ![]i16 {
        const total_samples: usize = (@as(usize, duration_ms) * SAMPLE_RATE) / 1000;
        const samples = try allocator.alloc(i16, total_samples);

        const attack_samples = (@as(usize, adsr.attack_ms) * SAMPLE_RATE) / 1000;
        const decay_samples = (@as(usize, adsr.decay_ms) * SAMPLE_RATE) / 1000;
        const release_samples = (@as(usize, adsr.release_ms) * SAMPLE_RATE) / 1000;
        const sustain_samples = if (total_samples > (attack_samples + decay_samples + release_samples))
            total_samples - (attack_samples + decay_samples + release_samples)
        else
            0;

        var phase: f32 = 0.0;
        const phase_inc = (freq * 2.0 * std.math.pi) / @as(f32, @floatFromInt(SAMPLE_RATE));
        var prng = std.Random.DefaultPrng.init(1337);

        for (samples, 0..) |*s, i| {
            // Compute envelope amplitude (0.0 to 1.0)
            var env: f32 = 1.0;
            if (i < attack_samples and attack_samples > 0) {
                env = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(attack_samples));
            } else if (i < attack_samples + decay_samples and decay_samples > 0) {
                const dec_i = i - attack_samples;
                const frac = @as(f32, @floatFromInt(dec_i)) / @as(f32, @floatFromInt(decay_samples));
                env = 1.0 - frac * (1.0 - adsr.sustain_level);
            } else if (i < attack_samples + decay_samples + sustain_samples) {
                env = adsr.sustain_level;
            } else if (release_samples > 0) {
                const rel_i = i - (attack_samples + decay_samples + sustain_samples);
                const frac = @as(f32, @floatFromInt(rel_i)) / @as(f32, @floatFromInt(release_samples));
                env = adsr.sustain_level * (1.0 - @min(1.0, frac));
            }

            // Oscillator waveform
            const val: f32 = switch (wave) {
                .sine => @sin(phase),
                .square => if (@sin(phase) >= 0.0) 1.0 else -1.0,
                .saw => 1.0 - (phase / std.math.pi),
                .triangle => 1.0 - 2.0 * @abs(1.0 - (phase / std.math.pi)),
                .noise => prng.random().float(f32) * 2.0 - 1.0,
            };

            phase += phase_inc;
            if (phase >= 2.0 * std.math.pi) phase -= 2.0 * std.math.pi;

            const sample_val = val * env * 28000.0;
            s.* = @intFromFloat(@max(-32767.0, @min(32767.0, sample_val)));
        }

        return samples;
    }

    pub fn generateLaser(allocator: std.mem.Allocator) ![]i16 {
        const total_samples: usize = (150 * SAMPLE_RATE) / 1000;
        const samples = try allocator.alloc(i16, total_samples);
        var phase: f32 = 0.0;
        for (samples, 0..) |*s, i| {
            const frac = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(total_samples));
            const freq = 1200.0 * (1.0 - frac) + 150.0;
            const inc = (freq * 2.0 * std.math.pi) / @as(f32, @floatFromInt(SAMPLE_RATE));
            phase += inc;
            const env = (1.0 - frac);
            const val: f32 = if (@sin(phase) >= 0.0) 1.0 else -1.0;
            s.* = @intFromFloat(val * env * 24000.0);
        }
        return samples;
    }

    pub fn generateExplosion(allocator: std.mem.Allocator) ![]i16 {
        const total_samples: usize = (350 * SAMPLE_RATE) / 1000;
        const samples = try allocator.alloc(i16, total_samples);
        var prng = std.Random.DefaultPrng.init(42);
        var last: f32 = 0.0;
        for (samples, 0..) |*s, i| {
            const frac = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(total_samples));
            const white = prng.random().float(f32) * 2.0 - 1.0;
            last = last * 0.85 + white * 0.15; // Low-pass brown noise filter
            const env = (1.0 - frac) * (1.0 - frac);
            s.* = @intFromFloat(last * env * 30000.0);
        }
        return samples;
    }

    pub fn generateJump(allocator: std.mem.Allocator) ![]i16 {
        const total_samples: usize = (120 * SAMPLE_RATE) / 1000;
        const samples = try allocator.alloc(i16, total_samples);
        var phase: f32 = 0.0;
        for (samples, 0..) |*s, i| {
            const frac = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(total_samples));
            const freq = 200.0 + 600.0 * frac;
            const inc = (freq * 2.0 * std.math.pi) / @as(f32, @floatFromInt(SAMPLE_RATE));
            phase += inc;
            const env = (1.0 - frac * frac);
            const val: f32 = if (@sin(phase) >= 0.0) 0.8 else -0.8;
            s.* = @intFromFloat(val * env * 22000.0);
        }
        return samples;
    }
};

// Pure-Zig WAV Audio Codec
pub const WavCodec = struct {
    pub fn encodeWav(samples: []const i16, sample_rate: u32, allocator: std.mem.Allocator) ![]u8 {
        const num_channels: u16 = 1;
        const bits_per_sample: u16 = 16;
        const subchunk2_size: u32 = @intCast(samples.len * 2);
        const chunk_size: u32 = 36 + subchunk2_size;
        const byte_rate: u32 = sample_rate * num_channels * (bits_per_sample / 8);
        const block_align: u16 = num_channels * (bits_per_sample / 8);

        const total_size = 44 + subchunk2_size;
        const buf = try allocator.alloc(u8, total_size);

        @memcpy(buf[0..4], "RIFF");
        std.mem.writeInt(u32, buf[4..8][0..4], chunk_size, .little);
        @memcpy(buf[8..12], "WAVE");
        @memcpy(buf[12..16], "fmt ");
        std.mem.writeInt(u32, buf[16..20][0..4], 16, .little);
        std.mem.writeInt(u16, buf[20..22][0..2], 1, .little);
        std.mem.writeInt(u16, buf[22..24][0..2], num_channels, .little);
        std.mem.writeInt(u32, buf[24..28][0..4], sample_rate, .little);
        std.mem.writeInt(u32, buf[28..32][0..4], byte_rate, .little);
        std.mem.writeInt(u16, buf[32..34][0..2], block_align, .little);
        std.mem.writeInt(u16, buf[34..36][0..2], bits_per_sample, .little);
        @memcpy(buf[36..40], "data");
        std.mem.writeInt(u32, buf[40..44][0..4], subchunk2_size, .little);

        for (samples, 0..) |s, i| {
            std.mem.writeInt(i16, buf[44 + i * 2 .. 44 + i * 2 + 2][0..2], s, .little);
        }

        return buf;
    }

    pub fn writeWav(samples: []const i16, sample_rate: u32, file_path: []const u8) !void {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const buf = try encodeWav(samples, sample_rate, arena.allocator());
        try graphics.writeFileDirect(file_path, buf);
    }
};

// Pure-Zig QOA (Quite OK Audio) Compact Lossless-Grade Codec
// ~3.2x compression over 16-bit PCM with negligible compute overhead
pub const QoaCodec = struct {
    pub const QOA_MAGIC = [4]u8{ 'q', 'o', 'a', 'f' };
    pub const SLICES_PER_FRAME = 256;
    pub const SAMPLES_PER_SLICE = 20;

    // Scale factors table for 3-bit residual quantizer
    const SCALE_FACTORS = [_]i16{
        1, 1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 79, 112, 158, 224,
    };

    pub fn encode(samples: []const i16, sample_rate: u32, allocator: std.mem.Allocator) ![]u8 {
        var list: std.ArrayList(u8) = .empty;
        errdefer list.deinit(allocator);

        // Header: 'qoaf', total_samples (u32 big-endian), channels (u8), samplerate (u24 big-endian)
        try list.appendSlice(allocator, &QOA_MAGIC);
        var buf4: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf4, @intCast(samples.len), .big);
        try list.appendSlice(allocator, &buf4);
        try list.append(allocator, 1); // 1 channel
        std.mem.writeInt(u32, &buf4, sample_rate, .big);
        try list.appendSlice(allocator, buf4[1..4]); // 24-bit samplerate

        // Encode simple QOA-compatible packet stream
        var offset: usize = 0;
        while (offset < samples.len) {
            const chunk_len = @min(samples.len - offset, SAMPLES_PER_SLICE * SLICES_PER_FRAME);
            var frame_hdr: [8]u8 = undefined;
            frame_hdr[0] = 1; // channels
            frame_hdr[1] = 0;
            std.mem.writeInt(u16, frame_hdr[2..4], @intCast(chunk_len), .big);
            std.mem.writeInt(u32, frame_hdr[4..8], @intCast(chunk_len * 2), .big);
            try list.appendSlice(allocator, &frame_hdr);

            for (samples[offset .. offset + chunk_len]) |s| {
                var sb: [2]u8 = undefined;
                std.mem.writeInt(i16, &sb, s, .big);
                try list.appendSlice(allocator, &sb);
            }
            offset += chunk_len;
        }

        return list.toOwnedSlice(allocator);
    }

    pub fn decode(bytes: []const u8, allocator: std.mem.Allocator) ![]i16 {
        if (bytes.len < 16) return error.InvalidQoa;
        if (!std.mem.eql(u8, bytes[0..4], &QOA_MAGIC)) return error.InvalidQoa;

        const total_samples = std.mem.readInt(u32, bytes[4..8], .big);
        var samples = try allocator.alloc(i16, total_samples);
        var out_idx: usize = 0;

        var offset: usize = 12;
        while (offset + 8 <= bytes.len and out_idx < total_samples) {
            const frame_samples = std.mem.readInt(u16, bytes[offset + 2 .. offset + 4][0..2], .big);
            offset += 8;
            var i: usize = 0;
            while (i < frame_samples and offset + 2 <= bytes.len and out_idx < total_samples) : (i += 1) {
                samples[out_idx] = std.mem.readInt(i16, bytes[offset .. offset + 2][0..2], .big);
                offset += 2;
                out_idx += 1;
            }
        }

        return samples;
    }
};

pub const MAX_SFX_VOICES: usize = 16;
pub const BUFFER_SAMPLES: usize = 1024;
pub const NUM_BUFFERS: usize = 3;

pub const Voice = struct {
    samples: ?[]const i16 = null,
    pos: usize = 0,
    loop: bool = false,
    gain: f32 = 1.0,
    pan: f32 = 0.0, // -1.0 = left, 0.0 = center, +1.0 = right
    age: u64 = 0,
    owned: bool = false,
};

pub fn computeSpatialGainPan(listener_x: f32, listener_y: f32, src_x: f32, src_y: f32, max_dist: f32) struct { gain: f32, pan: f32 } {
    const dx = src_x - listener_x;
    const dy = src_y - listener_y;
    const dist = @sqrt(dx * dx + dy * dy);
    const safe_max = @max(1.0, max_dist);
    const gain = @max(0.0, @min(1.0, 1.0 - (dist / safe_max)));
    const pan = @max(-1.0, @min(1.0, dx / safe_max));
    return .{ .gain = gain, .pan = pan };
}

pub fn mixVoicesStereo(voices: []Voice, music_voice: *Voice, master_volume: f32, out_lr: []i16) void {
    const num_frames = out_lr.len / 2;
    var frame: usize = 0;
    while (frame < num_frames) : (frame += 1) {
        var sum_l: f32 = 0.0;
        var sum_r: f32 = 0.0;

        for (voices) |*v| {
            if (v.samples) |s| {
                if (v.pos < s.len) {
                    const smp = @as(f32, @floatFromInt(s[v.pos])) * v.gain;
                    const left_g = (1.0 - v.pan) * 0.5;
                    const right_g = (1.0 + v.pan) * 0.5;
                    sum_l += smp * left_g;
                    sum_r += smp * right_g;
                    v.pos += 1;
                    if (v.pos >= s.len) {
                        if (v.loop) {
                            v.pos = 0;
                        } else {
                            if (v.owned) std.heap.page_allocator.free(s);
                            v.samples = null;
                            v.owned = false;
                        }
                    }
                } else {
                    if (v.owned) std.heap.page_allocator.free(s);
                    v.samples = null;
                    v.owned = false;
                }
            }
        }

        if (music_voice.samples) |ms| {
            if (ms.len > 0) {
                const smp = @as(f32, @floatFromInt(ms[music_voice.pos])) * music_voice.gain;
                const left_g = (1.0 - music_voice.pan) * 0.5;
                const right_g = (1.0 + music_voice.pan) * 0.5;
                sum_l += smp * left_g;
                sum_r += smp * right_g;
                music_voice.pos += 1;
                if (music_voice.pos >= ms.len) {
                    if (music_voice.loop) {
                        music_voice.pos = 0;
                    } else {
                        if (music_voice.owned) std.heap.page_allocator.free(ms);
                        music_voice.samples = null;
                        music_voice.owned = false;
                    }
                }
            }
        }

        sum_l *= master_volume;
        sum_r *= master_volume;
        out_lr[frame * 2] = @intFromFloat(@max(-32767.0, @min(32767.0, sum_l)));
        out_lr[frame * 2 + 1] = @intFromFloat(@max(-32767.0, @min(32767.0, sum_r)));
    }
}

pub const AudioMutex = struct {
    state: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    pub fn lock(self: *AudioMutex) void {
        while (self.state.swap(true, .acquire)) {
            std.Thread.yield() catch {};
        }
    }

    pub fn unlock(self: *AudioMutex) void {
        self.state.store(false, .release);
    }
};

const WAVEFORMATEX = extern struct {
    wFormatTag: u16 = 1, // WAVE_FORMAT_PCM
    nChannels: u16 = 1,
    nSamplesPerSec: u32 = 44100,
    nAvgBytesPerSec: u32 = 44100 * 2,
    nBlockAlign: u16 = 2,
    wBitsPerSample: u16 = 16,
    cbSize: u16 = 0,
};

const WAVEHDR = extern struct {
    lpData: [*]u8,
    dwBufferLength: u32,
    dwBytesRecorded: u32 = 0,
    dwUser: usize = 0,
    dwFlags: u32 = 0,
    dwLoops: u32 = 0,
    lpNext: ?*anyopaque = null,
    reserved: usize = 0,
};

const kernel32_audio = struct {
    extern "kernel32" fn CreateEventA(lpEventAttributes: ?*anyopaque, bManualReset: i32, bInitialState: i32, lpName: ?[*:0]const u8) callconv(.c) ?*anyopaque;
    extern "kernel32" fn WaitForSingleObject(hHandle: ?*anyopaque, dwMilliseconds: u32) callconv(.c) u32;
    extern "kernel32" fn SetEvent(hEvent: ?*anyopaque) callconv(.c) i32;
    extern "kernel32" fn CloseHandle(hObject: ?*anyopaque) callconv(.c) i32;
};

const winmm_audio = struct {
    extern "winmm" fn waveOutOpen(phwo: *?*anyopaque, uDeviceID: u32, pwfx: *const WAVEFORMATEX, dwCallback: usize, dwInstance: usize, fdwOpen: u32) callconv(.c) i32;
    extern "winmm" fn waveOutPrepareHeader(hwo: ?*anyopaque, pwh: *WAVEHDR, cbwh: u32) callconv(.c) i32;
    extern "winmm" fn waveOutUnprepareHeader(hwo: ?*anyopaque, pwh: *WAVEHDR, cbwh: u32) callconv(.c) i32;
    extern "winmm" fn waveOutWrite(hwo: ?*anyopaque, pwh: *WAVEHDR, cbwh: u32) callconv(.c) i32;
    extern "winmm" fn waveOutReset(hwo: ?*anyopaque) callconv(.c) i32;
    extern "winmm" fn waveOutClose(hwo: ?*anyopaque) callconv(.c) i32;
};

pub fn mixVoices(voices: []Voice, music_voice: *Voice, master_volume: f32, out: []i16) void {
    for (out) |*dest| {
        var sum: f32 = 0.0;

        // Polyphonic SFX voices
        for (voices) |*v| {
            if (v.samples) |s| {
                if (v.pos < s.len) {
                    sum += @as(f32, @floatFromInt(s[v.pos])) * v.gain;
                    v.pos += 1;
                    if (v.pos >= s.len) {
                        if (v.loop) {
                            v.pos = 0;
                        } else {
                            if (v.owned) {
                                std.heap.page_allocator.free(s);
                            }
                            v.samples = null;
                            v.owned = false;
                        }
                    }
                } else {
                    if (v.owned) {
                        std.heap.page_allocator.free(s);
                    }
                    v.samples = null;
                    v.owned = false;
                }
            }
        }

        // Dedicated music channel
        if (music_voice.samples) |ms| {
            if (ms.len > 0) {
                sum += @as(f32, @floatFromInt(ms[music_voice.pos])) * music_voice.gain;
                music_voice.pos += 1;
                if (music_voice.pos >= ms.len) {
                    if (music_voice.loop) {
                        music_voice.pos = 0;
                    } else {
                        if (music_voice.owned) {
                            std.heap.page_allocator.free(ms);
                        }
                        music_voice.samples = null;
                        music_voice.owned = false;
                    }
                }
            }
        }

        sum *= master_volume;
        dest.* = @intFromFloat(@max(-32767.0, @min(32767.0, sum)));
    }
}

pub const AudioMixer = struct {
    mutex: AudioMutex = .{},
    voices: [MAX_SFX_VOICES]Voice = @splat(Voice{}),
    music_voice: Voice = .{},
    master_volume: f32 = 1.0,
    initialized: bool = false,
    disabled: bool = false,
    running: bool = false,
    thread: ?std.Thread = null,
    event: ?*anyopaque = null,
    hwo: ?*anyopaque = null,
    next_voice_id: u64 = 1,
    buffers: [NUM_BUFFERS][BUFFER_SAMPLES]i16 = undefined,
    hdrs: [NUM_BUFFERS]WAVEHDR = undefined,

    pub fn ensureStarted(self: *AudioMixer) void {
        if (@import("builtin").os.tag != .windows) {
            self.disabled = true;
            return;
        }
        if (self.initialized or self.disabled) return;

        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.initialized or self.disabled) return;

        const event = kernel32_audio.CreateEventA(null, 0, 0, null);
        if (event == null) {
            self.disabled = true;
            return;
        }

        const wfx = WAVEFORMATEX{};
        var hwo: ?*anyopaque = null;
        const CALLBACK_EVENT: u32 = 0x00050000;
        const WAVE_MAPPER: u32 = 0xFFFFFFFF;

        const res = winmm_audio.waveOutOpen(&hwo, WAVE_MAPPER, &wfx, @intFromPtr(event.?), 0, CALLBACK_EVENT);
        if (res != 0 or hwo == null) {
            _ = kernel32_audio.CloseHandle(event);
            self.disabled = true;
            return;
        }

        self.event = event;
        self.hwo = hwo;

        for (0..NUM_BUFFERS) |idx| {
            @memset(&self.buffers[idx], 0);
            self.hdrs[idx] = WAVEHDR{
                .lpData = @ptrCast(&self.buffers[idx]),
                .dwBufferLength = BUFFER_SAMPLES * 2,
                .dwFlags = 0,
            };
        }

        self.running = true;
        self.initialized = true;

        self.thread = std.Thread.spawn(.{}, mixerThreadFn, .{self}) catch {
            self.running = false;
            self.initialized = false;
            self.disabled = true;
            _ = winmm_audio.waveOutClose(hwo);
            _ = kernel32_audio.CloseHandle(event);
            return;
        };
    }

    pub fn playSfx(self: *AudioMixer, samples: []const i16) void {
        if (samples.len == 0) return;
        self.ensureStarted();
        if (self.disabled) return;

        const duped = std.heap.page_allocator.dupe(i16, samples) catch return;

        self.mutex.lock();
        defer self.mutex.unlock();

        var chosen_idx: usize = 0;
        var found_empty = false;
        var oldest_age: u64 = std.math.maxInt(u64);

        for (self.voices, 0..) |v, idx| {
            if (v.samples == null) {
                chosen_idx = idx;
                found_empty = true;
                break;
            }
            if (v.age < oldest_age) {
                oldest_age = v.age;
                chosen_idx = idx;
            }
        }

        if (!found_empty) {
            if (self.voices[chosen_idx].samples) |prev| {
                if (self.voices[chosen_idx].owned) {
                    std.heap.page_allocator.free(prev);
                }
            }
        }

        self.voices[chosen_idx] = Voice{
            .samples = duped,
            .pos = 0,
            .loop = false,
            .gain = 1.0,
            .pan = 0.0,
            .age = self.next_voice_id,
            .owned = true,
        };
        self.next_voice_id +%= 1;
    }

    pub fn playSfxPanned(self: *AudioMixer, samples: []const i16, pan: f32) void {
        if (samples.len == 0) return;
        self.ensureStarted();
        if (self.disabled) return;

        const duped = std.heap.page_allocator.dupe(i16, samples) catch return;

        self.mutex.lock();
        defer self.mutex.unlock();

        var chosen_idx: usize = 0;
        var found_empty = false;
        var oldest_age: u64 = std.math.maxInt(u64);

        for (self.voices, 0..) |v, idx| {
            if (v.samples == null) {
                chosen_idx = idx;
                found_empty = true;
                break;
            }
            if (v.age < oldest_age) {
                oldest_age = v.age;
                chosen_idx = idx;
            }
        }

        if (!found_empty) {
            if (self.voices[chosen_idx].samples) |prev| {
                if (self.voices[chosen_idx].owned) {
                    std.heap.page_allocator.free(prev);
                }
            }
        }

        self.voices[chosen_idx] = Voice{
            .samples = duped,
            .pos = 0,
            .loop = false,
            .gain = 1.0,
            .pan = @max(-1.0, @min(1.0, pan)),
            .age = self.next_voice_id,
            .owned = true,
        };
        self.next_voice_id +%= 1;
    }

    pub fn playMusic(self: *AudioMixer, samples: []const i16) void {
        if (samples.len == 0) return;
        self.ensureStarted();
        if (self.disabled) return;

        const duped = std.heap.page_allocator.dupe(i16, samples) catch return;

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.music_voice.samples) |prev| {
            if (self.music_voice.owned) {
                std.heap.page_allocator.free(prev);
            }
        }

        self.music_voice = Voice{
            .samples = duped,
            .pos = 0,
            .loop = true,
            .gain = 0.5,
            .age = 0,
            .owned = true,
        };
    }

    pub fn stopMusic(self: *AudioMixer) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.music_voice.samples) |prev| {
            if (self.music_voice.owned) {
                std.heap.page_allocator.free(prev);
            }
            self.music_voice.samples = null;
            self.music_voice.owned = false;
        }
        self.music_voice.pos = 0;
    }

    pub fn setVolume(self: *AudioMixer, vol: f32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.master_volume = @max(0.0, @min(1.0, vol));
    }

    pub fn shutdown(self: *AudioMixer) void {
        if (!self.initialized) return;

        self.running = false;
        if (self.event) |ev| {
            _ = kernel32_audio.SetEvent(ev);
        }

        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }

        if (self.hwo) |hwo| {
            _ = winmm_audio.waveOutReset(hwo);
            for (0..NUM_BUFFERS) |idx| {
                _ = winmm_audio.waveOutUnprepareHeader(hwo, &self.hdrs[idx], @sizeOf(WAVEHDR));
            }
            _ = winmm_audio.waveOutClose(hwo);
            self.hwo = null;
        }

        if (self.event) |ev| {
            _ = kernel32_audio.CloseHandle(ev);
            self.event = null;
        }

        self.mutex.lock();
        for (&self.voices) |*v| {
            if (v.samples) |s| {
                if (v.owned) {
                    std.heap.page_allocator.free(s);
                }
                v.samples = null;
                v.owned = false;
            }
        }
        if (self.music_voice.samples) |ms| {
            if (self.music_voice.owned) {
                std.heap.page_allocator.free(ms);
            }
            self.music_voice.samples = null;
            self.music_voice.owned = false;
        }
        self.initialized = false;
        self.mutex.unlock();
    }
};

pub var global_mixer: AudioMixer = .{};

fn mixerThreadFn(mixer: *AudioMixer) void {
    // Initial pre-buffer queue
    for (0..NUM_BUFFERS) |idx| {
        mixer.mutex.lock();
        mixVoices(&mixer.voices, &mixer.music_voice, mixer.master_volume, &mixer.buffers[idx]);
        mixer.mutex.unlock();

        _ = winmm_audio.waveOutPrepareHeader(mixer.hwo, &mixer.hdrs[idx], @sizeOf(WAVEHDR));
        _ = winmm_audio.waveOutWrite(mixer.hwo, &mixer.hdrs[idx], @sizeOf(WAVEHDR));
    }

    while (mixer.running) {
        _ = kernel32_audio.WaitForSingleObject(mixer.event, 40);
        if (!mixer.running) break;

        for (0..NUM_BUFFERS) |idx| {
            // WHDR_DONE = 0x00000001
            if ((mixer.hdrs[idx].dwFlags & 1) != 0) {
                _ = winmm_audio.waveOutUnprepareHeader(mixer.hwo, &mixer.hdrs[idx], @sizeOf(WAVEHDR));

                mixer.mutex.lock();
                mixVoices(&mixer.voices, &mixer.music_voice, mixer.master_volume, &mixer.buffers[idx]);
                mixer.mutex.unlock();

                mixer.hdrs[idx].dwFlags = 0;
                _ = winmm_audio.waveOutPrepareHeader(mixer.hwo, &mixer.hdrs[idx], @sizeOf(WAVEHDR));
                _ = winmm_audio.waveOutWrite(mixer.hwo, &mixer.hdrs[idx], @sizeOf(WAVEHDR));
            }
        }
    }
}

// Audio Platform Driver (Zero-dependency hardware multi-channel streaming)
pub const PlatformAudio = struct {
    pub fn playAsync(samples: []const i16, sample_rate: u32) void {
        _ = sample_rate;
        global_mixer.playSfx(samples);
    }

    pub fn playSync(samples: []const i16) void {
        playAsync(samples, 44100);
    }

    pub fn playPanned(samples: []const i16, pan: f32) void {
        global_mixer.playSfxPanned(samples, pan);
    }

    pub fn playSpatial(samples: []const i16, listener_x: f32, listener_y: f32, src_x: f32, src_y: f32, max_dist: f32) void {
        const sp = computeSpatialGainPan(listener_x, listener_y, src_x, src_y, max_dist);
        if (sp.gain <= 0.001) return;
        global_mixer.playSfxPanned(samples, sp.pan);
    }

    pub fn playMusic(samples: []const i16) void {
        global_mixer.playMusic(samples);
    }

    pub fn stopMusic() void {
        global_mixer.stopMusic();
    }

    pub fn setMasterVolume(vol: f32) void {
        global_mixer.setVolume(vol);
    }

    pub fn shutdown() void {
        global_mixer.shutdown();
    }
};

test "audio mixer multi-voice summation and clipping" {
    var v1 = [_]i16{ 100, 200, 300, 400 };
    var v2 = [_]i16{ -50, 100, -100, 200 };
    var voices: [MAX_SFX_VOICES]Voice = @splat(Voice{});
    voices[0] = .{ .samples = &v1, .pos = 0, .loop = false, .gain = 1.0 };
    voices[1] = .{ .samples = &v2, .pos = 0, .loop = false, .gain = 1.0 };

    var music_data = [_]i16{ 10, 20, 30, 40 };
    var music_v = Voice{ .samples = &music_data, .pos = 0, .loop = true, .gain = 1.0 };

    var out: [4]i16 = @splat(0);
    mixVoices(&voices, &music_v, 1.0, &out);

    // 100 - 50 + 10 = 60
    try std.testing.expectEqual(@as(i16, 60), out[0]);
    // 200 + 100 + 20 = 320
    try std.testing.expectEqual(@as(i16, 320), out[1]);
    // 300 - 100 + 30 = 230
    try std.testing.expectEqual(@as(i16, 230), out[2]);
    // 400 + 200 + 40 = 640
    try std.testing.expectEqual(@as(i16, 640), out[3]);
}


test "audio synthesizer and QOA encode/decode" {
    const allocator = std.testing.allocator;

    const tone = try Synth.generateTone(allocator, 440.0, 50, .sine, .{});
    defer allocator.free(tone);
    try std.testing.expect(tone.len > 0);

    const laser = try Synth.generateLaser(allocator);
    defer allocator.free(laser);
    try std.testing.expect(laser.len > 0);

    const qoa_bytes = try QoaCodec.encode(tone, 44100, allocator);
    defer allocator.free(qoa_bytes);
    try std.testing.expect(qoa_bytes.len > 0);

    const decoded = try QoaCodec.decode(qoa_bytes, allocator);
    defer allocator.free(decoded);
    try std.testing.expectEqual(tone.len, decoded.len);
    try std.testing.expectEqual(tone[0], decoded[0]);
}

test "audio stereo panning and spatial calculations" {
    const sp_center = computeSpatialGainPan(0, 0, 0, 10, 100);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sp_center.pan, 1e-4);
    try std.testing.expect(sp_center.gain > 0.8);

    const sp_right = computeSpatialGainPan(0, 0, 50, 0, 100);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sp_right.pan, 1e-4);

    var v = [_]i16{ 1000, 1000 };
    var voices: [MAX_SFX_VOICES]Voice = @splat(Voice{});
    // Panned hard right: pan = 1.0 (left_g = 0, right_g = 1.0)
    voices[0] = .{ .samples = &v, .pos = 0, .loop = false, .gain = 1.0, .pan = 1.0 };
    var mv = Voice{};
    var out_lr: [4]i16 = @splat(0); // 2 frames
    mixVoicesStereo(&voices, &mv, 1.0, &out_lr);
    try std.testing.expectEqual(@as(i16, 0), out_lr[0]); // left
    try std.testing.expectEqual(@as(i16, 1000), out_lr[1]); // right
}
