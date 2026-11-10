//! Aether Runtime Platform Subsystem
//! Native pure-Zig desktop windowing (Win32), TUI canvas fallback, and unified input/event loop.

const std = @import("std");
const graphics = @import("runtime_graphics.zig");
const Color = graphics.Color;
const Framebuffer = graphics.Framebuffer;
const tui = @import("runtime_tui.zig");

pub const Backend = enum {
    auto,
    window,
    tui,
    headless,
};

pub const PlatformConfig = struct {
    width: usize = 320,
    height: usize = 240,
    scale: usize = 2,
    title: []const u8 = "Aether",
    target_fps: u32 = 60,
    backend: Backend = .auto,
    fallback: bool = true,
    use_hardware_swapchain: bool = false,
};

pub const KeyCode = struct {
    pub const space: u16 = 32;
    pub const left: u16 = 37;
    pub const up: u16 = 38;
    pub const right: u16 = 39;
    pub const down: u16 = 40;
    pub const a: u16 = 65;
    pub const d: u16 = 68;
    pub const s: u16 = 83;
    pub const w: u16 = 87;
    pub const r: u16 = 82;
    pub const escape: u16 = 27;
    pub const enter: u16 = 13;
};

pub const GamepadButton = struct {
    pub const dpad_up: u16 = 0x0001;
    pub const dpad_down: u16 = 0x0002;
    pub const dpad_left: u16 = 0x0004;
    pub const dpad_right: u16 = 0x0008;
    pub const start: u16 = 0x0010;
    pub const back: u16 = 0x0020;
    pub const left_thumb: u16 = 0x0040;
    pub const right_thumb: u16 = 0x0080;
    pub const left_shoulder: u16 = 0x0100;
    pub const right_shoulder: u16 = 0x0200;
    pub const a: u16 = 0x1000;
    pub const b: u16 = 0x2000;
    pub const x: u16 = 0x4000;
    pub const y: u16 = 0x8000;
};

pub const GamepadState = struct {
    connected: bool = false,
    buttons: u16 = 0,
    prev_buttons: u16 = 0,
    lx: f32 = 0.0,
    ly: f32 = 0.0,
    rx: f32 = 0.0,
    ry: f32 = 0.0,
    lt: f32 = 0.0,
    rt: f32 = 0.0,
};

pub const ActionBinding = struct {
    primary_key: u16 = 0,
    secondary_key: u16 = 0,
    gamepad_button: u16 = 0,
};

const XINPUT_GAMEPAD = extern struct {
    wButtons: u16,
    bLeftTrigger: u8,
    bRightTrigger: u8,
    sThumbLX: i16,
    sThumbLY: i16,
    sThumbRX: i16,
    sThumbRY: i16,
};
const XINPUT_STATE = extern struct {
    dwPacketNumber: u32,
    Gamepad: XINPUT_GAMEPAD,
};
const XInputGetStateFn = *const fn (u32, *XINPUT_STATE) callconv(.c) u32;

var xinput_get_state: ?XInputGetStateFn = null;
var xinput_probed = false;

fn probeXInput() ?XInputGetStateFn {
    if (xinput_probed) return xinput_get_state;
    xinput_probed = true;
    if (@import("builtin").os.tag != .windows) return null;
    const kernel32_dll = struct {
        extern "kernel32" fn LoadLibraryA([*:0]const u8) callconv(.c) ?*anyopaque;
        extern "kernel32" fn GetProcAddress(?*anyopaque, [*:0]const u8) callconv(.c) ?*const anyopaque;
    };
    const dll_names = [_][*:0]const u8{ "xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll" };
    for (dll_names) |name| {
        if (kernel32_dll.LoadLibraryA(name)) |handle| {
            if (kernel32_dll.GetProcAddress(handle, "XInputGetState")) |proc| {
                const fn_ptr: XInputGetStateFn = @ptrCast(proc);
                xinput_get_state = fn_ptr;
                return fn_ptr;
            }
        }
    }
    return null;
}

pub fn showCrashDialog(title: []const u8, message: []const u8) void {
    if (@import("builtin").os.tag == .windows) {
        const user32_box = struct {
            extern "user32" fn MessageBoxW(?*anyopaque, [*:0]const u16, [*:0]const u16, u32) callconv(.c) i32;
        };
        var title_utf16: [256]u16 = undefined;
        const t_len = std.unicode.utf8ToUtf16Le(&title_utf16, title) catch 0;
        title_utf16[t_len] = 0;

        var msg_utf16: [1024]u16 = undefined;
        const m_len = std.unicode.utf8ToUtf16Le(&msg_utf16, message) catch 0;
        msg_utf16[m_len] = 0;

        _ = user32_box.MessageBoxW(null, &msg_utf16, &title_utf16, 0x00000010); // MB_ICONERROR
    } else {
        std.debug.print("CRASH [{s}]: {s}\n", .{ title, message });
    }
}

const kernel32 = struct {
    extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.c) i32;
    extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.c) i32;
    extern "kernel32" fn Sleep(dwMilliseconds: u32) callconv(.c) void;
};

pub fn getTimestampSeconds() f64 {
    if (@import("builtin").os.tag == .windows) {
        var count: i64 = 0;
        var freq: i64 = 1;
        _ = kernel32.QueryPerformanceCounter(&count);
        _ = kernel32.QueryPerformanceFrequency(&freq);
        return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(freq));
    } else {
        const ns = std.time.nanoTimestamp();
        return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
    }
}

pub fn sleepMs(ms: u32) void {
    if (@import("builtin").os.tag == .windows) {
        kernel32.Sleep(ms);
    } else {
        std.Thread.sleep(@as(u64, ms) * std.time.ns_per_ms);
    }
}

pub fn printStdoutDirect(msg: []const u8) void {
    if (@import("builtin").os.tag == .windows) {
        const kernel32_out = struct {
            extern "kernel32" fn GetStdHandle(nStdHandle: i32) callconv(.c) ?*anyopaque;
            extern "kernel32" fn WriteFile(hFile: ?*anyopaque, lpBuffer: [*]const u8, nNumberOfBytesToWrite: u32, lpNumberOfBytesWritten: ?*u32, lpOverlapped: ?*anyopaque) callconv(.c) i32;
        };
        const handle = kernel32_out.GetStdHandle(-11); // STD_OUTPUT_HANDLE
        var written: u32 = 0;
        _ = kernel32_out.WriteFile(handle, msg.ptr, @intCast(msg.len), &written, null);
    } else {
        std.debug.print("{s}", .{msg});
    }
}

pub const Platform = struct {
    config: PlatformConfig,
    active_backend: Backend,
    hwnd: ?*anyopaque = null,
    hdc: ?*anyopaque = null,
    keys: [256]bool = @splat(false),
    prev_keys: [256]bool = @splat(false),
    mouse_x: i32 = 0,
    mouse_y: i32 = 0,
    mouse_buttons: [3]bool = @splat(false),
    prev_mouse_buttons: [3]bool = @splat(false),
    mouse_clicks: [3]bool = @splat(false),
    gamepad: GamepadState = .{},
    bindings: [32]ActionBinding = @splat(ActionBinding{}),
    should_close: bool = false,
    client_w: i32 = 0,
    client_h: i32 = 0,
    resized: bool = false,
    tui_canvas: ?tui.TuiCanvas = null,
    last_frame_time_s: f64 = 0.0,
    dt: f64 = 0.0166667,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: PlatformConfig) !Platform {
        var plat = Platform{
            .config = config,
            .active_backend = config.backend,
            .allocator = allocator,
            .last_frame_time_s = getTimestampSeconds(),
        };

        const want_window = (config.backend == .auto or config.backend == .window);
        if (want_window and @import("builtin").os.tag == .windows) {
            const win_ok = plat.initWin32Window();
            if (win_ok) {
                plat.active_backend = .window;
                return plat;
            }
            if (!config.fallback and config.backend == .window) {
                return error.WindowInitializationFailed;
            }
        }

        if (config.backend == .headless) {
            plat.active_backend = .headless;
        } else {
            // Default or fallback to TUI
            plat.active_backend = .tui;
            plat.tui_canvas = try tui.TuiCanvas.init(allocator, config.width, config.height);
        }

        return plat;
    }

    pub fn deinit(self: *Platform) void {
        if (self.active_backend == .window and @import("builtin").os.tag == .windows and self.hwnd != null) {
            self.deinitWin32Window();
        }
        if (self.tui_canvas) |*tc| {
            tc.deinit();
        }
    }

    pub fn pollEvents(self: *Platform) void {
        global_platform_instance = self;
        @memcpy(&self.prev_keys, &self.keys);
        @memcpy(&self.prev_mouse_buttons, &self.mouse_buttons);
        @memset(&self.mouse_clicks, false);

        if (self.active_backend == .window and @import("builtin").os.tag == .windows) {
            self.pollWin32Events();
        } else if (self.active_backend == .tui) {
            self.pollTuiEvents();
        }

        self.gamepad.prev_buttons = self.gamepad.buttons;
        if (probeXInput()) |get_state| {
            var state: XINPUT_STATE = undefined;
            if (get_state(0, &state) == 0) {
                self.gamepad.connected = true;
                self.gamepad.buttons = state.Gamepad.wButtons;
                const deadzone: i16 = 7849;
                const lx_raw = state.Gamepad.sThumbLX;
                const ly_raw = state.Gamepad.sThumbLY;
                self.gamepad.lx = if (@abs(lx_raw) > deadzone) @as(f32, @floatFromInt(lx_raw)) / 32767.0 else 0.0;
                self.gamepad.ly = if (@abs(ly_raw) > deadzone) @as(f32, @floatFromInt(ly_raw)) / 32767.0 else 0.0;
                self.gamepad.lt = @as(f32, @floatFromInt(state.Gamepad.bLeftTrigger)) / 255.0;
                self.gamepad.rt = @as(f32, @floatFromInt(state.Gamepad.bRightTrigger)) / 255.0;
            } else {
                self.gamepad.connected = false;
            }
        }

        const now = getTimestampSeconds();
        self.dt = @max(0.0001, now - self.last_frame_time_s);
        self.last_frame_time_s = now;
    }

    pub fn isKeyDown(self: *const Platform, vk: u16) bool {
        if (vk < 256) return self.keys[vk];
        return false;
    }

    pub fn isKeyPressed(self: *const Platform, vk: u16) bool {
        if (vk < 256) return self.keys[vk] and !self.prev_keys[vk];
        return false;
    }

    pub fn isMouseButtonDown(self: *const Platform, btn: usize) bool {
        if (btn < 3) return self.mouse_buttons[btn];
        return false;
    }

    pub fn isMouseButtonPressed(self: *const Platform, btn: usize) bool {
        if (btn < 3) return self.mouse_clicks[btn] or (self.mouse_buttons[btn] and !self.prev_mouse_buttons[btn]);
        return false;
    }

    pub fn bindAction(self: *Platform, action_id: usize, primary: u16, secondary: u16, pad_btn: u16) void {
        if (action_id < 32) {
            self.bindings[action_id] = .{
                .primary_key = primary,
                .secondary_key = secondary,
                .gamepad_button = pad_btn,
            };
        }
    }

    pub fn isActionDown(self: *const Platform, action_id: usize) bool {
        if (action_id >= 32) return false;
        const b = self.bindings[action_id];
        if (b.primary_key != 0 and self.isKeyDown(b.primary_key)) return true;
        if (b.secondary_key != 0 and self.isKeyDown(b.secondary_key)) return true;
        if (b.gamepad_button != 0 and (self.gamepad.buttons & b.gamepad_button) != 0) return true;
        return false;
    }

    pub fn isActionPressed(self: *const Platform, action_id: usize) bool {
        if (action_id >= 32) return false;
        const b = self.bindings[action_id];
        if (b.primary_key != 0 and self.isKeyPressed(b.primary_key)) return true;
        if (b.secondary_key != 0 and self.isKeyPressed(b.secondary_key)) return true;
        if (b.gamepad_button != 0 and (self.gamepad.buttons & b.gamepad_button) != 0 and (self.gamepad.prev_buttons & b.gamepad_button) == 0) return true;
        return false;
    }

    pub fn present(self: *Platform, fb: *const Framebuffer) !void {
        if (self.active_backend == .window and @import("builtin").os.tag == .windows) {
            self.presentWin32(fb);
        } else if (self.active_backend == .tui) {
            if (self.tui_canvas) |*tc| {
                @memcpy(tc.fb.pixels, fb.pixels);
                var stdout_buf: [16384]u8 = undefined;
                var bw = tui.BufferWriter{ .buf = &stdout_buf };
                tui.clearScreen(&bw) catch {};
                tc.render(&bw) catch {};
                printStdoutDirect(bw.buf[0..bw.len]);
            }
        }

        // Frame rate limiter
        if (self.config.target_fps > 0) {
            const target_dt = 1.0 / @as(f64, @floatFromInt(self.config.target_fps));
            const elapsed = getTimestampSeconds() - self.last_frame_time_s;
            if (elapsed < target_dt) {
                const sleep_ms: u32 = @intFromFloat((target_dt - elapsed) * 1000.0);
                if (sleep_ms > 0) sleepMs(sleep_ms);
            }
        }
    }

    // Windows Native GDI/DIB Implementation
    fn initWin32Window(self: *Platform) bool {
        if (@import("builtin").os.tag != .windows) return false;

        const user32 = struct {
            pub const WNDCLASSEXW = extern struct {
                cbSize: u32 = @sizeOf(WNDCLASSEXW),
                style: u32 = 0x0003, // CS_HREDRAW | CS_VREDRAW
                lpfnWndProc: *const fn (?*anyopaque, u32, usize, isize) callconv(.c) isize,
                cbClsExtra: i32 = 0,
                cbWndExtra: i32 = 0,
                hInstance: ?*anyopaque = null,
                hIcon: ?*anyopaque = null,
                hCursor: ?*anyopaque = null,
                hbrBackground: ?*anyopaque = null,
                lpszMenuName: ?[*:0]const u16 = null,
                lpszClassName: [*:0]const u16,
                hIconSm: ?*anyopaque = null,
            };
            extern "user32" fn RegisterClassExW(*const WNDCLASSEXW) callconv(.c) u16;
            extern "user32" fn CreateWindowExW(u32, [*:0]const u16, [*:0]const u16, u32, i32, i32, i32, i32, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
            extern "user32" fn ShowWindow(?*anyopaque, i32) callconv(.c) i32;
            extern "user32" fn UpdateWindow(?*anyopaque) callconv(.c) i32;
            extern "user32" fn SetFocus(?*anyopaque) callconv(.c) ?*anyopaque;
            extern "user32" fn LoadCursorA(?*anyopaque, ?[*:0]const u8) callconv(.c) ?*anyopaque;
            extern "user32" fn GetDC(?*anyopaque) callconv(.c) ?*anyopaque;
            extern "user32" fn DefWindowProcW(?*anyopaque, u32, usize, isize) callconv(.c) isize;
        };

        const class_name = std.unicode.utf8ToUtf16LeStringLiteral("AetherEngineWindowClass");
        var title_buf: [256]u16 = undefined;
        const title_len = std.unicode.utf8ToUtf16Le(&title_buf, self.config.title) catch 0;
        title_buf[title_len] = 0;
        const title_utf16: [*:0]const u16 = title_buf[0..title_len :0];

        const wc = user32.WNDCLASSEXW{
            .lpfnWndProc = win32WndProc,
            .hCursor = user32.LoadCursorA(null, @ptrFromInt(32512)), // IDC_ARROW
            .lpszClassName = class_name,
        };

        _ = user32.RegisterClassExW(&wc);

        const w: i32 = @intCast(self.config.width * self.config.scale);
        const h: i32 = @intCast(self.config.height * self.config.scale);
        const style: u32 = 0x00CF0000; // WS_OVERLAPPEDWINDOW

        const hwnd = user32.CreateWindowExW(
            0,
            class_name,
            title_utf16,
            style,
            100,
            100,
            w + 16,
            h + 39,
            null,
            null,
            null,
            null,
        );

        if (hwnd == null) return false;

        self.hwnd = hwnd;
        self.hdc = user32.GetDC(hwnd);
        self.client_w = w;
        self.client_h = h;

        _ = user32.ShowWindow(hwnd, 1);
        _ = user32.UpdateWindow(hwnd);
        _ = user32.SetFocus(hwnd);
        global_platform_instance = self;

        return true;
    }

    fn deinitWin32Window(self: *Platform) void {
        const user32 = struct {
            extern "user32" fn ReleaseDC(?*anyopaque, ?*anyopaque) callconv(.c) i32;
            extern "user32" fn DestroyWindow(?*anyopaque) callconv(.c) i32;
        };
        if (self.hdc != null and self.hwnd != null) {
            _ = user32.ReleaseDC(self.hwnd, self.hdc);
        }
        if (self.hwnd != null) {
            _ = user32.DestroyWindow(self.hwnd);
        }
        global_platform_instance = null;
    }

    fn pollWin32Events(self: *Platform) void {
        const user32 = struct {
            pub const MSG = extern struct {
                hwnd: ?*anyopaque,
                message: u32,
                wParam: usize,
                lParam: isize,
                time: u32,
                pt: extern struct { x: i32, y: i32 },
            };
            extern "user32" fn PeekMessageW(*MSG, ?*anyopaque, u32, u32, u32) callconv(.c) i32;
            extern "user32" fn TranslateMessage(*const MSG) callconv(.c) i32;
            extern "user32" fn DispatchMessageW(*const MSG) callconv(.c) isize;
        };

        var msg: user32.MSG = undefined;
        while (user32.PeekMessageW(&msg, null, 0, 0, 1) != 0) {
            if (msg.message == 0x0012) { // WM_QUIT
                self.should_close = true;
                break;
            }
            _ = user32.TranslateMessage(&msg);
            _ = user32.DispatchMessageW(&msg);
        }
    }

    fn pollTuiEvents(self: *Platform) void {
        // Poll Windows console keys if on Windows
        if (@import("builtin").os.tag == .windows) {
            const vk_codes = [_]u16{
                KeyCode.space, KeyCode.left, KeyCode.up, KeyCode.right, KeyCode.down,
                KeyCode.w,     KeyCode.a,    KeyCode.s,  KeyCode.d,     KeyCode.r,
                KeyCode.escape,
            };
            for (vk_codes) |vk| {
                self.keys[vk] = tui.WindowsConsoleInput.isKeyDown(vk);
            }
            if (self.keys[KeyCode.escape]) self.should_close = true;
        }
    }

    fn presentWin32(self: *Platform, fb: *const Framebuffer) void {
        if (self.hdc == null) return;
        const gdi32 = struct {
            pub const BITMAPINFO = extern struct {
                biSize: u32 = 40,
                biWidth: i32,
                biHeight: i32,
                biPlanes: u16 = 1,
                biBitCount: u16 = 32,
                biCompression: u32 = 0, // BI_RGB
                biSizeImage: u32 = 0,
                biXPelsPerMeter: i32 = 0,
                biYPelsPerMeter: i32 = 0,
                biClrUsed: u32 = 0,
                biClrImportant: u32 = 0,
            };
            extern "gdi32" fn StretchDIBits(?*anyopaque, i32, i32, i32, i32, i32, i32, i32, i32, ?*const anyopaque, *const BITMAPINFO, u32, u32) callconv(.c) i32;
        };

        // Note: DIB with negative height renders top-down
        const bmi = gdi32.BITMAPINFO{
            .biWidth = @intCast(fb.width),
            .biHeight = -@as(i32, @intCast(fb.height)),
        };

        const dst_w: i32 = if (self.client_w > 0) self.client_w else @intCast(fb.width * self.config.scale);
        const dst_h: i32 = if (self.client_h > 0) self.client_h else @intCast(fb.height * self.config.scale);

        _ = gdi32.StretchDIBits(
            self.hdc,
            0,
            0,
            dst_w,
            dst_h,
            0,
            0,
            @intCast(fb.width),
            @intCast(fb.height),
            fb.pixels.ptr,
            &bmi,
            0,
            0x00CC0020, // SRCCOPY
        );
    }
};

pub var global_platform_instance: ?*Platform = null;

fn mapMouseCoords(plat: *Platform, raw_x: i32, raw_y: i32) void {
    const cw: i32 = if (plat.client_w > 0) plat.client_w else @intCast(plat.config.width * plat.config.scale);
    const ch: i32 = if (plat.client_h > 0) plat.client_h else @intCast(plat.config.height * plat.config.scale);
    if (cw > 0 and ch > 0) {
        const fw: i32 = @intCast(plat.config.width);
        const fh: i32 = @intCast(plat.config.height);
        plat.mouse_x = std.math.clamp(@divTrunc(raw_x * fw, cw), 0, fw - 1);
        plat.mouse_y = std.math.clamp(@divTrunc(raw_y * fh, ch), 0, fh - 1);
    }
}

fn win32WndProc(hwnd: ?*anyopaque, msg: u32, wparam: usize, lparam: isize) callconv(.c) isize {
    const user32 = struct {
        extern "user32" fn DefWindowProcW(?*anyopaque, u32, usize, isize) callconv(.c) isize;
        extern "user32" fn PostQuitMessage(i32) callconv(.c) void;
    };

    if (global_platform_instance) |plat| {
        switch (msg) {
            0x0005 => { // WM_SIZE
                const cw: i32 = @as(i16, @truncate(lparam));
                const ch: i32 = @as(i16, @truncate(lparam >> 16));
                if (cw > 0 and ch > 0) {
                    plat.client_w = cw;
                    plat.client_h = ch;
                    plat.resized = true;
                }
                return 0;
            },
            0x0014 => { // WM_ERASEBKGND
                return 1;
            },
            0x0010 => { // WM_CLOSE
                plat.should_close = true;
                return 0;
            },
            0x0002 => { // WM_DESTROY
                user32.PostQuitMessage(0);
                plat.should_close = true;
                return 0;
            },
            0x0100 => { // WM_KEYDOWN
                if (wparam < 256) plat.keys[wparam] = true;
                return 0;
            },
            0x0101 => { // WM_KEYUP
                if (wparam < 256) plat.keys[wparam] = false;
                return 0;
            },
            0x0200 => { // WM_MOUSEMOVE
                const raw_x: i32 = @as(i16, @truncate(lparam));
                const raw_y: i32 = @as(i16, @truncate(lparam >> 16));
                mapMouseCoords(plat, raw_x, raw_y);
                return 0;
            },
            0x0201 => { // WM_LBUTTONDOWN
                const raw_x: i32 = @as(i16, @truncate(lparam));
                const raw_y: i32 = @as(i16, @truncate(lparam >> 16));
                mapMouseCoords(plat, raw_x, raw_y);
                plat.mouse_buttons[0] = true;
                plat.mouse_clicks[0] = true;
                return 0;
            },
            0x0202 => { // WM_LBUTTONUP
                const raw_x: i32 = @as(i16, @truncate(lparam));
                const raw_y: i32 = @as(i16, @truncate(lparam >> 16));
                mapMouseCoords(plat, raw_x, raw_y);
                plat.mouse_buttons[0] = false;
                return 0;
            },
            0x0204 => { // WM_RBUTTONDOWN
                const raw_x: i32 = @as(i16, @truncate(lparam));
                const raw_y: i32 = @as(i16, @truncate(lparam >> 16));
                mapMouseCoords(plat, raw_x, raw_y);
                plat.mouse_buttons[1] = true;
                return 0;
            },
            0x0205 => { // WM_RBUTTONUP
                const raw_x: i32 = @as(i16, @truncate(lparam));
                const raw_y: i32 = @as(i16, @truncate(lparam >> 16));
                mapMouseCoords(plat, raw_x, raw_y);
                plat.mouse_buttons[1] = false;
                return 0;
            },
            else => {},
        }
    }

    return user32.DefWindowProcW(hwnd, msg, wparam, lparam);
}

test "platform headless initialization" {
    var plat = try Platform.init(std.testing.allocator, .{
        .width = 64,
        .height = 64,
        .backend = .headless,
    });
    defer plat.deinit();

    try std.testing.expectEqual(Backend.headless, plat.active_backend);
    plat.pollEvents();
    try std.testing.expect(!plat.should_close);
}

test "platform mouse click and coordinates simulation" {
    var plat = try Platform.init(std.testing.allocator, .{
        .width = 360,
        .height = 420,
        .backend = .headless,
        .scale = 1,
    });
    defer plat.deinit();

    global_platform_instance = &plat;
    // Simulate WM_MOUSEMOVE to (100, 200)
    const lparam_move: isize = (200 << 16) | 100;
    _ = win32WndProc(null, 0x0200, 0, lparam_move);
    try std.testing.expectEqual(@as(i32, 100), plat.mouse_x);
    try std.testing.expectEqual(@as(i32, 200), plat.mouse_y);

    // Simulate WM_LBUTTONDOWN at (90, 120)
    const lparam_down: isize = (120 << 16) | 90;
    _ = win32WndProc(null, 0x0201, 0, lparam_down);
    try std.testing.expectEqual(@as(i32, 90), plat.mouse_x);
    try std.testing.expectEqual(@as(i32, 120), plat.mouse_y);
    try std.testing.expect(plat.isMouseButtonPressed(0));
    try std.testing.expect(plat.isMouseButtonDown(0));
}

test "platform window resize and scaled mouse coordinates" {
    var plat = try Platform.init(std.testing.allocator, .{
        .width = 360,
        .height = 420,
        .backend = .headless,
        .scale = 1,
    });
    defer plat.deinit();

    global_platform_instance = &plat;
    // Simulate WM_SIZE to 720 x 840 (2x resize)
    const lparam_size: isize = (840 << 16) | 720;
    _ = win32WndProc(null, 0x0005, 0, lparam_size);
    try std.testing.expectEqual(@as(i32, 720), plat.client_w);
    try std.testing.expectEqual(@as(i32, 840), plat.client_h);
    try std.testing.expect(plat.resized);

    // Mouse move to (200, 400) in resized window should scale back to (100, 200) in virtual coords
    const lparam_move: isize = (400 << 16) | 200;
    _ = win32WndProc(null, 0x0200, 0, lparam_move);
    try std.testing.expectEqual(@as(i32, 100), plat.mouse_x);
    try std.testing.expectEqual(@as(i32, 200), plat.mouse_y);
}

test "platform action bindings and gamepad simulation" {
    var plat = try Platform.init(std.testing.allocator, .{
        .width = 64,
        .height = 64,
        .backend = .headless,
    });
    defer plat.deinit();

    // Bind action 0 (e.g. Jump) to Space (32) and Gamepad A (0x1000)
    plat.bindAction(0, KeyCode.space, KeyCode.w, GamepadButton.a);

    try std.testing.expect(!plat.isActionDown(0));

    plat.keys[KeyCode.space] = true;
    try std.testing.expect(plat.isActionDown(0));
    try std.testing.expect(plat.isActionPressed(0));

    plat.keys[KeyCode.space] = false;
    plat.gamepad.buttons = GamepadButton.a;
    try std.testing.expect(plat.isActionDown(0));
}
