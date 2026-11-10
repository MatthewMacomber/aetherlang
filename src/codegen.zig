//! Aether Code Generator
//! Validates tensor shapes at compile-time and emits Zig code targeting runtime.zig.

const std = @import("std");
const parser = @import("parser.zig");
const types = @import("types.zig");
const autodiff = @import("autodiff.zig");
const autofree = @import("autofree.zig");
const rt_mem = @import("runtime_mem.zig");
const Node = parser.Node;
const Type = types.Type;

pub const TensorShape = struct {
    dims: []const usize,
};

pub const CodegenError = error{
    InvalidSyntax,
    ShapeMismatch,
    UndefinedSymbol,
    OutOfMemory,
    ArenaEscape,
};

pub const EngineBuiltin = struct {
    name: []const u8,
    rt_name: []const u8,
    min_args: usize,
};

pub const ENGINE_BUILTINS = [_]EngineBuiltin{
    .{ .name = "free", .rt_name = "rt.free", .min_args = 1 },
    .{ .name = "mem_live", .rt_name = "rt.memLive", .min_args = 0 },
    .{ .name = "memLive", .rt_name = "rt.memLive", .min_args = 0 },
    .{ .name = "gc_collect", .rt_name = "rt.gcCollect", .min_args = 0 },
    .{ .name = "gcCollect", .rt_name = "rt.gcCollect", .min_args = 0 },
    .{ .name = "gfx.init", .rt_name = "rt.gfxInit", .min_args = 5 },
    .{ .name = "gfx_init", .rt_name = "rt.gfxInit", .min_args = 5 },
    .{ .name = "gfxInit", .rt_name = "rt.gfxInit", .min_args = 5 },
    .{ .name = "game.init", .rt_name = "rt.gfxInit", .min_args = 5 },
    .{ .name = "game_init", .rt_name = "rt.gfxInit", .min_args = 5 },
    .{ .name = "gameInit", .rt_name = "rt.gfxInit", .min_args = 5 },
    .{ .name = "gfx.clear", .rt_name = "rt.gfxClear", .min_args = 1 },
    .{ .name = "gfx_clear", .rt_name = "rt.gfxClear", .min_args = 1 },
    .{ .name = "gfxClear", .rt_name = "rt.gfxClear", .min_args = 1 },
    .{ .name = "gfx.rgb", .rt_name = "rt.gfxRgb", .min_args = 3 },
    .{ .name = "gfx_rgb", .rt_name = "rt.gfxRgb", .min_args = 3 },
    .{ .name = "gfxRgb", .rt_name = "rt.gfxRgb", .min_args = 3 },
    .{ .name = "gfx.pixel", .rt_name = "rt.gfxDrawPixel", .min_args = 3 },
    .{ .name = "gfx_pixel", .rt_name = "rt.gfxDrawPixel", .min_args = 3 },
    .{ .name = "gfx.draw_pixel", .rt_name = "rt.gfxDrawPixel", .min_args = 3 },
    .{ .name = "gfxDrawPixel", .rt_name = "rt.gfxDrawPixel", .min_args = 3 },
    .{ .name = "gfx.line", .rt_name = "rt.gfxDrawLine", .min_args = 5 },
    .{ .name = "gfx_line", .rt_name = "rt.gfxDrawLine", .min_args = 5 },
    .{ .name = "gfx.draw_line", .rt_name = "rt.gfxDrawLine", .min_args = 5 },
    .{ .name = "gfxDrawLine", .rt_name = "rt.gfxDrawLine", .min_args = 5 },
    .{ .name = "gfx.rect", .rt_name = "rt.gfxDrawRect", .min_args = 6 },
    .{ .name = "gfx_rect", .rt_name = "rt.gfxDrawRect", .min_args = 6 },
    .{ .name = "gfx.draw_rect", .rt_name = "rt.gfxDrawRect", .min_args = 6 },
    .{ .name = "gfxDrawRect", .rt_name = "rt.gfxDrawRect", .min_args = 6 },
    .{ .name = "gfx.circle", .rt_name = "rt.gfxDrawCircle", .min_args = 5 },
    .{ .name = "gfx_circle", .rt_name = "rt.gfxDrawCircle", .min_args = 5 },
    .{ .name = "gfx.draw_circle", .rt_name = "rt.gfxDrawCircle", .min_args = 5 },
    .{ .name = "gfxDrawCircle", .rt_name = "rt.gfxDrawCircle", .min_args = 5 },
    .{ .name = "gfx.text", .rt_name = "rt.gfxDrawText", .min_args = 4 },
    .{ .name = "gfx_text", .rt_name = "rt.gfxDrawText", .min_args = 4 },
    .{ .name = "gfx.draw_text", .rt_name = "rt.gfxDrawText", .min_args = 4 },
    .{ .name = "gfxDrawText", .rt_name = "rt.gfxDrawText", .min_args = 4 },
    .{ .name = "gfx.export_bmp", .rt_name = "rt.gfxExportBMP", .min_args = 1 },
    .{ .name = "gfx_export_bmp", .rt_name = "rt.gfxExportBMP", .min_args = 1 },
    .{ .name = "gfxExportBMP", .rt_name = "rt.gfxExportBMP", .min_args = 1 },
    .{ .name = "gfx.draw_tensor", .rt_name = "rt.gfxDrawTensor", .min_args = 5 },
    .{ .name = "gfx_draw_tensor", .rt_name = "rt.gfxDrawTensor", .min_args = 5 },
    .{ .name = "gfxDrawTensor", .rt_name = "rt.gfxDrawTensor", .min_args = 5 },
    .{ .name = "gfx.screen_to_tensor", .rt_name = "rt.gfxScreenToTensor", .min_args = 0 },
    .{ .name = "gfx_screen_to_tensor", .rt_name = "rt.gfxScreenToTensor", .min_args = 0 },
    .{ .name = "gfxScreenToTensor", .rt_name = "rt.gfxScreenToTensor", .min_args = 0 },
    .{ .name = "tui.render", .rt_name = "rt.tuiRender", .min_args = 0 },
    .{ .name = "tui_render", .rt_name = "rt.tuiRender", .min_args = 0 },
    .{ .name = "tuiRender", .rt_name = "rt.tuiRender", .min_args = 0 },
    .{ .name = "tui.sparkline", .rt_name = "rt.tuiSparkline", .min_args = 1 },
    .{ .name = "tui_sparkline", .rt_name = "rt.tuiSparkline", .min_args = 1 },
    .{ .name = "tuiSparkline", .rt_name = "rt.tuiSparkline", .min_args = 1 },
    .{ .name = "replay_init", .rt_name = "rt.replayInit", .min_args = 1 },
    .{ .name = "replay_step", .rt_name = "rt.replayStep", .min_args = 0 },
    .{ .name = "replay_snapshot", .rt_name = "rt.replaySnapshot", .min_args = 1 },
    .{ .name = "replay_rewind", .rt_name = "rt.replayRewind", .min_args = 1 },
    .{ .name = "replay_rand", .rt_name = "rt.replayRandom", .min_args = 2 },
    .{ .name = "game.poll", .rt_name = "rt.gamePoll", .min_args = 0 },
    .{ .name = "game_poll", .rt_name = "rt.gamePoll", .min_args = 0 },
    .{ .name = "gamePoll", .rt_name = "rt.gamePoll", .min_args = 0 },
    .{ .name = "game.present", .rt_name = "rt.gamePresent", .min_args = 0 },
    .{ .name = "game_present", .rt_name = "rt.gamePresent", .min_args = 0 },
    .{ .name = "gamePresent", .rt_name = "rt.gamePresent", .min_args = 0 },
    .{ .name = "game.should_close", .rt_name = "rt.gameShouldClose", .min_args = 0 },
    .{ .name = "game_should_close", .rt_name = "rt.gameShouldClose", .min_args = 0 },
    .{ .name = "gameShouldClose", .rt_name = "rt.gameShouldClose", .min_args = 0 },
    .{ .name = "game.dt", .rt_name = "rt.gameDt", .min_args = 0 },
    .{ .name = "game_dt", .rt_name = "rt.gameDt", .min_args = 0 },
    .{ .name = "gameDt", .rt_name = "rt.gameDt", .min_args = 0 },
    .{ .name = "game.key_down", .rt_name = "rt.gameIsKeyDown", .min_args = 1 },
    .{ .name = "game_key_down", .rt_name = "rt.gameIsKeyDown", .min_args = 1 },
    .{ .name = "gameIsKeyDown", .rt_name = "rt.gameIsKeyDown", .min_args = 1 },
    .{ .name = "game.key_pressed", .rt_name = "rt.gameIsKeyPressed", .min_args = 1 },
    .{ .name = "game_key_pressed", .rt_name = "rt.gameIsKeyPressed", .min_args = 1 },
    .{ .name = "gameIsKeyPressed", .rt_name = "rt.gameIsKeyPressed", .min_args = 1 },
    .{ .name = "game.mouse_x", .rt_name = "rt.gameMouseX", .min_args = 0 },
    .{ .name = "game_mouse_x", .rt_name = "rt.gameMouseX", .min_args = 0 },
    .{ .name = "gameMouseX", .rt_name = "rt.gameMouseX", .min_args = 0 },
    .{ .name = "game.mouse_y", .rt_name = "rt.gameMouseY", .min_args = 0 },
    .{ .name = "game_mouse_y", .rt_name = "rt.gameMouseY", .min_args = 0 },
    .{ .name = "gameMouseY", .rt_name = "rt.gameMouseY", .min_args = 0 },
    .{ .name = "game.mouse_down", .rt_name = "rt.gameIsMouseDown", .min_args = 0 },
    .{ .name = "game_mouse_down", .rt_name = "rt.gameIsMouseDown", .min_args = 0 },
    .{ .name = "gameIsMouseDown", .rt_name = "rt.gameIsMouseDown", .min_args = 0 },
    .{ .name = "game.mouse_pressed", .rt_name = "rt.gameIsMousePressed", .min_args = 0 },
    .{ .name = "game_mouse_pressed", .rt_name = "rt.gameIsMousePressed", .min_args = 0 },
    .{ .name = "gameIsMousePressed", .rt_name = "rt.gameIsMousePressed", .min_args = 0 },
    .{ .name = "game.window_width", .rt_name = "rt.gameWindowWidth", .min_args = 0 },
    .{ .name = "game_window_width", .rt_name = "rt.gameWindowWidth", .min_args = 0 },
    .{ .name = "gameWindowWidth", .rt_name = "rt.gameWindowWidth", .min_args = 0 },
    .{ .name = "game.window_height", .rt_name = "rt.gameWindowHeight", .min_args = 0 },
    .{ .name = "game_window_height", .rt_name = "rt.gameWindowHeight", .min_args = 0 },
    .{ .name = "gameWindowHeight", .rt_name = "rt.gameWindowHeight", .min_args = 0 },
    .{ .name = "game.is_window_resized", .rt_name = "rt.gameIsWindowResized", .min_args = 0 },
    .{ .name = "game_is_window_resized", .rt_name = "rt.gameIsWindowResized", .min_args = 0 },
    .{ .name = "gameIsWindowResized", .rt_name = "rt.gameIsWindowResized", .min_args = 0 },
    .{ .name = "gfx.resize", .rt_name = "rt.gfxResize", .min_args = 2 },
    .{ .name = "gfx_resize", .rt_name = "rt.gfxResize", .min_args = 2 },
    .{ .name = "gfxResize", .rt_name = "rt.gfxResize", .min_args = 2 },
    .{ .name = "game.resize", .rt_name = "rt.gfxResize", .min_args = 2 },
    .{ .name = "game_resize", .rt_name = "rt.gfxResize", .min_args = 2 },
    .{ .name = "gameResize", .rt_name = "rt.gfxResize", .min_args = 2 },
    .{ .name = "engine.burst_particles", .rt_name = "rt.engineBurstParticles", .min_args = 4 },
    .{ .name = "engine_burst_particles", .rt_name = "rt.engineBurstParticles", .min_args = 4 },
    .{ .name = "engineBurstParticles", .rt_name = "rt.engineBurstParticles", .min_args = 4 },
    .{ .name = "phys.check_aabb", .rt_name = "rt.physCheckAABB", .min_args = 8 },
    .{ .name = "phys_check_aabb", .rt_name = "rt.physCheckAABB", .min_args = 8 },
    .{ .name = "physCheckAABB", .rt_name = "rt.physCheckAABB", .min_args = 8 },
    .{ .name = "phys.check_circle", .rt_name = "rt.physCheckCircle", .min_args = 6 },
    .{ .name = "phys_check_circle", .rt_name = "rt.physCheckCircle", .min_args = 6 },
    .{ .name = "physCheckCircle", .rt_name = "rt.physCheckCircle", .min_args = 6 },
    .{ .name = "phys.raycast_aabb", .rt_name = "rt.physRaycastAABB", .min_args = 8 },
    .{ .name = "phys_raycast_aabb", .rt_name = "rt.physRaycastAABB", .min_args = 8 },
    .{ .name = "physRaycastAABB", .rt_name = "rt.physRaycastAABB", .min_args = 8 },
    .{ .name = "audio.tone", .rt_name = "rt.audioPlayTone", .min_args = 2 },
    .{ .name = "audio_tone", .rt_name = "rt.audioPlayTone", .min_args = 2 },
    .{ .name = "audioPlayTone", .rt_name = "rt.audioPlayTone", .min_args = 2 },
    .{ .name = "audio.sfx_beep", .rt_name = "rt.audioPlayTone", .min_args = 2 },
    .{ .name = "audio_sfx_beep", .rt_name = "rt.audioPlayTone", .min_args = 2 },
    .{ .name = "audio.beep", .rt_name = "rt.audioPlayTone", .min_args = 2 },
    .{ .name = "audio_beep", .rt_name = "rt.audioPlayTone", .min_args = 2 },
    .{ .name = "audio.laser", .rt_name = "rt.audioPlayLaser", .min_args = 0 },
    .{ .name = "audio_laser", .rt_name = "rt.audioPlayLaser", .min_args = 0 },
    .{ .name = "audioPlayLaser", .rt_name = "rt.audioPlayLaser", .min_args = 0 },
    .{ .name = "audio.explosion", .rt_name = "rt.audioPlayExplosion", .min_args = 0 },
    .{ .name = "audio_explosion", .rt_name = "rt.audioPlayExplosion", .min_args = 0 },
    .{ .name = "audioPlayExplosion", .rt_name = "rt.audioPlayExplosion", .min_args = 0 },
    .{ .name = "audio.jump", .rt_name = "rt.audioPlayJump", .min_args = 0 },
    .{ .name = "audio_jump", .rt_name = "rt.audioPlayJump", .min_args = 0 },
    .{ .name = "audioPlayJump", .rt_name = "rt.audioPlayJump", .min_args = 0 },
    .{ .name = "audio.export_wav", .rt_name = "rt.audioExportWav", .min_args = 1 },
    .{ .name = "audio_export_wav", .rt_name = "rt.audioExportWav", .min_args = 1 },
    .{ .name = "audioExportWav", .rt_name = "rt.audioExportWav", .min_args = 1 },
    .{ .name = "audio.export_qoa", .rt_name = "rt.audioExportQoa", .min_args = 1 },
    .{ .name = "audio_export_qoa", .rt_name = "rt.audioExportQoa", .min_args = 1 },
    .{ .name = "audioExportQoa", .rt_name = "rt.audioExportQoa", .min_args = 1 },
    .{ .name = "gui.button", .rt_name = "rt.guiButton", .min_args = 6 },
    .{ .name = "gui_button", .rt_name = "rt.guiButton", .min_args = 6 },
    .{ .name = "guiButton", .rt_name = "rt.guiButton", .min_args = 6 },
    .{ .name = "gui.slider", .rt_name = "rt.guiSlider", .min_args = 9 },
    .{ .name = "gui_slider", .rt_name = "rt.guiSlider", .min_args = 9 },
    .{ .name = "guiSlider", .rt_name = "rt.guiSlider", .min_args = 9 },
    .{ .name = "gui.panel", .rt_name = "rt.guiPanel", .min_args = 5 },
    .{ .name = "gui_panel", .rt_name = "rt.guiPanel", .min_args = 5 },
    .{ .name = "guiPanel", .rt_name = "rt.guiPanel", .min_args = 5 },
    // Dynamic Collections Builtins
    .{ .name = "append", .rt_name = "rt.listAppend", .min_args = 2 },
    .{ .name = "list_append", .rt_name = "rt.listAppend", .min_args = 2 },
    .{ .name = "list.append", .rt_name = "rt.listAppend", .min_args = 2 },
    .{ .name = "concat", .rt_name = "rt.listConcat", .min_args = 2 },
    .{ .name = "list_concat", .rt_name = "rt.listConcat", .min_args = 2 },
    .{ .name = "list.concat", .rt_name = "rt.listConcat", .min_args = 2 },
    .{ .name = "get", .rt_name = "rt.listGet", .min_args = 2 },
    .{ .name = "index", .rt_name = "rt.listGet", .min_args = 2 },
    .{ .name = "list_get", .rt_name = "rt.listGet", .min_args = 2 },
    .{ .name = "list.get", .rt_name = "rt.listGet", .min_args = 2 },
    .{ .name = "set_at", .rt_name = "rt.listSet", .min_args = 3 },
    .{ .name = "list_set", .rt_name = "rt.listSet", .min_args = 3 },
    .{ .name = "list.set", .rt_name = "rt.listSet", .min_args = 3 },
    .{ .name = "length", .rt_name = "rt.listLength", .min_args = 1 },
    .{ .name = "len", .rt_name = "rt.listLength", .min_args = 1 },
    .{ .name = "list_len", .rt_name = "rt.listLength", .min_args = 1 },
    .{ .name = "list.len", .rt_name = "rt.listLength", .min_args = 1 },
    .{ .name = "slice", .rt_name = "rt.listSlice", .min_args = 3 },
    .{ .name = "list_slice", .rt_name = "rt.listSlice", .min_args = 3 },
    .{ .name = "list.slice", .rt_name = "rt.listSlice", .min_args = 3 },
    .{ .name = "pop", .rt_name = "rt.listPop", .min_args = 1 },
    .{ .name = "list_pop", .rt_name = "rt.listPop", .min_args = 1 },
    .{ .name = "list.pop", .rt_name = "rt.listPop", .min_args = 1 },
    .{ .name = "map", .rt_name = "rt.mapCreate", .min_args = 0 },
    .{ .name = "dict", .rt_name = "rt.mapCreate", .min_args = 0 },
    .{ .name = "map_create", .rt_name = "rt.mapCreate", .min_args = 0 },
    .{ .name = "map.new", .rt_name = "rt.mapCreate", .min_args = 0 },
    .{ .name = "map_set", .rt_name = "rt.mapSet", .min_args = 3 },
    .{ .name = "map.set", .rt_name = "rt.mapSet", .min_args = 3 },
    .{ .name = "map_get", .rt_name = "rt.mapGet", .min_args = 2 },
    .{ .name = "map.get", .rt_name = "rt.mapGet", .min_args = 2 },
    .{ .name = "map_has", .rt_name = "rt.mapHas", .min_args = 2 },
    .{ .name = "map.has", .rt_name = "rt.mapHas", .min_args = 2 },
    .{ .name = "map_delete", .rt_name = "rt.mapDelete", .min_args = 2 },
    .{ .name = "map.delete", .rt_name = "rt.mapDelete", .min_args = 2 },
    .{ .name = "map_keys", .rt_name = "rt.mapKeys", .min_args = 1 },
    .{ .name = "map.keys", .rt_name = "rt.mapKeys", .min_args = 1 },
    .{ .name = "map_values", .rt_name = "rt.mapValues", .min_args = 1 },
    .{ .name = "map.values", .rt_name = "rt.mapValues", .min_args = 1 },
    .{ .name = "map_size", .rt_name = "rt.mapSize", .min_args = 1 },
    .{ .name = "map.size", .rt_name = "rt.mapSize", .min_args = 1 },
    .{ .name = "set", .rt_name = "rt.setCreate", .min_args = 0 },
    .{ .name = "set_create", .rt_name = "rt.setCreate", .min_args = 0 },
    .{ .name = "set.new", .rt_name = "rt.setCreate", .min_args = 0 },
    .{ .name = "set_add", .rt_name = "rt.setAdd", .min_args = 2 },
    .{ .name = "set.add", .rt_name = "rt.setAdd", .min_args = 2 },
    .{ .name = "set_contains", .rt_name = "rt.setContains", .min_args = 2 },
    .{ .name = "set_has", .rt_name = "rt.setContains", .min_args = 2 },
    .{ .name = "set.has", .rt_name = "rt.setContains", .min_args = 2 },
    .{ .name = "set_remove", .rt_name = "rt.setRemove", .min_args = 2 },
    .{ .name = "set.remove", .rt_name = "rt.setRemove", .min_args = 2 },
    .{ .name = "set_size", .rt_name = "rt.setSize", .min_args = 1 },
    .{ .name = "set.size", .rt_name = "rt.setSize", .min_args = 1 },
    .{ .name = "set_to_list", .rt_name = "rt.setToList", .min_args = 1 },
    .{ .name = "set.to_list", .rt_name = "rt.setToList", .min_args = 1 },
    .{ .name = "deque", .rt_name = "rt.dequeCreate", .min_args = 0 },
    .{ .name = "deque_create", .rt_name = "rt.dequeCreate", .min_args = 0 },
    .{ .name = "deque.new", .rt_name = "rt.dequeCreate", .min_args = 0 },
    .{ .name = "push_back", .rt_name = "rt.dequePushBack", .min_args = 2 },
    .{ .name = "deque_push_back", .rt_name = "rt.dequePushBack", .min_args = 2 },
    .{ .name = "deque.push_back", .rt_name = "rt.dequePushBack", .min_args = 2 },
    .{ .name = "push_front", .rt_name = "rt.dequePushFront", .min_args = 2 },
    .{ .name = "deque_push_front", .rt_name = "rt.dequePushFront", .min_args = 2 },
    .{ .name = "deque.push_front", .rt_name = "rt.dequePushFront", .min_args = 2 },
    .{ .name = "pop_back", .rt_name = "rt.dequePopBack", .min_args = 1 },
    .{ .name = "deque_pop_back", .rt_name = "rt.dequePopBack", .min_args = 1 },
    .{ .name = "deque.pop_back", .rt_name = "rt.dequePopBack", .min_args = 1 },
    .{ .name = "pop_front", .rt_name = "rt.dequePopFront", .min_args = 1 },
    .{ .name = "deque_pop_front", .rt_name = "rt.dequePopFront", .min_args = 1 },
    .{ .name = "deque.pop_front", .rt_name = "rt.dequePopFront", .min_args = 1 },
    .{ .name = "deque_len", .rt_name = "rt.dequeLength", .min_args = 1 },
    .{ .name = "deque.len", .rt_name = "rt.dequeLength", .min_args = 1 },
    .{ .name = "deque_get", .rt_name = "rt.dequeGet", .min_args = 2 },
    .{ .name = "deque.get", .rt_name = "rt.dequeGet", .min_args = 2 },
    // Tensor Autodiff & Linear Algebra Builtins
    .{ .name = "backward", .rt_name = "rt.tensorBackward", .min_args = 1 },
    .{ .name = "tensor_backward", .rt_name = "rt.tensorBackward", .min_args = 1 },
    .{ .name = "tensor.backward", .rt_name = "rt.tensorBackward", .min_args = 1 },
    .{ .name = "grad", .rt_name = "rt.tensorGrad", .min_args = 1 },
    .{ .name = "tensor_grad", .rt_name = "rt.tensorGrad", .min_args = 1 },
    .{ .name = "tensor.grad", .rt_name = "rt.tensorGrad", .min_args = 1 },
    .{ .name = "clear_grads", .rt_name = "rt.tensorClearGrads", .min_args = 0 },
    .{ .name = "tensor_clear_grads", .rt_name = "rt.tensorClearGrads", .min_args = 0 },
    .{ .name = "tensor.clear_grads", .rt_name = "rt.tensorClearGrads", .min_args = 0 },
    .{ .name = "transpose", .rt_name = "rt.tensorTranspose", .min_args = 1 },
    .{ .name = "tensor_transpose", .rt_name = "rt.tensorTranspose", .min_args = 1 },
    .{ .name = "tensor.transpose", .rt_name = "rt.tensorTranspose", .min_args = 1 },
    .{ .name = "reshape", .rt_name = "rt.tensorReshape", .min_args = 2 },
    .{ .name = "tensor_reshape", .rt_name = "rt.tensorReshape", .min_args = 2 },
    .{ .name = "tensor.reshape", .rt_name = "rt.tensorReshape", .min_args = 2 },
    .{ .name = "tensor_slice", .rt_name = "rt.tensorSlice", .min_args = 3 },
    .{ .name = "tensor.slice", .rt_name = "rt.tensorSlice", .min_args = 3 },
    .{ .name = "argmax", .rt_name = "rt.tensorArgmax", .min_args = 1 },
    .{ .name = "tensor_argmax", .rt_name = "rt.tensorArgmax", .min_args = 1 },
    .{ .name = "tensor.argmax", .rt_name = "rt.tensorArgmax", .min_args = 1 },
    .{ .name = "softmax", .rt_name = "rt.tensorSoftmax", .min_args = 1 },
    .{ .name = "tensor_softmax", .rt_name = "rt.tensorSoftmax", .min_args = 1 },
    .{ .name = "tensor.softmax", .rt_name = "rt.tensorSoftmax", .min_args = 1 },
    .{ .name = "sgd_step", .rt_name = "rt.optimizerSGD", .min_args = 3 },
    .{ .name = "optimizer_sgd", .rt_name = "rt.optimizerSGD", .min_args = 3 },
    .{ .name = "sgd", .rt_name = "rt.optimizerSGD", .min_args = 3 },
    .{ .name = "adam_step", .rt_name = "rt.optimizerAdam", .min_args = 9 },
    .{ .name = "optimizer_adam", .rt_name = "rt.optimizerAdam", .min_args = 9 },
    .{ .name = "adam", .rt_name = "rt.optimizerAdam", .min_args = 9 },
    // Distributed Computing Builtins
    .{ .name = "cluster_start", .rt_name = "rt.clusterStart", .min_args = 1 },
    .{ .name = "cluster.start", .rt_name = "rt.clusterStart", .min_args = 1 },
    .{ .name = "cluster_stop", .rt_name = "rt.clusterStop", .min_args = 0 },
    .{ .name = "cluster.stop", .rt_name = "rt.clusterStop", .min_args = 0 },
    .{ .name = "cluster_spawn", .rt_name = "rt.clusterSpawn", .min_args = 3 },
    .{ .name = "cluster.spawn", .rt_name = "rt.clusterSpawn", .min_args = 3 },
    .{ .name = "cluster_send", .rt_name = "rt.clusterSend", .min_args = 5 },
    .{ .name = "cluster.send", .rt_name = "rt.clusterSend", .min_args = 5 },
    .{ .name = "fused_matmul_relu", .rt_name = "rt.fusedMatmulRelu", .min_args = 2 },
    .{ .name = "fused_dense_relu", .rt_name = "rt.fusedDenseRelu", .min_args = 3 },
    // Math & Trigonometry Stdlib Builtins
    .{ .name = "asin", .rt_name = "rt.mathAsin", .min_args = 1 },
    .{ .name = "math_asin", .rt_name = "rt.mathAsin", .min_args = 1 },
    .{ .name = "math.asin", .rt_name = "rt.mathAsin", .min_args = 1 },
    .{ .name = "acos", .rt_name = "rt.mathAcos", .min_args = 1 },
    .{ .name = "math_acos", .rt_name = "rt.mathAcos", .min_args = 1 },
    .{ .name = "math.acos", .rt_name = "rt.mathAcos", .min_args = 1 },
    .{ .name = "atan2", .rt_name = "rt.mathAtan2", .min_args = 2 },
    .{ .name = "math_atan2", .rt_name = "rt.mathAtan2", .min_args = 2 },
    .{ .name = "math.atan2", .rt_name = "rt.mathAtan2", .min_args = 2 },
    .{ .name = "log2", .rt_name = "rt.mathLog2", .min_args = 1 },
    .{ .name = "math_log2", .rt_name = "rt.mathLog2", .min_args = 1 },
    .{ .name = "math.log2", .rt_name = "rt.mathLog2", .min_args = 1 },
    .{ .name = "log10", .rt_name = "rt.mathLog10", .min_args = 1 },
    .{ .name = "math_log10", .rt_name = "rt.mathLog10", .min_args = 1 },
    .{ .name = "math.log10", .rt_name = "rt.mathLog10", .min_args = 1 },
    .{ .name = "clamp", .rt_name = "rt.mathClamp", .min_args = 3 },
    .{ .name = "math_clamp", .rt_name = "rt.mathClamp", .min_args = 3 },
    .{ .name = "math.clamp", .rt_name = "rt.mathClamp", .min_args = 3 },
    .{ .name = "sin", .rt_name = "rt.mathSin", .min_args = 1 },
    .{ .name = "math_sin", .rt_name = "rt.mathSin", .min_args = 1 },
    .{ .name = "math.sin", .rt_name = "rt.mathSin", .min_args = 1 },
    .{ .name = "cos", .rt_name = "rt.mathCos", .min_args = 1 },
    .{ .name = "math_cos", .rt_name = "rt.mathCos", .min_args = 1 },
    .{ .name = "math.cos", .rt_name = "rt.mathCos", .min_args = 1 },
    .{ .name = "tan", .rt_name = "rt.mathTan", .min_args = 1 },
    .{ .name = "math_tan", .rt_name = "rt.mathTan", .min_args = 1 },
    .{ .name = "math.tan", .rt_name = "rt.mathTan", .min_args = 1 },
    .{ .name = "exp", .rt_name = "rt.mathExp", .min_args = 1 },
    .{ .name = "math_exp", .rt_name = "rt.mathExp", .min_args = 1 },
    .{ .name = "math.exp", .rt_name = "rt.mathExp", .min_args = 1 },
    .{ .name = "log", .rt_name = "rt.mathLog", .min_args = 1 },
    .{ .name = "math_log", .rt_name = "rt.mathLog", .min_args = 1 },
    .{ .name = "math.log", .rt_name = "rt.mathLog", .min_args = 1 },
    .{ .name = "sqrt", .rt_name = "rt.mathSqrt", .min_args = 1 },
    .{ .name = "math_sqrt", .rt_name = "rt.mathSqrt", .min_args = 1 },
    .{ .name = "math.sqrt", .rt_name = "rt.mathSqrt", .min_args = 1 },
    .{ .name = "abs", .rt_name = "rt.mathAbs", .min_args = 1 },
    .{ .name = "math_abs", .rt_name = "rt.mathAbs", .min_args = 1 },
    .{ .name = "math.abs", .rt_name = "rt.mathAbs", .min_args = 1 },
    .{ .name = "pow", .rt_name = "rt.mathPow", .min_args = 2 },
    .{ .name = "math_pow", .rt_name = "rt.mathPow", .min_args = 2 },
    .{ .name = "math.pow", .rt_name = "rt.mathPow", .min_args = 2 },
    .{ .name = "floor", .rt_name = "rt.mathFloor", .min_args = 1 },
    .{ .name = "math_floor", .rt_name = "rt.mathFloor", .min_args = 1 },
    .{ .name = "math.floor", .rt_name = "rt.mathFloor", .min_args = 1 },
    .{ .name = "ceil", .rt_name = "rt.mathCeil", .min_args = 1 },
    .{ .name = "math_ceil", .rt_name = "rt.mathCeil", .min_args = 1 },
    .{ .name = "math.ceil", .rt_name = "rt.mathCeil", .min_args = 1 },
    .{ .name = "min", .rt_name = "rt.mathMin", .min_args = 2 },
    .{ .name = "math_min", .rt_name = "rt.mathMin", .min_args = 2 },
    .{ .name = "math.min", .rt_name = "rt.mathMin", .min_args = 2 },
    .{ .name = "max", .rt_name = "rt.mathMax", .min_args = 2 },
    .{ .name = "math_max", .rt_name = "rt.mathMax", .min_args = 2 },
    .{ .name = "math.max", .rt_name = "rt.mathMax", .min_args = 2 },
    // Statistics Stdlib Builtins
    .{ .name = "mean", .rt_name = "rt.statMean", .min_args = 1 },
    .{ .name = "stat_mean", .rt_name = "rt.statMean", .min_args = 1 },
    .{ .name = "stat.mean", .rt_name = "rt.statMean", .min_args = 1 },
    .{ .name = "variance", .rt_name = "rt.statVariance", .min_args = 1 },
    .{ .name = "stat_variance", .rt_name = "rt.statVariance", .min_args = 1 },
    .{ .name = "stat.variance", .rt_name = "rt.statVariance", .min_args = 1 },
    .{ .name = "stddev", .rt_name = "rt.statStdDev", .min_args = 1 },
    .{ .name = "stat_stddev", .rt_name = "rt.statStdDev", .min_args = 1 },
    .{ .name = "stat.stddev", .rt_name = "rt.statStdDev", .min_args = 1 },
    // File I/O Stdlib Builtins
    .{ .name = "file_read_str", .rt_name = "rt.fileReadStr", .min_args = 1 },
    .{ .name = "file.read_str", .rt_name = "rt.fileReadStr", .min_args = 1 },
    .{ .name = "file_read_text", .rt_name = "rt.fileReadStr", .min_args = 1 },
    .{ .name = "file.read_text", .rt_name = "rt.fileReadStr", .min_args = 1 },
    .{ .name = "file_write_str", .rt_name = "rt.fileWriteStr", .min_args = 2 },
    .{ .name = "file.write_str", .rt_name = "rt.fileWriteStr", .min_args = 2 },
    .{ .name = "file_write_text", .rt_name = "rt.fileWriteStr", .min_args = 2 },
    .{ .name = "file.write_text", .rt_name = "rt.fileWriteStr", .min_args = 2 },
    // String Utilities Stdlib Builtins
    .{ .name = "str_split", .rt_name = "rt.strSplit", .min_args = 2 },
    .{ .name = "str.split", .rt_name = "rt.strSplit", .min_args = 2 },
    .{ .name = "str_join", .rt_name = "rt.strJoin", .min_args = 2 },
    .{ .name = "str.join", .rt_name = "rt.strJoin", .min_args = 2 },
    .{ .name = "str_trim", .rt_name = "rt.strTrim", .min_args = 1 },
    .{ .name = "str.trim", .rt_name = "rt.strTrim", .min_args = 1 },
    .{ .name = "str_contains", .rt_name = "rt.strContains", .min_args = 2 },
    .{ .name = "str.contains", .rt_name = "rt.strContains", .min_args = 2 },
    .{ .name = "str_to_num", .rt_name = "rt.strToNum", .min_args = 1 },
    .{ .name = "str.to_num", .rt_name = "rt.strToNum", .min_args = 1 },
    .{ .name = "strToNum", .rt_name = "rt.strToNum", .min_args = 1 },
    .{ .name = "file_exists", .rt_name = "rt.fileExists", .min_args = 1 },
    .{ .name = "file.exists", .rt_name = "rt.fileExists", .min_args = 1 },
    .{ .name = "fileExists", .rt_name = "rt.fileExists", .min_args = 1 },
    // Multi-track Audio & Music Builtins
    .{ .name = "audio.music_play", .rt_name = "rt.audioMusicPlay", .min_args = 2 },
    .{ .name = "audio_music_play", .rt_name = "rt.audioMusicPlay", .min_args = 2 },
    .{ .name = "audioMusicPlay", .rt_name = "rt.audioMusicPlay", .min_args = 2 },
    .{ .name = "audio.music_stop", .rt_name = "rt.audioMusicStop", .min_args = 0 },
    .{ .name = "audio_music_stop", .rt_name = "rt.audioMusicStop", .min_args = 0 },
    .{ .name = "audioMusicStop", .rt_name = "rt.audioMusicStop", .min_args = 0 },
    .{ .name = "audio.volume", .rt_name = "rt.audioVolume", .min_args = 1 },
    .{ .name = "audio_volume", .rt_name = "rt.audioVolume", .min_args = 1 },
    .{ .name = "audioVolume", .rt_name = "rt.audioVolume", .min_args = 1 },
    // Compiler Toolchain & CLI Builtins
    .{ .name = "to_str", .rt_name = "rt.valToString", .min_args = 1 },
    .{ .name = "to_string", .rt_name = "rt.valToString", .min_args = 1 },
    .{ .name = "str", .rt_name = "rt.valToString", .min_args = 1 },
    .{ .name = "str_slice", .rt_name = "rt.listSlice", .min_args = 3 },
    .{ .name = "str.slice", .rt_name = "rt.listSlice", .min_args = 3 },
    .{ .name = "cli_args", .rt_name = "rt.cliArgs", .min_args = 0 },
    .{ .name = "sys_args", .rt_name = "rt.cliArgs", .min_args = 0 },
    .{ .name = "sys_exec", .rt_name = "rt.sysExec", .min_args = 1 },
    .{ .name = "sys_run", .rt_name = "rt.sysExec", .min_args = 1 },
    // FFI, ONNX, HTTP & ASQL Builtins
    .{ .name = "ffi_open", .rt_name = "rt.ffiOpen", .min_args = 1 },
    .{ .name = "ffi.open", .rt_name = "rt.ffiOpen", .min_args = 1 },
    .{ .name = "ffi_call", .rt_name = "rt.ffiCall", .min_args = 3 },
    .{ .name = "ffi.call", .rt_name = "rt.ffiCall", .min_args = 3 },
    .{ .name = "ffi_close", .rt_name = "rt.ffiClose", .min_args = 1 },
    .{ .name = "ffi.close", .rt_name = "rt.ffiClose", .min_args = 1 },
    .{ .name = "onnx_save", .rt_name = "rt.onnxSave", .min_args = 2 },
    .{ .name = "onnx.save", .rt_name = "rt.onnxSave", .min_args = 2 },
    .{ .name = "onnx_load", .rt_name = "rt.onnxLoad", .min_args = 1 },
    .{ .name = "onnx.load", .rt_name = "rt.onnxLoad", .min_args = 1 },
    .{ .name = "http_server_start", .rt_name = "rt.httpServerStart", .min_args = 1 },
    .{ .name = "http.server_start", .rt_name = "rt.httpServerStart", .min_args = 1 },
    .{ .name = "http_route", .rt_name = "rt.httpRoute", .min_args = 3 },
    .{ .name = "http.route", .rt_name = "rt.httpRoute", .min_args = 3 },
    .{ .name = "http_response", .rt_name = "rt.httpResponse", .min_args = 3 },
    .{ .name = "http.response", .rt_name = "rt.httpResponse", .min_args = 3 },
    .{ .name = "http_get", .rt_name = "rt.httpGet", .min_args = 1 },
    .{ .name = "http.get", .rt_name = "rt.httpGet", .min_args = 1 },
    .{ .name = "ws_frame", .rt_name = "rt.wsFrame", .min_args = 1 },
    .{ .name = "ws.frame", .rt_name = "rt.wsFrame", .min_args = 1 },
    .{ .name = "db_open", .rt_name = "rt.dbOpen", .min_args = 1 },
    .{ .name = "db.open", .rt_name = "rt.dbOpen", .min_args = 1 },
    .{ .name = "db_exec", .rt_name = "rt.dbExec", .min_args = 2 },
    .{ .name = "db.exec", .rt_name = "rt.dbExec", .min_args = 2 },
    .{ .name = "db_query", .rt_name = "rt.dbQuery", .min_args = 2 },
    .{ .name = "db.query", .rt_name = "rt.dbQuery", .min_args = 2 },
    .{ .name = "orm_save", .rt_name = "rt.ormSave", .min_args = 3 },
    .{ .name = "orm.save", .rt_name = "rt.ormSave", .min_args = 3 },
    .{ .name = "orm_find", .rt_name = "rt.ormFind", .min_args = 3 },
    .{ .name = "orm.find", .rt_name = "rt.ormFind", .min_args = 3 },
};

pub fn findEngineBuiltin(name: []const u8) ?EngineBuiltin {
    for (ENGINE_BUILTINS) |b| {
        if (std.mem.eql(u8, b.name, name)) return b;
    }
    return null;
}

pub const Codegen = struct {
    allocator: std.mem.Allocator,
    out: std.ArrayList(u8),
    indent_level: usize = 0,
    tensor_shapes: std.StringHashMap(TensorShape),
    local_types: std.StringHashMap([]const u8),
    fn_param_types: std.StringHashMap([]const []const u8),
    local_vars: std.StringHashMap(void),
    current_captures: ?*const std.StringHashMap(void) = null,
    mem_mode: rt_mem.MemMode = .autofree,
    autofree_result: ?autofree.AutoFreeResult = null,
    gc_counter: usize = 0,
    region_depth: usize = 0,
    region_locals: std.ArrayList(std.StringHashMap(void)),

    pub fn init(allocator: std.mem.Allocator) Codegen {
        return .{
            .allocator = allocator,
            .out = .empty,
            .tensor_shapes = std.StringHashMap(TensorShape).init(allocator),
            .local_types = std.StringHashMap([]const u8).init(allocator),
            .fn_param_types = std.StringHashMap([]const []const u8).init(allocator),
            .local_vars = std.StringHashMap(void).init(allocator),
            .region_locals = .empty,
        };
    }

    pub fn deinit(self: *Codegen) void {
        self.out.deinit(self.allocator);
        self.tensor_shapes.deinit();
        self.local_types.deinit();
        self.fn_param_types.deinit();
        self.local_vars.deinit();
        for (self.region_locals.items) |*map| {
            map.deinit();
        }
        self.region_locals.deinit(self.allocator);
    }

    fn writeIndent(self: *Codegen) !void {
        for (0..self.indent_level) |_| {
            try self.out.appendSlice(self.allocator, "    ");
        }
    }

    fn mangle(self: *Codegen, name: []const u8) ![]const u8 {
        var buf: std.ArrayList(u8) = .empty;
        try buf.appendSlice(self.allocator, "ae_");
        for (name) |c| {
            if (c == '-' or c == '.' or c == ':') {
                try buf.append(self.allocator, '_');
            } else {
                try buf.append(self.allocator, c);
            }
        }
        return try buf.toOwnedSlice(self.allocator);
    }

    pub fn generate(self: *Codegen, ast_nodes: []const *Node) ![]const u8 {
        try self.out.appendSlice(self.allocator,
            \\// Generated by Aether Compiler (Zig Backend)
            \\const std = @import("std");
            \\const rt = @import("runtime.zig");
            \\
            \\var g_io: std.Io = undefined;
            \\
            \\
        );

        // First pass: generate struct and enum definitions
        for (ast_nodes) |node| {
            switch (node.data) {
                .list => |children| {
                    if (children.len > 0 and children[0].data == .atom and children[0].data.atom == .symbol) {
                        const sym = children[0].data.atom.symbol;
                        if (std.mem.eql(u8, sym, "struct") or std.mem.eql(u8, sym, "linear-struct")) {
                            try self.generateStruct(children);
                            try self.out.appendSlice(self.allocator, "\n");
                        } else if (std.mem.eql(u8, sym, "enum")) {
                            try self.generateEnum(children);
                            try self.out.appendSlice(self.allocator, "\n");
                        } else if (std.mem.eql(u8, sym, "model")) {
                            try self.generateModel(children);
                            try self.out.appendSlice(self.allocator, "\n");
                        }
                    }
                },
                else => {},
            }
        }

        // Second pass: generate function declarations & definitions
        for (ast_nodes) |node| {
            switch (node.data) {
                .list => |children| {
                    if (children.len > 0 and children[0].data == .atom and children[0].data.atom == .symbol) {
                        const sym = children[0].data.atom.symbol;
                        if (std.mem.eql(u8, sym, "func")) {
                            try self.generateFunction(children);
                            try self.out.appendSlice(self.allocator, "\n");
                        } else if (std.mem.eql(u8, sym, "kernel")) {
                            try self.generateKernelCPUFallback(children);
                            try self.out.appendSlice(self.allocator, "\n");
                        }
                    }
                },
                else => {},
            }
        }

        // Generate runner main
        try self.out.appendSlice(self.allocator,
            \\pub fn main(init: std.process.Init) !void {
            \\    g_io = init.io;
            \\    rt.initMain(init);
            \\    rt.mem.mode = .
        );
        try self.out.appendSlice(self.allocator, @tagName(self.mem_mode));
        try self.out.appendSlice(self.allocator,
            \\;
            \\    const result = ae_main();
            \\    rt.deinitRuntime();
            \\    std.process.exit(rt.toExitCode(result));
            \\}
            \\
        );

        return try self.out.toOwnedSlice(self.allocator);
    }

    fn isTerminating(node: *Node) bool {
        switch (node.data) {
            .list => |children| {
                if (children.len == 0) return false;
                if (children[0].data == .atom and children[0].data.atom == .symbol) {
                    const sym = children[0].data.atom.symbol;
                    if (std.mem.eql(u8, sym, "return")) return true;
                    if (std.mem.eql(u8, sym, "if") and children.len >= 4) {
                        return isTerminating(children[2]) and isTerminating(children[3]);
                    }
                    if (std.mem.eql(u8, sym, "match") and children.len >= 3) {
                        for (children[2..]) |case_node| {
                            if (case_node.data != .list or case_node.data.list.len < 2) return false;
                            const last_stmt = case_node.data.list[case_node.data.list.len - 1];
                            if (!isTerminating(last_stmt)) return false;
                        }
                        return true;
                    }
                }
                return false;
            },
            else => return false,
        }
    }

    fn generateStruct(self: *Codegen, children: []const *Node) !void {
        if (children.len < 2) return CodegenError.InvalidSyntax;
        const name = children[1].data.atom.symbol;
        const mangled_name = try self.mangle(name);

        try self.out.appendSlice(self.allocator, "pub const ");
        try self.out.appendSlice(self.allocator, mangled_name);
        try self.out.appendSlice(self.allocator, " = struct {\n");
        self.indent_level += 1;

        for (children[2..]) |f_node| {
            if (f_node.data != .list or f_node.data.list.len < 2) continue;
            const f_name = f_node.data.list[0].data.atom.symbol;
            const f_type = f_node.data.list[1].data.atom.symbol;
            const mangled_f = try self.mangle(f_name);

            try self.writeIndent();
            try self.out.appendSlice(self.allocator, mangled_f);
            try self.out.appendSlice(self.allocator, ": ");
            if (std.mem.eql(u8, f_type, "i32")) {
                try self.out.appendSlice(self.allocator, "i32,\n");
            } else if (std.mem.eql(u8, f_type, "i64") or std.mem.eql(u8, f_type, "int")) {
                try self.out.appendSlice(self.allocator, "i64,\n");
            } else if (std.mem.eql(u8, f_type, "f32")) {
                try self.out.appendSlice(self.allocator, "f32,\n");
            } else if (std.mem.eql(u8, f_type, "f64") or std.mem.eql(u8, f_type, "float")) {
                try self.out.appendSlice(self.allocator, "f64,\n");
            } else if (std.mem.eql(u8, f_type, "bool")) {
                try self.out.appendSlice(self.allocator, "bool,\n");
            } else if (std.mem.eql(u8, f_type, "string")) {
                try self.out.appendSlice(self.allocator, "[]const u8,\n");
            } else {
                try self.out.appendSlice(self.allocator, "rt.Value,\n");
            }
        }

        self.indent_level -= 1;
        try self.out.appendSlice(self.allocator, "};\n");
    }

    fn generateEnum(self: *Codegen, children: []const *Node) !void {
        if (children.len < 2) return CodegenError.InvalidSyntax;
        const name = children[1].data.atom.symbol;
        const mangled_name = try self.mangle(name);

        try self.out.appendSlice(self.allocator, "pub const ");
        try self.out.appendSlice(self.allocator, mangled_name);
        try self.out.appendSlice(self.allocator, " = enum {\n");
        self.indent_level += 1;

        for (children[2..]) |v_node| {
            if (v_node.data == .atom and v_node.data.atom == .symbol) {
                const mangled_v = try self.mangle(v_node.data.atom.symbol);
                try self.writeIndent();
                try self.out.appendSlice(self.allocator, mangled_v);
                try self.out.appendSlice(self.allocator, ",\n");
            }
        }

        self.indent_level -= 1;
        try self.out.appendSlice(self.allocator, "};\n");
    }

    fn generateModel(self: *Codegen, children: []const *Node) !void {
        if (children.len < 2) return CodegenError.InvalidSyntax;
        const name = children[1].data.atom.symbol;
        const mangled_name = try self.mangle(name);

        try self.out.appendSlice(self.allocator, "pub const ");
        try self.out.appendSlice(self.allocator, mangled_name);
        try self.out.appendSlice(self.allocator, " = struct {\n");
        self.indent_level += 1;

        var layer_idx: usize = 0;
        for (children[2..]) |l_node| {
            if (l_node.data != .list or l_node.data.list.len < 2) continue;
            try self.writeIndent();
            var buf: [32]u8 = undefined;
            const w_name = std.fmt.bufPrint(&buf, "w{d}: rt.Value,\n", .{layer_idx}) catch "w: rt.Value,\n";
            try self.out.appendSlice(self.allocator, w_name);
            layer_idx += 1;
        }

        try self.writeIndent();
        try self.out.appendSlice(self.allocator, "pub fn forward(self: ");
        try self.out.appendSlice(self.allocator, mangled_name);
        try self.out.appendSlice(self.allocator, ", x: rt.Value) rt.Value {\n");
        self.indent_level += 1;

        try self.writeIndent();
        try self.out.appendSlice(self.allocator, "var cur = x;\n");

        layer_idx = 0;
        for (children[2..]) |l_node| {
            if (l_node.data != .list or l_node.data.list.len < 2) continue;
            const l_parts = l_node.data.list;
            var act: ?[]const u8 = null;
            if (l_parts.len >= 5 and l_parts[4].data == .atom and l_parts[4].data.atom == .symbol) {
                act = l_parts[4].data.atom.symbol;
            }

            try self.writeIndent();
            try self.out.appendSlice(self.allocator, "cur = ");
            if (act) |a| {
                if (std.mem.eql(u8, a, "relu")) try self.out.appendSlice(self.allocator, "rt.relu(")
                else if (std.mem.eql(u8, a, "sigmoid")) try self.out.appendSlice(self.allocator, "rt.sigmoid(")
                else if (std.mem.eql(u8, a, "tanh")) try self.out.appendSlice(self.allocator, "rt.tanh(");
            }
            var buf: [32]u8 = undefined;
            const step = std.fmt.bufPrint(&buf, "rt.matmul(cur, self.w{d})", .{layer_idx}) catch "cur";
            try self.out.appendSlice(self.allocator, step);
            if (act != null) {
                try self.out.appendSlice(self.allocator, ")");
            }
            try self.out.appendSlice(self.allocator, ";\n");
            layer_idx += 1;
        }

        try self.writeIndent();
        try self.out.appendSlice(self.allocator, "return cur;\n");
        self.indent_level -= 1;
        try self.writeIndent();
        try self.out.appendSlice(self.allocator, "}\n");

        self.indent_level -= 1;
        try self.writeIndent();
        try self.out.appendSlice(self.allocator, "};\n");
    }

    fn generateFunction(self: *Codegen, children: []const *Node) !void {
        // (func name (params...) [ret_type] body...)
        if (children.len < 3) return CodegenError.InvalidSyntax;
        const name_node = children[1];
        if (name_node.data != .atom or name_node.data.atom != .symbol) return CodegenError.InvalidSyntax;
        const fn_name = name_node.data.atom.symbol;
        const mangled_fn = try self.mangle(fn_name);

        const params_node = children[2];
        if (params_node.data != .list and params_node.data != .bracket_list) return CodegenError.InvalidSyntax;
        const params = if (params_node.data == .list) params_node.data.list else params_node.data.bracket_list;

        try self.out.appendSlice(self.allocator, "pub fn ");
        try self.out.appendSlice(self.allocator, mangled_fn);
        try self.out.appendSlice(self.allocator, "(");

        self.local_types.clearRetainingCapacity();
        self.local_vars.clearRetainingCapacity();
        var ptypes: std.ArrayList([]const u8) = .empty;

        for (params, 0..) |param_node, i| {
            if (i > 0) try self.out.appendSlice(self.allocator, ", ");
            var param_type_str: []const u8 = "rt.Value";
            var param_sym: []const u8 = "";

            if (param_node.data == .list and param_node.data.list.len >= 2) {
                param_sym = param_node.data.list[0].data.atom.symbol;
                const t_name = param_node.data.list[1].data.atom.symbol;
                if (std.mem.eql(u8, t_name, "i32")) param_type_str = "i32";
                if (std.mem.eql(u8, t_name, "i64") or std.mem.eql(u8, t_name, "int")) param_type_str = "i64";
                if (std.mem.eql(u8, t_name, "f32")) param_type_str = "f32";
                if (std.mem.eql(u8, t_name, "f64") or std.mem.eql(u8, t_name, "float")) param_type_str = "f64";
                if (std.mem.eql(u8, t_name, "bool")) param_type_str = "bool";
                if (std.mem.eql(u8, t_name, "string")) param_type_str = "[]const u8";
            } else if (param_node.data == .atom and param_node.data.atom == .symbol) {
                param_sym = param_node.data.atom.symbol;
            } else return CodegenError.InvalidSyntax;

            try self.local_types.put(param_sym, param_type_str);
            try ptypes.append(self.allocator, param_type_str);

            const param_name = try self.mangle(param_sym);
            try self.out.appendSlice(self.allocator, param_name);
            try self.out.appendSlice(self.allocator, ": ");
            try self.out.appendSlice(self.allocator, param_type_str);
        }
        try self.fn_param_types.put(fn_name, try ptypes.toOwnedSlice(self.allocator));

        var ret_type_str: []const u8 = "rt.Value";
        var body_start_idx: usize = 3;
        if (children.len > 3 and children[3].data == .atom and children[3].data.atom == .symbol) {
            const sym = children[3].data.atom.symbol;
            if (std.mem.eql(u8, sym, "i32")) { ret_type_str = "i32"; body_start_idx = 4; }
            else if (std.mem.eql(u8, sym, "i64") or std.mem.eql(u8, sym, "int")) { ret_type_str = "i64"; body_start_idx = 4; }
            else if (std.mem.eql(u8, sym, "f32")) { ret_type_str = "f32"; body_start_idx = 4; }
            else if (std.mem.eql(u8, sym, "f64") or std.mem.eql(u8, sym, "float")) { ret_type_str = "f64"; body_start_idx = 4; }
            else if (std.mem.eql(u8, sym, "bool")) { ret_type_str = "bool"; body_start_idx = 4; }
            else if (std.mem.eql(u8, sym, "string")) { ret_type_str = "[]const u8"; body_start_idx = 4; }
            else if (std.mem.eql(u8, sym, "void")) { ret_type_str = "void"; body_start_idx = 4; }
        }

        try self.out.appendSlice(self.allocator, ") ");
        try self.out.appendSlice(self.allocator, ret_type_str);
        try self.out.appendSlice(self.allocator, " {\n");
        self.indent_level += 1;

        if (self.mem_mode == .gc or self.mem_mode == .gc_stress or self.mem_mode == .autofree) {
            for (params) |param_node| {
                var psym: []const u8 = "";
                if (param_node.data == .list and param_node.data.list.len >= 1 and param_node.data.list[0].data == .atom and param_node.data.list[0].data.atom == .symbol) {
                    psym = param_node.data.list[0].data.atom.symbol;
                } else if (param_node.data == .atom and param_node.data.atom == .symbol) {
                    psym = param_node.data.atom.symbol;
                }
                if (psym.len > 0) {
                    const mangled_p = try self.mangle(psym);
                    try self.writeIndent();
                    try self.out.appendSlice(self.allocator, "_ = rt.mem.gcPin(");
                    try self.out.appendSlice(self.allocator, mangled_p);
                    try self.out.appendSlice(self.allocator, ");\n");
                }
            }
            const gc_id = self.gc_counter;
            self.gc_counter += 1;
            try self.writeIndent();
            var gbuf: [64]u8 = undefined;
            const gmark_str = std.fmt.bufPrint(&gbuf, "const _gc{d} = rt.mem.gcMark();\n", .{gc_id}) catch "const _gc = rt.mem.gcMark();\n";
            try self.out.appendSlice(self.allocator, gmark_str);
            try self.writeIndent();
            const gsp_str = std.fmt.bufPrint(&gbuf, "rt.mem.gcSafepoint(_gc{d});\n", .{gc_id}) catch "rt.mem.gcSafepoint(_gc);\n";
            try self.out.appendSlice(self.allocator, gsp_str);
        }

        // Generate statements in body
        var ends_with_term = false;
        const body_stmts = children[body_start_idx..];
        for (body_stmts, 0..) |stmt, idx| {
            if (idx == body_stmts.len - 1 and isTerminating(stmt)) {
                ends_with_term = true;
            }
            try self.generateStatement(stmt);
        }

        if (!ends_with_term) {
            try self.writeIndent();
            if (std.mem.eql(u8, ret_type_str, "void")) {
                try self.out.appendSlice(self.allocator, "return;\n");
            } else if (std.mem.eql(u8, ret_type_str, "rt.Value")) {
                try self.out.appendSlice(self.allocator, "return rt.nil();\n");
            } else if (std.mem.eql(u8, ret_type_str, "i64") or std.mem.eql(u8, ret_type_str, "i32")) {
                try self.out.appendSlice(self.allocator, "return 0;\n");
            } else if (std.mem.eql(u8, ret_type_str, "f64") or std.mem.eql(u8, ret_type_str, "f32")) {
                try self.out.appendSlice(self.allocator, "return 0.0;\n");
            } else if (std.mem.eql(u8, ret_type_str, "bool")) {
                try self.out.appendSlice(self.allocator, "return false;\n");
            } else {
                try self.out.appendSlice(self.allocator, "return rt.nil();\n");
            }
        }

        self.indent_level -= 1;
        try self.out.appendSlice(self.allocator, "}\n");
    }

    fn generateKernelCPUFallback(self: *Codegen, children: []const *Node) anyerror!void {
        if (children.len < 4) return;
        const name = children[1].data.atom.symbol;
        const mangled_name = try self.mangle(name);
        const params_node = children[2];
        const params = if (params_node.data == .list) params_node.data.list else params_node.data.bracket_list;

        try self.writeIndent();
        try self.out.appendSlice(self.allocator, "pub fn ");
        try self.out.appendSlice(self.allocator, mangled_name);
        try self.out.appendSlice(self.allocator, "(");

        var param_names: std.ArrayList([]const u8) = .empty;
        for (params, 0..) |p, idx| {
            if (idx > 0) try self.out.appendSlice(self.allocator, ", ");
            var pname: []const u8 = "arg";
            if (p.data == .list and p.data.list.len >= 1 and p.data.list[0].data == .atom and p.data.list[0].data.atom == .symbol) {
                pname = p.data.list[0].data.atom.symbol;
            } else if (p.data == .atom and p.data.atom == .symbol) {
                pname = p.data.atom.symbol;
            }
            try param_names.append(self.allocator, pname);
            const mangled_p = try self.mangle(pname);
            try self.out.appendSlice(self.allocator, mangled_p);
            try self.out.appendSlice(self.allocator, ": rt.Value");
            try self.local_vars.put(pname, {});
        }
        try self.out.appendSlice(self.allocator, ") rt.Value {\n");
        self.indent_level += 1;

        if (param_names.items.len > 0) {
            const first_p = try self.mangle(param_names.items[0]);
            try self.writeIndent();
            try self.out.appendSlice(self.allocator, "const _n = if (");
            try self.out.appendSlice(self.allocator, first_p);
            try self.out.appendSlice(self.allocator, " == .tensor) ");
            try self.out.appendSlice(self.allocator, first_p);
            try self.out.appendSlice(self.allocator, ".tensor.data.len else 0;\n");

            try self.writeIndent();
            try self.out.appendSlice(self.allocator, "var _idx: usize = 0;\n");
            try self.writeIndent();
            try self.out.appendSlice(self.allocator, "while (_idx < _n) : (_idx += 1) {\n");
            self.indent_level += 1;

            try self.writeIndent();
            try self.out.appendSlice(self.allocator, "const idx = rt.number(@floatFromInt(_idx));\n");
            try self.writeIndent();
            try self.out.appendSlice(self.allocator, "const global_id = idx;\n");
            try self.writeIndent();
            try self.out.appendSlice(self.allocator, "_ = global_id;\n");

            for (children[3..]) |stmt| {
                if (stmt.data == .list and stmt.data.list.len >= 3 and stmt.data.list[0].data == .atom and std.mem.eql(u8, stmt.data.list[0].data.atom.symbol, "let")) {
                    const var_n = stmt.data.list[1].data.atom.symbol;
                    const val_node = stmt.data.list[2];
                    const is_gid = (val_node.data == .atom and val_node.data.atom == .symbol and (std.mem.eql(u8, val_node.data.atom.symbol, "global_id") or std.mem.eql(u8, val_node.data.atom.symbol, "idx")));
                    const mangled_v = try self.mangle(var_n);
                    try self.writeIndent();
                    if (is_gid) {
                        try self.out.appendSlice(self.allocator, "const ");
                        try self.out.appendSlice(self.allocator, mangled_v);
                        try self.out.appendSlice(self.allocator, " = rt.number(@floatFromInt(_idx));\n");
                    } else {
                        try self.out.appendSlice(self.allocator, "var ");
                        try self.out.appendSlice(self.allocator, mangled_v);
                        try self.out.appendSlice(self.allocator, " = ");
                        try self.generateExpression(val_node);
                        try self.out.appendSlice(self.allocator, ";\n");
                    }
                    try self.writeIndent();
                    try self.out.appendSlice(self.allocator, "_ = &");
                    try self.out.appendSlice(self.allocator, mangled_v);
                    try self.out.appendSlice(self.allocator, ";\n");
                    try self.local_vars.put(var_n, {});
                } else if (stmt.data == .list and stmt.data.list.len >= 4 and stmt.data.list[0].data == .atom and std.mem.eql(u8, stmt.data.list[0].data.atom.symbol, "set")) {
                    const target = stmt.data.list[1].data.atom.symbol;
                    const mangled_t = try self.mangle(target);
                    const idx_expr = stmt.data.list[2];
                    const val_expr = stmt.data.list[3];

                    try self.writeIndent();
                    try self.out.appendSlice(self.allocator, "_ = rt.listSet(");
                    try self.out.appendSlice(self.allocator, mangled_t);
                    try self.out.appendSlice(self.allocator, ", ");
                    try self.generateExpression(idx_expr);
                    try self.out.appendSlice(self.allocator, ", ");
                    try self.generateExpression(val_expr);
                    try self.out.appendSlice(self.allocator, ");\n");
                } else {
                    try self.generateStatement(stmt);
                }
            }

            self.indent_level -= 1;
            try self.writeIndent();
            try self.out.appendSlice(self.allocator, "}\n");
        }

        try self.writeIndent();
        if (param_names.items.len > 0) {
            const last_p = try self.mangle(param_names.items[param_names.items.len - 1]);
            try self.out.appendSlice(self.allocator, "return ");
            try self.out.appendSlice(self.allocator, last_p);
            try self.out.appendSlice(self.allocator, ";\n");
        } else {
            try self.out.appendSlice(self.allocator, "return rt.nil();\n");
        }

        self.indent_level -= 1;
        try self.writeIndent();
        try self.out.appendSlice(self.allocator, "}\n");
    }

    fn generateStatement(self: *Codegen, node: *Node) anyerror!void {
        switch (node.data) {
            .list => |children| {
                if (children.len == 0) return;
                if (children[0].data == .atom and children[0].data.atom == .symbol) {
                    const op = children[0].data.atom.symbol;
                    if (std.mem.eql(u8, op, "let")) {
                        // (let name expr)
                        if (children.len < 3) return CodegenError.InvalidSyntax;
                        const var_name = children[1].data.atom.symbol;
                        const mangled_var = try self.mangle(var_name);

                        // Check if expr is a tensor creation to record shape
                        try self.inspectAndRecordTensorShape(var_name, children[2]);
                        try self.local_vars.put(var_name, {});

                        if (self.region_depth > 0 and self.region_locals.items.len > 0) {
                            try self.region_locals.items[self.region_locals.items.len - 1].put(var_name, {});
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "var ");
                        try self.out.appendSlice(self.allocator, mangled_var);
                        try self.out.appendSlice(self.allocator, " = ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ";\n");

                        // ponytail: _ = &var allows all Aether locals to be 'var' without tripping Zig's unused/unmutated checks
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "_ = &");
                        try self.out.appendSlice(self.allocator, mangled_var);
                        try self.out.appendSlice(self.allocator, ";\n");

                        if (self.mem_mode == .gc or self.mem_mode == .gc_stress or self.mem_mode == .autofree) {
                            try self.writeIndent();
                            try self.out.appendSlice(self.allocator, "rt.mem.gcRoot(&");
                            try self.out.appendSlice(self.allocator, mangled_var);
                            try self.out.appendSlice(self.allocator, ");\n");
                            try self.writeIndent();
                            try self.out.appendSlice(self.allocator, "defer rt.mem.gcUnroot(&");
                            try self.out.appendSlice(self.allocator, mangled_var);
                            try self.out.appendSlice(self.allocator, ");\n");
                        }
                        if (self.mem_mode == .autofree and self.autofree_result != null) {
                            if (self.autofree_result.?.shouldFree(node)) {
                                try self.writeIndent();
                                try self.out.appendSlice(self.allocator, "defer rt.mem.autoFree(");
                                try self.out.appendSlice(self.allocator, mangled_var);
                                try self.out.appendSlice(self.allocator, ");\n");
                            }
                        }
                        return;
                    } else if (std.mem.eql(u8, op, "set") or std.mem.eql(u8, op, "set!")) {
                        // (set name expr)
                        if (children.len < 3) return CodegenError.InvalidSyntax;
                        const var_name = children[1].data.atom.symbol;
                        const mangled_var = try self.mangle(var_name);

                        if (self.region_depth > 0 and self.region_locals.items.len > 0) {
                            if (!self.region_locals.items[self.region_locals.items.len - 1].contains(var_name)) {
                                return CodegenError.ArenaEscape;
                            }
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, mangled_var);
                        try self.out.appendSlice(self.allocator, " = ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ";\n");
                        return;
                    } else if (std.mem.eql(u8, op, "return")) {
                        // (return [expr])
                        if (self.region_depth > 0) {
                            return CodegenError.ArenaEscape;
                        }
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "return ");
                        const use_pin = (self.mem_mode == .gc or self.mem_mode == .gc_stress or self.mem_mode == .autofree);
                        if (use_pin) {
                            try self.out.appendSlice(self.allocator, "rt.mem.gcPin(");
                        }
                        if (children.len >= 2) {
                            try self.generateExpression(children[1]);
                        } else {
                            try self.out.appendSlice(self.allocator, "rt.nil()");
                        }
                        if (use_pin) {
                            try self.out.appendSlice(self.allocator, ")");
                        }
                        try self.out.appendSlice(self.allocator, ";\n");
                        return;
                    } else if (std.mem.eql(u8, op, "block") or std.mem.eql(u8, op, "begin") or std.mem.eql(u8, op, "do")) {
                        for (children[1..]) |child_stmt| {
                            try self.generateStatement(child_stmt);
                        }
                        return;
                    } else if (std.mem.eql(u8, op, "if")) {
                        // (if cond then [else])
                        if (children.len < 3) return CodegenError.InvalidSyntax;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "if (rt.isTruthy(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")) {\n");

                        self.indent_level += 1;
                        try self.generateStatement(children[2]);
                        self.indent_level -= 1;

                        if (children.len >= 4) {
                            try self.writeIndent();
                            try self.out.appendSlice(self.allocator, "} else {\n");
                            self.indent_level += 1;
                            try self.generateStatement(children[3]);
                            self.indent_level -= 1;
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}\n");
                        return;
                    } else if (std.mem.eql(u8, op, "arena")) {
                        // (arena body...)
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "{\n");
                        self.indent_level += 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "rt.mem.regionEnter();\n");
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "defer _ = rt.mem.regionExit(rt.nil());\n");

                        self.region_depth += 1;
                        const rmap = std.StringHashMap(void).init(self.allocator);
                        try self.region_locals.append(self.allocator, rmap);
                        defer {
                            self.region_depth -= 1;
                            if (self.region_locals.items.len > 0) {
                                self.region_locals.items[self.region_locals.items.len - 1].deinit();
                                _ = self.region_locals.pop();
                            }
                        }

                        for (children[1..]) |body_stmt| {
                            try self.generateStatement(body_stmt);
                        }

                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}\n");
                        return;
                    } else if (std.mem.eql(u8, op, "while")) {
                        // (while cond body...)
                        if (children.len < 2) return CodegenError.InvalidSyntax;
                        const gc_id = self.gc_counter;
                        self.gc_counter += 1;
                        var gbuf: [64]u8 = undefined;

                        try self.writeIndent();
                        if (self.mem_mode == .gc or self.mem_mode == .gc_stress or self.mem_mode == .autofree) {
                            const gmark_str = std.fmt.bufPrint(&gbuf, "const _gc{d} = rt.mem.gcMark();\n", .{gc_id}) catch "";
                            try self.out.appendSlice(self.allocator, gmark_str);
                            try self.writeIndent();
                        }
                        try self.out.appendSlice(self.allocator, "while (rt.isTruthy(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")) {\n");

                        self.indent_level += 1;
                        for (children[2..]) |body_stmt| {
                            try self.generateStatement(body_stmt);
                        }
                        if (self.mem_mode == .gc or self.mem_mode == .gc_stress or self.mem_mode == .autofree) {
                            try self.writeIndent();
                            const gsp_str = std.fmt.bufPrint(&gbuf, "rt.mem.gcSafepoint(_gc{d});\n", .{gc_id}) catch "";
                            try self.out.appendSlice(self.allocator, gsp_str);
                        }
                        self.indent_level -= 1;

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}\n");
                        return;
                    } else if (std.mem.eql(u8, op, "for")) {
                        // (for var (range ...) body...)
                        if (children.len < 3) return CodegenError.InvalidSyntax;
                        const var_node = children[1];
                        if (var_node.data != .atom or var_node.data.atom != .symbol) return CodegenError.InvalidSyntax;
                        const var_name = var_node.data.atom.symbol;
                        const mangled_var = try self.mangle(var_name);
                        try self.local_vars.put(var_name, {});

                        const iter_node = children[2];
                        var start_node: ?*Node = null;
                        var limit_node: ?*Node = null;
                        var step_num: f64 = 1.0;

                        if (iter_node.data == .list and iter_node.data.list.len >= 2 and iter_node.data.list[0].data == .atom and std.mem.eql(u8, iter_node.data.list[0].data.atom.symbol, "range")) {
                            const r_args = iter_node.data.list[1..];
                            if (r_args.len == 1) {
                                limit_node = r_args[0];
                            } else if (r_args.len >= 2) {
                                start_node = r_args[0];
                                limit_node = r_args[1];
                                if (r_args.len >= 3 and r_args[2].data == .atom and r_args[2].data.atom == .number) {
                                    step_num = r_args[2].data.atom.number;
                                }
                            }
                        } else {
                            limit_node = iter_node;
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "{\n");
                        self.indent_level += 1;

                        const gc_id = self.gc_counter;
                        self.gc_counter += 1;
                        if (self.mem_mode == .gc or self.mem_mode == .gc_stress or self.mem_mode == .autofree) {
                            try self.writeIndent();
                            var gbuf: [64]u8 = undefined;
                            const gmark_str = std.fmt.bufPrint(&gbuf, "const _gc{d} = rt.mem.gcMark();\n", .{gc_id}) catch "";
                            try self.out.appendSlice(self.allocator, gmark_str);
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "var ");
                        try self.out.appendSlice(self.allocator, mangled_var);
                        try self.out.appendSlice(self.allocator, " = ");
                        if (start_node) |sn| {
                            try self.generateExpression(sn);
                        } else {
                            try self.out.appendSlice(self.allocator, "rt.number(0)");
                        }
                        try self.out.appendSlice(self.allocator, ";\n");

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "_ = &");
                        try self.out.appendSlice(self.allocator, mangled_var);
                        try self.out.appendSlice(self.allocator, ";\n");

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "while (");
                        try self.out.appendSlice(self.allocator, mangled_var);
                        try self.out.appendSlice(self.allocator, ".number.val < (");
                        if (limit_node) |ln| {
                            try self.generateExpression(ln);
                        } else {
                            try self.out.appendSlice(self.allocator, "rt.number(0)");
                        }
                        try self.out.appendSlice(self.allocator, ").number.val) : (");
                        try self.out.appendSlice(self.allocator, mangled_var);
                        var buf_step: [64]u8 = undefined;
                        const s_step = std.fmt.bufPrint(&buf_step, " = rt.number({s}.number.val + {d})) {{\n", .{ mangled_var, step_num }) catch " = rt.number(0)) {\n";
                        try self.out.appendSlice(self.allocator, s_step);

                        self.indent_level += 1;
                        for (children[3..]) |body_stmt| {
                            try self.generateStatement(body_stmt);
                        }
                        if (self.mem_mode == .gc or self.mem_mode == .gc_stress or self.mem_mode == .autofree) {
                            try self.writeIndent();
                            const gsp_str = std.fmt.bufPrint(&buf_step, "rt.mem.gcSafepoint(_gc{d});\n", .{gc_id}) catch "";
                            try self.out.appendSlice(self.allocator, gsp_str);
                        }
                        self.indent_level -= 1;

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}\n");

                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}\n");
                        return;
                    } else if (std.mem.eql(u8, op, "parallel_for") or std.mem.eql(u8, op, "@parallel for") or std.mem.eql(u8, op, "parallel-for")) {
                        // (parallel_for var (range start end) body...) or (parallel_for var limit body...)
                        if (children.len < 3) return CodegenError.InvalidSyntax;
                        const var_node = children[1];
                        if (var_node.data != .atom or var_node.data.atom != .symbol) return CodegenError.InvalidSyntax;
                        const var_name = var_node.data.atom.symbol;
                        const mangled_var = try self.mangle(var_name);
                        try self.local_vars.put(var_name, {});

                        const iter_node = children[2];
                        var start_node: ?*Node = null;
                        var limit_node: ?*Node = null;

                        if (iter_node.data == .list and iter_node.data.list.len >= 2 and iter_node.data.list[0].data == .atom and std.mem.eql(u8, iter_node.data.list[0].data.atom.symbol, "range")) {
                            const r_args = iter_node.data.list[1..];
                            if (r_args.len == 1) {
                                limit_node = r_args[0];
                            } else if (r_args.len >= 2) {
                                start_node = r_args[0];
                                limit_node = r_args[1];
                            }
                        } else {
                            limit_node = iter_node;
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "{\n");
                        self.indent_level += 1;

                        const gc_id = self.gc_counter;
                        self.gc_counter += 1;
                        if (self.mem_mode == .gc or self.mem_mode == .gc_stress or self.mem_mode == .autofree) {
                            try self.writeIndent();
                            var gbuf: [64]u8 = undefined;
                            const gmark_str = std.fmt.bufPrint(&gbuf, "const _gc{d} = rt.mem.gcMark();\n", .{gc_id}) catch "";
                            try self.out.appendSlice(self.allocator, gmark_str);
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "const _p_start: usize = @intFromFloat(@max(0.0, (");
                        if (start_node) |sn| {
                            try self.generateExpression(sn);
                        } else {
                            try self.out.appendSlice(self.allocator, "rt.number(0)");
                        }
                        try self.out.appendSlice(self.allocator, ").number.val));\n");

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "const _p_end: usize = @intFromFloat(@max(0.0, (");
                        if (limit_node) |ln| {
                            try self.generateExpression(ln);
                        } else {
                            try self.out.appendSlice(self.allocator, "rt.number(0)");
                        }
                        try self.out.appendSlice(self.allocator, ").number.val));\n");

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "var _p_i: usize = _p_start;\n");
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "while (_p_i < _p_end) : (_p_i += 1) {\n");
                        self.indent_level += 1;

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "var ");
                        try self.out.appendSlice(self.allocator, mangled_var);
                        try self.out.appendSlice(self.allocator, " = rt.number(@floatFromInt(_p_i));\n");
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "_ = &");
                        try self.out.appendSlice(self.allocator, mangled_var);
                        try self.out.appendSlice(self.allocator, ";\n");

                        for (children[3..]) |body_stmt| {
                            try self.generateStatement(body_stmt);
                        }
                        if (self.mem_mode == .gc or self.mem_mode == .gc_stress or self.mem_mode == .autofree) {
                            try self.writeIndent();
                            var gbuf: [64]u8 = undefined;
                            const gsp_str = std.fmt.bufPrint(&gbuf, "rt.mem.gcSafepoint(_gc{d});\n", .{gc_id}) catch "";
                            try self.out.appendSlice(self.allocator, gsp_str);
                        }

                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}\n");

                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}\n");
                        return;
                    } else if (std.mem.eql(u8, op, "assert") and children.len >= 2) {
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "if (!rt.isTruthy(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")) @panic(\"Assertion failed\");\n");
                        return;
                    } else if (std.mem.eql(u8, op, "match")) {
                        // (match target (case1 body...) (case2 body...) ...)
                        if (children.len < 2) return CodegenError.InvalidSyntax;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "{\n");
                        self.indent_level += 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "const _match_val = ");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ";\n");

                        var first = true;
                        for (children[2..]) |case_node| {
                            if (case_node.data != .list or case_node.data.list.len < 2) continue;
                            const case_parts = case_node.data.list;
                            const pat = case_parts[0];

                            try self.writeIndent();
                            const is_wildcard = (pat.data == .atom and pat.data.atom == .symbol and std.mem.eql(u8, pat.data.atom.symbol, "_"));
                            if (is_wildcard) {
                                if (!first) {
                                    try self.out.appendSlice(self.allocator, "} else {\n");
                                } else {
                                    try self.out.appendSlice(self.allocator, "{\n");
                                }
                            } else {
                                if (!first) {
                                    try self.out.appendSlice(self.allocator, "} else if (rt.isTruthy(rt.eq(_match_val, ");
                                } else {
                                    try self.out.appendSlice(self.allocator, "if (rt.isTruthy(rt.eq(_match_val, ");
                                }
                                try self.generateExpression(pat);
                                try self.out.appendSlice(self.allocator, "))) {\n");
                            }
                            first = false;

                            self.indent_level += 1;
                            for (case_parts[1..]) |stmt| {
                                try self.generateStatement(stmt);
                            }
                            self.indent_level -= 1;
                        }
                        if (!first) {
                            try self.writeIndent();
                            try self.out.appendSlice(self.allocator, "}\n");
                        }

                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}\n");
                        return;
                    }
                }

                // Default statement: expression evaluated and discarded
                try self.writeIndent();
                try self.out.appendSlice(self.allocator, "_ = ");
                try self.generateExpression(node);
                try self.out.appendSlice(self.allocator, ";\n");
            },
            else => {
                try self.writeIndent();
                try self.out.appendSlice(self.allocator, "_ = ");
                try self.generateExpression(node);
                try self.out.appendSlice(self.allocator, ";\n");
            },
        }
    }

    fn inspectAndRecordTensorShape(self: *Codegen, var_name: []const u8, expr: *Node) !void {
        if (expr.data != .list) return;
        const children = expr.data.list;
        if (children.len < 2) return;
        if (children[0].data != .atom or children[0].data.atom != .symbol) return;
        const op = children[0].data.atom.symbol;

        if (std.mem.eql(u8, op, "tensor")) {
            // (tensor [dims...] [data...])
            const shape_node = children[1];
            if (shape_node.data == .bracket_list or shape_node.data == .list) {
                const dims = if (shape_node.data == .bracket_list) shape_node.data.bracket_list else shape_node.data.list;
                var dim_list: std.ArrayList(usize) = .empty;
                for (dims) |d| {
                    if (d.data == .atom and d.data.atom == .number) {
                        try dim_list.append(self.allocator, @intFromFloat(d.data.atom.number));
                    }
                }
                const shape = try dim_list.toOwnedSlice(self.allocator);
                try self.tensor_shapes.put(var_name, .{ .dims = shape });
            }
        } else if (std.mem.eql(u8, op, "matmul")) {
            // (matmul a b)
            if (children.len >= 3 and children[1].data == .atom and children[2].data == .atom) {
                const a_name = children[1].data.atom.symbol;
                const b_name = children[2].data.atom.symbol;
                const a_shape = self.tensor_shapes.get(a_name);
                const b_shape = self.tensor_shapes.get(b_name);

                if (a_shape != null and b_shape != null) {
                    const ash = a_shape.?.dims;
                    const bsh = b_shape.?.dims;
                    if (ash.len == 2 and bsh.len == 2) {
                        // Compile-time shape checking!
                        if (ash[1] != bsh[0]) {
                            std.debug.print("Aether Compile Error: matmul inner dimension mismatch: [{d}, {d}] vs [{d}, {d}]\n", .{ ash[0], ash[1], bsh[0], bsh[1] });
                            return CodegenError.ShapeMismatch;
                        }
                        const out_shape = try self.allocator.dupe(usize, &[_]usize{ ash[0], bsh[1] });
                        try self.tensor_shapes.put(var_name, .{ .dims = out_shape });
                    }
                }
            }
        }
    }

    fn collectFreeVars(self: *Codegen, node: *Node, params: []const *Node, free_vars: *std.StringHashMap(void)) anyerror!void {
        switch (node.data) {
            .atom => |a| {
                if (a == .symbol) {
                    const s = a.symbol;
                    var is_param = false;
                    for (params) |p| {
                        if (p.data == .atom and p.data.atom == .symbol and std.mem.eql(u8, p.data.atom.symbol, s)) {
                            is_param = true;
                            break;
                        }
                    }
                    if (!is_param and self.local_vars.contains(s)) {
                        try free_vars.put(s, {});
                    }
                }
            },
            .list => |children| {
                var skip_idx: ?usize = null;
                if (children.len >= 2 and children[0].data == .atom and children[0].data.atom == .symbol and std.mem.eql(u8, children[0].data.atom.symbol, "let")) {
                    skip_idx = 1;
                }
                for (children, 0..) |c, idx| {
                    if (skip_idx != null and idx == skip_idx.?) continue;
                    try self.collectFreeVars(c, params, free_vars);
                }
            },
            .bracket_list => |children| {
                for (children) |c| {
                    try self.collectFreeVars(c, params, free_vars);
                }
            },
        }
    }

    fn generateExpression(self: *Codegen, node: *Node) anyerror!void {
        switch (node.data) {
            .atom => |a| switch (a) {
                .nil => try self.out.appendSlice(self.allocator, "rt.nil()"),
                .boolean => |b| try self.out.appendSlice(self.allocator, if (b) "rt.boolean(true)" else "rt.boolean(false)"),
                .number => |n| {
                    var buf: [64]u8 = undefined;
                    const s = std.fmt.bufPrint(&buf, "rt.number({d})", .{n}) catch "rt.number(0)";
                    try self.out.appendSlice(self.allocator, s);
                },
                .string => |s| {
                    try self.out.appendSlice(self.allocator, "rt.string(\"");
                    // Escape string literal for Zig
                    for (s) |ch| {
                        switch (ch) {
                            '\n' => try self.out.appendSlice(self.allocator, "\\n"),
                            '\r' => try self.out.appendSlice(self.allocator, "\\r"),
                            '\t' => try self.out.appendSlice(self.allocator, "\\t"),
                            '\\' => try self.out.appendSlice(self.allocator, "\\\\"),
                            '"' => try self.out.appendSlice(self.allocator, "\\\""),
                            else => try self.out.append(self.allocator, ch),
                        }
                    }
                    try self.out.appendSlice(self.allocator, "\")");
                },
                .symbol => |s| {
                    const mangled = try self.mangle(s);
                    if (self.current_captures) |caps| {
                        if (caps.contains(s)) {
                            try self.out.appendSlice(self.allocator, "self.");
                            try self.out.appendSlice(self.allocator, mangled);
                            return;
                        }
                    }
                    if (self.local_types.get(s)) |t| {
                        if (std.mem.eql(u8, t, "i64") or std.mem.eql(u8, t, "i32")) {
                            try self.out.appendSlice(self.allocator, "rt.number(@floatFromInt(");
                            try self.out.appendSlice(self.allocator, mangled);
                            try self.out.appendSlice(self.allocator, "))");
                            return;
                        } else if (std.mem.eql(u8, t, "f64") or std.mem.eql(u8, t, "f32")) {
                            try self.out.appendSlice(self.allocator, "rt.number(");
                            try self.out.appendSlice(self.allocator, mangled);
                            try self.out.appendSlice(self.allocator, ")");
                            return;
                        } else if (std.mem.eql(u8, t, "bool")) {
                            try self.out.appendSlice(self.allocator, "rt.boolean(");
                            try self.out.appendSlice(self.allocator, mangled);
                            try self.out.appendSlice(self.allocator, ")");
                            return;
                        }
                    }
                    try self.out.appendSlice(self.allocator, mangled);
                },
            },
            .bracket_list => |children| {
                // Bracket list as an expression: emit dynamic List
                try self.out.appendSlice(self.allocator, "rt.list(&[_]rt.Value{ ");
                for (children, 0..) |child, i| {
                    if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                    try self.generateExpression(child);
                }
                try self.out.appendSlice(self.allocator, " })");
            },
            .list => |children| {
                if (children.len == 0) {
                    try self.out.appendSlice(self.allocator, "rt.nil()");
                    return;
                }

                const head = children[0];
                if (head.data == .atom and head.data.atom == .symbol) {
                    const op = head.data.atom.symbol;

                    if (std.mem.startsWith(u8, op, ".") and op.len > 1 and children.len >= 2) {
                        // Field access: (.x obj)
                        const field_name = op[1..];
                        const mangled_f = try self.mangle(field_name);
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ".");
                        try self.out.appendSlice(self.allocator, mangled_f);
                        return;
                    }

                    // Built-in binary math ops
                    if (std.mem.eql(u8, op, "+") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.add(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "-") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.sub(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "*") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.mul(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "/") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.div(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if ((std.mem.eql(u8, op, "%") or std.mem.eql(u8, op, "mod")) and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.mod(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "<=") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.le(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "<") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.lt(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, ">=") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.ge(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, ">") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.gt(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if ((std.mem.eql(u8, op, "==") or std.mem.eql(u8, op, "=")) and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.eq(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "!=") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.ne(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "and") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.logicAnd(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "or") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.logicOr(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "not") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.logicNot(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "matmul") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.matmul(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "sum") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.sum(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "fused_matmul_relu") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.fusedMatmulRelu(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "fused_dense_relu") and children.len >= 4) {
                        try self.out.appendSlice(self.allocator, "rt.fusedDenseRelu(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[3]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "relu") and children.len >= 2) {
                        const operand = children[1];
                        if (operand.data == .list and operand.data.list.len >= 3 and operand.data.list[0].data == .atom and std.mem.eql(u8, operand.data.list[0].data.atom.symbol, "+")) {
                            const p1 = operand.data.list[1];
                            const p2 = operand.data.list[2];
                            var mm_node: ?*Node = null;
                            var bias_node: ?*Node = null;
                            if (p1.data == .list and p1.data.list.len >= 3 and p1.data.list[0].data == .atom and std.mem.eql(u8, p1.data.list[0].data.atom.symbol, "matmul")) {
                                mm_node = p1;
                                bias_node = p2;
                            } else if (p2.data == .list and p2.data.list.len >= 3 and p2.data.list[0].data == .atom and std.mem.eql(u8, p2.data.list[0].data.atom.symbol, "matmul")) {
                                mm_node = p2;
                                bias_node = p1;
                            }
                            if (mm_node != null and bias_node != null) {
                                try self.out.appendSlice(self.allocator, "rt.fusedDenseRelu(");
                                try self.generateExpression(mm_node.?.data.list[1]);
                                try self.out.appendSlice(self.allocator, ", ");
                                try self.generateExpression(mm_node.?.data.list[2]);
                                try self.out.appendSlice(self.allocator, ", ");
                                try self.generateExpression(bias_node.?);
                                try self.out.appendSlice(self.allocator, ")");
                                return;
                            }
                        }
                        if (operand.data == .list and operand.data.list.len >= 3 and operand.data.list[0].data == .atom and std.mem.eql(u8, operand.data.list[0].data.atom.symbol, "matmul")) {
                            // Operator fusion: (relu (matmul a b)) -> rt.fusedMatmulRelu(a, b)
                            try self.out.appendSlice(self.allocator, "rt.fusedMatmulRelu(");
                            try self.generateExpression(operand.data.list[1]);
                            try self.out.appendSlice(self.allocator, ", ");
                            try self.generateExpression(operand.data.list[2]);
                            try self.out.appendSlice(self.allocator, ")");
                            return;
                        }
                        try self.out.appendSlice(self.allocator, "rt.relu(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "sigmoid") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.sigmoid(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "tanh") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.tanh(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "mse_loss") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.mse_loss(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "cross_entropy") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.cross_entropy(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "quantize") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.quantize(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "uncertain") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.uncertain(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "Normal") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.distNormal(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "Uniform") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.distUniform(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "Bernoulli") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.distBernoulli(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "sample") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.sample(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "observe") and children.len >= 4) {
                        try self.out.appendSlice(self.allocator, "rt.observe(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[3]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "gradient") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.gradient(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "diff") and children.len >= 3) {
                        // Static compile-time autodiff: (diff expr wrt)
                        const wrt_node = children[2];
                        if (wrt_node.data == .atom and wrt_node.data.atom == .symbol) {
                            const wrt_sym = wrt_node.data.atom.symbol;
                            var sd = autodiff.StaticDiff.init(self.allocator);
                            const differentiated = try sd.differentiate(children[1], wrt_sym);
                            try self.generateExpression(differentiated);
                            return;
                        }
                    } else if (std.mem.eql(u8, op, "assert") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "blk: { if (!rt.isTruthy(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")) @panic(\"Assertion failed\"); break :blk rt.boolean(true); }");
                        return;
                    } else if (std.mem.eql(u8, op, "arena")) {
                        if (children.len < 2) return CodegenError.InvalidSyntax;
                        try self.out.appendSlice(self.allocator, "blk: {\n");
                        self.indent_level += 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "rt.mem.regionEnter();\n");

                        self.region_depth += 1;
                        const rmap = std.StringHashMap(void).init(self.allocator);
                        try self.region_locals.append(self.allocator, rmap);
                        defer {
                            self.region_depth -= 1;
                            if (self.region_locals.items.len > 0) {
                                self.region_locals.items[self.region_locals.items.len - 1].deinit();
                                _ = self.region_locals.pop();
                            }
                        }

                        for (children[1 .. children.len - 1]) |body_stmt| {
                            try self.generateStatement(body_stmt);
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "const _arena_res = ");
                        try self.generateExpression(children[children.len - 1]);
                        try self.out.appendSlice(self.allocator, ";\n");

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "break :blk rt.mem.regionExit(_arena_res);\n");

                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}");
                        return;
                    } else if (std.mem.eql(u8, op, "autodiff.forward") and children.len >= 2) {
                        try self.generateExpression(children[1]);
                        return;
                    } else if (std.mem.eql(u8, op, "sqrt") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.mathSqrt(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "sin") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.mathSin(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "cos") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.mathCos(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "tan") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.mathTan(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "exp") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.mathExp(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "log") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.mathLog(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "abs") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.mathAbs(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "pow") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.mathPow(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "min") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.mathMin(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "max") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.mathMax(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "floor") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.mathFloor(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "ceil") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.mathCeil(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "vec3") and children.len >= 4) {
                        try self.out.appendSlice(self.allocator, "rt.vec3(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[3]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "dot") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.vec3Dot(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "cross") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.vec3Cross(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "norm") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.vec3Norm(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "spawn") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.spawnActor(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "send") and children.len >= 4) {
                        try self.out.appendSlice(self.allocator, "rt.sendActor(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[3]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "receive") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.receiveActor(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "actor_state") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.getActorState(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "set_actor_state") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.setActorState(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "replay_init") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.replayInit(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "replay_step")) {
                        try self.out.appendSlice(self.allocator, "rt.replayStep()");
                        return;
                    } else if (std.mem.eql(u8, op, "replay_rand") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "rt.replayRandom(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(children[2]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "replay_snapshot") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.replaySnapshot(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "replay_rewind") and children.len >= 2) {
                        try self.out.appendSlice(self.allocator, "rt.replayRewind(");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (findEngineBuiltin(op)) |b| {
                        try self.out.appendSlice(self.allocator, b.rt_name);
                        try self.out.appendSlice(self.allocator, "(");
                        for (children[1..], 0..) |arg, i| {
                            if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                            try self.generateExpression(arg);
                        }
                        var cur_arg_count = children.len - 1;
                        while (cur_arg_count < b.min_args) : (cur_arg_count += 1) {
                            if (cur_arg_count > 0 or children.len > 1) try self.out.appendSlice(self.allocator, ", ");
                            try self.out.appendSlice(self.allocator, "rt.nil()");
                        }
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    } else if (std.mem.eql(u8, op, "lambda") and children.len >= 3) {
                        const params = if (children[1].data == .list) children[1].data.list else children[1].data.bracket_list;
                        var free_vars = std.StringHashMap(void).init(self.allocator);
                        defer free_vars.deinit();
                        for (children[2..]) |body_node| {
                            try self.collectFreeVars(body_node, params, &free_vars);
                        }

                        try self.out.appendSlice(self.allocator, "(struct {\n");
                        self.indent_level += 1;
                        var it = free_vars.keyIterator();
                        while (it.next()) |k| {
                            try self.writeIndent();
                            const mangled = try self.mangle(k.*);
                            try self.out.appendSlice(self.allocator, mangled);
                            try self.out.appendSlice(self.allocator, ": rt.Value,\n");
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "pub fn call(self: @This()");
                        for (params) |p| {
                            try self.out.appendSlice(self.allocator, ", ");
                            const pname = try self.mangle(p.data.atom.symbol);
                            try self.out.appendSlice(self.allocator, pname);
                            try self.out.appendSlice(self.allocator, ": rt.Value");
                        }
                        try self.out.appendSlice(self.allocator, ") rt.Value {\n");
                        self.indent_level += 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "_ = &self;\n");

                        const prev_caps = self.current_captures;
                        self.current_captures = &free_vars;
                        for (children[2 .. children.len - 1]) |stmt| {
                            try self.generateStatement(stmt);
                        }
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "return ");
                        try self.generateExpression(children[children.len - 1]);
                        try self.out.appendSlice(self.allocator, ";\n");
                        self.current_captures = prev_caps;

                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}\n");
                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}{\n");
                        self.indent_level += 1;
                        it = free_vars.keyIterator();
                        while (it.next()) |k| {
                            try self.writeIndent();
                            const mangled = try self.mangle(k.*);
                            try self.out.appendSlice(self.allocator, ".");
                            try self.out.appendSlice(self.allocator, mangled);
                            try self.out.appendSlice(self.allocator, " = ");
                            try self.out.appendSlice(self.allocator, mangled);
                            try self.out.appendSlice(self.allocator, ",\n");
                        }
                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "})");
                        return;
                    } else if (std.mem.eql(u8, op, "match") and children.len >= 3) {
                        try self.out.appendSlice(self.allocator, "blk: {\n");
                        self.indent_level += 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "const _match_val = ");
                        try self.generateExpression(children[1]);
                        try self.out.appendSlice(self.allocator, ";\n");

                        var first = true;
                        for (children[2..]) |case_node| {
                            if (case_node.data != .list or case_node.data.list.len < 2) continue;
                            const case_parts = case_node.data.list;
                            const pat = case_parts[0];

                            try self.writeIndent();
                            const is_wildcard = (pat.data == .atom and pat.data.atom == .symbol and std.mem.eql(u8, pat.data.atom.symbol, "_"));
                            if (is_wildcard) {
                                if (!first) {
                                    try self.out.appendSlice(self.allocator, "} else {\n");
                                } else {
                                    try self.out.appendSlice(self.allocator, "{\n");
                                }
                            } else {
                                if (!first) {
                                    try self.out.appendSlice(self.allocator, "} else if (rt.isTruthy(rt.eq(_match_val, ");
                                } else {
                                    try self.out.appendSlice(self.allocator, "if (rt.isTruthy(rt.eq(_match_val, ");
                                }
                                try self.generateExpression(pat);
                                try self.out.appendSlice(self.allocator, "))) {\n");
                            }
                            first = false;

                            self.indent_level += 1;
                            for (case_parts[1 .. case_parts.len - 1]) |stmt| {
                                try self.generateStatement(stmt);
                            }
                            try self.writeIndent();
                            try self.out.appendSlice(self.allocator, "break :blk ");
                            try self.generateExpression(case_parts[case_parts.len - 1]);
                            try self.out.appendSlice(self.allocator, ";\n");
                            self.indent_level -= 1;
                        }
                        if (!first) {
                            try self.writeIndent();
                            try self.out.appendSlice(self.allocator, "}\n");
                        }

                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "break :blk rt.nil();\n");
                        self.indent_level -= 1;
                        try self.writeIndent();
                        try self.out.appendSlice(self.allocator, "}");
                        return;
                    } else if (std.mem.eql(u8, op, "list")) {
                        try self.out.appendSlice(self.allocator, "rt.list(&[_]rt.Value{ ");
                        for (children[1..], 0..) |arg, i| {
                            if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                            try self.generateExpression(arg);
                        }
                        try self.out.appendSlice(self.allocator, " })");
                        return;
                    } else if (std.mem.eql(u8, op, "tensor") and children.len >= 3) {
                        // (tensor [dims...] [data...])
                        try self.out.appendSlice(self.allocator, "rt.tensor(&[_]usize{ ");
                        const shape_node = children[1];
                        const dims = if (shape_node.data == .bracket_list) shape_node.data.bracket_list else shape_node.data.list;
                        for (dims, 0..) |d, i| {
                            if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                            if (d.data == .atom and d.data.atom == .number) {
                                var buf: [32]u8 = undefined;
                                const s = std.fmt.bufPrint(&buf, "{d}", .{@as(usize, @intFromFloat(d.data.atom.number))}) catch "0";
                                try self.out.appendSlice(self.allocator, s);
                            }
                        }
                        try self.out.appendSlice(self.allocator, " }, &[_]f64{ ");

                        const data_node = children[2];
                        const data_elems = if (data_node.data == .bracket_list) data_node.data.bracket_list else data_node.data.list;
                        for (data_elems, 0..) |elem, i| {
                            if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                            if (elem.data == .atom and elem.data.atom == .number) {
                                var buf: [64]u8 = undefined;
                                const s = std.fmt.bufPrint(&buf, "{d}", .{elem.data.atom.number}) catch "0";
                                try self.out.appendSlice(self.allocator, s);
                            }
                        }
                        try self.out.appendSlice(self.allocator, " })");
                        return;
                    } else if (std.mem.eql(u8, op, "call")) {
                        // (call func_name args...) or (call print args...)
                        if (children.len < 2) return CodegenError.InvalidSyntax;
                        const target = children[1];
                        if (target.data != .atom or target.data.atom != .symbol) return CodegenError.InvalidSyntax;
                        const target_name = target.data.atom.symbol;

                        if (std.mem.eql(u8, target_name, "print")) {
                            try self.out.appendSlice(self.allocator, "rt.print(g_io, &[_]rt.Value{ ");
                            for (children[2..], 0..) |arg, i| {
                                if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                                try self.generateExpression(arg);
                            }
                            try self.out.appendSlice(self.allocator, " })");
                            return;
                        }

                        if (std.mem.eql(u8, target_name, "list")) {
                            try self.out.appendSlice(self.allocator, "rt.list(&[_]rt.Value{ ");
                            for (children[2..], 0..) |arg, i| {
                                if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                                try self.generateExpression(arg);
                            }
                            try self.out.appendSlice(self.allocator, " })");
                            return;
                        }

                        if (findEngineBuiltin(target_name)) |b| {
                            try self.out.appendSlice(self.allocator, b.rt_name);
                            try self.out.appendSlice(self.allocator, "(");
                            for (children[2..], 0..) |arg, i| {
                                if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                                try self.generateExpression(arg);
                            }
                            var cur_arg_count = children.len - 2;
                            while (cur_arg_count < b.min_args) : (cur_arg_count += 1) {
                                if (cur_arg_count > 0 or children.len > 2) try self.out.appendSlice(self.allocator, ", ");
                                try self.out.appendSlice(self.allocator, "rt.nil()");
                            }
                            try self.out.appendSlice(self.allocator, ")");
                            return;
                        }

                        // Check if calling a local closure/variable
                        const mangled_call = try self.mangle(target_name);
                        if (self.local_vars.contains(target_name)) {
                            try self.out.appendSlice(self.allocator, mangled_call);
                            try self.out.appendSlice(self.allocator, ".call(");
                            for (children[2..], 0..) |arg, i| {
                                if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                                try self.generateExpression(arg);
                            }
                            try self.out.appendSlice(self.allocator, ")");
                            return;
                        }

                        // User function call
                        const expected_params = self.fn_param_types.get(target_name);
                        try self.out.appendSlice(self.allocator, mangled_call);
                        try self.out.appendSlice(self.allocator, "(");
                        for (children[2..], 0..) |arg, i| {
                            if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                            var exp_type: ?[]const u8 = null;
                            if (expected_params) |p_types| {
                                if (i < p_types.len) exp_type = p_types[i];
                            }
                            if (exp_type) |et| {
                                if (std.mem.eql(u8, et, "i64") or std.mem.eql(u8, et, "i32")) {
                                    try self.out.appendSlice(self.allocator, "rt.toI64(");
                                    try self.generateExpression(arg);
                                    try self.out.appendSlice(self.allocator, ")");
                                    continue;
                                } else if (std.mem.eql(u8, et, "f64") or std.mem.eql(u8, et, "f32")) {
                                    try self.out.appendSlice(self.allocator, "rt.toF64(");
                                    try self.generateExpression(arg);
                                    try self.out.appendSlice(self.allocator, ")");
                                    continue;
                                }
                            }
                            try self.generateExpression(arg);
                        }
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    }

                    // Direct function call: (func_name arg1 arg2 ...)
                    const mangled_direct = try self.mangle(op);
                    if (self.local_vars.contains(op)) {
                        try self.out.appendSlice(self.allocator, mangled_direct);
                        try self.out.appendSlice(self.allocator, ".call(");
                        for (children[1..], 0..) |arg, i| {
                            if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                            try self.generateExpression(arg);
                        }
                        try self.out.appendSlice(self.allocator, ")");
                        return;
                    }
                    try self.out.appendSlice(self.allocator, mangled_direct);
                    try self.out.appendSlice(self.allocator, "(");
                    for (children[1..], 0..) |arg, i| {
                        if (i > 0) try self.out.appendSlice(self.allocator, ", ");
                        try self.generateExpression(arg);
                    }
                    try self.out.appendSlice(self.allocator, ")");
                    return;
                }

                try self.out.appendSlice(self.allocator, "rt.nil()");
            },
        }
    }
};

test "codegen emits valid Zig for basic function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(func add (a b)
        \\  (return (+ a b)))
        \\(func main ()
        \\  (let res (call add 10 20))
        \\  (call print "Result:" res)
        \\  (return 0))
    ;

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var cg = Codegen.init(arena.allocator());
    defer cg.deinit();

    const zig_code = try cg.generate(nodes);
    try std.testing.expect(std.mem.indexOf(u8, zig_code, "pub fn ae_add") != null);
    try std.testing.expect(std.mem.indexOf(u8, zig_code, "pub fn ae_main") != null);
}

test "codegen emits match statement and expression" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(func classify (x)
        \\  (let label (match x
        \\    (1 "one")
        \\    (2 "two")
        \\    (_ "other")))
        \\  (return label))
        \\(func main ()
        \\  (match 10
        \\    (10 (call print "matched 10"))
        \\    (_ (call print "not 10")))
        \\  (return 0))
    ;

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var cg = Codegen.init(arena.allocator());
    defer cg.deinit();

    const zig_code = try cg.generate(nodes);
    try std.testing.expect(std.mem.indexOf(u8, zig_code, "const _match_val =") != null);
    try std.testing.expect(std.mem.indexOf(u8, zig_code, "break :blk") != null);
}
