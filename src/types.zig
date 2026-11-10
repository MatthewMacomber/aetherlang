//! Aether Type System & Semantic Checker
//! Provides gradual typing, static inference, struct/enum definitions,
//! dependent tensor shape checking, and linear resource tracking.

const std = @import("std");
const parser = @import("parser.zig");
const Node = parser.Node;

pub const TypeKind = enum {
    dynamic,
    i32,
    i64,
    usize_t,
    f32,
    f64,
    boolean,
    string,
    void_t,
    tensor,
    custom_struct,
    custom_enum,
    function,
    list_t,
    map_t,
    set_t,
    deque_t,
};

pub const TensorType = struct {
    elem_type: TypeKind = .f64,
    shape: []const ?usize, // None represents dynamic dimension
};

pub const Field = struct {
    name: []const u8,
    type_info: Type,
};

pub const StructDef = struct {
    name: []const u8,
    is_linear: bool = false,
    fields: []const Field,
};

pub const EnumDef = struct {
    name: []const u8,
    variants: []const []const u8,
};

pub const FunctionType = struct {
    params: []const Type,
    ret: *Type,
};

pub const Type = struct {
    kind: TypeKind,
    tensor_info: ?TensorType = null,
    struct_name: ?[]const u8 = null,
    enum_name: ?[]const u8 = null,
    func_info: ?FunctionType = null,
    is_linear: bool = false,

    pub const DYNAMIC = Type{ .kind = .dynamic };
    pub const I32 = Type{ .kind = .i32 };
    pub const I64 = Type{ .kind = .i64 };
    pub const USIZE = Type{ .kind = .usize_t };
    pub const F32 = Type{ .kind = .f32 };
    pub const F64 = Type{ .kind = .f64 };
    pub const BOOL = Type{ .kind = .boolean };
    pub const STRING = Type{ .kind = .string };
    pub const VOID = Type{ .kind = .void_t };
    pub const TENSOR = Type{ .kind = .tensor };
    pub const LIST = Type{ .kind = .list_t };
    pub const MAP = Type{ .kind = .map_t };
    pub const SET = Type{ .kind = .set_t };
    pub const DEQUE = Type{ .kind = .deque_t };

    pub fn isNumeric(self: Type) bool {
        return switch (self.kind) {
            .i32, .i64, .usize_t, .f32, .f64 => true,
            else => false,
        };
    }

    pub fn isFloat(self: Type) bool {
        return self.kind == .f32 or self.kind == .f64;
    }

    pub fn isInteger(self: Type) bool {
        return self.kind == .i32 or self.kind == .i64 or self.kind == .usize_t;
    }

    pub fn equals(self: Type, other: Type) bool {
        if (self.kind == .dynamic or other.kind == .dynamic) return true; // gradual typing matches
        if (self.kind != other.kind) return false;
        return switch (self.kind) {
            .tensor => blk: {
                if (self.tensor_info == null or other.tensor_info == null) break :blk true;
                const a = self.tensor_info.?;
                const b = other.tensor_info.?;
                if (a.shape.len != b.shape.len) break :blk false;
                for (a.shape, b.shape) |d1, d2| {
                    if (d1 != null and d2 != null and d1.? != d2.?) return false;
                }
                break :blk true;
            },
            .custom_struct => std.mem.eql(u8, self.struct_name orelse "", other.struct_name orelse ""),
            .custom_enum => std.mem.eql(u8, self.enum_name orelse "", other.enum_name orelse ""),
            else => true,
        };
    }

    pub fn toZigType(self: Type) []const u8 {
        return switch (self.kind) {
            .dynamic => "rt.Value",
            .i32 => "i32",
            .i64 => "i64",
            .usize_t => "usize",
            .f32 => "f32",
            .f64 => "f64",
            .boolean => "bool",
            .string => "[]const u8",
            .void_t => "void",
            .tensor => "rt.Value",
            .custom_struct => self.struct_name orelse "rt.Value",
            .custom_enum => self.enum_name orelse "rt.Value",
            .function => "rt.Value",
            .list_t, .map_t, .set_t, .deque_t => "rt.Value",
        };
    }
};

pub const Symbol = struct {
    name: []const u8,
    type_info: Type,
    is_linear: bool = false,
    consumed: bool = false,
};

pub const Scope = struct {
    parent: ?*Scope = null,
    symbols: std.StringHashMap(Symbol),

    pub fn init(allocator: std.mem.Allocator, parent: ?*Scope) Scope {
        return .{
            .parent = parent,
            .symbols = std.StringHashMap(Symbol).init(allocator),
        };
    }

    pub fn deinit(self: *Scope) void {
        self.symbols.deinit();
    }

    pub fn put(self: *Scope, name: []const u8, sym: Symbol) !void {
        try self.symbols.put(name, sym);
    }

    pub fn get(self: *Scope, name: []const u8) ?*Symbol {
        if (self.symbols.getPtr(name)) |s| return s;
        if (self.parent) |p| return p.get(name);
        return null;
    }
};

pub const TypeChecker = struct {
    allocator: std.mem.Allocator,
    structs: std.StringHashMap(StructDef),
    enums: std.StringHashMap(EnumDef),
    functions: std.StringHashMap(FunctionType),

    pub fn init(allocator: std.mem.Allocator) TypeChecker {
        return .{
            .allocator = allocator,
            .structs = std.StringHashMap(StructDef).init(allocator),
            .enums = std.StringHashMap(EnumDef).init(allocator),
            .functions = std.StringHashMap(FunctionType).init(allocator),
        };
    }

    pub fn deinit(self: *TypeChecker) void {
        self.structs.deinit();
        self.enums.deinit();
        self.functions.deinit();
    }

    pub fn parseTypeString(self: *TypeChecker, name: []const u8) Type {
        _ = self;
        if (std.mem.eql(u8, name, "i32")) return Type.I32;
        if (std.mem.eql(u8, name, "i64") or std.mem.eql(u8, name, "int")) return Type.I64;
        if (std.mem.eql(u8, name, "usize")) return Type.USIZE;
        if (std.mem.eql(u8, name, "f32")) return Type.F32;
        if (std.mem.eql(u8, name, "f64") or std.mem.eql(u8, name, "float")) return Type.F64;
        if (std.mem.eql(u8, name, "bool") or std.mem.eql(u8, name, "boolean")) return Type.BOOL;
        if (std.mem.eql(u8, name, "string") or std.mem.eql(u8, name, "str")) return Type.STRING;
        if (std.mem.eql(u8, name, "void") or std.mem.eql(u8, name, "nil")) return Type.VOID;
        if (std.mem.eql(u8, name, "tensor") or std.mem.eql(u8, name, "Tensor")) return Type.TENSOR;
        if (std.mem.eql(u8, name, "list") or std.mem.eql(u8, name, "List")) return Type.LIST;
        if (std.mem.eql(u8, name, "map") or std.mem.eql(u8, name, "Map") or std.mem.eql(u8, name, "dict")) return Type.MAP;
        if (std.mem.eql(u8, name, "set") or std.mem.eql(u8, name, "Set")) return Type.SET;
        if (std.mem.eql(u8, name, "deque") or std.mem.eql(u8, name, "Deque")) return Type.DEQUE;
        if (std.mem.eql(u8, name, "dyn") or std.mem.eql(u8, name, "dynamic")) return Type.DYNAMIC;
        return .{ .kind = .custom_struct, .struct_name = name };
    }

    pub fn checkProgram(self: *TypeChecker, nodes: []const *Node) !void {
        var global_scope = Scope.init(self.allocator, null);
        defer global_scope.deinit();

        // Pass 1: Collect struct, enum, and function signatures
        for (nodes) |node| {
            if (node.data != .list) continue;
            const children = node.data.list;
            if (children.len == 0 or children[0].data != .atom or children[0].data.atom != .symbol) continue;
            const kw = children[0].data.atom.symbol;

            if (std.mem.eql(u8, kw, "struct") or std.mem.eql(u8, kw, "linear-struct")) {
                try self.registerStruct(children, std.mem.eql(u8, kw, "linear-struct"));
            } else if (std.mem.eql(u8, kw, "enum")) {
                try self.registerEnum(children);
            } else if (std.mem.eql(u8, kw, "func")) {
                try self.registerFunction(children);
            }
        }

        // Pass 2: Type check function bodies and enforce linear consumption
        for (nodes) |node| {
            if (node.data != .list) continue;
            const children = node.data.list;
            if (children.len == 0 or children[0].data != .atom or children[0].data.atom != .symbol) continue;
            if (std.mem.eql(u8, children[0].data.atom.symbol, "func")) {
                try self.checkFunctionBody(&global_scope, children);
            }
        }
    }

    fn registerStruct(self: *TypeChecker, children: []const *Node, is_linear: bool) !void {
        if (children.len < 2) return error.InvalidStructSyntax;
        const name = children[1].data.atom.symbol;
        var fields: std.ArrayList(Field) = .empty;

        for (children[2..]) |field_node| {
            if (field_node.data != .list or field_node.data.list.len < 2) continue;
            const f_name = field_node.data.list[0].data.atom.symbol;
            const f_type_name = field_node.data.list[1].data.atom.symbol;
            try fields.append(self.allocator, .{
                .name = f_name,
                .type_info = self.parseTypeString(f_type_name),
            });
        }

        try self.structs.put(name, .{
            .name = name,
            .is_linear = is_linear,
            .fields = try fields.toOwnedSlice(self.allocator),
        });
    }

    fn registerEnum(self: *TypeChecker, children: []const *Node) !void {
        if (children.len < 2) return error.InvalidEnumSyntax;
        const name = children[1].data.atom.symbol;
        var variants: std.ArrayList([]const u8) = .empty;

        for (children[2..]) |v_node| {
            if (v_node.data == .atom and v_node.data.atom == .symbol) {
                try variants.append(self.allocator, v_node.data.atom.symbol);
            }
        }

        try self.enums.put(name, .{
            .name = name,
            .variants = try variants.toOwnedSlice(self.allocator),
        });
    }

    fn registerFunction(self: *TypeChecker, children: []const *Node) !void {
        if (children.len < 3) return error.InvalidFunctionSyntax;
        const name = children[1].data.atom.symbol;
        const params_node = children[2];

        var param_types: std.ArrayList(Type) = .empty;
        const params = if (params_node.data == .list) params_node.data.list else params_node.data.bracket_list;

        for (params) |p| {
            if (p.data == .list and p.data.list.len >= 2) {
                // Typed parameter: (name type)
                const type_name = p.data.list[1].data.atom.symbol;
                try param_types.append(self.allocator, self.parseTypeString(type_name));
            } else {
                // Untyped parameter: dynamic
                try param_types.append(self.allocator, Type.DYNAMIC);
            }
        }

        var ret_type = Type.DYNAMIC;
        if (children.len >= 4 and children[3].data == .atom and children[3].data.atom == .symbol) {
            const sym = children[3].data.atom.symbol;
            if (std.mem.eql(u8, sym, "i32") or std.mem.eql(u8, sym, "i64") or
                std.mem.eql(u8, sym, "f32") or std.mem.eql(u8, sym, "f64") or
                std.mem.eql(u8, sym, "bool") or std.mem.eql(u8, sym, "string") or
                std.mem.eql(u8, sym, "void") or std.mem.eql(u8, sym, "tensor"))
            {
                ret_type = self.parseTypeString(sym);
            }
        }

        const ret_ptr = try self.allocator.create(Type);
        ret_ptr.* = ret_type;

        try self.functions.put(name, .{
            .params = try param_types.toOwnedSlice(self.allocator),
            .ret = ret_ptr,
        });
    }

    fn checkFunctionBody(self: *TypeChecker, parent_scope: *Scope, children: []const *Node) !void {
        var fn_scope = Scope.init(self.allocator, parent_scope);
        defer fn_scope.deinit();

        const params_node = children[2];
        const params = if (params_node.data == .list) params_node.data.list else params_node.data.bracket_list;

        for (params) |p| {
            if (p.data == .list and p.data.list.len >= 2) {
                const p_name = p.data.list[0].data.atom.symbol;
                const p_type_str = p.data.list[1].data.atom.symbol;
                const is_lin = std.mem.startsWith(u8, p_type_str, "linear-");
                const clean_type_str = if (is_lin) p_type_str["linear-".len..] else p_type_str;
                const p_type = self.parseTypeString(clean_type_str);
                try fn_scope.put(p_name, .{
                    .name = p_name,
                    .type_info = p_type,
                    .is_linear = is_lin,
                });
            } else if (p.data == .atom and p.data.atom == .symbol) {
                const p_name = p.data.atom.symbol;
                try fn_scope.put(p_name, .{
                    .name = p_name,
                    .type_info = Type.DYNAMIC,
                });
            }
        }

        // Determine body start index and declared return type
        var body_start: usize = 3;
        var declared_ret = Type.DYNAMIC;
        if (children.len > 3 and children[3].data == .atom and children[3].data.atom == .symbol) {
            const sym = children[3].data.atom.symbol;
            if (std.mem.eql(u8, sym, "i32") or std.mem.eql(u8, sym, "i64") or
                std.mem.eql(u8, sym, "f32") or std.mem.eql(u8, sym, "f64") or
                std.mem.eql(u8, sym, "bool") or std.mem.eql(u8, sym, "string") or
                std.mem.eql(u8, sym, "void") or std.mem.eql(u8, sym, "tensor"))
            {
                declared_ret = self.parseTypeString(sym);
                body_start = 4;
            }
        }

        for (children[body_start..]) |stmt| {
            const stmt_t = try self.inferNode(&fn_scope, stmt);
            // Check return type matching against declared signature
            if (stmt.data == .list and stmt.data.list.len >= 2 and stmt.data.list[0].data == .atom and stmt.data.list[0].data.atom == .symbol) {
                const op = stmt.data.list[0].data.atom.symbol;
                if (std.mem.eql(u8, op, "return") or std.mem.eql(u8, op, "ret")) {
                    if (declared_ret.kind != .dynamic and declared_ret.kind != .void_t) {
                        if (!stmt_t.equals(declared_ret)) {
                            std.debug.print("Type Error: Function declared return type '{s}', but returned '{s}'\n", .{ declared_ret.toZigType(), stmt_t.toZigType() });
                            return error.ReturnTypeMismatch;
                        }
                    }
                }
            }
        }

        // Linear type check: ensure all linear symbols in this scope were consumed!
        var it = fn_scope.symbols.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.is_linear and !entry.value_ptr.consumed) {
                std.debug.print("Linear Type Error: Resource '{s}' was dropped without being consumed!\n", .{entry.key_ptr.*});
                return error.LinearResourceNotConsumed;
            }
        }
    }

    pub fn inferNode(self: *TypeChecker, scope: *Scope, node: *Node) !Type {
        switch (node.data) {
            .atom => |a| switch (a) {
                .nil => return Type.VOID,
                .boolean => return Type.BOOL,
                .number => return Type.F64,
                .string => return Type.STRING,
                .symbol => |s| {
                    if (scope.get(s)) |sym| {
                        if (sym.is_linear) {
                            if (sym.consumed) {
                                std.debug.print("Linear Type Error: Use of previously consumed resource '{s}'!\n", .{s});
                                return error.UseOfConsumedLinearValue;
                            }
                            sym.consumed = true; // Consumed by being used!
                        }
                        return sym.type_info;
                    }
                    return Type.DYNAMIC;
                },
            },
            .bracket_list => {
                return Type.TENSOR;
            },
            .list => |children| {
                if (children.len == 0) return Type.VOID;
                const head = children[0];
                if (head.data != .atom or head.data.atom != .symbol) return Type.DYNAMIC;
                const op = head.data.atom.symbol;

                if (std.mem.eql(u8, op, "let")) {
                    if (children.len < 3) return error.InvalidSyntax;
                    var var_name: []const u8 = "";
                    var decl_type = Type.DYNAMIC;
                    var is_lin = false;

                    if (children[1].data == .list and children[1].data.list.len >= 2) {
                        // (let (x type) expr)
                        var_name = children[1].data.list[0].data.atom.symbol;
                        const type_str = children[1].data.list[1].data.atom.symbol;
                        if (std.mem.startsWith(u8, type_str, "linear-")) {
                            is_lin = true;
                            decl_type = self.parseTypeString(type_str["linear-".len..]);
                        } else {
                            decl_type = self.parseTypeString(type_str);
                        }
                    } else if (children[1].data == .atom and children[1].data.atom == .symbol) {
                        // (let x expr)
                        var_name = children[1].data.atom.symbol;
                    }

                    const val_type = try self.inferNode(scope, children[2]);
                    const final_type = if (decl_type.kind != .dynamic) decl_type else val_type;

                    try scope.put(var_name, .{
                        .name = var_name,
                        .type_info = final_type,
                        .is_linear = is_lin,
                    });
                    return Type.VOID;
                } else if (std.mem.eql(u8, op, "matmul")) {
                    // Dependent shape check for matmul
                    if (children.len >= 3) {
                        const t1 = try self.inferNode(scope, children[1]);
                        const t2 = try self.inferNode(scope, children[2]);

                        if (t1.tensor_info != null and t2.tensor_info != null) {
                            const s1 = t1.tensor_info.?.shape;
                            const s2 = t2.tensor_info.?.shape;
                            if (s1.len == 2 and s2.len == 2) {
                                if (s1[1] != null and s2[0] != null and s1[1].? != s2[0].?) {
                                    std.debug.print("Dependent Type Error: Incompatible tensor shapes for matmul: [{?}, {?}] and [{?}, {?}]\n", .{ s1[0], s1[1], s2[0], s2[1] });
                                    return error.DependentShapeMismatch;
                                }
                                const out_shape = try self.allocator.dupe(?usize, &[_]?usize{ s1[0], s2[1] });
                                return Type{
                                    .kind = .tensor,
                                    .tensor_info = .{ .shape = out_shape },
                                };
                            }
                        }
                    }
                    return Type.TENSOR;
                } else if (std.mem.eql(u8, op, "tensor")) {
                    // (tensor [dims...] [data...])
                    if (children.len >= 3) {
                        const shape_node = children[1];
                        const dims = if (shape_node.data == .bracket_list) shape_node.data.bracket_list else shape_node.data.list;
                        var shape_list: std.ArrayList(?usize) = .empty;
                        for (dims) |d| {
                            if (d.data == .atom and d.data.atom == .number) {
                                try shape_list.append(self.allocator, @intFromFloat(d.data.atom.number));
                            } else {
                                try shape_list.append(self.allocator, null);
                            }
                        }
                        return Type{
                            .kind = .tensor,
                            .tensor_info = .{ .shape = try shape_list.toOwnedSlice(self.allocator) },
                        };
                    }
                    return Type.TENSOR;
                } else if (std.mem.eql(u8, op, "match")) {
                    if (children.len < 2) return error.InvalidSyntax;
                    _ = try self.inferNode(scope, children[1]);
                    var ret_t = Type.DYNAMIC;
                    for (children[2..]) |case_node| {
                        if (case_node.data != .list or case_node.data.list.len < 2) continue;
                        const case_parts = case_node.data.list;
                        for (case_parts[1..]) |body_node| {
                            ret_t = try self.inferNode(scope, body_node);
                        }
                    }
                    return ret_t;
                } else if (std.mem.eql(u8, op, "diff") and children.len >= 3) {
                    return try self.inferNode(scope, children[1]);
                } else if (std.mem.eql(u8, op, "relu") or std.mem.eql(u8, op, "sigmoid") or
                    std.mem.eql(u8, op, "tanh") or std.mem.eql(u8, op, "quantize"))
                {
                    if (children.len >= 2) return try self.inferNode(scope, children[1]);
                    return Type.DYNAMIC;
                } else if (std.mem.eql(u8, op, "mse_loss") or std.mem.eql(u8, op, "cross_entropy")) {
                    return Type.F64;
                } else if (std.mem.eql(u8, op, "sample") or std.mem.eql(u8, op, "uncertain")) {
                    return Type.DYNAMIC;
                } else if (std.mem.eql(u8, op, "Normal") or std.mem.eql(u8, op, "Uniform") or std.mem.eql(u8, op, "Bernoulli")) {
                    return Type.DYNAMIC;
                } else if (std.mem.eql(u8, op, "observe")) {
                    return Type.BOOL;
                } else if (std.mem.eql(u8, op, "for")) {
                    if (children.len >= 3 and children[1].data == .atom and children[1].data.atom == .symbol) {
                        const loop_var = children[1].data.atom.symbol;
                        try scope.put(loop_var, .{
                            .name = loop_var,
                            .type_info = Type.F64,
                        });
                        for (children[3..]) |body_stmt| {
                            _ = try self.inferNode(scope, body_stmt);
                        }
                    }
                    return Type.VOID;
                } else if (std.mem.eql(u8, op, "return") or std.mem.eql(u8, op, "ret")) {
                    if (children.len >= 2) {
                        return try self.inferNode(scope, children[1]);
                    }
                    return Type.VOID;
                } else if (std.mem.eql(u8, op, "autodiff.forward") or std.mem.eql(u8, op, "lambda")) {
                    return Type.DYNAMIC;
                } else if (std.mem.eql(u8, op, "call")) {
                    for (children[2..]) |arg| {
                        _ = try self.inferNode(scope, arg);
                    }
                    if (children.len >= 2 and children[1].data == .atom and children[1].data.atom == .symbol) {
                        const target = children[1].data.atom.symbol;
                        if (self.functions.get(target)) |fn_type| {
                            return fn_type.ret.*;
                        }
                    }
                    return Type.DYNAMIC;
                } else if (std.mem.eql(u8, op, "+") or std.mem.eql(u8, op, "-") or
                    std.mem.eql(u8, op, "*") or std.mem.eql(u8, op, "/"))
                {
                    if (children.len >= 3) {
                        const l = try self.inferNode(scope, children[1]);
                        const r = try self.inferNode(scope, children[2]);
                        if (l.kind == .tensor or r.kind == .tensor) {
                            if (l.kind == .tensor and r.kind == .tensor and l.tensor_info != null and r.tensor_info != null) {
                                const s1 = l.tensor_info.?.shape;
                                const s2 = r.tensor_info.?.shape;
                                const max_len = @max(s1.len, s2.len);
                                var out_shape = try self.allocator.alloc(?usize, max_len);
                                var k: usize = 0;
                                while (k < max_len) : (k += 1) {
                                    const d1: ?usize = if (k < max_len - s1.len) null else s1[k - (max_len - s1.len)];
                                    const d2: ?usize = if (k < max_len - s2.len) null else s2[k - (max_len - s2.len)];
                                    if (d1 != null and d2 != null) {
                                        const v1 = d1.?;
                                        const v2 = d2.?;
                                        if (v1 != v2 and v1 != 1 and v2 != 1) {
                                            std.debug.print("Dependent Type Error: Incompatible tensor broadcast shapes for {s}\n", .{op});
                                            return error.DependentShapeMismatch;
                                        }
                                        out_shape[k] = @max(v1, v2);
                                    } else {
                                        out_shape[k] = d1 orelse d2;
                                    }
                                }
                                return Type{
                                    .kind = .tensor,
                                    .tensor_info = .{ .shape = out_shape },
                                };
                            } else if (l.kind == .tensor) {
                                return l;
                            } else {
                                return r;
                            }
                        }
                        if (l.isFloat() or r.isFloat()) return Type.F64;
                        if (l.isInteger() and r.isInteger()) return l;
                    }
                    return Type.DYNAMIC;
                } else if (std.mem.eql(u8, op, "<") or std.mem.eql(u8, op, "<=") or
                    std.mem.eql(u8, op, ">") or std.mem.eql(u8, op, ">=") or
                    std.mem.eql(u8, op, "==") or std.mem.eql(u8, op, "!="))
                {
                    return Type.BOOL;
                } else if (std.mem.eql(u8, op, "if")) {
                    if (children.len >= 4) {
                        _ = try self.inferNode(scope, children[1]);
                        const t = try self.inferNode(scope, children[2]);
                        const f = try self.inferNode(scope, children[3]);
                        if (t.equals(f)) return t;
                    }
                    return Type.DYNAMIC;
                }

                return Type.DYNAMIC;
            },
        }
    }
};

test "type checker validates typed functions and dependent shapes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(func add ((a i64) (b i64)) i64
        \\  (return (+ a b)))
        \\(func matmul_test ()
        \\  (let a (tensor [2 3] [1 2 3 4 5 6]))
        \\  (let b (tensor [3 2] [1 2 3 4 5 6]))
        \\  (let c (matmul a b))
        \\  (return c))
    ;

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var tc = TypeChecker.init(arena.allocator());
    defer tc.deinit();

    try tc.checkProgram(nodes);
}

test "type checker catches linear resource duplication error" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(func linear_test ((file linear-FileHandle)) void
        \\  (call close file)
        \\  (call close file))
    ;

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var tc = TypeChecker.init(arena.allocator());
    defer tc.deinit();

    const result = tc.checkProgram(nodes);
    try std.testing.expectError(error.UseOfConsumedLinearValue, result);
}

test "type checker catches dependent shape mismatch at compile time" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(func invalid_matmul ()
        \\  (let a (tensor [2 3] [1 2 3 4 5 6]))
        \\  (let b (tensor [4 2] [1 2 3 4 5 6 7 8]))
        \\  (let c (matmul a b))
        \\  (return c))
    ;

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var tc = TypeChecker.init(arena.allocator());
    defer tc.deinit();

    const result = tc.checkProgram(nodes);
    try std.testing.expectError(error.DependentShapeMismatch, result);
}

test "type checker catches return type mismatch" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const src =
        \\(func bad_return () i64
        \\  (return "this is a string, not i64"))
    ;

    var p = try parser.Parser.init(src, arena.allocator());
    const nodes = try p.parseAll();

    var tc = TypeChecker.init(arena.allocator());
    defer tc.deinit();

    const result = tc.checkProgram(nodes);
    try std.testing.expectError(error.ReturnTypeMismatch, result);
}

test "type checker validates tensor broadcasting" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    // Valid broadcasting: [2 1] + [2 3] -> [2 3]
    const valid_src =
        \\(func broadcast_ok ()
        \\  (let a (tensor [2 1] [10 20]))
        \\  (let b (tensor [2 3] [1 2 3 4 5 6]))
        \\  (let c (+ a b))
        \\  (return c))
    ;
    var p1 = try parser.Parser.init(valid_src, arena.allocator());
    const n1 = try p1.parseAll();
    var tc1 = TypeChecker.init(arena.allocator());
    defer tc1.deinit();
    try tc1.checkProgram(n1);

    // Invalid broadcasting: [2 3] + [2 4] -> DependentShapeMismatch
    const invalid_src =
        \\(func broadcast_bad ()
        \\  (let a (tensor [2 3] [1 2 3 4 5 6]))
        \\  (let b (tensor [2 4] [1 2 3 4 5 6 7 8]))
        \\  (let c (+ a b))
        \\  (return c))
    ;
    var p2 = try parser.Parser.init(invalid_src, arena.allocator());
    const n2 = try p2.parseAll();
    var tc2 = TypeChecker.init(arena.allocator());
    defer tc2.deinit();
    try std.testing.expectError(error.DependentShapeMismatch, tc2.checkProgram(n2));
}
