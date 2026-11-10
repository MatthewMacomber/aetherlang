/*
 * Aether Scripting Engine - C / C++ Embedding API
 * Copyright (c) 2026 CrypticTek / Aether Authors. MIT License.
 */

#ifndef AETHER_H
#define AETHER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Value Type Tag
 */
typedef enum {
    AE_TYPE_NIL = 0,
    AE_TYPE_BOOLEAN = 1,
    AE_TYPE_NUMBER = 2,
    AE_TYPE_STRING = 3,
    AE_TYPE_PTR = 4,
    AE_TYPE_TENSOR = 5,
    AE_TYPE_OBJECT = 6
} AeType;

/*
 * 32-Byte Value Representation passed across the ABI
 */
typedef struct AeValue {
    int32_t type;
    int32_t _pad;
    double num;
    const void* ptr;
    size_t len;
} AeValue;

/*
 * Opaque Virtual Machine State
 */
typedef struct AetherVM AetherVM;

/*
 * Host Callback Function Signature
 */
typedef AeValue (*AeHostFn)(size_t argc, const AeValue* argv, void* userdata);

/*
 * Lifecycle Management
 */
AetherVM* ae_vm_create(void);
void ae_vm_destroy(AetherVM* vm);
const char* ae_last_error(const AetherVM* vm);

/*
 * Script Execution
 */
AeValue ae_eval(AetherVM* vm, const char* source);
AeValue ae_load_file(AetherVM* vm, const char* path);
AeValue ae_call(AetherVM* vm, const char* fn_name, size_t argc, const AeValue* argv);

/*
 * Environment & Host Interoperability
 */
void ae_set_global(AetherVM* vm, const char* name, AeValue val);
AeValue ae_get_global(AetherVM* vm, const char* name);
void ae_register_fn(AetherVM* vm, const char* name, AeHostFn func, void* userdata);
AeValue ae_tensor_create_1d(AetherVM* vm, const double* data, size_t len);
void ae_gc_collect(AetherVM* vm);

/*
 * Inline C Constructors & Accessors
 */
static inline AeValue ae_nil(void) {
    AeValue v;
    memset(&v, 0, sizeof(v));
    v.type = AE_TYPE_NIL;
    return v;
}

static inline AeValue ae_bool(bool b) {
    AeValue v;
    memset(&v, 0, sizeof(v));
    v.type = AE_TYPE_BOOLEAN;
    v.num = b ? 1.0 : 0.0;
    return v;
}

static inline AeValue ae_number(double n) {
    AeValue v;
    memset(&v, 0, sizeof(v));
    v.type = AE_TYPE_NUMBER;
    v.num = n;
    return v;
}

static inline AeValue ae_string(const char* s) {
    AeValue v;
    memset(&v, 0, sizeof(v));
    v.type = AE_TYPE_STRING;
    v.ptr = (const void*)s;
    v.len = s ? strlen(s) : 0;
    return v;
}

static inline AeValue ae_ptr(void* p) {
    AeValue v;
    memset(&v, 0, sizeof(v));
    v.type = AE_TYPE_PTR;
    v.ptr = (const void*)p;
    return v;
}

static inline double ae_as_number(AeValue v) {
    return (v.type == AE_TYPE_NUMBER || v.type == AE_TYPE_BOOLEAN) ? v.num : 0.0;
}

static inline int64_t ae_as_int(AeValue v) {
    return (int64_t)ae_as_number(v);
}

static inline bool ae_as_bool(AeValue v) {
    if (v.type == AE_TYPE_BOOLEAN || v.type == AE_TYPE_NUMBER) return v.num != 0.0;
    if (v.type == AE_TYPE_STRING) return v.len > 0;
    if (v.type == AE_TYPE_PTR) return v.ptr != NULL;
    return false;
}

static inline const char* ae_as_string(AeValue v) {
    return (v.type == AE_TYPE_STRING) ? (const char*)v.ptr : "";
}

static inline void* ae_as_ptr(AeValue v) {
    return (v.type == AE_TYPE_PTR) ? (void*)v.ptr : NULL;
}

#ifdef __cplusplus
} // extern "C"

#include <string>
#include <vector>
#include <initializer_list>
#include <stdexcept>

namespace aether {

class VM {
private:
    AetherVM* handle;

public:
    VM() : handle(ae_vm_create()) {
        if (!handle) throw std::runtime_error("Failed to create Aether VM");
    }

    ~VM() {
        if (handle) {
            ae_vm_destroy(handle);
            handle = nullptr;
        }
    }

    VM(const VM&) = delete;
    VM& operator=(const VM&) = delete;

    VM(VM&& other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    VM& operator=(VM&& other) noexcept {
        if (this != &other) {
            if (handle) ae_vm_destroy(handle);
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }

    AetherVM* raw() const { return handle; }

    AeValue eval(const std::string& code) {
        return ae_eval(handle, code.c_str());
    }

    AeValue load_file(const std::string& path) {
        return ae_load_file(handle, path.c_str());
    }

    AeValue call(const std::string& fn_name, const std::vector<AeValue>& args = {}) {
        return ae_call(handle, fn_name.c_str(), args.size(), args.data());
    }

    AeValue call(const std::string& fn_name, std::initializer_list<AeValue> args) {
        std::vector<AeValue> v(args);
        return ae_call(handle, fn_name.c_str(), v.size(), v.data());
    }

    void set_global(const std::string& name, AeValue val) {
        ae_set_global(handle, name.c_str(), val);
    }

    void set_global(const std::string& name, double val) {
        ae_set_global(handle, name.c_str(), ae_number(val));
    }

    void set_global(const std::string& name, const std::string& val) {
        ae_set_global(handle, name.c_str(), ae_string(val.c_str()));
    }

    void set_global(const std::string& name, bool val) {
        ae_set_global(handle, name.c_str(), ae_bool(val));
    }

    void set_global(const std::string& name, void* val) {
        ae_set_global(handle, name.c_str(), ae_ptr(val));
    }

    AeValue get_global(const std::string& name) const {
        return ae_get_global(handle, name.c_str());
    }

    void register_fn(const std::string& name, AeHostFn fn, void* userdata = nullptr) {
        ae_register_fn(handle, name.c_str(), fn, userdata);
    }

    const char* last_error() const {
        return ae_last_error(handle);
    }
};

} // namespace aether

#endif // __cplusplus

#endif // AETHER_H

