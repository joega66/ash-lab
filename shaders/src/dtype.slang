#pragma once

// The element types a generic kernel may be instantiated with.
//
// Slang type-checks a generic body once against its constraints, rather than once per
// instantiation the way the preprocessor did, so a kernel may only do what its constraints
// allow. Converting between two type parameters therefore needs an interface that declares
// the conversion, and no built-in interface spans both integers and floats: `IFloat` has
// `toFloat`, `IInteger` has `toInt`, and `IArithmetic` has neither.
//
// Declaring that interface here and attaching it to `float`/`int`/`uint` with an
// `extension` does not work: `slangc -specialize` resolves its type argument without user
// extensions in scope and rejects the conformance. A struct that declares the interface
// directly does specialize, so each element type is wrapped in one.
//
// Each wrapper holds exactly one scalar and nothing else, so it has the size, alignment and
// layout of that scalar: a buffer of wrappers is a buffer of scalars, and the host binds
// plain `f32`/`i32`/`u8`/... against them. `bool` is the one exception on the host side: it
// is stored as a 32-bit word, so the host binds `gpu::Bool32` rather than a Rust `bool`. `convert` goes through the source's own native
// type rather than a common bridge, which keeps every conversion exact — there is no type
// that could serve as a bridge here in any case, since Metal has no `double`.
interface IDType : IArithmetic {
    // One constructor per native scalar, so that `convert` can hand its value over
    // unchanged whatever the source type is.
    __init(bool v);
    __init(int8_t v);
    __init(uint8_t v);
    __init(int16_t v);
    __init(uint16_t v);
    __init(int v);
    __init(uint v);
    __init(int64_t v);
    __init(uint64_t v);
    __init(float v);

    // This value as any other element type, converted as the scalar cast would.
    To convert<To : IDType>();
}

// `ARITH` is the type the arithmetic is done in. It is the scalar itself everywhere but
// `bool`, which has no arithmetic of its own: a `B32` is computed as an `int` and cast back,
// so any nonzero result is `true`, the same as the scalar cast.
#define DTYPE_WRAPPER(NAME, SCALAR, ARITH)                                                 \
    struct NAME : IDType {                                                                 \
        SCALAR value;                                                                      \
                                                                                           \
        /* `IArithmetic`'s init requirement is itself generic. Narrowing it to `IDType`    \
           still satisfies the conformance, and is what makes `Dst(src)` between two type  \
           parameters a compile error rather than a silent zero: the supported spelling is \
           `dtype_cast`. The concrete overloads below win resolution for `T(10)`. */       \
        __init<T : IDType>(T v) {                                                          \
            this = v.convert<NAME>();                                                      \
        }                                                                                  \
                                                                                           \
        __init(bool v) {                                                                   \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
        __init(int8_t v) {                                                                 \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
        __init(uint8_t v) {                                                                \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
        __init(int16_t v) {                                                                \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
        __init(uint16_t v) {                                                               \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
        __init(int v) {                                                                    \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
        __init(uint v) {                                                                   \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
        __init(int64_t v) {                                                                \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
        __init(uint64_t v) {                                                               \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
        __init(float v) {                                                                  \
            value = (SCALAR)v;                                                             \
        }                                                                                  \
                                                                                           \
        To convert<To : IDType>() {                                                        \
            return To(this.value);                                                         \
        }                                                                                  \
                                                                                           \
        NAME add(NAME o) {                                                                 \
            return NAME((SCALAR)((ARITH)this.value + (ARITH)o.value));                     \
        }                                                                                  \
        NAME sub(NAME o) {                                                                 \
            return NAME((SCALAR)((ARITH)this.value - (ARITH)o.value));                     \
        }                                                                                  \
        NAME mul(NAME o) {                                                                 \
            return NAME((SCALAR)((ARITH)this.value * (ARITH)o.value));                     \
        }                                                                                  \
        NAME div(NAME o) {                                                                 \
            return NAME((SCALAR)((ARITH)this.value / (ARITH)o.value));                     \
        }                                                                                  \
        NAME mod(NAME o) {                                                                 \
            return NAME((SCALAR)((ARITH)this.value % (ARITH)o.value));                     \
        }                                                                                  \
        NAME neg() {                                                                       \
            return NAME((SCALAR)(-(ARITH)this.value));                                     \
        }                                                                                  \
                                                                                           \
        bool equals(NAME o) {                                                              \
            return (ARITH)this.value == (ARITH)o.value;                                    \
        }                                                                                  \
        bool lessThan(NAME o) {                                                            \
            return (ARITH)this.value < (ARITH)o.value;                                     \
        }                                                                                  \
        bool lessThanOrEquals(NAME o) {                                                    \
            return (ARITH)this.value <= (ARITH)o.value;                                    \
        }                                                                                  \
    }

DTYPE_WRAPPER(B32, bool, int);
DTYPE_WRAPPER(I8, int8_t, int8_t);
DTYPE_WRAPPER(U8, uint8_t, uint8_t);
DTYPE_WRAPPER(I16, int16_t, int16_t);
DTYPE_WRAPPER(U16, uint16_t, uint16_t);
DTYPE_WRAPPER(I32, int, int);
DTYPE_WRAPPER(U32, uint, uint);
DTYPE_WRAPPER(I64, int64_t, int64_t);
DTYPE_WRAPPER(U64, uint64_t, uint64_t);
DTYPE_WRAPPER(F32, float, float);

#undef DTYPE_WRAPPER

// Converts between the element types a kernel was instantiated with.
To dtype_cast<To : IDType, From : IDType>(From value) {
    return value.convert<To>();
}
