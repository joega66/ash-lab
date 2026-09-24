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
// plain `f32`/`i32`/`u32` against them. `convert` goes through the source's own native
// type rather than a common bridge, which keeps every conversion exact — there is no type
// that could serve as a bridge here in any case, since Metal has no `double`.
interface IDType : IArithmetic {
    __init(float v);
    __init(int v);
    __init(uint v);

    // This value as any other element type, converted as the scalar cast would.
    To convert<To : IDType>();
}

#define DTYPE_WRAPPER(NAME, SCALAR, NATIVE_CONVERT)                                        \
    struct NAME : IDType {                                                                 \
        SCALAR value;                                                                      \
                                                                                           \
        /* `IArithmetic`'s init requirement is itself generic. Narrowing it to `IDType`  \
           still satisfies the conformance, and is what makes `Dst(src)` between two type  \
           parameters a compile error rather than a silent zero: the supported spelling is \
           `dtype_cast`. The concrete overloads below win resolution for `T(10)`. */       \
        __init<T : IDType>(T v) { this = v.convert<NAME>(); }                                \
                                                                                           \
        __init(float v) { value = (SCALAR)v; }                                             \
        __init(int v) { value = (SCALAR)v; }                                               \
        __init(uint v) { value = (SCALAR)v; }                                              \
                                                                                           \
        To convert<To : IDType>() { return To(NATIVE_CONVERT); }                           \
                                                                                           \
        NAME add(NAME o) { return NAME(this.value + o.value); }                            \
        NAME sub(NAME o) { return NAME(this.value - o.value); }                            \
        NAME mul(NAME o) { return NAME(this.value * o.value); }                            \
        NAME div(NAME o) { return NAME(this.value / o.value); }                            \
        NAME mod(NAME o) { return NAME(this.value % o.value); }                            \
        NAME neg() { return NAME(-this.value); }                                           \
                                                                                           \
        bool equals(NAME o) { return this.value == o.value; }                              \
        bool lessThan(NAME o) { return this.value < o.value; }                              \
        bool lessThanOrEquals(NAME o) { return this.value <= o.value; }                    \
    }

DTYPE_WRAPPER(F32, float, this.value);
DTYPE_WRAPPER(I32, int, this.value);
DTYPE_WRAPPER(U32, uint, this.value);

#undef DTYPE_WRAPPER

// Converts between the element types a kernel was instantiated with.
To dtype_cast<To : IDType, From : IDType>(From value) {
    return value.convert<To>();
}