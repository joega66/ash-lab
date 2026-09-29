//! Compile-time values a generic device function may be specialized with, lifted to types.
//!
//! A `function!` declared as `Name<T, const N: u32> for [(f32, 4), (u32, 8)]` registers
//! exactly those combinations, and proves it the same way it does for element types: with a
//! [`DTypeOf`](crate::DTypeOf) impl on the tuple of its arguments. A value cannot stand in a
//! tuple, so each is carried there as one of these types instead: `(f32, ConstU32<4>)`.
//!
//! Rust has no `struct Const<T, const V: T>`, so there is one marker per scalar a Slang
//! `let` parameter can have. `function!` picks the marker from the parameter's declared type.

macro_rules! const_param {
    ($($(#[$doc:meta])* $name:ident: $ty:ty;)+) => {$(
        $(#[$doc])*
        pub struct $name<const V: $ty>;
    )+};
}

const_param! {
    /// A `const B: bool` argument, as `let B : bool` in Slang.
    ConstBool: bool;
    /// A `const N: i8` argument, as `let N : int8_t` in Slang.
    ConstI8: i8;
    /// A `const N: u8` argument, as `let N : uint8_t` in Slang.
    ConstU8: u8;
    /// A `const N: i16` argument, as `let N : int16_t` in Slang.
    ConstI16: i16;
    /// A `const N: u16` argument, as `let N : uint16_t` in Slang.
    ConstU16: u16;
    /// A `const N: i32` argument, as `let N : int` in Slang.
    ConstI32: i32;
    /// A `const N: u32` argument, as `let N : uint` in Slang.
    ConstU32: u32;
    /// A `const N: i64` argument, as `let N : int64_t` in Slang.
    ConstI64: i64;
    /// A `const N: u64` argument, as `let N : uint64_t` in Slang.
    ConstU64: u64;
}
