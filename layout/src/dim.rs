//! A single shape or stride dimension, known either at compile time or at run time.

use core::fmt;

/// One entry of a shape or stride tuple.
///
/// The two concrete implementations are [`Const<N>`], whose extent is baked into the type, and
/// `usize`, whose extent is only known at run time. `STATIC` is what lets the rest of the crate
/// fold compile-time dimensions away: sizes, strides and cosizes stay `Some(..)` for as long as
/// every dimension feeding into them is static.
pub trait Dim: Copy {
    /// `Some(n)` if this dimension is known at compile time, `None` if it is a runtime value.
    const STATIC: Option<usize>;

    /// This dimension's extent.
    fn value(self) -> usize;
}

/// A dimension whose extent is part of its type. Zero-sized, so static dimensions cost no storage.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Const<const N: usize>;

impl<const N: usize> Dim for Const<N> {
    const STATIC: Option<usize> = Some(N);

    #[inline(always)]
    fn value(self) -> usize {
        N
    }
}

impl<const N: usize> fmt::Display for Const<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{N}")
    }
}

/// A dimension supplied at run time.
impl Dim for usize {
    const STATIC: Option<usize> = None;

    #[inline(always)]
    fn value(self) -> usize {
        self
    }
}

/// The product of two dimensions, static whenever both of its operands are.
///
/// [`row_major`](crate::row_major) and [`col_major`](crate::col_major) build strides out of these,
/// so a stride over static dimensions is itself a zero-sized compile-time constant, and a stride
/// that happens to mix in a runtime dimension degrades to a single multiply.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Prod<A, B>(pub A, pub B);

impl<A: Dim, B: Dim> Dim for Prod<A, B> {
    const STATIC: Option<usize> = match (A::STATIC, B::STATIC) {
        (Some(a), Some(b)) => Some(a * b),
        _ => None,
    };

    #[inline(always)]
    fn value(self) -> usize {
        // Monomorphizes to a constant when both halves are static.
        match Self::STATIC {
            Some(n) => n,
            None => self.0.value() * self.1.value(),
        }
    }
}

/// Product of a list of per-dimension compile-time extents, or `None` if any of them is dynamic.
pub const fn static_product(dims: &[Option<usize>]) -> Option<usize> {
    let mut acc = 1;
    let mut i = 0;
    while i < dims.len() {
        match dims[i] {
            Some(d) => acc *= d,
            None => return None,
        }
        i += 1;
    }
    Some(acc)
}

/// Size of the memory region spanned by a shape/stride pair, or `None` if any dimension is dynamic.
pub const fn static_cosize(shape: &[Option<usize>], stride: &[Option<usize>]) -> Option<usize> {
    let mut acc = 1;
    let mut i = 0;
    while i < shape.len() {
        let (s, d) = match (shape[i], stride[i]) {
            (Some(s), Some(d)) => (s, d),
            _ => return None,
        };
        if s == 0 {
            return Some(0);
        }
        acc += (s - 1) * d;
        i += 1;
    }
    Some(acc)
}

/// Whether every dimension in `dims` is known at compile time.
pub const fn all_static(dims: &[Option<usize>]) -> bool {
    let mut i = 0;
    while i < dims.len() {
        if dims[i].is_none() {
            return false;
        }
        i += 1;
    }
    true
}
