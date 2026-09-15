//! The [`Layout`] type: a shape/stride pair mapping logical coordinates to linear indices.

use core::fmt;

use crate::dim::{all_static, static_cosize};
use crate::shape::{DimAt, Shape};

/// A mapping from logical coordinates to a linear memory index, defined by a shape and a stride.
///
/// The rank is the arity of the shape tuple and so is always known at compile time, while each
/// dimension is independently static ([`Const<N>`](crate::Const)) or dynamic (`usize`). A layout
/// over entirely static dimensions is zero-sized and every query on it — [`size`](Self::size),
/// [`cosize`](Self::cosize), [`crd2idx`](Self::crd2idx) — folds to constants; mixing in a runtime
/// dimension only makes the terms that depend on it dynamic.
///
/// The stride's `Coords` is tied to the shape's, which is how equal ranks are enforced at compile
/// time.
///
/// ```
/// use layout::{Const, Layout, row_major};
///
/// // 4 x 8, row-major: index (i, j) lives at 8i + j.
/// let l = row_major((Const::<4>, Const::<8>));
/// assert_eq!(l.crd2idx([1, 3]), 11);
/// assert_eq!(l.size(), 32);
/// assert_eq!(size_of_val(&l), 0);
///
/// // The same mapping with a runtime column count.
/// let n = 8;
/// let l = row_major((Const::<4>, n));
/// assert_eq!(l.crd2idx([1, 3]), 11);
/// assert_eq!(Layout::<(Const<4>, usize), (usize, Const<1>)>::STATIC_PRODUCT, None);
/// ```
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct Layout<S, D> {
    /// The logical coordinate space.
    pub shape: S,
    /// The memory step taken per unit of each coordinate.
    pub stride: D,
}

impl<S, D> Layout<S, D> {
    /// Builds a layout from an explicit shape and stride.
    #[inline]
    pub const fn new(shape: S, stride: D) -> Self {
        Self { shape, stride }
    }
}

impl<S: Shape, D: Shape<Coords = S::Coords>> Layout<S, D> {
    /// Number of dimensions.
    pub const RANK: usize = S::RANK;

    /// Number of dimensions after flattening nested coordinates.
    ///
    /// This crate's shapes are flat, so this is always [`RANK`](Self::RANK); it exists so code
    /// ported from Mojo keeps reading the same.
    pub const FLAT_RANK: usize = S::RANK;

    /// Whether every shape dimension is known at compile time.
    pub const SHAPE_KNOWN: bool = S::ALL_KNOWN;

    /// Whether every stride dimension is known at compile time.
    pub const STRIDE_KNOWN: bool = D::ALL_KNOWN;

    /// Whether every shape *and* stride dimension is known at compile time.
    pub const ALL_DIMS_KNOWN: bool = all_static(S::STATIC) && all_static(D::STATIC);

    /// Compile-time product of all shape dimensions, or `None` if any of them is dynamic.
    pub const STATIC_PRODUCT: Option<usize> = S::STATIC_PRODUCT;

    /// Compile-time size of the memory region this layout spans, or `None` if any dimension is
    /// dynamic.
    pub const STATIC_COSIZE: Option<usize> = static_cosize(S::STATIC, D::STATIC);

    /// Compile-time extent of shape dimension `i`, or `None` if it is a runtime value.
    #[inline]
    pub const fn static_shape(i: usize) -> Option<usize> {
        S::STATIC[i]
    }

    /// Compile-time value of stride dimension `i`, or `None` if it is a runtime value.
    #[inline]
    pub const fn static_stride(i: usize) -> Option<usize> {
        D::STATIC[i]
    }

    /// Number of dimensions, for when naming the layout's type to reach [`RANK`](Self::RANK) is
    /// inconvenient.
    #[inline]
    pub fn rank(&self) -> usize {
        S::RANK
    }

    /// The full shape.
    #[inline]
    pub fn shape_coord(&self) -> S {
        self.shape
    }

    /// The full stride.
    #[inline]
    pub fn stride_coord(&self) -> D {
        self.stride
    }

    /// The extent of shape dimension `i`.
    #[inline]
    pub fn shape_at(&self, i: usize) -> usize {
        self.shape.get(i)
    }

    /// The value of stride dimension `i`.
    #[inline]
    pub fn stride_at(&self, i: usize) -> usize {
        self.stride.get(i)
    }

    /// Shape dimension `I`, keeping its static-ness: this gives back a zero-sized
    /// [`Const<N>`](crate::Const) for a compile-time dimension.
    #[inline]
    pub fn shape_dim<const I: usize>(&self) -> <S as DimAt<I>>::Dim
    where
        S: DimAt<I>,
    {
        self.shape.dim_at()
    }

    /// Stride dimension `I`, keeping its static-ness.
    #[inline]
    pub fn stride_dim<const I: usize>(&self) -> <D as DimAt<I>>::Dim
    where
        D: DimAt<I>,
    {
        self.stride.dim_at()
    }

    /// Total number of elements in the layout's domain.
    #[inline]
    pub fn product(&self) -> usize {
        self.shape.product()
    }

    /// Total number of elements in the layout's domain. Alias for [`product`](Self::product).
    #[inline]
    pub fn size(&self) -> usize {
        self.product()
    }

    /// Size of the memory region the layout spans: `(m - 1) * r + (n - 1) * s + 1` for shape
    /// `(m, n)` and stride `(r, s)`, or 0 for an empty domain.
    #[inline]
    pub fn cosize(&self) -> usize {
        if let Some(n) = Self::STATIC_COSIZE {
            return n;
        }
        let shape = self.shape.dims();
        let stride = self.stride.dims();
        let mut acc = 1;
        for (&s, &d) in shape.as_ref().iter().zip(stride.as_ref()) {
            if s == 0 {
                return 0;
            }
            acc += (s - 1) * d;
        }
        acc
    }

    /// Maps logical coordinates to a linear memory index.
    #[inline]
    pub fn crd2idx(&self, crd: S::Coords) -> usize {
        let stride = self.stride.dims();
        let mut idx = 0;
        for (&c, &d) in crd.as_ref().iter().zip(stride.as_ref()) {
            idx += c * d;
        }
        idx
    }

    /// Maps logical coordinates to a linear memory index. The equivalent of Mojo's `__call__`.
    #[inline]
    pub fn call(&self, crd: S::Coords) -> usize {
        self.crd2idx(crd)
    }

    /// Inverse of [`crd2idx`](Self::crd2idx): maps a linear index back to logical coordinates via
    /// `crd[i] = (idx / stride[i]) % shape[i]`.
    ///
    /// This is exact for layouts that are injective over their domain; for others it returns the
    /// representative coordinate the formula yields. A zero stride is treated as 1 so that
    /// broadcast dimensions do not divide by zero.
    #[inline]
    pub fn idx2crd(&self, idx: usize) -> S::Coords {
        let shape = self.shape.dims();
        let stride = self.stride.dims();
        let mut crd = S::zero_coords();
        for i in 0..S::RANK {
            let (s, d) = (shape.as_ref()[i], stride.as_ref()[i]);
            crd.as_mut()[i] = if s == 0 { 0 } else { (idx / d.max(1)) % s };
        }
        crd
    }

    /// Maps the `linear_idx`-th element of the domain, walked with the leftmost coordinate varying
    /// fastest, to its memory index.
    ///
    /// Iterating `0..layout.size()` through this visits every element of the layout exactly once.
    #[inline]
    pub fn linear2idx(&self, linear_idx: usize) -> usize {
        let shape = self.shape.dims();
        let stride = self.stride.dims();
        let mut rest = linear_idx;
        let mut idx = 0;
        for (&s, &d) in shape.as_ref().iter().zip(stride.as_ref()) {
            if s == 0 {
                return 0;
            }
            idx += (rest % s) * d;
            rest /= s;
        }
        idx
    }

    /// Every memory index in the layout's domain, in the order used by
    /// [`linear2idx`](Self::linear2idx).
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.size()).map(move |i| self.linear2idx(i))
    }

    /// Reverses the dimension order, turning a row-major layout into a column-major one.
    #[inline]
    pub fn reverse(&self) -> Layout<S::Reversed, D::Reversed> {
        Layout::new(self.shape.reversed(), self.stride.reversed())
    }

    /// Transposes the layout by reversing its dimensions; for rank 2 this swaps rows and columns.
    #[inline]
    pub fn transpose(&self) -> Layout<S::Reversed, D::Reversed> {
        self.reverse()
    }

    /// Materializes every dimension as a runtime value, erasing compile-time extents.
    #[inline]
    pub fn make_dynamic(&self) -> Layout<S::Dynamic, D::Dynamic> {
        Layout::new(self.shape.to_dynamic(), self.stride.to_dynamic())
    }

    /// Whether two layouts describe the same mapping, regardless of which dimensions each one
    /// knows at compile time.
    #[inline]
    pub fn dims_eq<S2, D2>(&self, other: &Layout<S2, D2>) -> bool
    where
        S2: Shape,
        D2: Shape<Coords = S2::Coords>,
    {
        self.shape.dims().as_ref() == other.shape.dims().as_ref()
            && self.stride.dims().as_ref() == other.stride.dims().as_ref()
    }
}

impl<S: Shape, D: Shape<Coords = S::Coords>> fmt::Display for Layout<S, D> {
    /// Writes the layout as `(shape:stride)`, e.g. `((4, 8):(8, 1))`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        self.shape.write_to(f)?;
        f.write_str(":")?;
        self.stride.write_to(f)?;
        f.write_str(")")
    }
}

impl<S: Shape, D: Shape<Coords = S::Coords>> fmt::Debug for Layout<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
