use core::fmt;

use crate::dim::{ConstDim, all_static, static_cosize};
use crate::shape::{DimAt, Shape};

/// A layout that supports mixed compile-time and runtime dimensions.
///
/// This layout provides a unified interface for layouts where some dimensions
//  are known at compile time and others are determined at runtime. It enables
/// more ergonomic layout definitions while maintaining performance.
///
/// A Layout's shape and strides must be non-negative.
#[repr(C)]
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
    pub const FLAT_RANK: usize = S::RANK;

    /// Whether all shape dimensions are known at compile time.
    pub const SHAPE_KNOWN: bool = S::ALL_KNOWN;

    /// Whether all stride dimensions are known at compile time.
    pub const STRIDE_KNOWN: bool = D::ALL_KNOWN;

    /// Whether all shape *and* stride dimensions are known at compile time.
    pub const ALL_DIMS_KNOWN: bool = all_static(S::STATIC) && all_static(D::STATIC);

    /// Compile-time product of all shape dimensions, or `None` if any of them is dynamic.
    pub const STATIC_PRODUCT: Option<usize> = S::STATIC_PRODUCT;

    /// Compile-time size of the memory region this layout spans, or `None` if any dimension is
    /// dynamic.
    pub const STATIC_COSIZE: Option<usize> = static_cosize(S::STATIC, D::STATIC);

    /// Returns the compile-time value of the i-th flattened shape dimension.
    #[inline]
    pub const fn static_shape<const I: usize>() -> usize
    where
        S: DimAt<I>,
        <S as DimAt<I>>::Dim: ConstDim,
    {
        <<S as DimAt<I>>::Dim as ConstDim>::VALUE
    }

    /// Returns the compile-time value of the i-th flattened stride dimension.
    #[inline]
    pub const fn static_stride<const I: usize>() -> usize
    where
        D: DimAt<I>,
        <D as DimAt<I>>::Dim: ConstDim,
    {
        <<D as DimAt<I>>::Dim as ConstDim>::VALUE
    }

    /// The number of dimensions in the layout.
    #[inline]
    pub fn rank(&self) -> usize {
        S::RANK
    }

    /// Returns the full shape as a Coord.
    #[inline]
    pub fn shape_coord(&self) -> S {
        self.shape
    }

    /// Returns the full stride as a Coord.
    #[inline]
    pub fn stride_coord(&self) -> D {
        self.stride
    }

    /// Returns the total number of elements in the layout's domain.
    ///
    /// For a layout with shape(m, n), this returns m*n, representing the total
    /// number of valid coordinates in the layout.
    #[inline]
    pub fn product(&self) -> usize {
        self.shape.product()
    }

    /// Returns the total number of elements in the layout's domain.
    /// Alias for `product()`.
    #[inline]
    pub fn size(&self) -> usize {
        self.product()
    }

    /// Returns the size of the memory region spanned by the layout.
    ///
    /// For a layout with `shape(m, n)` and `stride(r, s)`, this returns `(m - 1) * r + (n - 1) * s + 1`,
    /// representing the memory footprint.
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

    /// Converts multi-dimensional coordinates to a linear index.
    ///
    /// This function is the inverse of `idx2crd` , transforming a set of coordinates into a flat index
    /// based on the provided shape and stride information. This is essential for mapping multi-
    /// dimensional tensor elements to linear memory.
    #[inline]
    pub fn crd2idx(&self, crd: S::Coords) -> usize {
        let stride = self.stride.dims();
        let mut idx = 0;
        for (&c, &d) in crd.as_ref().iter().zip(stride.as_ref()) {
            idx += c * d;
        }
        idx
    }

    /// Converts a linear index to multi-dimensional coordinates. This function transforms a flat
    /// index into coordinate values based on the provided shape and stride information. This is
    /// essential for mapping linear memory accesses to multi-dimensional tensor elements.
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

    /// Reverse the order of dimensions in the layout.
    /// Turns row-major into column-major ordering where the stride-1 dimension comes first,
    /// enabling coalesced scalar iteration.
    #[inline]
    pub fn reverse(&self) -> Layout<S::Reversed, D::Reversed> {
        Layout::new(self.shape.reversed(), self.stride.reversed())
    }

    /// Transposes the layout by reversing the order of dimensions.
    /// For an n-dimensional layout, this reverses the order of both shapes and strides. For 2D layouts,
    /// this swaps rows and columns, converting row-major to column-major and vice versa.
    #[inline]
    pub fn transpose(&self) -> Layout<S::Reversed, D::Reversed> {
        self.reverse()
    }

    /// Convert all elements in shape and stride to Scalar<DType>.
    #[inline]
    pub fn make_dynamic(&self) -> Layout<S::Dynamic, D::Dynamic> {
        Layout::new(self.shape.to_dynamic(), self.stride.to_dynamic())
    }

    /// Shape dimension `I`, keeping its static-ness.
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
