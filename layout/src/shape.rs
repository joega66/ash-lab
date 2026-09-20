use core::fmt;

use crate::dim::{Dim, all_static, static_product};

/// A rank-N tuple of dimensions, e.g. `(Const<4>, usize)` for a 4 x N shape.
pub trait Shape: Copy {
    /// Number of dimensions.
    const RANK: usize;

    /// Per-dimension compile-time extents; `None` where a dimension is only known at run time.
    const STATIC: &'static [Option<usize>];

    /// Product of all extents, or `None` if any dimension is dynamic.
    const STATIC_PRODUCT: Option<usize> = static_product(Self::STATIC);

    /// Whether every dimension is known at compile time.
    const ALL_KNOWN: bool = all_static(Self::STATIC);

    /// `[usize; RANK]`.
    type Coords: Copy + AsRef<[usize]> + AsMut<[usize]> + fmt::Debug + PartialEq + Eq;

    /// This shape with the dimension order reversed.
    type Reversed: Shape<Coords = Self::Coords>;

    /// This shape with every dimension turned into a runtime `usize`.
    type Dynamic: Shape<Coords = Self::Coords>;

    /// The extent of every dimension.
    fn dims(self) -> Self::Coords;

    /// Reverses the dimension order.
    fn reversed(self) -> Self::Reversed;

    /// Materializes every dimension as a runtime value.
    fn to_dynamic(self) -> Self::Dynamic;

    /// An all-zero coordinate.
    fn zero_coords() -> Self::Coords;

    /// The extent of dimension `i`.
    #[inline]
    fn get(self, i: usize) -> usize {
        self.dims().as_ref()[i]
    }

    /// The compile-time extent of dimension `i`, or `None` if it is a runtime value.
    #[inline]
    fn static_at(i: usize) -> Option<usize> {
        Self::STATIC[i]
    }

    /// Product of all extents.
    #[inline]
    fn product(self) -> usize {
        match Self::STATIC_PRODUCT {
            Some(n) => n,
            None => self.dims().as_ref().iter().product(),
        }
    }

    /// Writes the tuple as `(d0, d1, ...)`.
    fn write_to(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        for (i, d) in self.dims().as_ref().iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str(")")
    }
}

/// Typed access to one dimension of a [`Shape`], preserving whether it is static.
pub trait DimAt<const I: usize>: Shape {
    /// The type of dimension `I`.
    type Dim: Dim;

    /// Dimension `I`.
    fn dim_at(self) -> Self::Dim;
}

/// Expands to `usize` for any dimension type.
macro_rules! as_dynamic {
    ($d:ident) => {
        usize
    };
}

macro_rules! impl_shape {
    ($rank:literal; $($d:ident),+; $($idx:tt),+; $($rd:ident),+; $($ridx:tt),+) => {
        impl<$($d: Dim),+> Shape for ($($d,)+) {
            const RANK: usize = $rank;
            const STATIC: &'static [Option<usize>] = &[$($d::STATIC),+];

            type Coords = [usize; $rank];
            type Reversed = ($($rd,)+);
            type Dynamic = ($(as_dynamic!($d),)+);

            #[inline]
            fn dims(self) -> Self::Coords {
                [$(self.$idx.value()),+]
            }

            #[inline]
            fn reversed(self) -> Self::Reversed {
                ($(self.$ridx,)+)
            }

            #[inline]
            fn to_dynamic(self) -> Self::Dynamic {
                ($(self.$idx.value(),)+)
            }

            #[inline]
            fn zero_coords() -> Self::Coords {
                [0; $rank]
            }
        }
    };
}

/// The dim list is passed as a single token tree so that it can be re-expanded per index.
macro_rules! impl_dim_at {
    ($dims:tt; $($idx:tt: $sel:ident),+) => {
        $(impl_dim_at!(@one $dims, $idx, $sel);)+
    };
    (@one ($($d:ident),+), $idx:tt, $sel:ident) => {
        impl<$($d: Dim),+> DimAt<$idx> for ($($d,)+) {
            type Dim = $sel;

            #[inline]
            fn dim_at(self) -> Self::Dim {
                self.$idx
            }
        }
    };
}

impl_shape!(1; A0; 0; A0; 0);
impl_shape!(2; A0, A1; 0, 1; A1, A0; 1, 0);
impl_shape!(3; A0, A1, A2; 0, 1, 2; A2, A1, A0; 2, 1, 0);
impl_shape!(4; A0, A1, A2, A3; 0, 1, 2, 3; A3, A2, A1, A0; 3, 2, 1, 0);
impl_shape!(5; A0, A1, A2, A3, A4; 0, 1, 2, 3, 4; A4, A3, A2, A1, A0; 4, 3, 2, 1, 0);

impl_dim_at!((A0); 0: A0);
impl_dim_at!((A0, A1); 0: A0, 1: A1);
impl_dim_at!((A0, A1, A2); 0: A0, 1: A1, 2: A2);
impl_dim_at!((A0, A1, A2, A3); 0: A0, 1: A1, 2: A2, 3: A3);
impl_dim_at!((A0, A1, A2, A3, A4); 0: A0, 1: A1, 2: A2, 3: A3, 4: A4);
