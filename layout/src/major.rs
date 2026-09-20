//! Row-major and column-major stride generation.

use crate::dim::{Const, Dim, Prod};
use crate::layout::Layout;
use crate::shape::Shape;

/// Shapes that can be given row-major (C-order) strides: the rightmost dimension has stride 1 and
/// each preceding dimension has stride equal to the product of all following dimensions.
pub trait RowMajor: Shape {
    /// The row-major stride tuple for this shape.
    type Stride: Shape<Coords = Self::Coords>;

    /// Computes the row-major strides.
    fn row_major_stride(self) -> Self::Stride;
}

/// Shapes that can be given column-major (glm-order) strides: the leftmost dimension has
/// stride 1 and each following dimension has stride equal to the product of all preceding ones.
pub trait ColMajor: Shape {
    /// The column-major stride tuple for this shape.
    type Stride: Shape<Coords = Self::Coords>;

    /// Computes the column-major strides.
    fn col_major_stride(self) -> Self::Stride;
}

/// A row-major layout over `S`.
pub type RowMajorLayout<S> = Layout<S, <S as RowMajor>::Stride>;

/// A column-major layout over `S`.
pub type ColMajorLayout<S> = Layout<S, <S as ColMajor>::Stride>;

/// Creates a row-major layout from a shape Coord.
/// Row-major means the rightmost dimension has stride 1, and each preceding dimension has
/// stride equal to the product of all following dimensions.
///
/// For shape(M, N, K):
///     row_major strides : (N*K, K, 1)
///     col_major strides : (1, M, M*N)
#[inline]
pub fn row_major<S: RowMajor>(shape: S) -> RowMajorLayout<S> {
    Layout::new(shape, shape.row_major_stride())
}

/// Create a column-major layout from variadic arguments.
/// Column-major means the first dimension has stride 1, and each subsequent dimension has
/// stride equal to the product of all previous dimensions.
#[inline]
pub fn col_major<S: ColMajor>(shape: S) -> ColMajorLayout<S> {
    Layout::new(shape, shape.col_major_stride())
}

/// The type of the product of a (possibly empty) list of dimensions.
macro_rules! prod_ty {
    () => { Const<1> };
    ($only:ident) => { $only };
    ($head:ident $($tail:ident)+) => { Prod<$head, prod_ty!($($tail)+)> };
}

/// The value of the product of a (possibly empty) list of dimensions.
macro_rules! prod_val {
    () => { Const };
    ($only:ident) => { $only };
    ($head:ident $($tail:ident)+) => { Prod($head, prod_val!($($tail)+)) };
}

/// Implements one of the majorness traits, given the product feeding each stride entry.
macro_rules! impl_major {
    ($trait:ident::$method:ident; $($d:ident),+; $([$($factor:ident)*]),+) => {
        impl<$($d: Dim),+> $trait for ($($d,)+) {
            type Stride = ($(prod_ty!($($factor)*),)+);

            #[inline]
            #[allow(non_snake_case, unused_variables)]
            fn $method(self) -> Self::Stride {
                let ($($d,)+) = self;
                ($(prod_val!($($factor)*),)+)
            }
        }
    };
}

// Row-major: stride i is the product of the dimensions after i.
impl_major!(RowMajor::row_major_stride; A0; []);
impl_major!(RowMajor::row_major_stride; A0, A1; [A1], []);
impl_major!(RowMajor::row_major_stride; A0, A1, A2; [A1 A2], [A2], []);
impl_major!(RowMajor::row_major_stride; A0, A1, A2, A3; [A1 A2 A3], [A2 A3], [A3], []);
impl_major!(
    RowMajor::row_major_stride;
    A0, A1, A2, A3, A4;
    [A1 A2 A3 A4], [A2 A3 A4], [A3 A4], [A4], []
);

// Column-major: stride i is the product of the dimensions before i.
impl_major!(ColMajor::col_major_stride; A0; []);
impl_major!(ColMajor::col_major_stride; A0, A1; [], [A0]);
impl_major!(ColMajor::col_major_stride; A0, A1, A2; [], [A0], [A0 A1]);
impl_major!(ColMajor::col_major_stride; A0, A1, A2, A3; [], [A0], [A0 A1], [A0 A1 A2]);
impl_major!(
    ColMajor::col_major_stride;
    A0, A1, A2, A3, A4;
    [], [A0], [A0 A1], [A0 A1 A2], [A0 A1 A2 A3]
);
