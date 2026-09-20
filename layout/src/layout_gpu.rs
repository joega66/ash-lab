use core::mem::{offset_of, size_of};

use gpu::{FieldLayout, ScalarKind, ShaderType, TypeLayout, bytemuck};

use crate::dim::{Const, Dim, Prod};
use crate::layout::Layout;
use crate::shape::Shape;

/// The [`TypeLayout`] of [`Dim`].
pub unsafe trait DimType: Dim {
    /// This dimension as slangc would describe its Slang counterpart.
    fn type_layout() -> TypeLayout;
}

/// A compile-time dimension occupies no bytes on either side: `Const<N>` is a zero-sized struct in
/// Rust and a fieldless one in Slang.
unsafe impl<const N: usize> DimType for Const<N> {
    fn type_layout() -> TypeLayout {
        TypeLayout::Struct {
            name: "Const",
            size: 0,
            fields: Vec::new(),
        }
    }
}

/// A runtime dimension is a `usize` here and a `uint64_t` there.
unsafe impl DimType for usize {
    fn type_layout() -> TypeLayout {
        TypeLayout::Scalar(ScalarKind::UInt64)
    }
}

unsafe impl<A: DimType, B: DimType> DimType for Prod<A, B> {
    fn type_layout() -> TypeLayout {
        TypeLayout::Struct {
            name: "Prod",
            size: size_of::<Self>() as u32,
            fields: vec![
                FieldLayout {
                    name: "a",
                    offset: offset_of!(Self, 0) as u32,
                    ty: A::type_layout(),
                },
                FieldLayout {
                    name: "b",
                    offset: offset_of!(Self, 1) as u32,
                    ty: B::type_layout(),
                },
            ],
        }
    }
}

/// The [`TypeLayout`] of [`Shape`].
pub unsafe trait ShapeType: Shape {
    /// This shape as slangc would describe the matching `ShapeN` struct.
    fn type_layout() -> TypeLayout;
}

/// Shapes are tuples here and `ShapeN` structs there, so the tuple's positional fields are
/// described under the names the Slang struct gives them.
macro_rules! impl_shape_type {
    ($name:literal; $($d:ident),+; $($idx:tt),+; $($field:literal),+) => {
        unsafe impl<$($d: DimType),+> ShapeType for ($($d,)+) {
            fn type_layout() -> TypeLayout {
                TypeLayout::Struct {
                    name: $name,
                    size: size_of::<Self>() as u32,
                    fields: vec![
                        $(FieldLayout {
                            name: $field,
                            offset: offset_of!(Self, $idx) as u32,
                            ty: $d::type_layout(),
                        }),+
                    ],
                }
            }
        }
    };
}

impl_shape_type!("Shape1"; A0; 0; "d0");
impl_shape_type!("Shape2"; A0, A1; 0, 1; "d0", "d1");
impl_shape_type!("Shape3"; A0, A1, A2; 0, 1, 2; "d0", "d1", "d2");
impl_shape_type!("Shape4"; A0, A1, A2, A3; 0, 1, 2, 3; "d0", "d1", "d2", "d3");
impl_shape_type!("Shape5"; A0, A1, A2, A3, A4; 0, 1, 2, 3, 4; "d0", "d1", "d2", "d3", "d4");

impl<S: ShapeType, D: ShapeType<Coords = S::Coords>> ShaderType for Layout<S, D> {
    fn type_layout() -> TypeLayout {
        TypeLayout::Struct {
            name: "Layout",
            size: size_of::<Self>() as u32,
            fields: vec![
                FieldLayout {
                    name: "shape",
                    offset: offset_of!(Self, shape) as u32,
                    ty: S::type_layout(),
                },
                FieldLayout {
                    name: "stride",
                    offset: offset_of!(Self, stride) as u32,
                    ty: D::type_layout(),
                },
            ],
        }
    }
}

unsafe impl<S: ShapeType, D: ShapeType<Coords = S::Coords>> bytemuck::Zeroable for Layout<S, D> {}
unsafe impl<S, D> bytemuck::Pod for Layout<S, D>
where
    S: ShapeType + 'static,
    D: ShapeType<Coords = S::Coords> + 'static,
{
}
