//! Layouts: shape/stride pairs that map logical coordinates to linear memory indices, in the style
//! of Mojo's [`layout.tile_layout`](https://max.modular.com/api/mojo/layout/tile_layout/Layout/)
//! (and, behind it, CuTe).
//!
//! A [`Layout`] is a tuple of shape dimensions plus a tuple of stride dimensions. The rank is the
//! tuple arity, so it is always known at compile time, while each dimension is independently
//! either compile-time ([`Const<N>`], zero-sized) or run-time (`usize`):
//!
//! ```
//! use layout::{Const, row_major};
//!
//! // Fully static: the layout is zero-sized and every query folds to a constant.
//! let tile = row_major((Const::<4>, Const::<8>));
//! assert_eq!(size_of_val(&tile), 0);
//! assert_eq!(tile.size(), 32);
//! assert_eq!(tile.crd2idx([1, 3]), 11);
//!
//! // Mixed: rows known at compile time, columns only at run time. Only the terms that depend on
//! // the runtime extent stay dynamic — here the column count and the row stride it implies.
//! let cols = 8;
//! let rows = row_major((Const::<4>, cols));
//! assert_eq!(size_of_val(&rows), 2 * size_of::<usize>());
//! assert!(rows.dims_eq(&tile));
//! ```
//!
//! Static-ness propagates through everything derived from a layout — strides built by
//! [`row_major`] / [`col_major`], [`Layout::STATIC_PRODUCT`], [`Layout::STATIC_COSIZE`],
//! [`Layout::shape_dim`] — so a dimension known at compile time stays known.
//!
//! # Relationship to Mojo's API
//!
//! | Mojo | here |
//! |---|---|
//! | `Layout[shape_types, stride_types]` | [`Layout<S, D>`] |
//! | `Layout(shape, stride)` | [`Layout::new`] |
//! | `row_major(...)` / `col_major(...)` | [`row_major`] / [`col_major`] |
//! | `RowMajorLayout` / `ColMajorLayout` | [`RowMajorLayout`] / [`ColMajorLayout`] |
//! | `rank` / `flat_rank` | [`Layout::RANK`] / [`Layout::FLAT_RANK`] |
//! | `shape_known` / `stride_known` / `all_dims_known` | [`Layout::SHAPE_KNOWN`] / [`Layout::STRIDE_KNOWN`] / [`Layout::ALL_DIMS_KNOWN`] |
//! | `static_product` / `static_cosize` | [`Layout::STATIC_PRODUCT`] / [`Layout::STATIC_COSIZE`] |
//! | `static_shape[i]` / `static_stride[i]` | [`Layout::static_shape`] / [`Layout::static_stride`] |
//! | `shape[i]()` / `stride[i]()` | [`Layout::shape_dim`] / [`Layout::stride_dim`] |
//! | `shape_coord()` / `stride_coord()` | [`Layout::shape_coord`] / [`Layout::stride_coord`] |
//! | `product()` / `size()` / `cosize()` | [`Layout::product`] / [`Layout::size`] / [`Layout::cosize`] |
//! | `__call__(coord)` | [`Layout::call`] / [`Layout::crd2idx`] |
//! | `idx2crd(idx)` | [`Layout::idx2crd`] |
//! | `reverse()` / `transpose()` | [`Layout::reverse`] / [`Layout::transpose`] |
//! | `make_dynamic()` | [`Layout::make_dynamic`] |
//! | `write_to()` | `Display`, as `(shape:stride)` |
//!
//! Shapes here are flat: nested (hierarchical) shapes and the layout algebra built on them
//! (`coalesce`, `blocked_product`, `zipped_divide`) are not implemented, which is why `flat_rank`
//! always equals `rank`.

mod dim;
mod layout;
mod major;
mod shape;

pub use dim::{Const, Dim, Prod, all_static, static_cosize, static_product};
pub use layout::Layout;
pub use major::{ColMajor, ColMajorLayout, RowMajor, RowMajorLayout, col_major, row_major};
pub use shape::{DimAt, Shape};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_layouts_are_zero_sized_and_fold() {
        const L: Layout<(Const<4>, Const<8>), (Const<8>, Const<1>)> =
            Layout::new((Const, Const), (Const, Const));

        assert_eq!(size_of_val(&L), 0);
        const { assert!(<Layout<(Const<4>, Const<8>), (Const<8>, Const<1>)>>::ALL_DIMS_KNOWN) };
        assert_eq!(
            <Layout<(Const<4>, Const<8>), (Const<8>, Const<1>)>>::STATIC_PRODUCT,
            Some(32)
        );
        assert_eq!(
            <Layout<(Const<4>, Const<8>), (Const<8>, Const<1>)>>::STATIC_COSIZE,
            Some(32)
        );
        assert_eq!(L.crd2idx([2, 5]), 21);
    }

    #[test]
    fn row_major_strides() {
        let l = row_major((Const::<2>, Const::<3>, Const::<4>));
        assert_eq!(l.stride.dims(), [12, 4, 1]);
        assert_eq!(l.size(), 24);
        assert_eq!(l.cosize(), 24);
        assert_eq!(l.crd2idx([1, 2, 3]), 12 + 8 + 3);
        assert_eq!(format!("{l}"), "((2, 3, 4):(12, 4, 1))");
    }

    #[test]
    fn col_major_strides() {
        let l = col_major((Const::<2>, Const::<3>, Const::<4>));
        assert_eq!(l.stride.dims(), [1, 2, 6]);
        assert_eq!(l.crd2idx([1, 2, 3]), 1 + 4 + 18);
        assert_eq!(l.size(), 24);
        assert_eq!(l.cosize(), 24);
    }

    #[test]
    fn mixed_static_and_dynamic_dims() {
        let n = 7;
        let l = row_major((Const::<4>, n));

        // Only runtime dims are stored: the dynamic column count, and the row stride it implies.
        assert_eq!(size_of_val(&l), 2 * size_of::<usize>());
        assert_eq!(size_of_val(&l.shape), size_of::<usize>());
        assert_eq!(size_of_val(&l.stride), size_of::<usize>());

        const { assert!(!<Layout<(Const<4>, usize), (usize, Const<1>)>>::ALL_DIMS_KNOWN) };
        assert_eq!(
            <Layout<(Const<4>, usize), (usize, Const<1>)>>::static_shape(0),
            Some(4)
        );
        assert_eq!(
            <Layout<(Const<4>, usize), (usize, Const<1>)>>::static_shape(1),
            None
        );
        assert_eq!(
            <Layout<(Const<4>, usize), (usize, Const<1>)>>::static_stride(1),
            Some(1)
        );

        assert_eq!(l.size(), 28);
        assert_eq!(l.cosize(), 28);
        assert_eq!(l.crd2idx([3, 6]), 27);
    }

    #[test]
    fn static_product_of_a_dynamic_stride_is_unknown() {
        // A static shape dimension multiplied by a dynamic one stays dynamic.
        assert_eq!(<(Const<4>, usize) as Shape>::STATIC_PRODUCT, None);
        assert_eq!(<(Const<4>, Const<8>) as Shape>::STATIC_PRODUCT, Some(32));
        assert_eq!(<Prod<Const<4>, Const<8>> as Dim>::STATIC, Some(32));
        assert_eq!(<Prod<Const<4>, usize> as Dim>::STATIC, None);
    }

    #[test]
    fn typed_dimension_access_preserves_staticness() {
        let l = row_major((Const::<4>, 7usize));

        let rows: Const<4> = l.shape_dim::<0>();
        let cols: usize = l.shape_dim::<1>();
        let unit: Const<1> = l.stride_dim::<1>();

        assert_eq!(rows.value(), 4);
        assert_eq!(cols, 7);
        assert_eq!(unit.value(), 1);
        assert_eq!(size_of_val(&rows), 0);
    }

    #[test]
    fn idx2crd_inverts_crd2idx() {
        let l = row_major((Const::<3>, Const::<5>));
        for i in 0..3 {
            for j in 0..5 {
                let idx = l.crd2idx([i, j]);
                assert_eq!(l.idx2crd(idx), [i, j]);
            }
        }
    }

    #[test]
    fn linear2idx_walks_the_whole_domain() {
        let l = row_major((Const::<3>, Const::<4>));
        let visited: Vec<_> = l.iter().collect();

        assert_eq!(visited.len(), l.size());
        let mut sorted = visited.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());
        // Leftmost coordinate varies fastest.
        assert_eq!(&visited[..4], &[0, 4, 8, 1]);
    }

    #[test]
    fn transpose_reverses_dimensions() {
        let l = row_major((Const::<2>, 5usize));
        let t = l.transpose();

        assert_eq!(t.shape.dims(), [5, 2]);
        assert_eq!(t.stride.dims(), [1, 5]);
        assert_eq!(t.crd2idx([3, 1]), l.crd2idx([1, 3]));

        // Reversing a row-major layout gives the column-major layout of the reversed shape.
        assert!(t.dims_eq(&col_major((5usize, Const::<2>))));
    }

    #[test]
    fn make_dynamic_erases_static_extents() {
        let l = row_major((Const::<4>, Const::<8>));
        let d = l.make_dynamic();

        const { assert!(<Layout<(Const<4>, Const<8>), (Const<8>, Const<1>)>>::ALL_DIMS_KNOWN) };
        const { assert!(!<Layout<(usize, usize), (usize, usize)>>::ALL_DIMS_KNOWN) };
        assert!(d.dims_eq(&l));
        assert_eq!(d.shape, (4, 8));
        assert_eq!(d.stride, (8, 1));
    }

    #[test]
    fn non_contiguous_layouts() {
        // A 4 x 4 window into a 4 x 16 row-major buffer.
        let l = Layout::new((Const::<4>, Const::<4>), (Const::<16>, Const::<1>));

        assert_eq!(l.size(), 16);
        assert_eq!(l.cosize(), 3 * 16 + 3 + 1);
        assert_eq!(l.crd2idx([3, 3]), 51);
    }

    #[test]
    fn empty_domain_has_zero_cosize() {
        let l = row_major((Const::<0>, Const::<4>));
        assert_eq!(l.size(), 0);
        assert_eq!(l.cosize(), 0);

        let l = row_major((0usize, 4usize));
        assert_eq!(l.size(), 0);
        assert_eq!(l.cosize(), 0);
    }

    #[test]
    fn rank_one_and_high_rank() {
        let l = row_major((Const::<7>,));
        assert_eq!(l.stride.dims(), [1]);
        assert_eq!(l.crd2idx([3]), 3);
        assert_eq!(format!("{l}"), "((7):(1))");

        let l = row_major((
            Const::<2>, Const::<3>, Const::<4>, Const::<5>, Const::<6>, Const::<7>, 8usize,
            Const::<9>,
        ));
        assert_eq!(l.rank(), 8);
        assert_eq!(
            l.stride.dims(),
            [
                9 * 8 * 7 * 6 * 5 * 4 * 3,
                9 * 8 * 7 * 6 * 5 * 4,
                9 * 8 * 7 * 6 * 5,
                9 * 8 * 7 * 6,
                9 * 8 * 7,
                9 * 8,
                9,
                1
            ]
        );
        assert_eq!(l.size(), 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9);
    }

    #[test]
    fn defaults_zero_runtime_dims() {
        let l = <Layout<(Const<4>, usize), (usize, Const<1>)>>::default();
        assert_eq!(l.shape.dims(), [4, 0]);
        assert_eq!(l.stride.dims(), [0, 1]);
    }
}
