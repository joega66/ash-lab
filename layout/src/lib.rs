mod dim;
mod layout;
mod layout_gpu;
mod major;
mod shape;
mod tile_tensor;
mod tile_tensor_gpu;

pub use dim::{Const, ConstDim, Dim, Prod, all_static, static_cosize, static_product};
pub use layout::Layout;
pub use layout_gpu::{DimType, ShapeType};
pub use major::{ColMajor, ColMajorLayout, RowMajor, RowMajorLayout, col_major, row_major};
pub use shape::{DimAt, Shape};
pub use tile_tensor::*;
pub use tile_tensor_gpu::{DeviceTensor, RWDeviceTensor};

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
        // The static dimensions are readable as compile-time constants. Asking for the dynamic
        // ones -- `static_shape::<1>()` or `static_stride::<0>()` here -- is a compile error.
        const { assert!(<Layout<(Const<4>, usize), (usize, Const<1>)>>::static_shape::<0>() == 4) };
        const { assert!(<Layout<(Const<4>, usize), (usize, Const<1>)>>::static_stride::<1>() == 1) };

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
    }

    #[test]
    fn tensor_dim_reports_shape_extents() {
        let t = TileTensor::new(vec![0.0f32; 28], row_major((Const::<4>, 7usize)));

        // Runtime index: always a plain number.
        assert_eq!(t.dim(0), 4);
        assert_eq!(t.dim(1), 7);

        // Compile-time index: a static extent stays static and costs no storage.
        let rows: Const<4> = t.dim_at::<0>();
        let cols: usize = t.dim_at::<1>();
        assert_eq!(rows.value(), 4);
        assert_eq!(cols, 7);
        assert_eq!(size_of_val(&rows), 0);
    }

    #[test]
    fn tensor_indexes_by_coordinate() {
        let layout = row_major((Const::<3>, Const::<4>));
        let mut t = TileTensor::new(vec![0.0f32; 12], layout);

        t[[1, 2]] = 7.0;
        t[[2, 3]] = 9.0;

        assert_eq!(t[[1, 2]], 7.0);
        assert_eq!(t[[2, 3]], 9.0);
        // Row-major: (1, 2) is element 1 * 4 + 2.
        assert_eq!(t.storage()[6], 7.0);
        assert_eq!(t.storage()[11], 9.0);
    }

    #[test]
    fn tensor_indexing_follows_a_non_contiguous_layout() {
        // A 2 x 3 window into a 2 x 8 row-major buffer.
        let t = TileTensor::new(
            (0..16).map(|i| i as f32).collect::<Vec<_>>(),
            Layout::new((Const::<2>, Const::<3>), (Const::<8>, Const::<1>)),
        );

        assert_eq!(t[[0, 0]], 0.0);
        assert_eq!(t[[0, 2]], 2.0);
        assert_eq!(t[[1, 0]], 8.0);
        assert_eq!(t[[1, 2]], 10.0);
    }

    #[test]
    #[should_panic]
    fn tensor_dim_rejects_an_out_of_range_index() {
        let t = TileTensor::new(vec![0.0f32; 12], row_major((Const::<3>, Const::<4>)));
        let _ = t.dim(2);
    }

    #[test]
    fn defaults_zero_runtime_dims() {
        let l = <Layout<(Const<4>, usize), (usize, Const<1>)>>::default();
        assert_eq!(l.shape.dims(), [4, 0]);
        assert_eq!(l.stride.dims(), [0, 1]);
    }
}
