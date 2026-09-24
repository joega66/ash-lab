use gpu::*;
use gpu_reflect::*;
use layout::{Const, DeviceTensor, RWDeviceTensor, RowMajorLayout};

const SIZE: usize = 2;
const BLOCKS_PER_GRID: u32 = 1;
const THREADS_PER_BLOCK: (u32, u32) = (3, 3);
type LayoutType = RowMajorLayout<(Const<SIZE>, Const<SIZE>)>;

#[push]
pub struct Add102dPush<T: DType> {
    pub output: RWDeviceTensor<T, LayoutType>,
    pub a: DeviceTensor<T, LayoutType>,
}
function!(
    Add102d<T> for [f32, u32],
    push: Add102dPush<T>,
    "main",
    "p04.slang",
);

#[cfg(test)]
mod test {
    use super::*;
    use gpu::{DeviceContext, DeviceContextCreateInfo, U32Castable};
    use layout::TileTensor;
    use std::assert_eq;
    use std::fmt::Debug;

    trait TestScalar: DType + DTypeOf<Add102d<Self>> + U32Castable + Debug + PartialEq {
        fn from_index(index: usize) -> Self;
        fn plus_ten(self) -> Self;
        fn zero() -> Self;
    }

    impl TestScalar for f32 {
        fn from_index(index: usize) -> Self {
            index as f32
        }
        fn plus_ten(self) -> Self {
            self + 10.0
        }
        fn zero() -> Self {
            0.0
        }
    }

    impl TestScalar for u32 {
        fn from_index(index: usize) -> Self {
            index as u32
        }
        fn plus_ten(self) -> Self {
            self + 10
        }
        fn zero() -> Self {
            0
        }
    }

    fn add_10_2d<T: TestScalar>() {
        let layout = LayoutType::default();

        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo::default());

        let out_buf = ctx.enqueue_create_buffer::<T>("out_buf", SIZE * SIZE);
        ctx.enqueue_fill(&out_buf, T::zero());
        let out_tensor = TileTensor::new(out_buf.clone(), layout);
        println!("out shape:{}x{}", out_tensor.dim(0), out_tensor.dim(1));

        let mut expected = Vec::with_capacity(SIZE * SIZE);
        let mut a_host = Vec::with_capacity(SIZE * SIZE);
        for i in 0..SIZE * SIZE {
            a_host.push(T::from_index(i));
            expected.push(T::from_index(i).plus_ten());
        }

        let a = ctx.enqueue_create_buffer("a", SIZE * SIZE);
        ctx.enqueue_copy(a_host.as_slice(), &a);

        let a_tensor = TileTensor::new(a, layout);

        enqueue_function!(
            ctx,
            Add102d<T>,
            push: Add102dPush {
                output: out_tensor.as_ref().into(),
                a: a_tensor.as_ref().into(),
            },
            grid_dim: UInt3::splat(BLOCKS_PER_GRID),
        );

        let out_host = ctx.create_host_buffer("out_host", SIZE * SIZE);
        ctx.enqueue_copy(&out_buf, &out_host);

        ctx.execute(None).expect("failed to execute");

        ctx.synchronize();

        let out_host = out_host.map_to_host(&ctx);

        println!("out: {:?}", out_host);
        println!("expected: {:?}", expected);
        for i in 0..out_host.len() {
            assert_eq!(out_host[i], expected[i]);
        }
    }

    #[test]
    fn p04() {
        add_10_2d::<f32>();
        println!("Puzzle 04 complete ✅");
    }

    /// The same kernel, against the `uint` instantiation the declaration also registered.
    #[test]
    fn p04_u32() {
        add_10_2d::<u32>();
    }
}
