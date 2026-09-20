use gpu::*;
use gpu_reflect::*;
use layout::{Const, DeviceTensor, RWDeviceTensor, RowMajorLayout};

const SIZE: usize = 2;
const BLOCKS_PER_GRID: u32 = 1;
const THREADS_PER_BLOCK: (u32, u32) = (3, 3);
type DType = f32;
type LayoutType = RowMajorLayout<(Const<SIZE>, Const<SIZE>)>;

#[push]
pub struct Add102dPush {
    pub output: RWDeviceTensor<DType, LayoutType>,
    pub a: DeviceTensor<DType, LayoutType>,
}
function!(
    Add102d,
    push: Add102dPush,
    "main",
    "p04.slang",
);

#[cfg(test)]
mod test {
    use super::*;
    use gpu::{DeviceContext, DeviceContextCreateInfo};
    use layout::TileTensor;
    use std::assert_eq;

    #[test]
    fn p04() {
        let layout = LayoutType::default();

        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo::default());

        let out_buf = ctx.enqueue_create_buffer("out_buf", SIZE * SIZE);
        ctx.enqueue_fill(&out_buf, 0_f32);
        let out_tensor = TileTensor::new(out_buf.clone(), layout);
        println!("out shape:{}x{}", out_tensor.dim(0), out_tensor.dim(1));

        let mut expected = vec![0_f32; SIZE * SIZE];

        let mut a_host = vec![0_f32; SIZE * SIZE];
        for i in 0..SIZE * SIZE {
            a_host[i] = i as f32;
            expected[i] = a_host[i] + 10.0;
        }

        let a = ctx.enqueue_create_buffer("a", SIZE * SIZE);
        ctx.enqueue_copy(a_host.as_slice(), &a);

        let a_tensor = TileTensor::new(a, layout);

        enqueue_function!(
            ctx,
            Add102d,
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
        println!("Puzzle 04 complete ✅");
    }
}
