use gpu::function;
use gpu_reflect::push;
use layout::{Const, DeviceTensor, RWDeviceTensor, RowMajorLayout};

const SIZE: usize = 2;
const BLOCKS_PER_GRID: u32 = 1;
const THREADS_PER_BLOCK: (u32, u32) = (3, 3);
type LayoutType = RowMajorLayout<(Const<SIZE>, Const<SIZE>)>;

#[push]
pub struct Add102dPush {
    pub output: RWDeviceTensor<f32, LayoutType>,
    pub a: DeviceTensor<f32, LayoutType>,
}
function!(
    Add102d<const SIZE: u32, const X: u32, const Y: u32> for [(SIZE as u32, THREADS_PER_BLOCK.0 as u32, THREADS_PER_BLOCK.1 as u32)],
    push: Add102dPush,
    name: "main",
    path: "p04.slang",
);

#[cfg(test)]
mod test {
    use super::*;
    use gpu::{DeviceContext, DeviceContextCreateInfo, UInt3, enqueue_function};
    use layout::TileTensor;
    use std::assert_eq;

    #[test]
    fn p04() {
        let layout = LayoutType::default();

        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo::default());

        let mut out_buf = ctx.enqueue_create_buffer("out_buf", SIZE * SIZE);
        ctx.enqueue_fill(&out_buf, 0.0);
        let out_tensor = TileTensor::new(&mut out_buf, layout);
        println!("out shape:{}x{}", out_tensor.dim(0), out_tensor.dim(1));

        let mut expected = Vec::with_capacity(SIZE * SIZE);
        let mut a_host = Vec::with_capacity(SIZE * SIZE);
        for i in 0..SIZE * SIZE {
            a_host.push(i as f32);
            expected.push((i as f32) + 10.0);
        }

        let a = ctx.enqueue_create_buffer("a", SIZE * SIZE);
        ctx.enqueue_copy(&a_host, &a);

        let a_tensor = TileTensor::new(&a, layout);

        enqueue_function!(
            ctx,
            Add102d<{SIZE as u32}, {THREADS_PER_BLOCK.0 as u32}, {THREADS_PER_BLOCK.1 as u32}>,
            push: Add102dPush {
                output: out_tensor.read_write(),
                a: a_tensor.read_only(),
            },
            grid_dim: UInt3::splat(BLOCKS_PER_GRID as u32),
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
