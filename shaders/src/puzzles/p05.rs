use gpu::function;
use gpu_reflect::push;
use layout::{Const, DeviceTensor, RWDeviceTensor, RowMajorLayout};

const SIZE: usize = 2;
const BLOCKS_PER_GRID: u32 = 1;
const THREADS_PER_BLOCK: (u32, u32) = (3, 3);
type DType = f32;
type OutLayout = RowMajorLayout<(Const<SIZE>, Const<SIZE>)>;
type ALayout = RowMajorLayout<(Const<1>, Const<SIZE>)>;
type BLayout = RowMajorLayout<(Const<SIZE>, Const<1>)>;

#[push]
pub struct BroadcastAddPush {
    output: RWDeviceTensor<DType, OutLayout>,
    a: DeviceTensor<DType, ALayout>,
    b: DeviceTensor<DType, BLayout>,
}

function!(
    BroadcastAdd<const SIZE: u32, const X: u32, const Y: u32> for [(SIZE as u32, THREADS_PER_BLOCK.0 as u32, THREADS_PER_BLOCK.1 as u32)],
    push: BroadcastAddPush,
    name: "main",
    path: "p05.slang",
);

#[cfg(test)]
mod test {
    use super::*;
    use gpu::{DeviceContext, DeviceContextCreateInfo, UInt3, enqueue_function};
    use layout::TileTensor;

    #[test]
    fn p05() {
        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo::default());

        let mut out_buf = ctx.enqueue_create_buffer("out_buf", SIZE * SIZE);
        let out_tensor = TileTensor::new(&mut out_buf, OutLayout::default());
        println!("out shape:{}x{}", out_tensor.dim(0), out_tensor.dim(1));

        let mut expected_buf = vec![0_f32; SIZE * SIZE];
        let mut expected_tensor = TileTensor::new(&mut expected_buf, OutLayout::default());

        let mut a_host = Vec::new();
        let mut b_host = Vec::new();

        for i in 0..SIZE {
            a_host.push((i + 1) as f32);
            b_host.push((i * 10) as f32);
        }

        for i in 0..SIZE {
            for j in 0..SIZE {
                expected_tensor[[i, j]] = a_host[j] + b_host[i];
            }
        }

        let a = ctx.enqueue_create_buffer("a", a_host.len());
        ctx.enqueue_copy(&a_host, &a);
        let b = ctx.enqueue_create_buffer("b", b_host.len());
        ctx.enqueue_copy(&b_host, &b);

        let a_tensor = TileTensor::new(&a, ALayout::default());
        let b_tensor = TileTensor::new(&b, BLayout::default());

        enqueue_function!(
            ctx,
            BroadcastAdd<{SIZE as u32}, {THREADS_PER_BLOCK.0 as u32}, {THREADS_PER_BLOCK.1 as u32}>,
            push: BroadcastAddPush {
                output: out_tensor.read_write(),
                a: a_tensor.read_only(),
                b: b_tensor.read_only(),
            },
            grid_dim: UInt3::splat(BLOCKS_PER_GRID),
        );

        let out_buf_host = ctx.create_host_buffer("out_buf_host", out_buf.len());
        ctx.enqueue_copy(&out_buf, &out_buf_host);

        ctx.execute(None).expect("failed to execute");

        ctx.synchronize();

        let out_buf_host = out_buf_host.map_to_host(&ctx);
        println!("out:{:?}", out_buf_host);
        println!("expected:{:?}", expected_buf);
        for i in 0..SIZE {
            for j in 0..SIZE {
                assert_eq!(out_buf_host[i * SIZE + j], expected_buf[i * SIZE + j]);
            }
        }
        println!("Puzzle 05 complete ✅");
    }
}
