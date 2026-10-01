use gpu::{DeviceAddress, RWDeviceAddress, function};
use gpu_reflect::push;

const SIZE: usize = 9;
const BLOCKS_PER_GRID: usize = 3;
const THREADS_PER_BLOCK: usize = 4;
type DType = f32;

#[push]
pub struct Add10BlocksPush {
    pub output: RWDeviceAddress<DType>,
    pub a: DeviceAddress<DType>,
    pub size: u64,
}

function!(
    Add10Blocks<const X: u32> for [THREADS_PER_BLOCK as u32],
    push: Add10BlocksPush,
    name: "main",
    path: "p06.slang"
);

#[cfg(test)]
mod tests {
    use super::*;
    use gpu::{DeviceContext, DeviceContextCreateInfo, UInt3, enqueue_function};

    #[test]
    fn p06() {
        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo::default());

        let a = ctx.enqueue_create_buffer("a", SIZE);
        ctx.enqueue_copy((0..SIZE).map(|x| x as f32), &a);

        let out = ctx.enqueue_create_buffer("out", SIZE);

        enqueue_function!(
            ctx,
            Add10Blocks<{THREADS_PER_BLOCK as u32}>,
            push: Add10BlocksPush {
                output: out.read_write(),
                a: a.read_only(),
                size: out.len() as u64,
            },
            grid_dim: UInt3::new(BLOCKS_PER_GRID as u32, 1, 1),
        );

        let out_host = ctx.enqueue_create_host_buffer("out_host", SIZE);
        ctx.enqueue_copy(&out, &out_host);

        ctx.execute(None).expect("failed to execute");

        ctx.synchronize();

        let expected: Vec<_> = (0..SIZE).map(|x| (x as f32) + 10.0).collect();

        let out_host = out_host.map_to_host(&ctx);
        println!("out:{:?}", out_host);
        println!("expected:{:?}", expected);
        for i in 0..SIZE {
            assert_eq!(out_host[i], expected[i])
        }
        println!("Puzzle 06 complete ✅");
    }
}
