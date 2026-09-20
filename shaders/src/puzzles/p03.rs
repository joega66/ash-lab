use gpu::*;
use gpu_reflect::*;

#[push]
pub struct Add10GuardPush {
    pub output: RWDeviceAddress<f32>,
    pub a: DeviceAddress<f32>,
    pub len: u64,
}
#[spec]
pub struct Add10GuardSpec {
    pub block_dim_x: u32,
}
function!(
    Add10Guard,
    push: Add10GuardPush,
    spec: Add10GuardSpec,
    "main",
    "p03.slang",
);

#[cfg(test)]
mod test {
    use super::*;
    use gpu::{DeviceContext, DeviceContextCreateInfo};

    /// https://puzzles.modular.com/puzzle_03/puzzle_03.html
    #[test]
    fn p03() {
        const SIZE: usize = 4;
        const BLOCKS_PER_GRID: usize = 1;
        const THREADS_PER_BLOCK: usize = 8;

        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo::default());

        let a_host: Vec<_> = (0..SIZE).map(|i| i as f32).collect();
        let expected: Vec<_> = a_host.iter().map(|x| x + 10_f32).collect();

        let out = ctx.enqueue_create_buffer("out", SIZE);
        ctx.enqueue_fill(&out, 0_f32);

        let a = ctx.enqueue_create_buffer("a", SIZE);
        ctx.enqueue_copy(a_host.as_slice(), &a);

        let add_10_guard = ctx.compile_function::<Add10Guard>(
            &(),
            Some(Add10GuardSpec {
                block_dim_x: THREADS_PER_BLOCK as u32,
            }),
        );

        enqueue_function!(
            ctx,
            func: &add_10_guard,
            push: Add10GuardPush {
                output: out.as_ref().into(),
                a: a.as_ref().into(),
                len: a.len() as u64,
            },
            grid_dim: UInt3::new(BLOCKS_PER_GRID as u32, 1, 1),
        );

        let out_host = ctx.create_host_buffer("out_host", SIZE);
        ctx.enqueue_copy(&out, &out_host);

        ctx.execute(None).expect("failed to execute");

        ctx.synchronize();

        let out_host = out_host.map_to_host(&ctx);

        for i in 0..SIZE {
            assert_eq!(expected[i], out_host[i]);
        }

        println!("out: {:?}", out_host);
        println!("expected: {:?}", expected);
        println!("Puzzle 03 complete ✅")
    }
}
