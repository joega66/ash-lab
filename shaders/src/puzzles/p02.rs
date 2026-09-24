use gpu::*;
use gpu_reflect::*;

#[push]
pub struct AddPush {
    pub output: RWDeviceAddress<f32>,
    pub a: DeviceAddress<f32>,
    pub b: DeviceAddress<f32>,
}
#[spec]
pub struct AddSpec {
    pub block_dim_x: u32,
}
function!(
    Add,
    push: AddPush,
    spec: AddSpec,
    "main",
    "p02.slang",
);

#[cfg(test)]
mod test {
    use super::*;
    use gpu::{DeviceContext, DeviceContextCreateInfo};

    /// https://puzzles.modular.com/puzzle_02/puzzle_02.html
    #[test]
    fn p02() {
        const SIZE: usize = 4;

        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo::default());

        let mut a_host = Vec::new();
        let mut b_host = Vec::new();
        let mut expected = Vec::new();
        for i in 0..SIZE {
            let val = i as f32;
            a_host.push(val);
            b_host.push(val);
            expected.push(val + val);
        }

        let out = ctx.enqueue_create_buffer("out", SIZE);
        ctx.enqueue_fill(&out, 0_f32);

        let a = ctx.enqueue_create_buffer("a", SIZE);
        ctx.enqueue_copy(a_host.as_slice(), &a);

        let b = ctx.enqueue_create_buffer("b", SIZE);
        ctx.enqueue_copy(b_host.as_slice(), &b);

        let add = ctx.compile_function::<Add>(
            &(),
            Some(AddSpec {
                block_dim_x: SIZE as u32,
            }),
        );

        enqueue_function!(
            ctx,
            func: &add,
            push: AddPush {
                output: out.as_ref().into(),
                a: a.as_ref().into(),
                b: b.as_ref().into(),
            },
            grid_dim: UInt3::new(a.len() as u32, 1, 1),
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
        println!("Puzzle 02 complete ✅")
    }
}
