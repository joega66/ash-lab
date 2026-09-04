use bytemuck::{Pod, Zeroable};
use rhi::*;
use rhi_reflect::*;

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, ShaderType)]
pub struct Add10PushConstant {
    pub output: RWDeviceAddress<f32>,
    pub a: DeviceAddress<f32>,
    pub len: u32,
    pub _padding: u32,
}
kernel!(Add10, (), Add10PushConstant, "main", "add_10.slang");

#[cfg(test)]
mod test {
    use super::*;
    use rhi::{DeviceContext, DeviceContextCreateInfo};

    /// https://puzzles.modular.com/puzzle_01/raw.html
    #[test]
    fn add_10() {
        const SIZE: usize = 4;

        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo::default());

        let input: Vec<_> = (0..SIZE).map(|i| i as f32).collect();

        let output = ctx.enqueue_create_buffer("output", SIZE);
        ctx.enqueue_fill(output, 0_f32);

        let a = ctx.enqueue_create_buffer("a", SIZE);
        ctx.enqueue_copy(&input, a);

        // TODO: We need inline compilation for block sizes. This implies either
        // runtime shader compilation and caching, or static registration at call site. Not sure
        // if the latter is possible in Rust.
        ctx.enqueue_function::<Add10>(
            (),
            (),
            Add10PushConstant {
                output: output.into(),
                a: a.into(),
                len: a.len() as u32,
                _padding: Default::default(),
            },
            grid_dim_1d(a.len())
        );

        // TODO: Host readback is tedious.
        let output_host = ctx.create_host_buffer(SIZE);
        {
            let output_host = ctx.register_buffer("output_host", &output_host);
            ctx.enqueue_copy_buffer(output, output_host);
        }

        ctx.execute(None).expect("failed to execute");

        ctx.synchronize();

        let expected: Vec<_> = input.iter().map(|x| x + 10_f32).collect();

        let output_host = output_host.map_to_host(&ctx);

        for i in 0..SIZE {
            assert_eq!(expected[i], output_host[i]);
        }

        println!("output: {:?}", output_host);
        println!("expected: {:?}", expected);
    }
}
