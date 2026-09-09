use rhi::*;
use rhi_reflect::*;

#[push]
pub struct Add10Push {
    pub output: RWDeviceAddress<f32>,
    pub a: DeviceAddress<f32>,
}
#[spec]
pub struct Add10Spec {
    pub grid_size_x: u32,
}
function!(
    Add10,
    push: Add10Push,
    spec: Add10Spec,
    "main",
    "add_10.slang",
);

#[push]
pub struct AddPush {
    pub output: RWDeviceAddress<f32>,
    pub a: DeviceAddress<f32>,
    pub b: DeviceAddress<f32>,
}
#[spec]
pub struct AddSpec {
    pub grid_size_x: u32,
}
function!(
    Add,
    push: AddPush,
    spec: AddSpec,
    "main",
    "add.slang",
);

#[cfg(test)]
mod test {
    use super::*;
    use rhi::{DeviceContext, DeviceContextCreateInfo};

    /// https://puzzles.modular.com/puzzle_01/puzzle_01.html
    #[test]
    fn add_10() {
        const SIZE: usize = 4;

        let mut ctx = DeviceContext::new(&DeviceContextCreateInfo::default());

        let a_host: Vec<_> = (0..SIZE).map(|i| i as f32).collect();
        let expected: Vec<_> = a_host.iter().map(|x| x + 10_f32).collect();

        let output = ctx.enqueue_create_buffer("output", SIZE);
        ctx.enqueue_fill(output, 0_f32);

        let a = ctx.enqueue_create_buffer("a", SIZE);
        ctx.enqueue_copy(a_host.as_slice(), a);

        let add_10 = ctx.compile_function::<Add10>(
            &(),
            Some(Add10Spec {
                grid_size_x: SIZE as u32,
            }),
        );

        enqueue_function!(
            ctx,
            func: &add_10,
            push: Add10Push {
                output: output.into(),
                a: a.into(),
            },
            grid_dim: UInt3::new(a.len() as u32, 1, 1),
        );

        let output_host = ctx.create_host_buffer(SIZE);
        {
            let output_host = ctx.register_buffer("output_host", &output_host);
            ctx.enqueue_copy(output, output_host);
        }

        ctx.execute(None).expect("failed to execute");

        ctx.synchronize();

        let output_host = output_host.map_to_host(&ctx);

        for i in 0..SIZE {
            assert_eq!(expected[i], output_host[i]);
        }

        println!("output: {:?}", output_host);
        println!("expected: {:?}", expected);
    }

    /// https://puzzles.modular.com/puzzle_02/puzzle_02.html
    #[test]
    fn add() {
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

        let output = ctx.enqueue_create_buffer("output", SIZE);
        ctx.enqueue_fill(output, 0_f32);

        let a = ctx.enqueue_create_buffer("a", SIZE);
        ctx.enqueue_copy(a_host.as_slice(), a);

        let b = ctx.enqueue_create_buffer("b", SIZE);
        ctx.enqueue_copy(b_host.as_slice(), b);

        let add = ctx.compile_function::<Add>(
            &(),
            Some(AddSpec {
                grid_size_x: SIZE as u32,
            }),
        );

        enqueue_function!(
            ctx,
            func: &add,
            push: AddPush {
                output: output.into(),
                a: a.into(),
                b: b.into(),
            },
            grid_dim: UInt3::new(a.len() as u32, 1, 1),
        );

        let output_host = ctx.create_host_buffer(SIZE);
        {
            let output_host = ctx.register_buffer("output_host", &output_host);
            ctx.enqueue_copy(output, output_host);
        }

        ctx.execute(None).expect("failed to execute");

        ctx.synchronize();

        let output_host = output_host.map_to_host(&ctx);

        for i in 0..SIZE {
            assert_eq!(expected[i], output_host[i]);
        }

        println!("output: {:?}", output_host);
        println!("expected: {:?}", expected);
    }
}
