extern crate self as rhi;

// Re-exported so `shader_module!` can expand to `$crate::inventory::submit!`
// from any downstream crate without that crate needing its own `inventory` dependency.
pub use inventory;

mod device_context;
pub use device_context::*;

mod permutation;
pub use permutation::*;

mod shader_module;
pub use shader_module::*;

mod shader_parameter;
pub use shader_parameter::*;

mod shader_reflection;
pub use shader_reflection::*;

mod shader_type;
pub use shader_type::*;

mod vector;
pub use vector::*;

pub use ash::vk;

/// Divide-and-round-up helper for launching thread groups.
pub const fn div_round_up(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Returns the number of work groups to launch for a 1D kernel with the default work group size.
pub const fn grid_dim_1d(numel: usize) -> UInt3 {
    let grid_dim_x = div_round_up(numel, 64) as u32;
    UInt3::new(grid_dim_x, 1, 1)
}
