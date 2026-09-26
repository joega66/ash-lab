extern crate self as gpu;

// Re-exported so `shader_module!` can expand to `$crate::inventory::submit!`
// from any downstream crate without that crate needing its own `inventory` dependency.
pub use inventory;

// Same reason: `#[push_constant]` derives `bytemuck::Pod` through this path.
pub use bytemuck;

// `shader!` and `function!` live in the proc-macro crate, but are re-exported here so a
// shader module reaches them through the same `use gpu::*` as everything else they need.
pub use gpu_reflect::{function, shader};

mod device_context;
pub use device_context::{
    DeviceBuffer, DeviceContext, DeviceContextCreateInfo, DeviceFunction, DeviceImage,
    DeviceImageCreateInfo, HostMappedMemory, Swapchain, SwapchainImage, U32Castable,
    buffer_transfer_read, buffer_transfer_write, color_attachment, constant_buffer_read,
    sampled_fragment, storage_buffer_read, storage_buffer_read_write,
};

mod dtype;
pub use dtype::{DType, DTypeOf};

mod descriptor_pool;

mod shader_permutation;
pub use shader_permutation::{
    ShaderPermutationBool, ShaderPermutationDimension, ShaderPermutationEnumValue,
    ShaderPermutationInt,
};

mod shader_module;
pub use shader_module::{
    DeviceFunctionLike, DeviceFunctionRegistry, DynDeviceFunctionLike, DynShaderModuleLike,
    ShaderModuleLike, ShaderModuleRegistry,
};

mod shader_parameter;
pub use shader_parameter::{
    ConstantBuffer, DescriptorKind, DeviceAddress, DynShaderParameters, RWDeviceAddress,
    RWStructuredBuffer, ShaderParameterType, StructuredBuffer,
};

mod shader_reflection;
pub use shader_reflection::{Access, BaseShape, Binding, EntryPoint, ShaderReflection, Type};

mod shader_type;
pub use shader_type::{FieldLayout, ScalarKind, ShaderType, TypeLayout, format_mismatches};

mod utils;

mod vector;
pub use vector::*;

pub use ash::vk;

/// Divide-and-round-up helper for launching thread groups.
pub const fn div_round_up(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Returns the number of work groups to launch for a 1D compute shader with the default grid size.
pub const fn grid_dim_1d(numel: usize) -> UInt3 {
    let grid_dim_x = div_round_up(numel, 64) as u32;
    UInt3::new(grid_dim_x, 1, 1)
}
