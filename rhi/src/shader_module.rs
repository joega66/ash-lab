use crate::{
    ShaderParameterType, ShaderParametersTrait, ShaderPermutationMatrix, ShaderType, TypeLayout,
};
use ash::Device;
use bytemuck::Pod;
use std::any::TypeId;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Walks up from `start` to find the workspace root (the ancestor
/// directory whose `Cargo.toml` declares `[workspace]`) and returns its
/// `target` directory.
fn workspace_target_dir(start: &Path) -> PathBuf {
    let mut dir = start.to_path_buf();
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if let Ok(contents) = std::fs::read_to_string(&cargo_toml) {
            if contents.contains("[workspace]") {
                return dir.join("target");
            }
        }
        if !dir.pop() {
            panic!("failed to locate workspace root from {}", start.display());
        }
    }
}

/// Example usage:
/// shader!(MyShader, "MyShader.slang")
/// shader!(MyShader, "MyShader.slang", MyShaderPermutations)
pub trait ShaderModuleTrait {
    fn manifest_dir(&self) -> &'static str;

    fn relative_path(&self) -> &'static str;

    fn full_path(&self) -> PathBuf {
        Path::new(self.manifest_dir())
            .join("src")
            .join(self.relative_path())
    }

    fn build_dir(&self) -> PathBuf {
        let manifest_dir = Path::new(self.manifest_dir());
        let crate_name = manifest_dir.file_name().expect("manifest dir has no name");
        workspace_target_dir(manifest_dir)
            .join("shaders")
            .join(crate_name)
    }

    fn total_permutations(&self) -> usize;

    fn defines(&self, index: usize) -> Vec<(&'static str, String)>;

    #[allow(unused_variables)]
    fn should_compile(&self, index: usize) -> bool {
        true
    }

    #[allow(unused_variables)]
    fn should_create(&self, index: usize, device: &Device) -> bool {
        true
    }

    fn spirv_file_name(&self, index: usize) -> PathBuf {
        let relative_path = Path::new(self.relative_path());
        let mut new_file_name = relative_path
            .file_stem()
            .expect("missing file stem")
            .to_string_lossy()
            .into_owned();
        new_file_name.push_str(&format!("_{}", index));
        let new_file_name = Path::new(&new_file_name);
        let new_file_name = new_file_name.with_extension("spirv");
        relative_path.with_file_name(&new_file_name)
    }

    fn reflection_file_name(&self, index: usize) -> PathBuf {
        self.spirv_file_name(index).with_extension("json")
    }
}

pub trait ShaderModule: ShaderModuleTrait {
    type Permutations: ShaderPermutationMatrix;

    fn total_permutations(&self) -> usize {
        Self::Permutations::total_permutations()
    }

    fn defines(&self, index: usize) -> Vec<(&'static str, String)> {
        let perm = Self::Permutations::from_flat_index(index);
        perm.defines()
    }
}

#[macro_export]
macro_rules! shader {
    ($ty:ident, $path:expr) => {
        $crate::shader!($ty, $path, ());
    };
    ($ty:ident, $path:expr, $permutations:ty) => {
        pub struct $ty {}

        impl $crate::ShaderModule for $ty {
            type Permutations = $permutations;
        }

        impl $crate::ShaderModuleTrait for $ty {
            fn manifest_dir(&self) -> &'static str {
                env!("CARGO_MANIFEST_DIR")
            }
            fn relative_path(&self) -> &'static str {
                $path
            }
            fn total_permutations(&self) -> usize {
                $crate::ShaderModule::total_permutations(self)
            }
            fn defines(&self, index: usize) -> Vec<(&'static str, String)> {
                $crate::ShaderModule::defines(self, index)
            }
        }

        $crate::inventory::submit! {
            $crate::ShaderModuleRegistry {
                type_id: std::any::TypeId::of::<$ty>(),
                instantiate: || -> Box<dyn $crate::ShaderModuleTrait> { Box::new($ty {}) },
            }
        }
    };
}

pub struct ShaderModuleRegistry {
    pub type_id: TypeId,
    pub instantiate: fn() -> Box<dyn ShaderModuleTrait>,
}

inventory::collect!(ShaderModuleRegistry);

impl ShaderModuleRegistry {
    pub fn collect() -> HashMap<TypeId, Box<dyn ShaderModuleTrait>> {
        let map = inventory::iter::<ShaderModuleRegistry>()
            .map(|registration| (registration.type_id, (registration.instantiate)()))
            .collect();
        map
    }
}

pub trait KernelTrait {
    fn shader_type(&self) -> std::any::TypeId;
    fn parameter_types(&self) -> Vec<ShaderParameterType>;
    fn push_constant_layout(&self) -> TypeLayout;
    fn push_constant_range_size(&self) -> u32 {
        self.push_constant_layout().size().next_multiple_of(4)
    }
    fn entry_point(&self) -> &'static str;
}

pub trait Kernel: KernelTrait {
    type Shader: ShaderModuleTrait + 'static;
    type Params: ShaderParametersTrait;
    type PushConstant: ShaderType + Pod;

    fn shader_type(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self::Shader>()
    }

    fn parameter_types(&self) -> Vec<ShaderParameterType> {
        Self::Params::parameter_types()
    }

    fn push_constant_layout(&self) -> TypeLayout {
        Self::PushConstant::type_layout()
    }
}

#[macro_export]
macro_rules! kernel {
    ($shader_ty:ident, $params_ty:ty, $push_constant_ty:ty, $entry_point:expr, $path:expr) => {
        shader!($shader_ty, $path);
        impl $crate::Kernel for $shader_ty {
            type Shader = $shader_ty;
            type Params = $params_ty;
            type PushConstant = $push_constant_ty;
        }
        impl $crate::KernelTrait for $shader_ty {
            fn shader_type(&self) -> std::any::TypeId {
                $crate::Kernel::shader_type(self)
            }

            fn parameter_types(&self) -> Vec<$crate::ShaderParameterType> {
                $crate::Kernel::parameter_types(self)
            }

            fn push_constant_layout(&self) -> $crate::TypeLayout {
                $crate::Kernel::push_constant_layout(self)
            }

            fn entry_point(&self) -> &'static str {
                $entry_point
            }
        }
        $crate::inventory::submit! {
            {
                $crate::KernelRegistry {
                    kernel_type: std::any::TypeId::of::<$shader_ty>(),
                    instantiate: || -> Box<dyn $crate::KernelTrait> { Box::new($shader_ty {}) },
                }
            }
        }
    };
}

#[derive(Clone)]
pub struct KernelRegistry {
    pub kernel_type: TypeId,
    pub instantiate: fn() -> Box<dyn KernelTrait>,
}

inventory::collect!(KernelRegistry);

impl KernelRegistry {
    pub fn collect() -> HashMap<TypeId, Box<dyn KernelTrait>> {
        let map = inventory::iter::<KernelRegistry>()
            .map(|registration| (registration.kernel_type, (registration.instantiate)()))
            .collect();
        map
    }
}
