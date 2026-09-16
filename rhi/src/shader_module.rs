use crate::{
    DynShaderParameters, ShaderParameterType, ShaderPermutationMatrix, ShaderType, TypeLayout,
};
use ash::Device;
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
pub trait DynShaderModuleLike {
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

pub trait ShaderModuleLike: DynShaderModuleLike {
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

        impl $crate::ShaderModuleLike for $ty {
            type Permutations = $permutations;
        }

        impl $crate::DynShaderModuleLike for $ty {
            fn manifest_dir(&self) -> &'static str {
                env!("CARGO_MANIFEST_DIR")
            }
            fn relative_path(&self) -> &'static str {
                $path
            }
            fn total_permutations(&self) -> usize {
                $crate::ShaderModuleLike::total_permutations(self)
            }
            fn defines(&self, index: usize) -> Vec<(&'static str, String)> {
                $crate::ShaderModuleLike::defines(self, index)
            }
        }

        $crate::inventory::submit! {
            $crate::ShaderModuleRegistry {
                type_id: std::any::TypeId::of::<$ty>(),
                instantiate: || -> Box<dyn $crate::DynShaderModuleLike> { Box::new($ty {}) },
            }
        }
    };
}

pub struct ShaderModuleRegistry {
    pub type_id: TypeId,
    pub instantiate: fn() -> Box<dyn DynShaderModuleLike>,
}

inventory::collect!(ShaderModuleRegistry);

impl ShaderModuleRegistry {
    pub fn collect() -> HashMap<TypeId, Box<dyn DynShaderModuleLike>> {
        let map = inventory::iter::<ShaderModuleRegistry>()
            .map(|registration| (registration.type_id, (registration.instantiate)()))
            .collect();
        map
    }
}

pub trait DynDeviceFunctionLike {
    /// `new` is used for accessing the vtable.
    fn new() -> Self
    where
        Self: Sized;
    fn shader_type(&self) -> std::any::TypeId;
    fn parameter_types(&self) -> Vec<ShaderParameterType>;
    fn push_constant_layout(&self) -> TypeLayout;
    fn push_constant_range_size(&self) -> u32 {
        self.push_constant_layout().size().next_multiple_of(4)
    }
    fn spec_constant_layout(&self) -> TypeLayout;
    fn entry_point(&self) -> &'static str;
}

pub trait DeviceFunctionLike: DynDeviceFunctionLike {
    type Shader: DynShaderModuleLike + 'static;
    type Params: DynShaderParameters;
    type PushConstant: ShaderType;
    type SpecConstant: ShaderType;

    fn shader_type(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self::Shader>()
    }

    fn parameter_types(&self) -> Vec<ShaderParameterType> {
        Self::Params::parameter_types()
    }

    fn push_constant_layout(&self) -> TypeLayout {
        Self::PushConstant::type_layout()
    }

    fn spec_constant_layout(&self) -> TypeLayout {
        Self::SpecConstant::type_layout()
    }
}

#[macro_export]
macro_rules! function {
    ($shader_ty:ident, params: $params_ty:ty, push: $push:ty, $entry_point:expr, $path:expr $(,)?) => {
        $crate::function!(
            $shader_ty,
            params: $params_ty,
            push: $push,
            spec: (),
            $entry_point,
            $path,
        );
    };
    ($shader_ty:ident, params: $params_ty:ty, spec: $spec:ty, $entry_point:expr, $path:expr $(,)?) => {
        $crate::function!(
            $shader_ty,
            params: $params_ty,
            push: (),
            spec: $spec,
            $entry_point,
            $path,
        );
    };
    ($shader_ty:ident, push: $push:ty, spec: $spec:ty, $entry_point:expr, $path:expr $(,)?) => {
        $crate::function!(
            $shader_ty,
            params: (),
            push: $push,
            spec: $spec,
            $entry_point,
            $path,
        );
    };
    ($shader_ty:ident, push: $push:ty, $entry_point:expr, $path:expr $(,)?) => {
        $crate::function!(
            $shader_ty,
            params: (),
            push: $push,
            spec: (),
            $entry_point,
            $path,
        );
    };

    ($shader_ty:ident, params: $params_ty:ty, push: $push:ty, spec: $spec:ty, $entry_point:expr, $path:expr $(,)?) => {
        shader!($shader_ty, $path);
        impl $crate::DeviceFunctionLike for $shader_ty {
            type Shader = $shader_ty;
            type Params = $params_ty;
            type PushConstant = $push;
            type SpecConstant = $spec;
        }
        impl $crate::DynDeviceFunctionLike for $shader_ty {
            fn new() -> Self {
                Self {}
            }

            fn shader_type(&self) -> std::any::TypeId {
                $crate::DeviceFunctionLike::shader_type(self)
            }

            fn parameter_types(&self) -> Vec<$crate::ShaderParameterType> {
                $crate::DeviceFunctionLike::parameter_types(self)
            }

            fn push_constant_layout(&self) -> $crate::TypeLayout {
                $crate::DeviceFunctionLike::push_constant_layout(self)
            }

            fn spec_constant_layout(&self) -> $crate::TypeLayout {
                $crate::DeviceFunctionLike::spec_constant_layout(self)
            }

            fn entry_point(&self) -> &'static str {
                $entry_point
            }
        }
        $crate::inventory::submit! {
            {
                $crate::DeviceFunctionRegistry {
                    function_type: std::any::TypeId::of::<$shader_ty>(),
                    instantiate: || -> Box<dyn $crate::DynDeviceFunctionLike> { Box::new($shader_ty {}) },
                }
            }
        }
    };
}

#[derive(Clone)]
pub struct DeviceFunctionRegistry {
    pub function_type: TypeId,
    pub instantiate: fn() -> Box<dyn DynDeviceFunctionLike>,
}

inventory::collect!(DeviceFunctionRegistry);

impl DeviceFunctionRegistry {
    pub fn collect() -> HashMap<TypeId, Box<dyn DynDeviceFunctionLike>> {
        let map = inventory::iter::<DeviceFunctionRegistry>()
            .map(|registration| (registration.function_type, (registration.instantiate)()))
            .collect();
        map
    }
}
