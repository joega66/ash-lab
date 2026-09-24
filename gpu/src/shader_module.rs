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

/// The shader path is relative to the directory of the `.rs` file that declares it, so
/// a shader sits next to its declaring module rather than at the top of `src`.
///
/// Declared with [`shader!`](crate::shader), or by [`function!`](crate::function) when the
/// shader is also an entry point:
///
/// ```ignore
/// shader!(MyShader, "MyShader.slang");
/// shader!(MyShader, "MyShader.slang", MyShaderPermutations);
/// shader!(MyShader<T> for [f32, u32], "MyShader.slang");
/// ```
pub trait DynShaderModuleLike {
    fn manifest_dir(&self) -> &'static str;

    fn relative_path(&self) -> &'static str;

    /// The `.rs` file that declared this shader, as `file!()` reports it.
    fn source_file(&self) -> &'static str;

    /// The directory of [`Self::source_file`], relative to the crate's `src`, which is
    /// what [`Self::relative_path`] and the compiled artifacts are laid out against.
    /// Empty for a shader declared directly in `src`.
    ///
    /// `file!()` is relative to whichever directory cargo invoked rustc from — the
    /// workspace root, not the crate root — so it is anchored on its `src` component
    /// rather than joined onto a guessed base.
    fn source_dir(&self) -> PathBuf {
        let source_file = Path::new(self.source_file());
        let mut components = source_file.components();
        let found_src = components.any(|component| component.as_os_str() == "src");
        assert!(
            found_src,
            "{} is not under a `src` directory",
            source_file.display(),
        );
        let under_src = components.collect::<PathBuf>();
        under_src
            .parent()
            .expect("source file has no parent")
            .to_path_buf()
    }

    fn full_path(&self) -> PathBuf {
        Path::new(self.manifest_dir())
            .join("src")
            .join(self.source_dir())
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

    /// The Slang type arguments this instantiation specializes the entry point with, one
    /// per type parameter and in declaration order, e.g. `["float"]` or
    /// `["float", "uint"]`. Empty for a non-generic shader.
    ///
    /// The shader declares an ordinary Slang generic and the compiler passes these to
    /// slangc as `-specialize`, so the kernel reads as generic code rather than as a
    /// preprocessor sandwich. A generic entry point takes its push constant as a `uniform`
    /// parameter, since a module-scope global cannot name the entry point's type
    /// parameters; Slang still lowers that to a real SPIR-V push constant.
    ///
    /// Instantiations are an axis of their own rather than a [`ShaderPermutationMatrix`]
    /// dimension: a permutation is chosen at the call site by a value, an instantiation by
    /// a type. Each instantiation is its own shader module type, so the artifacts it builds
    /// are the full cross product of the two axes.
    fn specialization_args(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// A file-name-safe tag separating this instantiation's build artifacts from its
    /// siblings', derived from the element types it was instantiated with: `float`, or
    /// `float_uint` for a function over a pair. Empty for a non-generic shader.
    ///
    /// Taken from [`Self::specialization_args`] rather than stored beside it, so the two
    /// cannot disagree about what this instantiation is, whatever its arity.
    fn instantiation_tag(&self) -> String {
        self.specialization_args().join("_").to_lowercase()
    }

    #[allow(unused_variables)]
    fn should_compile(&self, index: usize) -> bool {
        true
    }

    #[allow(unused_variables)]
    fn should_create(&self, index: usize, device: &Device) -> bool {
        true
    }

    /// Mirrors the source layout under [`Self::build_dir`], so that shaders sharing a
    /// file name in different modules compile to distinct artifacts, and so that the
    /// instantiations of one generic shader do not overwrite each other.
    fn spirv_file_name(&self, index: usize) -> PathBuf {
        let relative_path = self.source_dir().join(self.relative_path());
        let mut new_file_name = relative_path
            .file_stem()
            .expect("missing file stem")
            .to_string_lossy()
            .into_owned();
        let tag = self.instantiation_tag();
        if !tag.is_empty() {
            new_file_name.push('_');
            new_file_name.push_str(&tag);
        }
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

    /// The permutation's defines. Instantiations do not contribute: they specialize the
    /// entry point's generic parameters instead, through [`Self::specialization_args`].
    fn defines(&self, index: usize) -> Vec<(&'static str, String)> {
        Self::Permutations::from_flat_index(index).defines()
    }
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
