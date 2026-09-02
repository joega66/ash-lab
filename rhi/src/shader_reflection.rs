use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ShaderReflection {
    #[serde(default)]
    pub parameters: Vec<Parameter>,
    #[serde(default, rename = "entryPoints")]
    pub entry_points: Vec<EntryPoint>,
    #[serde(rename = "bindlessSpaceIndex")]
    pub bindless_space_index: u32,
}

impl ShaderReflection {
    pub fn from_file(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("failed to read reflection JSON {}: {e}", path.display()));
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("failed to parse reflection JSON {}: {e}", path.display()))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Parameter {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub ty: Type,
    #[serde(default)]
    pub binding: Option<Binding>,
    #[serde(default, rename = "semanticName")]
    pub semantic_name: Option<String>,
    #[serde(default)]
    pub stage: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EntryPoint {
    pub name: String,
    pub stage: String,
    #[serde(default)]
    pub parameters: Vec<Parameter>,
    #[serde(default)]
    pub result: Option<Parameter>,
    #[serde(default, rename = "threadGroupSize")]
    pub thread_group_size: Option<[u32; 3]>,
    #[serde(default)]
    pub bindings: Vec<EntryPointBinding>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EntryPointBinding {
    pub name: String,
    pub binding: Binding,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Binding {
    DescriptorTableSlot {
        index: u32,
    },
    PushConstantBuffer {
        index: u32,
    },
    #[serde(rename_all = "camelCase")]
    Uniform {
        offset: u32,
        size: u32,
        #[serde(default)]
        element_stride: u32,
    },
    VaryingInput {
        index: u32,
    },
    VaryingOutput {
        index: u32,
    },
}

/// The Slang type of a parameter or field, which themselves can be composite
/// types (structs, constant buffers, resources, vectors, matrices).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Type {
    Struct {
        name: String,
        #[serde(default)]
        fields: Vec<Parameter>,
    },
    #[serde(rename_all = "camelCase")]
    ConstantBuffer {
        element_type: Box<Type>,
        #[serde(default)]
        container_var_layout: Option<ContainerVarLayout>,
        #[serde(default)]
        element_var_layout: Option<ElementVarLayout>,
    },
    #[serde(rename_all = "camelCase")]
    Resource {
        base_shape: BaseShape,
        #[serde(default)]
        access: Option<Access>,
        #[serde(default)]
        result_type: Option<Box<Type>>,
    },
    #[serde(rename_all = "camelCase")]
    Scalar {
        scalar_type: String,
    },
    #[serde(rename_all = "camelCase")]
    Vector {
        element_count: u32,
        element_type: Box<Type>,
    },
    #[serde(rename_all = "camelCase")]
    Array {
        element_count: u32,
        element_type: Box<Type>,
        #[serde(default)]
        uniform_stride: u32,
    },
    #[serde(rename_all = "camelCase")]
    Matrix {
        row_count: u32,
        column_count: u32,
        element_type: Box<Type>,
    },
    #[serde(rename_all = "camelCase")]
    Pointer {
        value_type: String,
    },
    SamplerState,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum BaseShape {
    StructuredBuffer,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum Access {
    ReadWrite,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContainerVarLayout {
    pub binding: Binding,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ElementVarLayout {
    #[serde(rename = "type")]
    pub ty: Box<Type>,
    pub binding: Binding,
}

#[cfg(test)]
mod tests {}
