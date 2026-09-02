use crate::{Binding, Type};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
}

impl ScalarKind {
    /// The `scalarType` string slangc writes into its reflection JSON.
    pub fn slang_name(self) -> &'static str {
        match self {
            ScalarKind::Bool => "bool",
            ScalarKind::Int8 => "int8",
            ScalarKind::UInt8 => "uint8",
            ScalarKind::Int16 => "int16",
            ScalarKind::UInt16 => "uint16",
            ScalarKind::Int32 => "int32",
            ScalarKind::UInt32 => "uint32",
            ScalarKind::Int64 => "int64",
            ScalarKind::UInt64 => "uint64",
            ScalarKind::Float32 => "float32",
            ScalarKind::Float64 => "float64",
        }
    }

    pub fn source_name(self) -> &'static str {
        match self {
            ScalarKind::Bool => "bool",
            ScalarKind::Int8 => "int8_t",
            ScalarKind::UInt8 => "uint8_t",
            ScalarKind::Int16 => "int16_t",
            ScalarKind::UInt16 => "uint16_t",
            ScalarKind::Int32 => "int",
            ScalarKind::UInt32 => "uint",
            ScalarKind::Int64 => "int64_t",
            ScalarKind::UInt64 => "uint64_t",
            ScalarKind::Float32 => "float",
            ScalarKind::Float64 => "double",
        }
    }

    pub fn size(self) -> u32 {
        match self {
            // Slang's `bool` is word-sized in a uniform buffer.
            ScalarKind::Bool => 4,
            ScalarKind::Int8 | ScalarKind::UInt8 => 1,
            ScalarKind::Int16 | ScalarKind::UInt16 => 2,
            ScalarKind::Int32 | ScalarKind::UInt32 | ScalarKind::Float32 => 4,
            ScalarKind::Int64 | ScalarKind::UInt64 | ScalarKind::Float64 => 8,
        }
    }
}

/// Location in the byte blob of a vk::DeviceAddress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressSlot {
    pub offset: u32,
    pub writable: bool,
}

/// One field of a [`TypeLayout::Struct`], at its real Rust offset.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldLayout {
    pub name: &'static str,
    pub offset: u32,
    pub ty: TypeLayout,
}

/// A host type described the way slangc describes a shader type.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeLayout {
    Unit,
    Scalar(ScalarKind),
    Vector {
        element_type: ScalarKind,
        element_count: u32,
    },
    Array {
        element_type: Box<TypeLayout>,
        element_count: u32,
    },
    Struct {
        name: &'static str,
        /// `size_of` the Rust struct, including trailing padding.
        size: u32,
        fields: Vec<FieldLayout>,
    },
    DeviceAddress {
        pointee: Box<TypeLayout>,
        writable: bool,
    },
}

impl TypeLayout {
    pub fn size(&self) -> u32 {
        match self {
            TypeLayout::Unit => 0,
            TypeLayout::Scalar(scalar) => scalar.size(),
            TypeLayout::Vector {
                element_type,
                element_count,
            } => element_type.size() * element_count,
            TypeLayout::Array {
                element_type,
                element_count,
            } => element_type.size() * element_count,
            TypeLayout::Struct { size, .. } => *size,
            TypeLayout::DeviceAddress { .. } => 8,
        }
    }

    pub fn address_slots(&self) -> Vec<AddressSlot> {
        let mut slots = Vec::new();
        self.collect_address_slots(0, &mut slots);
        slots
    }

    fn collect_address_slots(&self, base: u32, slots: &mut Vec<AddressSlot>) {
        match self {
            TypeLayout::DeviceAddress { writable, .. } => slots.push(AddressSlot {
                offset: base,
                writable: *writable,
            }),
            TypeLayout::Struct { fields, .. } => {
                for field in fields {
                    field.ty.collect_address_slots(base + field.offset, slots);
                }
            }
            TypeLayout::Array {
                element_type,
                element_count,
            } => {
                let stride = element_type.size();
                for i in 0..*element_count {
                    element_type.collect_address_slots(base + i * stride, slots);
                }
            }
            TypeLayout::Unit | TypeLayout::Scalar(_) | TypeLayout::Vector { .. } => {}
        }
    }

    /// What to call this in a mismatch message.
    fn kind_name(&self) -> &'static str {
        match self {
            TypeLayout::Unit => "unit",
            TypeLayout::Scalar(_) => "scalar",
            TypeLayout::Vector { .. } => "vector",
            TypeLayout::Array { .. } => "array",
            TypeLayout::Struct { .. } => "struct",
            TypeLayout::DeviceAddress { .. } => "device address",
        }
    }

    fn source_name(&self) -> Option<&'static str> {
        match self {
            TypeLayout::Scalar(scalar) => Some(scalar.source_name()),
            TypeLayout::Struct { name, .. } => Some(name),
            _ => None,
        }
    }

    fn root_path(&self) -> String {
        match self {
            TypeLayout::Struct { name, .. } => (*name).to_string(),
            TypeLayout::Unit => "()".to_string(),
            _ => "<value>".to_string(),
        }
    }

    /// Compares this host layout against a type slangc reflected.
    pub fn check(&self, reflected: &Type) -> Result<(), Vec<LayoutMismatch>> {
        let mut mismatches = Vec::new();
        check_type(self, reflected, &self.root_path(), &mut mismatches);
        finish(mismatches)
    }

    pub fn check_constant_buffer(&self, reflected: &Type) -> Result<(), Vec<LayoutMismatch>> {
        self.check_uniform_block(reflected, "constant buffer")
    }

    pub fn check_push_constant(&self, reflected: Option<&Type>) -> Result<(), Vec<LayoutMismatch>> {
        match (self, reflected) {
            (TypeLayout::Unit, None) => Ok(()),
            (TypeLayout::Unit, Some(_)) => Err(vec![mismatch(
                &self.root_path(),
                "the shader declares a push constant block, but the kernel's PushConstant type \
                 is `()`"
                    .to_string(),
            )]),
            (_, None) => Err(vec![mismatch(
                &self.root_path(),
                "the kernel declares a PushConstant, but the shader has no \
                 [vk::push_constant] parameter"
                    .to_string(),
            )]),
            (_, Some(reflected)) => self.check_uniform_block(reflected, "push constant block"),
        }
    }

    fn check_uniform_block(&self, reflected: &Type, noun: &str) -> Result<(), Vec<LayoutMismatch>> {
        let path = self.root_path();

        let Type::ConstantBuffer {
            element_type,
            element_var_layout,
            ..
        } = reflected
        else {
            return Err(vec![LayoutMismatch {
                path,
                message: format!(
                    "shader parameter is {}, not a {noun}",
                    reflected_kind_name(reflected)
                ),
            }]);
        };

        let mut mismatches = Vec::new();
        check_type(self, element_type, &path, &mut mismatches);

        if let Some(element_var_layout) = element_var_layout {
            if let Binding::Uniform { size, .. } = element_var_layout.binding {
                if self.size() > size {
                    mismatches.push(LayoutMismatch {
                        path: path.clone(),
                        message: format!(
                            "host type is {} bytes, but the shader reserved {size} bytes for the \
                             {noun}",
                            self.size()
                        ),
                    });
                }
            }
        }

        finish(mismatches)
    }

    pub fn check_structured_buffer(&self, reflected: &Type) -> Result<(), Vec<LayoutMismatch>> {
        let path = self.root_path();

        let Type::Resource { result_type, .. } = reflected else {
            return Err(vec![LayoutMismatch {
                path,
                message: format!(
                    "shader parameter is {}, not a structured buffer",
                    reflected_kind_name(reflected)
                ),
            }]);
        };

        let Some(result_type) = result_type else {
            return Err(vec![LayoutMismatch {
                path,
                message: "shader parameter does not report an element type".to_string(),
            }]);
        };

        let mut mismatches = Vec::new();
        check_type(self, result_type, &path, &mut mismatches);
        finish(mismatches)
    }
}

/// One disagreement between a host type and a reflected shader type.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutMismatch {
    /// Dotted path to the offending field, e.g. `CameraUniform.position`.
    pub path: String,
    pub message: String,
}

impl std::fmt::Display for LayoutMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

pub fn format_mismatches(mismatches: &[LayoutMismatch]) -> String {
    mismatches
        .iter()
        .map(|mismatch| format!("  {mismatch}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn finish(mismatches: Vec<LayoutMismatch>) -> Result<(), Vec<LayoutMismatch>> {
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(mismatches)
    }
}

fn reflected_kind_name(reflected: &Type) -> &'static str {
    match reflected {
        Type::Struct { .. } => "a struct",
        Type::ConstantBuffer { .. } => "a constant buffer",
        Type::Resource { .. } => "a resource",
        Type::Scalar { .. } => "a scalar",
        Type::Vector { .. } => "a vector",
        Type::Array { .. } => "an array",
        Type::Matrix { .. } => "a matrix",
        Type::Pointer { .. } => "a pointer",
        Type::SamplerState => "a sampler state",
    }
}

fn mismatch(path: &str, message: String) -> LayoutMismatch {
    LayoutMismatch {
        path: path.to_string(),
        message,
    }
}

fn check_type(
    host: &TypeLayout,
    reflected: &Type,
    path: &str,
    mismatches: &mut Vec<LayoutMismatch>,
) {
    match (host, reflected) {
        (TypeLayout::Scalar(host_scalar), Type::Scalar { scalar_type }) => {
            if host_scalar.slang_name() != scalar_type {
                mismatches.push(mismatch(
                    path,
                    format!(
                        "host type is {}, shader type is {scalar_type}",
                        host_scalar.slang_name()
                    ),
                ));
            }
        }

        (
            TypeLayout::Vector {
                element_type,
                element_count,
            },
            Type::Vector {
                element_type: reflected_element_type,
                element_count: reflected_element_count,
            },
        ) => {
            if element_count != reflected_element_count {
                mismatches.push(mismatch(
                    path,
                    format!(
                        "host type has {element_count} components, shader type has \
                         {reflected_element_count}"
                    ),
                ));
            }
            check_type(
                &TypeLayout::Scalar(*element_type),
                reflected_element_type,
                path,
                mismatches,
            );
        }

        (
            TypeLayout::Struct { fields, .. },
            Type::Struct {
                fields: reflected_fields,
                ..
            },
        ) => {
            check_fields(fields, reflected_fields, path, mismatches);
        }

        (
            TypeLayout::Array {
                element_type,
                element_count,
            },
            Type::Array {
                element_type: reflected_element_type,
                element_count: reflected_element_count,
                uniform_stride,
            },
        ) => {
            if element_count != reflected_element_count {
                mismatches.push(mismatch(
                    path,
                    format!(
                        "host type has {element_count} elements, shader type has \
                         {reflected_element_count}"
                    ),
                ));
            }

            // A Rust array is packed, so its stride is just the element size.
            // Slang's is the layout's, which for std140 rounds up to 16.
            let host_stride = element_type.size();
            if *uniform_stride != 0 && host_stride != *uniform_stride {
                mismatches.push(mismatch(
                    path,
                    format!(
                        "host elements are {host_stride} bytes apart, shader elements are \
                         {uniform_stride} bytes apart"
                    ),
                ));
            }

            check_type(element_type, reflected_element_type, path, mismatches);
        }

        (TypeLayout::DeviceAddress { pointee, .. }, Type::Pointer { value_type }) => {
            if let Some(host_name) = pointee.source_name() {
                if host_name != value_type {
                    mismatches.push(mismatch(
                        path,
                        format!("host points at {host_name}, shader points at {value_type}"),
                    ));
                }
            }
        }

        _ => {
            mismatches.push(mismatch(
                path,
                format!(
                    "host type is a {}, shader type is {}",
                    host.kind_name(),
                    reflected_kind_name(reflected)
                ),
            ));
        }
    }
}

fn check_fields(
    fields: &[FieldLayout],
    reflected_fields: &[crate::Parameter],
    path: &str,
    mismatches: &mut Vec<LayoutMismatch>,
) {
    if fields.len() != reflected_fields.len() {
        let host_names: Vec<&str> = fields.iter().map(|field| field.name).collect();
        let shader_names: Vec<&str> = reflected_fields
            .iter()
            .map(|field| field.name.as_deref().unwrap_or("<unnamed>"))
            .collect();
        mismatches.push(mismatch(
            path,
            format!(
                "host struct has {} fields [{}], shader struct has {} fields [{}]",
                fields.len(),
                host_names.join(", "),
                reflected_fields.len(),
                shader_names.join(", "),
            ),
        ));
    }

    for (field, reflected_field) in fields.iter().zip(reflected_fields) {
        let field_path = format!("{path}.{}", field.name);

        let reflected_name = reflected_field.name.as_deref().unwrap_or("<unnamed>");
        if field.name != reflected_name {
            mismatches.push(mismatch(
                &field_path,
                format!("shader field at this position is named `{reflected_name}`"),
            ));
        }

        match &reflected_field.binding {
            Some(Binding::Uniform { offset, size, .. }) => {
                if field.offset != *offset {
                    mismatches.push(mismatch(
                        &field_path,
                        format!(
                            "host field is at offset {}, shader field is at offset {offset}",
                            field.offset
                        ),
                    ));
                }
                check_field_size(&field.ty, *size, &field_path, mismatches);
            }
            Some(_) => mismatches.push(mismatch(
                &field_path,
                "shader field is not laid out in a uniform buffer".to_string(),
            )),
            None => mismatches.push(mismatch(
                &field_path,
                "shader field has no binding information".to_string(),
            )),
        }

        check_type(&field.ty, &reflected_field.ty, &field_path, mismatches);
    }
}

/// Leaf fields must be exactly the size the shader gives them. Composite
/// fields only have to fit, because Slang pads a nested struct out to its own
/// alignment and the field offsets already catch a real layout difference.
fn check_field_size(
    host: &TypeLayout,
    reflected_size: u32,
    path: &str,
    mismatches: &mut Vec<LayoutMismatch>,
) {
    let host_size = host.size();
    let disagrees = match host {
        TypeLayout::Scalar(_) | TypeLayout::Vector { .. } | TypeLayout::DeviceAddress { .. } => {
            host_size != reflected_size
        }
        // A unit field has no counterpart at all; `check_type` reports it.
        TypeLayout::Unit => false,
        TypeLayout::Array { .. } | TypeLayout::Struct { .. } => host_size > reflected_size,
    };
    if disagrees {
        mismatches.push(mismatch(
            path,
            format!("host field is {host_size} bytes, shader field is {reflected_size} bytes"),
        ));
    }
}

pub trait ShaderType {
    fn type_layout() -> TypeLayout;

    fn size() -> u32
    where
        Self: Sized,
    {
        size_of::<Self>() as u32
    }
}

/// The empty shader type, for a kernel that has no push constant.
impl ShaderType for () {
    fn type_layout() -> TypeLayout {
        TypeLayout::Unit
    }
}

macro_rules! scalar_shader_type {
    ($ty:ty, $scalar:expr) => {
        impl ShaderType for $ty {
            fn type_layout() -> TypeLayout {
                TypeLayout::Scalar($scalar)
            }
        }
    };
}

scalar_shader_type!(i8, ScalarKind::Int8);
scalar_shader_type!(u8, ScalarKind::UInt8);
scalar_shader_type!(i16, ScalarKind::Int16);
scalar_shader_type!(u16, ScalarKind::UInt16);
scalar_shader_type!(i32, ScalarKind::Int32);
scalar_shader_type!(u32, ScalarKind::UInt32);
scalar_shader_type!(i64, ScalarKind::Int64);
scalar_shader_type!(u64, ScalarKind::UInt64);
scalar_shader_type!(f32, ScalarKind::Float32);
scalar_shader_type!(f64, ScalarKind::Float64);

impl<T: ShaderType, const N: usize> ShaderType for [T; N] {
    fn type_layout() -> TypeLayout {
        TypeLayout::Array {
            element_type: Box::new(T::type_layout()),
            element_count: N as u32,
        }
    }
}

#[cfg(test)]
mod tests {}
