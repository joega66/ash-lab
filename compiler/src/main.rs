use gpu::*;
use std::assert_eq;
use std::fs;
use std::path::Path;
use std::unimplemented;

extern crate shaders as _;

/// Checks a device function's parameter's host type against the type slangc reflected for
/// the shader parameter it is bound to, failing the build on any disagreement.
fn check_element_layout(parameter: &ShaderParameterType, reflected: &Type, shader: &Path) {
    let layout = parameter.layout.expect(&format!(
        "parameter `{}` does not describe its element type; give it a {:?}<T> where T derives \
         ShaderType",
        parameter.name, parameter.kind,
    ));

    let result = match parameter.kind {
        DescriptorKind::ConstantBuffer => layout().check_constant_buffer(reflected),
        DescriptorKind::StructuredBuffer | DescriptorKind::RWStructuredBuffer => {
            layout().check_structured_buffer(reflected)
        }
    };

    if let Err(mismatches) = result {
        panic!(
            "{:?} `{}` of {} does not match the host type:\n{}",
            parameter.kind,
            parameter.name,
            shader.display(),
            format_mismatches(&mismatches),
        );
    }
}

/// Checks a device function's `PushConstant` type against the push constant the shader
/// declared, or `None` when it declared none.
fn check_push_constant_layout(
    function: &dyn DynDeviceFunctionLike,
    expected: Option<&Type>,
    shader: &Path,
) {
    let host = function.push_constant_layout();
    let result = host.check_push_constant(expected);

    if let Err(mismatches) = result {
        panic!(
            "the push constant of {}:{} does not match the host type:\n{}",
            shader.display(),
            function.entry_point(),
            format_mismatches(&mismatches),
        );
    }
}

/// Check's a device function's `SpecConstant` type against the `[SpecializationConstant]` block
/// the shader declared.
fn check_spec_constant_layout(
    function: &dyn DynDeviceFunctionLike,
    reflection: &ShaderReflection,
    shader: &Path,
) {
    let host = function.spec_constant_layout();
    let result = host.check_spec_constants(&reflection.specialization_constants());

    if let Err(mismatches) = result {
        panic!(
            "the specialization constants of {}:{} do not match the host type:\n{}",
            shader.display(),
            function.entry_point(),
            format_mismatches(&mismatches),
        );
    }
}

/// The Slang type the shader binds its push constant to, or `None` when it declares none.
///
/// A non-generic shader declares a module-scope `[vk::push_constant]` global, which slangc
/// reports as a binding on the entry point. A generic one cannot: a global cannot name the
/// entry point's type parameters, so the push constant is a `uniform` entry-point parameter
/// instead, and slangc reports it inline on the entry point. Both lower to the same SPIR-V
/// push constant, so both are checked against the same host type.
fn reflected_push_constant<'a>(
    reflection: &'a ShaderReflection,
    entry_point: &'a EntryPoint,
    function: &dyn DynDeviceFunctionLike,
    shader: &Path,
) -> Option<&'a Type> {
    let mut found: Option<&Type> = None;

    for binding in &entry_point.bindings {
        let Binding::PushConstantBuffer { index } = binding.binding else {
            continue;
        };
        assert_eq!(index, 0, "only one push constant range is supported");
        let parameter = reflection
            .parameters
            .iter()
            .find(|x| x.name.as_deref() == Some(binding.name.as_str()))
            .unwrap_or_else(|| panic!("missing binding {}", binding.name));
        assert!(
            found.replace(&parameter.ty).is_none(),
            "{} declares more than one push constant",
            shader.display(),
        );
    }

    for parameter in &entry_point.parameters {
        let Some(Binding::Uniform { offset, size, .. }) = parameter.binding else {
            continue;
        };
        let name = parameter.name.as_deref().unwrap_or("<unnamed>");
        // The host writes its push constant blob at offset 0 of the range, so the one
        // uniform parameter has to start there.
        assert_eq!(
            offset,
            0,
            "the uniform parameter `{name}` of {} does not start the push constant block; a \
             generic entry point takes exactly one `uniform` parameter",
            shader.display(),
        );
        let host_size = function.push_constant_range_size();
        assert!(
            host_size <= size,
            "the host push constant of {}:{} is {host_size} bytes, but the shader reserved \
             {size} bytes for `{name}`",
            shader.display(),
            function.entry_point(),
        );
        assert!(
            found.replace(&parameter.ty).is_none(),
            "{} declares more than one push constant",
            shader.display(),
        );
    }

    found
}

fn main() {
    let shaders = ShaderModuleRegistry::collect();
    let functions = DeviceFunctionRegistry::collect();

    for (shader_type, shader) in &shaders {
        let src_path = &shader.full_path();
        let build_dir_path = shader.build_dir();

        for index in 0..shader.total_permutations() {
            if !shader.should_compile(index) {
                continue;
            }

            let spirv_file_name = shader.spirv_file_name(index);
            let spirv_file_path = build_dir_path.join(&spirv_file_name);
            let json_file_path = spirv_file_path.with_extension("json");

            let needs_recompile = || {
                let Ok(dst_path_metadata) = fs::metadata(&spirv_file_path) else {
                    return true;
                };
                let src_path_metadata = match fs::metadata(&src_path) {
                    Ok(src_path_metadata) => src_path_metadata,
                    Err(err) => panic!("{}", err),
                };
                src_path_metadata.modified().unwrap() > dst_path_metadata.modified().unwrap()
            };

            if !needs_recompile() {
                continue;
            }

            println!("Compiling shader {}", spirv_file_name.display());

            if let Some(parent) = spirv_file_path.parent() {
                fs::create_dir_all(parent).expect("failed to create directories in build dir");
            }

            let mut command = std::process::Command::new("slangc");
            command
                .arg(&src_path)
                .arg("-o")
                .arg(&spirv_file_path)
                .arg("-target")
                .arg("spirv")
                .arg("-reflection-json")
                .arg(&json_file_path)
                .arg("-O3");

            for (define, value) in shader.defines(index) {
                let preprocessor_macro = format!("{define}={value}");
                command.arg("-D");
                command.arg(&preprocessor_macro);
            }

            let specialization_args = shader.specialization_args();
            if !specialization_args.is_empty() {
                let function = functions.get(shader_type).unwrap_or_else(|| {
                    panic!(
                        "{} is generic but has no device function to name its entry point",
                        src_path.display(),
                    )
                });
                command
                    .arg("-entry")
                    .arg(function.entry_point())
                    .arg("-stage")
                    .arg("compute");
                for arg in specialization_args {
                    command.arg("-specialize").arg(arg);
                }
            }

            let output = command.output().unwrap();

            if !output.status.success() {
                println!("slangc failed for {}", src_path.display());
                println!("{}", String::from_utf8_lossy(&output.stdout));
                panic!("{}", String::from_utf8_lossy(&output.stderr));
            }
        }
    }

    let functions = DeviceFunctionRegistry::collect();

    for (_, function) in &functions {
        let shader = shaders.get(&function.shader_type()).unwrap();

        let build_dir_path = shader.build_dir();

        let parameter_types = function.parameter_types();

        for index in 0..shader.total_permutations() {
            if !shader.should_compile(index) {
                continue;
            }

            let spirv_file_name = shader.spirv_file_name(index);
            let spirv_file_path = build_dir_path.join(&spirv_file_name);
            let json_file_path = spirv_file_path.with_extension("json");

            println!("Compiling function {}", spirv_file_name.display());

            let reflection = ShaderReflection::from_file(json_file_path);

            let entry_point = reflection
                .entry_points
                .iter()
                .find(|x| x.name == function.entry_point())
                .expect(&format!("missing entry point {}", function.entry_point()));

            let expected_push_constant =
                reflected_push_constant(&reflection, entry_point, function.as_ref(), &spirv_file_name);

            for binding in &entry_point.bindings {
                // Handled above, against the entry point as a whole.
                if let Binding::PushConstantBuffer { .. } = binding.binding {
                    continue;
                }

                // Specialization constants are module-scope, so they are
                // checked against the whole reflection below rather than one
                // entry point binding at a time.
                if let Binding::SpecializationConstant { .. } = binding.binding {
                    continue;
                }

                let expected_parameter = reflection
                    .parameters
                    .iter()
                    .find(|x| {
                        let Some(name) = x.name.as_ref() else {
                            return false;
                        };
                        *name == binding.name
                    })
                    .expect(&format!("missing binding {}", binding.name));

                let (parameter, parameter_index) = parameter_types
                    .iter()
                    .zip(0..(parameter_types.len() as u32))
                    .find(|(parameter, _)| parameter.name == binding.name)
                    .expect(&format!("missing parameter {}", binding.name));

                match binding.binding {
                    Binding::DescriptorTableSlot { index } => {
                        assert_eq!(index, parameter_index);
                        match parameter.kind {
                            DescriptorKind::ConstantBuffer => match &expected_parameter.ty {
                                Type::ConstantBuffer { .. } => {
                                    check_element_layout(
                                        parameter,
                                        &expected_parameter.ty,
                                        &spirv_file_name,
                                    );
                                }
                                _ => {
                                    panic!("expected a ConstantBuffer at {}", parameter.name);
                                }
                            },
                            DescriptorKind::StructuredBuffer => match &expected_parameter.ty {
                                Type::Resource {
                                    base_shape, access, ..
                                } => {
                                    assert_eq!(base_shape, &BaseShape::StructuredBuffer);
                                    assert_eq!(access, &None);
                                    check_element_layout(
                                        parameter,
                                        &expected_parameter.ty,
                                        &spirv_file_name,
                                    );
                                }
                                _ => {
                                    panic!("expected a StructuredBuffer at {}", parameter.name);
                                }
                            },
                            DescriptorKind::RWStructuredBuffer => match &expected_parameter.ty {
                                Type::Resource {
                                    base_shape, access, ..
                                } => {
                                    assert_eq!(base_shape, &BaseShape::StructuredBuffer);
                                    let access = access.as_ref().unwrap();
                                    assert_eq!(access, &Access::ReadWrite);
                                    check_element_layout(
                                        parameter,
                                        &expected_parameter.ty,
                                        &spirv_file_name,
                                    );
                                }
                                _ => {
                                    panic!("expected a RWStructuredBuffer at {}", parameter.name);
                                }
                            },
                        }
                    }
                    _ => {
                        unimplemented!()
                    }
                }
            }

            check_push_constant_layout(function.as_ref(), expected_push_constant, &spirv_file_name);
            check_spec_constant_layout(function.as_ref(), &reflection, &spirv_file_name);
        }
    }
}
