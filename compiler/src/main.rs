use rhi::*;
use std::assert_eq;
use std::fs;
use std::path::Path;
use std::unimplemented;

extern crate shaders as _;

/// Checks a kernel parameter's host type against the type slangc reflected for
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

/// Checks a kernel's `PushConstant` type against the `[vk::push_constant]`
/// block the shader declared, or `None` when the shader declared none.
fn check_push_constant_layout(
    kernel: &dyn KernelTrait,
    expected: Option<&Parameter>,
    shader: &Path,
) {
    let host = kernel.push_constant_layout();
    let result = host.check_push_constant(expected.map(|parameter| &parameter.ty));

    if let Err(mismatches) = result {
        panic!(
            "the push constant of {}:{} does not match the host type:\n{}",
            shader.display(),
            kernel.entry_point(),
            format_mismatches(&mismatches),
        );
    }
}

fn main() {
    let shaders = ShaderModuleRegistry::collect();

    for (_, shader) in &shaders {
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

            let output = command.output().unwrap();

            if !output.status.success() {
                println!("slangc failed for {}", src_path.display());
                println!("{}", String::from_utf8_lossy(&output.stdout));
                panic!("{}", String::from_utf8_lossy(&output.stderr));
            }
        }
    }

    let kernels = KernelRegistry::collect();

    for (_, kernel) in &kernels {
        let shader = shaders.get(&kernel.shader_type()).unwrap();

        let build_dir_path = shader.build_dir();

        let parameter_types = kernel.parameter_types();

        for index in 0..shader.total_permutations() {
            if !shader.should_compile(index) {
                continue;
            }

            let spirv_file_name = shader.spirv_file_name(index);
            let spirv_file_path = build_dir_path.join(&spirv_file_name);
            let json_file_path = spirv_file_path.with_extension("json");

            println!("Compiling kernel {}", spirv_file_name.display());

            let reflection = ShaderReflection::from_file(json_file_path);

            let entry_point = reflection
                .entry_points
                .iter()
                .find(|x| x.name == kernel.entry_point())
                .expect(&format!("missing entry point {}", kernel.entry_point()));

            // The shader's `[vk::push_constant]` parameter, if it has one.
            let mut expected_push_constant = None;

            for binding in &entry_point.bindings {
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

                if let Binding::PushConstantBuffer { index } = binding.binding {
                    assert_eq!(index, 0, "only one push constant range is supported");
                    assert!(
                        expected_push_constant.replace(expected_parameter).is_none(),
                        "{} declares more than one push constant",
                        spirv_file_name.display(),
                    );
                    continue;
                }

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

            check_push_constant_layout(kernel.as_ref(), expected_push_constant, &spirv_file_name);
        }
    }
}
