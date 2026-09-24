use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, ItemStruct, parse_macro_input};

mod device_function;

/// Declares a shader module.
/// Compiled once per permutation and, if it is generic, once per registered instantiation.
/// Example:
/// ```ignore
/// shader!(MyShader, "shader.slang");
/// shader!(MyShader, "shader.slang", MyShaderPermutations);
/// shader!(Gemm<In, Acc> for [(f32, f32), (u32, f32)], "gemm.slang");
/// ```
#[proc_macro]
pub fn shader(input: TokenStream) -> TokenStream {
    let decl = parse_macro_input!(input as device_function::ShaderDecl);
    TokenStream::from(device_function::shader_impl(decl))
}

/// Declares a device function, optionally generic over one or more [`DType`]s.
///
/// ```ignore
/// function!(Add10, push: Add10Push, spec: Add10Spec, "main", "p01.slang");
/// ```
///
/// A generic function is written `Name<T>`, with `T` usable in the push constant, parameter
/// and specialization constant types. That one declaration registers it for every `DType`,
/// so the shader is compiled and its layout checked against `Add102dPush<T>` once per
/// instantiation, before any call site exists:
///
/// ```ignore
/// #[push]
/// pub struct Add102dPush<T: DType> { /* .. */ }
///
/// function!(Add102d<T>, push: Add102dPush<T>, "main", "p04.slang");
/// ```
///
/// The shader declares an ordinary Slang generic, which the compiler specializes with
/// `-specialize`, one argument per type parameter in declaration order. A generic entry
/// point takes its push constant as a `uniform` parameter, because a module-scope global
/// cannot name the entry point's type parameters; Slang still lowers that to a real SPIR-V
/// push constant:
///
/// ```ignore
/// struct Add102dPush<T : IArithmetic> { /* .. */ };
///
/// [numthreads(8, 8, 1)]
/// void main<T : IArithmetic>(uint2 tid: SV_GroupThreadID, uniform Add102dPush<T> push) { }
/// ```
///
/// Slang checks a generic body once, against its constraints, rather than once per
/// instantiation, so the kernel may only do what its constraints allow. Converting between
/// two type parameters needs an interface that declares the conversion: `IFloat` supplies
/// `toFloat`, `IInteger` supplies `toInt`, and nothing built in spans both.
///
/// A kernel that only makes sense for some of them names its own set, and pays for only
/// what it names. The listed types are the only ones the function is declared for, so
/// `Add102d<i32>` is then a compile error at the call site rather than a missing pipeline
/// at dispatch:
///
/// ```ignore
/// function!(Add102d<T> for [f32, u32], push: Add102dPush<T>, "main", "p04.slang");
/// ```
///
/// A function over several element types lists the *combinations* it supports rather than a
/// set per parameter, so an unwanted pairing simply never exists. This registers
/// `Gemm<f32, f32>` and `Gemm<u32, f32>` and nothing else — `Gemm<f32, u32>` does not
/// compile — and builds the shader once per listed tuple, specialized
/// accordingly, into `gemm_<in>_<acc>_<permutation>.spirv`:
///
/// ```ignore
/// function!(
///     Gemm<In, Acc> for [(f32, f32), (u32, f32)],
///     push: GemmPush<In, Acc>,
///     "main",
///     "gemm.slang",
/// );
/// ```
#[proc_macro]
pub fn function(input: TokenStream) -> TokenStream {
    let decl = parse_macro_input!(input as device_function::FunctionDecl);
    TokenStream::from(device_function::function_impl(decl))
}

#[proc_macro_attribute]
pub fn push(args: TokenStream, input: TokenStream) -> TokenStream {
    shader_type(args, input)
}

#[proc_macro_attribute]
pub fn spec(args: TokenStream, input: TokenStream) -> TokenStream {
    shader_type(args, input)
}

fn shader_type(args: TokenStream, input: TokenStream) -> TokenStream {
    if !args.is_empty() {
        let args = proc_macro2::TokenStream::from(args);
        return syn::Error::new_spanned(args, "push_constant takes no arguments")
            .to_compile_error()
            .into();
    }

    let input = parse_macro_input!(input as ItemStruct);

    // bytemuck refuses to derive `Pod` for a type with generic parameters, because it cannot
    // see through them to prove the struct has no padding. A generic push constant therefore
    // gets the impls by hand; `ShaderType` makes up for the missing check by asserting
    // per instantiation that the fields tile the struct exactly.
    if !input.generics.params.is_empty() {
        let name = &input.ident;
        let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

        let expanded = quote! {
            #[repr(C)]
            #[derive(Clone, Copy, ::gpu_reflect::ShaderType)]
            #input

            unsafe impl #impl_generics ::gpu::bytemuck::Zeroable
                for #name #ty_generics #where_clause {}

            unsafe impl #impl_generics ::gpu::bytemuck::Pod
                for #name #ty_generics #where_clause {}
        };

        return TokenStream::from(expanded);
    }

    let expanded = quote! {
        #[repr(C)]
        #[derive(
            Clone,
            Copy,
            ::gpu::bytemuck::Zeroable,
            ::gpu::bytemuck::Pod,
            ::gpu_reflect::ShaderType,
        )]
        #[bytemuck(crate = "::gpu::bytemuck")]
        #input
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(ShaderParameters)]
pub fn derive_shader_parameters(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(
                    &name,
                    "ShaderParameters can only be derived for structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(
                &name,
                "ShaderParameters can only be derived for structs",
            )
            .to_compile_error()
            .into();
        }
    };

    let member_items = fields.iter().map(|f| {
        let field_name = f.ident.as_ref().unwrap().to_string();
        let ty = &f.ty;
        quote! {
            gpu::ShaderParameterType {
                name: #field_name,
                kind: <#ty as gpu::Descriptor>::kind(),
                layout: <#ty as gpu::Descriptor>::layout(),
            }
        }
    });

    let value_items = fields.iter().map(|f| {
        let field_ident = f.ident.as_ref().unwrap();
        let field_name = field_ident.to_string();
        let ty = &f.ty;
        quote! {
            gpu::ShaderParameter {
                name: #field_name,
                kind: <#ty as gpu::Descriptor>::kind(),
                handle: gpu::Descriptor::handle(&self.#field_ident),
            }
        }
    });

    let expanded = quote! {
        impl #impl_generics gpu::DynShaderParameters for #name #ty_generics #where_clause {
            fn parameter_types() -> Vec<gpu::ShaderParameterType> {
                vec![ #( #member_items ),* ]
            }

            fn parameters(&self) -> Vec<gpu::ShaderParameter> {
                vec![ #( #value_items ),* ]
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(ShaderType)]
pub fn derive_shader_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let name_string = name.to_string();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(
                    &name,
                    "ShaderType can only be derived for structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&name, "ShaderType can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    // A type whose field order the compiler is free to shuffle has no layout
    // worth comparing against a shader's.
    let has_c_layout = input.attrs.iter().any(|attr| {
        if !attr.path().is_ident("repr") {
            return false;
        }
        // `repr(C)`, `repr(transparent)`, `repr(C, align(16))`, ...
        let Ok(list) = attr.meta.require_list() else {
            return false;
        };
        list.tokens.to_string().split(',').any(|repr| {
            let repr = repr.trim();
            repr == "C" || repr == "transparent"
        })
    });
    if !has_c_layout {
        return syn::Error::new_spanned(
            &name,
            "ShaderType requires #[repr(C)]: a repr(Rust) struct has no layout the GPU can rely on",
        )
        .to_compile_error()
        .into();
    }

    let field_items = fields.iter().map(|f| {
        let field_ident = f.ident.as_ref().unwrap();
        let field_name = field_ident.to_string();
        let ty = &f.ty;
        quote! {
            gpu::FieldLayout {
                name: #field_name,
                offset: ::std::mem::offset_of!(Self, #field_ident) as u32,
                ty: <#ty as gpu::ShaderType>::type_layout(),
            }
        }
    });

    let field_sizes = fields.iter().map(|f| {
        let ty = &f.ty;
        quote! { + ::std::mem::size_of::<#ty>() }
    });

    // A padded type is unsound to treat as `Pod`, and a padded push constant would compare
    // its field offsets against a shader that packs them differently. bytemuck checks this
    // for a concrete struct; for a generic one only a monomorphization can, so the check
    // lives in an associated const that `type_layout` forces for every instantiation.
    let expanded = quote! {
        #[doc(hidden)]
        impl #impl_generics #name #ty_generics #where_clause {
            const __ASSERT_NO_PADDING: () = assert!(
                ::std::mem::size_of::<Self>() == 0 #( #field_sizes )*,
                concat!(
                    "`", #name_string, "` has padding between or after its fields, which a \
                     shader type may not: reorder the fields, or pad them explicitly",
                ),
            );
        }

        impl #impl_generics gpu::ShaderType for #name #ty_generics #where_clause {
            fn type_layout() -> gpu::TypeLayout {
                let () = Self::__ASSERT_NO_PADDING;
                gpu::TypeLayout::Struct {
                    name: #name_string,
                    size: ::std::mem::size_of::<Self>() as u32,
                    fields: vec![ #( #field_items ),* ],
                }
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(ShaderPermutation)]
pub fn derive_shader_permutation(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(
                    &name,
                    "ShaderPermutation can only be derived for structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(
                &name,
                "ShaderPermutation can only be derived for structs",
            )
            .to_compile_error()
            .into();
        }
    };

    let field_names: Vec<_> = fields.iter().map(|f| f.ident.clone().unwrap()).collect();
    let field_types: Vec<_> = fields.iter().map(|f| f.ty.clone()).collect();

    let flatten_stmts = field_names.iter().zip(&field_types).map(|(f, ty)| {
        quote! {
            index += gpu::ShaderPermutationDimension::value(&self.#f) * stride;
            stride *= <#ty as gpu::ShaderPermutationDimension>::len();
        }
    });

    let total_stmts = field_types.iter().map(|ty| {
        quote! {
            total *= <#ty as gpu::ShaderPermutationDimension>::len();
        }
    });

    let define_items = field_names.iter().zip(&field_types).map(|(f, ty)| {
        quote! {
            (
                <#ty as gpu::ShaderPermutationDimension>::name(),
                gpu::ShaderPermutationDimension::define_value(&self.#f),
            )
        }
    });

    let from_flat_index_stmts = field_names.iter().zip(&field_types).map(|(f, ty)| {
        quote! {
            let #f = {
                let len = <#ty as gpu::ShaderPermutationDimension>::len();
                let digit = index % len;
                index /= len;
                <#ty as gpu::ShaderPermutationDimension>::from_index(digit)
            };
        }
    });

    let expanded = quote! {
        impl #impl_generics gpu::ShaderPermutationMatrix for #name #ty_generics #where_clause {
            #[allow(unused_mut, unused_assignments)]
            fn flatten(&self) -> usize {
                let mut index: usize = 0;
                let mut stride: usize = 1;
                #( #flatten_stmts )*
                index
            }

            fn total_permutations() -> usize {
                let mut total: usize = 1;
                #( #total_stmts )*
                total
            }

            fn defines(&self) -> Vec<(&'static str, String)> {
                vec![ #( #define_items ),* ]
            }

            #[allow(unused_mut, unused_assignments)]
            fn from_flat_index(mut index: usize) -> Self {
                #( #from_flat_index_stmts )*
                Self {
                    #( #field_names ),*
                }
            }
        }
    };

    TokenStream::from(expanded)
}
