use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derives `rhi::ShaderParametersTrait` for a struct whose fields all implement
/// `rhi::Descriptor`, listing each field's name alongside its Slang
/// descriptor kind in declaration order.
#[proc_macro_derive(ShaderParameters)]
pub fn derive_descriptor_set(input: TokenStream) -> TokenStream {
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
            return syn::Error::new_spanned(&name, "ShaderParameters can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let member_items = fields.iter().map(|f| {
        let field_name = f.ident.as_ref().unwrap().to_string();
        let ty = &f.ty;
        quote! {
            rhi::ShaderParameterType {
                name: #field_name,
                kind: <#ty as rhi::Descriptor>::kind(),
                layout: <#ty as rhi::Descriptor>::layout(),
            }
        }
    });

    let value_items = fields.iter().map(|f| {
        let field_ident = f.ident.as_ref().unwrap();
        let field_name = field_ident.to_string();
        let ty = &f.ty;
        quote! {
            rhi::ShaderParameter {
                name: #field_name,
                kind: <#ty as rhi::Descriptor>::kind(),
                buffer: rhi::Descriptor::handle(&self.#field_ident),
            }
        }
    });

    let expanded = quote! {
        impl #impl_generics rhi::ShaderParametersTrait for #name #ty_generics #where_clause {
            fn parameter_types() -> Vec<rhi::ShaderParameterType> {
                vec![ #( #member_items ),* ]
            }

            fn parameters(&self) -> Vec<rhi::ShaderParameter> {
                vec![ #( #value_items ),* ]
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derives `rhi::ShaderType` for a `#[repr(C)]` struct whose fields all
/// implement `rhi::ShaderType`, reporting each field's real offset so the
/// layout can be checked against slangc's reflection JSON.
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
            rhi::FieldLayout {
                name: #field_name,
                offset: ::std::mem::offset_of!(Self, #field_ident) as u32,
                ty: <#ty as rhi::ShaderType>::type_layout(),
            }
        }
    });

    let expanded = quote! {
        impl #impl_generics rhi::ShaderType for #name #ty_generics #where_clause {
            fn type_layout() -> rhi::TypeLayout {
                rhi::TypeLayout::Struct {
                    name: #name_string,
                    size: ::std::mem::size_of::<Self>() as u32,
                    fields: vec![ #( #field_items ),* ],
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derives `rhi::ShaderPermutationMatrix` for a struct whose fields all
/// implement `rhi::ShaderPermutationDimension`, by enumerating the fields in declaration order.
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
            index += rhi::ShaderPermutationDimension::value(&self.#f) * stride;
            stride *= <#ty as rhi::ShaderPermutationDimension>::len();
        }
    });

    let total_stmts = field_types.iter().map(|ty| {
        quote! {
            total *= <#ty as rhi::ShaderPermutationDimension>::len();
        }
    });

    let define_items = field_names.iter().zip(&field_types).map(|(f, ty)| {
        quote! {
            (
                <#ty as rhi::ShaderPermutationDimension>::name(),
                rhi::ShaderPermutationDimension::define_value(&self.#f),
            )
        }
    });

    let from_flat_index_stmts = field_names.iter().zip(&field_types).map(|(f, ty)| {
        quote! {
            let #f = {
                let len = <#ty as rhi::ShaderPermutationDimension>::len();
                let digit = index % len;
                index /= len;
                <#ty as rhi::ShaderPermutationDimension>::from_index(digit)
            };
        }
    });

    let expanded = quote! {
        impl #impl_generics rhi::ShaderPermutationMatrix for #name #ty_generics #where_clause {
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
