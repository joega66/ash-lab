//! `shader!` and `function!`, which declare a device function and register every
//! instantiation of it ahead of time.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, Token, Type, bracketed, parse_quote};

/// The element types one instantiation is built for, one per type parameter.
pub struct Instantiation {
    types: Vec<Type>,
}

/// `Name`, `Name<T>` or `Name<Src, Dst> for [(u32, f32), (f32, f32)]`.
pub struct Head {
    name: Ident,
    /// The declared type parameters. Empty for a non-generic shader.
    params: Vec<Ident>,
    /// The instantiations to register. `None` means every [`DType`], which is only
    /// meaningful for a single type parameter.
    instantiations: Option<Vec<Instantiation>>,
}

impl Parse for Head {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;

        let mut params = Vec::new();
        if input.peek(Token![<]) {
            input.parse::<Token![<]>()?;
            loop {
                params.push(input.parse()?);
                if input.parse::<Option<Token![,]>>()?.is_none() {
                    break;
                }
            }
            input.parse::<Token![>]>()?;
        }

        let mut instantiations = None;
        if input.peek(Token![for]) {
            input.parse::<Token![for]>()?;
            let list;
            bracketed!(list in input);
            let items = Punctuated::<Type, Token![,]>::parse_terminated(&list)?;
            if params.is_empty() {
                return Err(syn::Error::new_spanned(
                    &name,
                    "only a generic shader has instantiations to list; give it type \
                     parameters, as in `Name<T> for [f32, u32]`",
                ));
            }
            instantiations = Some(
                items
                    .iter()
                    .map(|item| Instantiation::parse_item(item, params.len()))
                    .collect::<syn::Result<_>>()?,
            );
        }

        if instantiations.is_none() && params.len() > 1 {
            return Err(syn::Error::new_spanned(
                &name,
                format!(
                    "`{name}` takes {} element types, so there is no default set to register \
                     it for: list the combinations, as in `{name}<{}> for [({})]`",
                    params.len(),
                    params
                        .iter()
                        .map(Ident::to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                    vec!["f32"; params.len()].join(", "),
                ),
            ));
        }

        Ok(Head {
            name,
            params,
            instantiations,
        })
    }
}

impl Instantiation {
    /// One entry of a `for [..]` list. A function over a single type parameter lists bare
    /// types; one over several lists tuples, so that the list constrains the combination
    /// rather than each parameter on its own.
    fn parse_item(item: &Type, arity: usize) -> syn::Result<Self> {
        if arity == 1 {
            return Ok(Instantiation {
                types: vec![item.clone()],
            });
        }
        match item {
            Type::Tuple(tuple) if tuple.elems.len() == arity => Ok(Instantiation {
                types: tuple.elems.iter().cloned().collect(),
            }),
            _ => Err(syn::Error::new_spanned(
                item,
                format!("expected a tuple of {arity} element types, one per type parameter"),
            )),
        }
    }
}

/// The parts of a declaration that describe the shader itself.
pub struct ShaderDecl {
    head: Head,
    path: Expr,
    permutations: Type,
}

impl Parse for ShaderDecl {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let head: Head = input.parse()?;
        input.parse::<Token![,]>()?;
        let path: Expr = input.parse()?;

        let mut permutations: Type = parse_quote!(());
        if input.parse::<Option<Token![,]>>()?.is_some() && !input.is_empty() {
            permutations = input.parse()?;
            input.parse::<Option<Token![,]>>()?;
        }

        Ok(ShaderDecl {
            head,
            path,
            permutations,
        })
    }
}

/// A shader plus the entry point and parameter types that make it callable.
pub struct FunctionDecl {
    head: Head,
    params_ty: Type,
    push_ty: Type,
    spec_ty: Type,
    entry_point: Expr,
    path: Expr,
}

impl Parse for FunctionDecl {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let head: Head = input.parse()?;
        input.parse::<Token![,]>()?;

        let mut params_ty: Type = parse_quote!(());
        let mut spec_ty: Type = parse_quote!(());
        let mut push_ty: Type = parse_quote!(());

        // `params:`, `push:` and `spec:` are each optional and default to `()`. Nothing
        // that may follow them is an identifier followed by a colon, so a lookahead is
        // enough to tell a named argument from the entry point.
        while input.peek(Ident) && input.peek2(Token![:]) {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let value: Type = input.parse()?;
            match key.to_string().as_str() {
                "params" => params_ty = value,
                "push" => push_ty = value,
                "spec" => spec_ty = value,
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!("unknown argument `{other}`, expected `params`, `push` or `spec`"),
                    ));
                }
            }
            input.parse::<Token![,]>()?;
        }

        let entry_point: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let path: Expr = input.parse()?;
        input.parse::<Option<Token![,]>>()?;

        Ok(FunctionDecl {
            head,
            params_ty,
            push_ty,
            spec_ty,
            entry_point,
            path,
        })
    }
}

impl Head {
    fn is_generic(&self) -> bool {
        !self.params.is_empty()
    }

    /// `Name<A, B>`, or `Name` when the shader is not generic.
    fn self_ty(&self) -> TokenStream {
        let name = &self.name;
        if self.is_generic() {
            let params = &self.params;
            quote!(#name<#(#params),*>)
        } else {
            quote!(#name)
        }
    }

    /// The generics and where clause the generated impls carry: every parameter is a
    /// [`DType`], and the combination of them must be one this shader was registered for.
    fn impl_generics(&self) -> (TokenStream, TokenStream) {
        if !self.is_generic() {
            return (quote!(), quote!());
        }
        let params = &self.params;
        let self_ty = self.self_ty();
        let proof = self.proof_ty(&params.iter().map(|p| parse_quote!(#p)).collect::<Vec<_>>());
        (
            quote!(<#(#params),*>),
            quote! {
                where
                    #(#params: ::gpu::DType,)*
                    #proof: ::gpu::DTypeOf<#self_ty>
            },
        )
    }

    /// What carries the proof that an instantiation was registered.
    ///
    /// A single type parameter carries it itself; several carry it on the tuple, so that
    /// listing `(f16, f32)` does not also admit `(f32, f16)`.
    fn proof_ty(&self, types: &[Type]) -> TokenStream {
        if types.len() == 1 {
            let only = &types[0];
            quote!(#only)
        } else {
            quote!((#(#types),*))
        }
    }

    /// The struct, and the shader-module impls every instantiation shares.
    fn shader_module_items(&self, path: &Expr, permutations: &Type) -> TokenStream {
        let name = &self.name;
        let self_ty = self.self_ty();

        if !self.is_generic() {
            return quote! {
                pub struct #name {}

                impl ::gpu::ShaderModuleLike for #name {
                    type Permutations = #permutations;
                }

                impl ::gpu::DynShaderModuleLike for #name {
                    fn manifest_dir(&self) -> &'static str { env!("CARGO_MANIFEST_DIR") }
                    fn relative_path(&self) -> &'static str { #path }
                    fn source_file(&self) -> &'static str { file!() }
                    fn total_permutations(&self) -> usize {
                        ::gpu::ShaderModuleLike::total_permutations(self)
                    }
                    fn defines(&self, index: usize) -> Vec<(&'static str, String)> {
                        ::gpu::ShaderModuleLike::defines(self, index)
                    }
                }
            };
        }

        let params = &self.params;
        let (generics, where_clause) = self.impl_generics();

        // The Slang type arguments the entry point is specialized with, in the order its
        // generic parameters are declared.
        let specialization_args = params.iter().map(|param| {
            quote!(<#param as ::gpu::DType>::slang_name())
        });

        quote! {
            #[allow(dead_code)]
            pub struct #name<#(#params: ::gpu::DType),*>(
                ::core::marker::PhantomData<(#(#params),*)>
            );

            impl #generics ::gpu::ShaderModuleLike for #self_ty #where_clause {
                type Permutations = #permutations;
            }

            impl #generics ::gpu::DynShaderModuleLike for #self_ty #where_clause {
                fn manifest_dir(&self) -> &'static str { env!("CARGO_MANIFEST_DIR") }
                fn relative_path(&self) -> &'static str { #path }
                fn source_file(&self) -> &'static str { file!() }
                fn total_permutations(&self) -> usize {
                    ::gpu::ShaderModuleLike::total_permutations(self)
                }
                fn defines(&self, index: usize) -> Vec<(&'static str, String)> {
                    ::gpu::ShaderModuleLike::defines(self, index)
                }
                fn specialization_args(&self) -> Vec<&'static str> {
                    vec![#(#specialization_args),*]
                }
            }
        }
    }

    /// The registry entries, and the `DTypeOf` impls that make exactly these instantiations
    /// nameable at a call site.
    ///
    /// `inventory` registers through a static in a link section, which cannot be generic and
    /// is not monomorphized, so each entry has to be a concrete item — which is why the
    /// instantiations are enumerated at the declaration rather than discovered from the call
    /// sites, which are not known until the whole crate graph has type-checked.
    fn registrations(&self, with_function: bool) -> TokenStream {
        let name = &self.name;

        if !self.is_generic() {
            let function = with_function.then(|| {
                quote! {
                    ::gpu::inventory::submit! {
                        ::gpu::DeviceFunctionRegistry {
                            function_type: std::any::TypeId::of::<#name>(),
                            instantiate: || -> Box<dyn ::gpu::DynDeviceFunctionLike> {
                                Box::new(#name {})
                            },
                        }
                    }
                }
            });
            return quote! {
                ::gpu::inventory::submit! {
                    ::gpu::ShaderModuleRegistry {
                        type_id: std::any::TypeId::of::<#name>(),
                        instantiate: || -> Box<dyn ::gpu::DynShaderModuleLike> {
                            Box::new(#name {})
                        },
                    }
                }
                #function
            };
        }

        // Without a list, a single type parameter registers for every `DType`. The set
        // lives in `for_each_dtype!` and a proc macro cannot read it, so the expansion
        // hands the work back to that macro.
        let Some(instantiations) = &self.instantiations else {
            let callback = if with_function {
                format_ident!("__register_dtypes_function")
            } else {
                format_ident!("__register_dtypes_shader")
            };
            return quote! {
                ::gpu::for_each_dtype!(#callback!(#name));
            };
        };

        let entries = instantiations.iter().map(|instantiation| {
            let types = &instantiation.types;
            let proof = self.proof_ty(types);
            let function = with_function.then(|| {
                quote! {
                    ::gpu::inventory::submit! {
                        ::gpu::DeviceFunctionRegistry {
                            function_type: std::any::TypeId::of::<#name<#(#types),*>>(),
                            instantiate: || -> Box<dyn ::gpu::DynDeviceFunctionLike> {
                                Box::new(#name::<#(#types),*>(::core::marker::PhantomData))
                            },
                        }
                    }
                }
            });
            quote! {
                impl ::gpu::DTypeOf<#name<#(#types),*>> for #proof {}

                ::gpu::inventory::submit! {
                    ::gpu::ShaderModuleRegistry {
                        type_id: std::any::TypeId::of::<#name<#(#types),*>>(),
                        instantiate: || -> Box<dyn ::gpu::DynShaderModuleLike> {
                            Box::new(#name::<#(#types),*>(::core::marker::PhantomData))
                        },
                    }
                }

                #function
            }
        });

        quote!(#(#entries)*)
    }

    /// The impls that make the shader callable as a device function.
    fn device_function_items(
        &self,
        params_ty: &Type,
        push_ty: &Type,
        spec_ty: &Type,
        entry_point: &Expr,
    ) -> TokenStream {
        let name = &self.name;
        let self_ty = self.self_ty();
        let (generics, where_clause) = self.impl_generics();

        let new_body = if self.is_generic() {
            quote!(#name(::core::marker::PhantomData))
        } else {
            quote!(#name {})
        };

        quote! {
            impl #generics ::gpu::DeviceFunctionLike for #self_ty #where_clause {
                type Shader = #self_ty;
                type Params = #params_ty;
                type PushConstant = #push_ty;
                type SpecConstant = #spec_ty;
            }

            impl #generics ::gpu::DynDeviceFunctionLike for #self_ty #where_clause {
                fn new() -> Self {
                    #new_body
                }

                fn shader_type(&self) -> std::any::TypeId {
                    ::gpu::DeviceFunctionLike::shader_type(self)
                }

                fn parameter_types(&self) -> Vec<::gpu::ShaderParameterType> {
                    ::gpu::DeviceFunctionLike::parameter_types(self)
                }

                fn push_constant_layout(&self) -> ::gpu::TypeLayout {
                    ::gpu::DeviceFunctionLike::push_constant_layout(self)
                }

                fn spec_constant_layout(&self) -> ::gpu::TypeLayout {
                    ::gpu::DeviceFunctionLike::spec_constant_layout(self)
                }

                fn entry_point(&self) -> &'static str {
                    #entry_point
                }
            }
        }
    }
}

pub fn shader_impl(decl: ShaderDecl) -> TokenStream {
    let items = decl
        .head
        .shader_module_items(&decl.path, &decl.permutations);
    let registrations = decl.head.registrations(false);
    quote! {
        #items
        #registrations
    }
}

pub fn function_impl(decl: FunctionDecl) -> TokenStream {
    let permutations: Type = parse_quote!(());
    let shader = decl.head.shader_module_items(&decl.path, &permutations);
    let function = decl.head.device_function_items(
        &decl.params_ty,
        &decl.push_ty,
        &decl.spec_ty,
        &decl.entry_point,
    );
    let registrations = decl.head.registrations(true);
    quote! {
        #shader
        #function
        #registrations
    }
}
