//! `shader!` and `function!`, which declare a device function and register every
//! instantiation of it ahead of time.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Token, Type, bracketed, parenthesized, parse_quote};

/// One generic parameter of a declaration.
pub enum Param {
    /// `T`, an element type: a [`DType`], passed to slangc as its wrapper's name.
    Type(Ident),
    /// `const N: u32`, a value the entry point is specialized with, passed to slangc as a
    /// literal for a Slang `let N : uint` parameter.
    Const {
        name: Ident,
        ty: Ident,
        /// The `gpu::Const*` type that carries the value into the registration proof.
        marker: Ident,
    },
}

impl Param {
    fn name(&self) -> &Ident {
        match self {
            Param::Type(name) | Param::Const { name, .. } => name,
        }
    }

    /// How the parameter is declared on the generated struct and its impls.
    fn declaration(&self, bounded: bool) -> TokenStream {
        match self {
            Param::Type(name) if bounded => quote!(#name: ::gpu::DType),
            Param::Type(name) => quote!(#name),
            Param::Const { name, ty, .. } => quote!(const #name: #ty),
        }
    }

    /// What stands for this parameter in the tuple that proves an instantiation was
    /// registered: the type itself, or a value lifted into its marker type.
    fn proof(&self, arg: TokenStream) -> TokenStream {
        match self {
            Param::Type(_) => arg,
            Param::Const { marker, .. } => quote!(::gpu::#marker<#arg>),
        }
    }
}

impl Parse for Param {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.parse::<Option<Token![const]>>()?.is_none() {
            return Ok(Param::Type(input.parse()?));
        }
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty: Ident = input.parse()?;
        // The scalars a Slang `let` parameter can have, and that Rust allows as the type of
        // a const generic.
        let marker = match ty.to_string().as_str() {
            "bool" => "ConstBool",
            "i8" => "ConstI8",
            "u8" => "ConstU8",
            "i16" => "ConstI16",
            "u16" => "ConstU16",
            "i32" => "ConstI32",
            "u32" => "ConstU32",
            "i64" => "ConstI64",
            "u64" => "ConstU64",
            _ => {
                return Err(syn::Error::new_spanned(
                    &ty,
                    "a `const` parameter is one of bool, i8, u8, i16, u16, i32, u32, i64 or \
                     u64, the scalars a Slang `let` parameter can have",
                ));
            }
        };
        Ok(Param::Const {
            name,
            ty,
            marker: format_ident!("{marker}"),
        })
    }
}

/// One argument of an instantiation: an element type for a type parameter, or a value for a
/// `const` one.
pub enum Arg {
    Type(Type),
    Const(Expr),
}

impl Arg {
    fn parse_for(param: &Param, input: ParseStream) -> syn::Result<Self> {
        match param {
            Param::Type(_) => Ok(Arg::Type(input.parse()?)),
            Param::Const { .. } => Ok(Arg::Const(input.parse()?)),
        }
    }

    /// The argument as it is written between `Name<..>`. A value is braced, since only a
    /// literal or a lone identifier may stand there bare, and `-1` is neither.
    fn tokens(&self) -> TokenStream {
        match self {
            Arg::Type(ty) => quote!(#ty),
            Arg::Const(value) => quote!({ #value }),
        }
    }
}

/// The arguments one instantiation is built for, one per generic parameter.
pub struct Instantiation {
    args: Vec<Arg>,
}

/// `Name`, `Name<T>`, `Name<Src, Dst> for [(u32, f32), (f32, f32)]` or
/// `Name<T, const N: u32> for [(f32, 4), (u32, 8)]`.
pub struct Head {
    name: Ident,
    /// The declared generic parameters. Empty for a non-generic shader.
    params: Vec<Param>,
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
            let for_token = input.parse::<Token![for]>()?;
            if params.is_empty() {
                return Err(syn::Error::new_spanned(
                    for_token,
                    "only a generic shader has instantiations to list; give it generic \
                     parameters, as in `Name<T> for [f32, u32]`",
                ));
            }
            let list;
            bracketed!(list in input);
            let mut items = Vec::new();
            while !list.is_empty() {
                items.push(Instantiation::parse_item(&list, &params)?);
                if list.is_empty() {
                    break;
                }
                list.parse::<Token![,]>()?;
            }
            instantiations = Some(items);
        }

        if instantiations.is_none() {
            let has_default = match params.as_slice() {
                [] | [Param::Type(_)] => true,
                _ => false,
            };
            if !has_default {
                let names = params
                    .iter()
                    .map(|param| param.name().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(syn::Error::new_spanned(
                    &name,
                    format!(
                        "`{name}<{names}>` has no default set of instantiations to register: \
                         only a single element type does. List the ones to build, as in \
                         `{name}<..> for [(f32, 4), (u32, 8)]`",
                    ),
                ));
            }
        }

        Ok(Head {
            name,
            params,
            instantiations,
        })
    }
}

impl Instantiation {
    /// One entry of a `for [..]` list. A function over a single generic parameter lists
    /// bare arguments; one over several lists tuples, so that the list constrains the
    /// combination rather than each parameter on its own.
    fn parse_item(input: ParseStream, params: &[Param]) -> syn::Result<Self> {
        if let [param] = params {
            return Ok(Instantiation {
                args: vec![Arg::parse_for(param, input)?],
            });
        }

        let arity = params.len();
        let expected = || format!("expected a tuple of {arity} arguments, one per generic parameter");
        if !input.peek(syn::token::Paren) {
            return Err(input.error(expected()));
        }
        let tuple;
        parenthesized!(tuple in input);
        let mut args = Vec::with_capacity(arity);
        for (index, param) in params.iter().enumerate() {
            if index > 0 {
                tuple.parse::<Token![,]>().map_err(|_| tuple.error(expected()))?;
            }
            args.push(Arg::parse_for(param, &tuple)?);
        }
        tuple.parse::<Option<Token![,]>>()?;
        if !tuple.is_empty() {
            return Err(tuple.error(expected()));
        }
        Ok(Instantiation { args })
    }
}

/// Parses the `key: value` arguments that follow a declaration's head, in any order.
///
/// `arg` parses the value for `key` into its slot and reports whether that slot was
/// already filled, or returns `None` for a key it does not know; `expected` lists the
/// known keys for the error message.
fn parse_named_args(
    input: ParseStream,
    expected: &str,
    mut arg: impl FnMut(&str, ParseStream) -> syn::Result<Option<bool>>,
) -> syn::Result<()> {
    while input.parse::<Option<Token![,]>>()?.is_some() && !input.is_empty() {
        let key: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        match arg(&key.to_string(), input)? {
            Some(false) => {}
            Some(true) => {
                return Err(syn::Error::new_spanned(
                    &key,
                    format!("`{key}` is given more than once"),
                ));
            }
            None => {
                return Err(syn::Error::new_spanned(
                    &key,
                    format!("unknown argument `{key}`, expected {expected}"),
                ));
            }
        }
    }

    if !input.is_empty() {
        return Err(input.error("expected `,`"));
    }
    Ok(())
}

/// Fills `slot` from `input`, reporting whether it was already filled.
fn set<T: Parse>(slot: &mut Option<T>, input: ParseStream) -> syn::Result<Option<bool>> {
    Ok(Some(slot.replace(input.parse()?).is_some()))
}

fn missing(head: &Head, arg: &str, example: &str) -> syn::Error {
    syn::Error::new_spanned(
        &head.name,
        format!("missing `{arg}` argument, as in `{arg}: {example}`"),
    )
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

        let mut path: Option<Expr> = None;
        let mut permutations: Option<Type> = None;

        // `path` is required; `permutations` defaults to `()`.
        parse_named_args(input, "`path` or `permutations`", |key, input| match key {
            "path" => set(&mut path, input),
            "permutations" => set(&mut permutations, input),
            _ => Ok(None),
        })?;

        let path = path.ok_or_else(|| missing(&head, "path", "\"shader.slang\""))?;

        Ok(ShaderDecl {
            head,
            path,
            permutations: permutations.unwrap_or_else(|| parse_quote!(())),
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

        let mut params_ty: Option<Type> = None;
        let mut push_ty: Option<Type> = None;
        let mut spec_ty: Option<Type> = None;
        let mut entry_point: Option<Expr> = None;
        let mut path: Option<Expr> = None;

        // `name` and `path` are required; `params`, `push` and `spec` default to `()`.
        parse_named_args(
            input,
            "`name`, `path`, `params`, `push` or `spec`",
            |key, input| match key {
                "params" => set(&mut params_ty, input),
                "push" => set(&mut push_ty, input),
                "spec" => set(&mut spec_ty, input),
                "name" => set(&mut entry_point, input),
                "path" => set(&mut path, input),
                _ => Ok(None),
            },
        )?;

        let entry_point = entry_point.ok_or_else(|| missing(&head, "name", "\"main\""))?;
        let path = path.ok_or_else(|| missing(&head, "path", "\"shader.slang\""))?;

        let unit = || parse_quote!(());
        Ok(FunctionDecl {
            head,
            params_ty: params_ty.unwrap_or_else(unit),
            push_ty: push_ty.unwrap_or_else(unit),
            spec_ty: spec_ty.unwrap_or_else(unit),
            entry_point,
            path,
        })
    }
}

impl Head {
    fn is_generic(&self) -> bool {
        !self.params.is_empty()
    }

    /// `Name<A, N>`, or `Name` when the shader is not generic.
    fn self_ty(&self) -> TokenStream {
        let name = &self.name;
        if self.is_generic() {
            let names = self.params.iter().map(Param::name);
            quote!(#name<#(#names),*>)
        } else {
            quote!(#name)
        }
    }

    /// The type parameters alone, without the `const` ones.
    fn type_params(&self) -> impl Iterator<Item = &Ident> {
        self.params.iter().filter_map(|param| match param {
            Param::Type(name) => Some(name),
            Param::Const { .. } => None,
        })
    }

    /// The generics and where clause the generated impls carry: every type parameter is a
    /// [`DType`], and the combination of arguments must be one this shader was registered
    /// for.
    fn impl_generics(&self) -> (TokenStream, TokenStream) {
        if !self.is_generic() {
            return (quote!(), quote!());
        }
        let declarations = self.params.iter().map(|param| param.declaration(false));
        let type_params = self.type_params();
        let self_ty = self.self_ty();
        let proof = self.proof_ty(self.params.iter().map(|param| {
            let name = param.name();
            quote!(#name)
        }));
        (
            quote!(<#(#declarations),*>),
            quote! {
                where
                    #(#type_params: ::gpu::DType,)*
                    #proof: ::gpu::DTypeOf<#self_ty>
            },
        )
    }

    /// What carries the proof that an instantiation was registered, given one argument per
    /// generic parameter.
    ///
    /// A single type parameter carries it itself; several carry it on the tuple, so that
    /// listing `(f16, f32)` does not also admit `(f32, f16)`. A value is lifted into its
    /// marker type first, since a tuple can only hold types.
    fn proof_ty(&self, args: impl IntoIterator<Item = TokenStream>) -> TokenStream {
        let elements: Vec<_> = self
            .params
            .iter()
            .zip(args)
            .map(|(param, arg)| param.proof(arg))
            .collect();
        if let [only] = elements.as_slice() {
            quote!(#only)
        } else {
            quote!((#(#elements),*))
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

        let declarations = self.params.iter().map(|param| param.declaration(true));
        let type_params = self.type_params();
        let (generics, where_clause) = self.impl_generics();

        // The Slang generic arguments the entry point is specialized with, in the order its
        // generic parameters are declared: an element type's wrapper name, or a value as a
        // decimal literal, which slangc converts to the `let` parameter's type.
        let specialization_args = self.params.iter().map(|param| match param {
            Param::Type(name) => quote!(<#name as ::gpu::DType>::slang_name().to_string()),
            Param::Const { name, .. } => quote!(#name.to_string()),
        });

        quote! {
            #[allow(dead_code)]
            pub struct #name<#(#declarations),*>(
                ::core::marker::PhantomData<(#(#type_params,)*)>
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
                fn specialization_args(&self) -> Vec<String> {
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
            let types: Vec<_> = instantiation.args.iter().map(Arg::tokens).collect();
            let proof = self.proof_ty(types.iter().cloned());
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
