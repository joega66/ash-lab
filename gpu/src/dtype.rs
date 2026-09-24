use crate::{ScalarKind, ShaderType, bytemuck};

/// A scalar element type that a generic device function may be instantiated over.
///
/// The set is closed, and [`for_each_dtype!`] is the one place it is written down. A generic
/// `function!` is registered, compiled and layout-checked once per member of that set,
/// which is what lets an instantiation be built ahead of time: call sites are only known
/// after the whole crate graph type-checks, but the domain of `T` is known at its
/// declaration.
pub trait DType: ShaderType + bytemuck::Pod + 'static {
    const KIND: ScalarKind;

    /// How Slang spells this type: the name of its wrapper struct in `dtype.slang`, which
    /// is the `-specialize` argument the entry point is specialized with and the tag
    /// separating this instantiation's build artifacts.
    ///
    /// A kernel is instantiated with the wrapper rather than the bare scalar because
    /// `slangc -specialize` only accepts a type that declares its interfaces directly: an
    /// `extension float : IDType` is not in scope when it resolves the argument. The
    /// wrapper holds exactly one scalar, so it has that scalar's layout and the host still
    /// binds a plain `f32`/`i32`/`u32` against it.
    fn slang_name() -> &'static str {
        Self::KIND.wrapper_name().unwrap_or_else(|| {
            panic!(
                "`{}` has no element-type wrapper in dtype.slang",
                Self::KIND.source_name()
            )
        })
    }
}

/// Proof that a generic device function `F` was registered for this element type, or for
/// this combination of them.
///
/// A generic `function!` declares the impls for the entries it lists, so anything outside
/// that list leaves `F` without a [`DeviceFunctionLike`](crate::DeviceFunctionLike) impl
/// and the call site fails to compile, rather than dispatching to a pipeline that was never
/// built.
///
/// A function over one type parameter carries the proof on the type itself; one over
/// several carries it on the tuple, so that the list constrains the *combination* rather
/// than each parameter separately. `Gemm<f16, f32>` can be registered without
/// `Gemm<f32, f16>` coming along with it.
#[diagnostic::on_unimplemented(
    message = "`{F}` was not registered for `{Self}`",
    label = "no pipeline was compiled for this element type",
    note = "add `{Self}` to the `for [..]` list on the `function!` that declares `{F}`"
)]
pub trait DTypeOf<F: ?Sized> {}

/// Expands `$callback!` once for the whole set of [`DType`]s, appending
/// `; [<rust type> => <ScalarKind>] ...` to the arguments it is given.
///
/// Adding a type here instantiates every generic device function for it, so the set is kept
/// to the scalars shaders actually use. A function that should not cover all of them names
/// its own subset instead, with `function!(Name<T> for [f32, u32], ..)`.
///
/// ```ignore
/// for_each_dtype!(my_macro!(some, args));
/// // expands to:
/// my_macro!(some, args; [f32 => ScalarKind::Float32] [i32 => ...] [u32 => ...]);
/// ```
#[macro_export]
macro_rules! for_each_dtype {
    ($callback:ident ! ( $($args:tt)* )) => {
        $crate::$callback!(
            $($args)* ;
            [f32 => $crate::ScalarKind::Float32]
            [i32 => $crate::ScalarKind::Int32]
            [u32 => $crate::ScalarKind::UInt32]
        );
    };
}

/// Registers a generic shader module for every [`DType`], for the `Name<T>` form that
/// names no list of its own.
///
/// `function!` cannot enumerate the set itself — a proc macro has no way to read the list
/// in [`for_each_dtype!`] — so it hands the work back here.
#[doc(hidden)]
#[macro_export]
macro_rules! __register_dtypes_shader {
    ($ty:ident ; $([$dtype:ty => $kind:expr])+) => {
        $(
            impl $crate::DTypeOf<$ty<$dtype>> for $dtype {}

            $crate::inventory::submit! {
                $crate::ShaderModuleRegistry {
                    type_id: std::any::TypeId::of::<$ty<$dtype>>(),
                    instantiate: || -> Box<dyn $crate::DynShaderModuleLike> {
                        Box::new($ty::<$dtype>(::core::marker::PhantomData))
                    },
                }
            }
        )+
    };
}

/// [`__register_dtypes_shader!`], plus the device function entry that makes each
/// instantiation callable.
#[doc(hidden)]
#[macro_export]
macro_rules! __register_dtypes_function {
    ($ty:ident ; $([$dtype:ty => $kind:expr])+) => {
        $crate::__register_dtypes_shader!($ty ; $([$dtype => $kind])+);
        $(
            $crate::inventory::submit! {
                $crate::DeviceFunctionRegistry {
                    function_type: std::any::TypeId::of::<$ty<$dtype>>(),
                    instantiate: || -> Box<dyn $crate::DynDeviceFunctionLike> {
                        Box::new($ty::<$dtype>(::core::marker::PhantomData))
                    },
                }
            }
        )+
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __impl_dtype {
    (; $([$ty:ty => $kind:expr])+) => {
        $(
            impl $crate::DType for $ty {
                const KIND: $crate::ScalarKind = $kind;
            }
        )+
    };
}

for_each_dtype!(__impl_dtype!());
