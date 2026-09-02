use std::marker::PhantomData;

pub trait ShaderDefine {
    const NAME: &'static str;
}

/// Give a shader permutation dimension its preprocessor define name.
/// Example usage:
/// ```
/// shader_define!(EnableShadows, "ENABLE_SHADOWS");
/// ```
#[macro_export]
macro_rules! shader_define {
    ($ty:ident, $name:expr) => {
        pub struct $ty;

        impl $crate::ShaderDefine for $ty {
            const NAME: &'static str = $name;
        }
    };
}

/// A shader permutation dimension, i.e. one preprocessor `#define`.
pub trait ShaderPermutationDimension: Sized {
    fn name() -> &'static str;

    fn len() -> usize;

    fn value(&self) -> usize;

    fn define_value(&self) -> String;

    fn from_index(index: usize) -> Self;
}

pub struct ShaderPermutationBool<D: ShaderDefine> {
    value: bool,
    _define: PhantomData<D>,
}

impl<D: ShaderDefine> ShaderPermutationBool<D> {
    pub fn new(value: bool) -> Self {
        Self {
            value,
            _define: PhantomData,
        }
    }

    pub fn set(&mut self, value: bool) -> &mut Self {
        self.value = value;
        self
    }
}

impl<D: ShaderDefine> ShaderPermutationDimension for ShaderPermutationBool<D> {
    fn name() -> &'static str {
        D::NAME
    }

    fn len() -> usize {
        2
    }

    fn value(&self) -> usize {
        self.value as usize
    }

    fn define_value(&self) -> String {
        (self.value as usize).to_string()
    }

    fn from_index(index: usize) -> Self {
        Self::new(index != 0)
    }
}

/// An integer permutation dimension over the inclusive range `MIN..=MAX`
pub struct ShaderPermutationInt<D: ShaderDefine, const MIN: i32, const MAX: i32> {
    value: i32,
    _define: PhantomData<D>,
}

impl<D: ShaderDefine, const MIN: i32, const MAX: i32> ShaderPermutationInt<D, MIN, MAX> {
    pub fn new(value: i32) -> Self {
        assert!(
            MIN <= MAX,
            "shader permutation `{}` has an empty range",
            D::NAME
        );
        assert!(
            (MIN..=MAX).contains(&value),
            "shader permutation `{}` value {value} outside of range {MIN}..={MAX}",
            D::NAME
        );
        Self {
            value,
            _define: PhantomData,
        }
    }

    pub fn set(&mut self, value: i32) -> &mut Self {
        assert!(
            (MIN..=MAX).contains(&value),
            "shader permutation `{}` value {value} outside of range {MIN}..={MAX}",
            D::NAME
        );
        self.value = value;
        self
    }
}

impl<D: ShaderDefine, const MIN: i32, const MAX: i32> ShaderPermutationDimension
    for ShaderPermutationInt<D, MIN, MAX>
{
    fn name() -> &'static str {
        D::NAME
    }

    fn len() -> usize {
        (MAX - MIN + 1) as usize
    }

    fn value(&self) -> usize {
        (self.value - MIN) as usize
    }

    fn define_value(&self) -> String {
        self.value.to_string()
    }

    fn from_index(index: usize) -> Self {
        Self::new(MIN + index as i32)
    }
}

/// Implemented via [`shader_permutation_enum!`] rather than by hand.
pub trait ShaderPermutationEnum: Copy + PartialEq + 'static {
    const VARIANTS: &'static [Self];

    fn define_value(&self) -> &'static str;
}

pub struct ShaderPermutationEnumValue<D: ShaderDefine, E: ShaderPermutationEnum> {
    value: E,
    _define: PhantomData<D>,
}

impl<D: ShaderDefine, E: ShaderPermutationEnum> ShaderPermutationEnumValue<D, E> {
    pub fn new(value: E) -> Self {
        Self {
            value,
            _define: PhantomData,
        }
    }

    pub fn set(&mut self, value: E) -> &mut Self {
        self.value = value;
        self
    }
}

impl<D: ShaderDefine, E: ShaderPermutationEnum> ShaderPermutationDimension
    for ShaderPermutationEnumValue<D, E>
{
    fn name() -> &'static str {
        D::NAME
    }

    fn len() -> usize {
        E::VARIANTS.len()
    }

    fn value(&self) -> usize {
        E::VARIANTS
            .iter()
            .position(|v| v == &self.value)
            .expect("enum value missing from its own `ShaderPermutationEnum::VARIANTS`")
    }

    fn define_value(&self) -> String {
        self.value.define_value().to_string()
    }

    fn from_index(index: usize) -> Self {
        Self::new(E::VARIANTS[index])
    }
}

/// Implements [`ShaderPermutationEnum`] for an enum, using each
/// variant's identifier as its define value.
/// Example usage:
/// ```
/// use rhi_reflect::{ShaderPermutation};
/// use rhi::{shader_permutation_enum};
/// #[derive(Clone, Copy, PartialEq)]
/// enum LightingModel {
///     Phong,
///     Pbr,
///     Unlit,
/// }
/// shader_permutation_enum!(LightingModel { Phong, Pbr, Unlit });
/// ```
#[macro_export]
macro_rules! shader_permutation_enum {
    ($ty:ty { $($variant:ident),+ $(,)? }) => {
        impl $crate::ShaderPermutationEnum for $ty {
            const VARIANTS: &'static [Self] = &[$(Self::$variant),+];

            fn define_value(&self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($variant)),+
                }
            }
        }
    };
}

/// A shader module's full set of permutation dimensions, flattened into a
/// single linear index used to select a shader variant at
/// runtime.
///
/// Should be a plain struct whose fields all implement [`ShaderPermutationDimension`]
///
/// Example usage:
/// ```
/// use rhi_reflect::{ShaderPermutation};
/// use rhi::{shader_define, ShaderPermutationBool, ShaderPermutationInt};
/// shader_define!(EnableShadows, "ENABLE_SHADOWS");
/// shader_define!(CascadeCount, "CASCADE_COUNT");
///
/// #[derive(ShaderPermutation)]
/// struct MyPermutations {
///     shadows: ShaderPermutationBool<EnableShadows>,
///     cascades: ShaderPermutationInt<CascadeCount, 1, 4>,
/// }
/// ```
pub trait ShaderPermutationMatrix: Sized {
    fn flatten(&self) -> usize;

    fn total_permutations() -> usize;

    fn defines(&self) -> Vec<(&'static str, String)>;

    fn from_flat_index(index: usize) -> Self;
}

/// Unit shader permutation with no preprocessor defines.
impl ShaderPermutationMatrix for () {
    fn flatten(&self) -> usize {
        0
    }

    fn total_permutations() -> usize {
        1
    }

    fn defines(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    fn from_flat_index(_index: usize) -> Self {}
}

#[cfg(test)]
mod tests {}
