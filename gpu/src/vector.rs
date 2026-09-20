use crate::{ScalarKind, ShaderType, TypeLayout};
use bytemuck::{Pod, Zeroable};

macro_rules! vector {
    ($name:ident, $element:ty, $scalar:expr, $count:literal, [$($component:ident : $index:literal),+]) => {
        #[repr(C)]
        #[derive(Debug, Default, Copy, Clone, PartialEq, Pod, Zeroable)]
        pub struct $name {
            elements: [$element; $count],
        }

        impl $name {
            pub const fn new($($component: $element),+) -> Self {
                Self { elements: [$($component),+] }
            }

            /// Every component set to `value`.
            pub const fn splat(value: $element) -> Self {
                Self { elements: [value; $count] }
            }

            $(
                pub const fn $component(&self) -> $element {
                    self.elements[$index]
                }
            )+

            pub const fn as_array(&self) -> &[$element; $count] {
                &self.elements
            }

            pub const fn as_mut_array(&mut self) -> &mut [$element; $count] {
                &mut self.elements
            }
        }

        impl From<[$element; $count]> for $name {
            fn from(elements: [$element; $count]) -> Self {
                Self { elements }
            }
        }

        impl From<$name> for [$element; $count] {
            fn from(vector: $name) -> Self {
                vector.elements
            }
        }

        impl std::ops::Index<usize> for $name {
            type Output = $element;
            fn index(&self, index: usize) -> &$element {
                &self.elements[index]
            }
        }

        impl std::ops::IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut $element {
                &mut self.elements[index]
            }
        }

        impl ShaderType for $name {
            fn type_layout() -> TypeLayout {
                TypeLayout::Vector {
                    element_type: $scalar,
                    element_count: $count,
                }
            }
        }
    };
}

/// Rust vs. Slang difference:
/// Slang's `bool` is word-sized in a uniform block, while Rust's is byte-sized.
macro_rules! bool_vector {
    ($name:ident, $count:literal, [$($component:ident : $index:literal),+]) => {
        #[repr(C)]
        #[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Pod, Zeroable)]
        pub struct $name {
            elements: [u32; $count],
        }

        impl $name {
            pub const fn new($($component: bool),+) -> Self {
                Self { elements: [$($component as u32),+] }
            }

            /// Every component set to `value`.
            pub const fn splat(value: bool) -> Self {
                Self { elements: [value as u32; $count] }
            }

            $(
                pub const fn $component(&self) -> bool {
                    self.elements[$index] != 0
                }
            )+

            pub const fn as_array(&self) -> &[u32; $count] {
                &self.elements
            }
        }

        impl From<[bool; $count]> for $name {
            fn from(elements: [bool; $count]) -> Self {
                Self { elements: elements.map(|element| element as u32) }
            }
        }

        impl From<$name> for [bool; $count] {
            fn from(vector: $name) -> Self {
                vector.elements.map(|element| element != 0)
            }
        }

        impl std::ops::Index<usize> for $name {
            type Output = u32;
            fn index(&self, index: usize) -> &u32 {
                &self.elements[index]
            }
        }

        impl ShaderType for $name {
            fn type_layout() -> TypeLayout {
                TypeLayout::Vector {
                    element_type: ScalarKind::Bool,
                    element_count: $count,
                }
            }
        }
    };
}

vector!(Float2, f32, ScalarKind::Float32, 2, [x: 0, y: 1]);
vector!(Float3, f32, ScalarKind::Float32, 3, [x: 0, y: 1, z: 2]);
vector!(Float4, f32, ScalarKind::Float32, 4, [x: 0, y: 1, z: 2, w: 3]);

vector!(Double2, f64, ScalarKind::Float64, 2, [x: 0, y: 1]);
vector!(Double3, f64, ScalarKind::Float64, 3, [x: 0, y: 1, z: 2]);
vector!(Double4, f64, ScalarKind::Float64, 4, [x: 0, y: 1, z: 2, w: 3]);

vector!(Int2, i32, ScalarKind::Int32, 2, [x: 0, y: 1]);
vector!(Int3, i32, ScalarKind::Int32, 3, [x: 0, y: 1, z: 2]);
vector!(Int4, i32, ScalarKind::Int32, 4, [x: 0, y: 1, z: 2, w: 3]);

vector!(UInt2, u32, ScalarKind::UInt32, 2, [x: 0, y: 1]);
vector!(UInt3, u32, ScalarKind::UInt32, 3, [x: 0, y: 1, z: 2]);
vector!(UInt4, u32, ScalarKind::UInt32, 4, [x: 0, y: 1, z: 2, w: 3]);

bool_vector!(Bool2, 2, [x: 0, y: 1]);
bool_vector!(Bool3, 3, [x: 0, y: 1, z: 2]);
bool_vector!(Bool4, 4, [x: 0, y: 1, z: 2, w: 3]);

#[cfg(test)]
mod tests {}
