use crate::{AnyRgBuffer, RgBuffer, ShaderType, TypeLayout, describe_tagged, tag_buffer_ref};
use ash::vk;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    ConstantBuffer,
    StructuredBuffer,
    RWStructuredBuffer,
}

impl DescriptorKind {
    pub fn is_buffer(&self) -> bool {
        match self {
            DescriptorKind::ConstantBuffer
            | DescriptorKind::StructuredBuffer
            | DescriptorKind::RWStructuredBuffer => true,
        }
    }
}

impl From<DescriptorKind> for vk::DescriptorType {
    fn from(value: DescriptorKind) -> Self {
        match value {
            DescriptorKind::ConstantBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            DescriptorKind::StructuredBuffer | DescriptorKind::RWStructuredBuffer => {
                vk::DescriptorType::STORAGE_BUFFER
            }
        }
    }
}

pub trait Descriptor {
    fn kind() -> DescriptorKind;

    fn layout() -> Option<fn() -> TypeLayout> {
        None
    }

    fn handle(&self) -> AnyRgBuffer;
}

pub struct ConstantBuffer<T: ShaderType> {
    handle: RgBuffer<T>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ShaderType> Descriptor for ConstantBuffer<T> {
    fn kind() -> DescriptorKind {
        DescriptorKind::ConstantBuffer
    }
    fn layout() -> Option<fn() -> TypeLayout> {
        Some(<T as ShaderType>::type_layout)
    }
    fn handle(&self) -> AnyRgBuffer {
        self.handle.into()
    }
}

pub struct StructuredBuffer<T: ShaderType> {
    handle: RgBuffer<T>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ShaderType> Descriptor for StructuredBuffer<T> {
    fn kind() -> DescriptorKind {
        DescriptorKind::StructuredBuffer
    }
    fn layout() -> Option<fn() -> TypeLayout> {
        Some(<T as ShaderType>::type_layout)
    }
    fn handle(&self) -> AnyRgBuffer {
        self.handle.into()
    }
}

pub struct RWStructuredBuffer<T: ShaderType> {
    handle: RgBuffer<T>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ShaderType> Descriptor for RWStructuredBuffer<T> {
    fn kind() -> DescriptorKind {
        DescriptorKind::RWStructuredBuffer
    }
    fn layout() -> Option<fn() -> TypeLayout> {
        Some(<T as ShaderType>::type_layout)
    }
    fn handle(&self) -> AnyRgBuffer {
        self.handle.into()
    }
}

impl<T: ShaderType> From<RgBuffer<T>> for ConstantBuffer<T> {
    fn from(value: RgBuffer<T>) -> Self {
        Self {
            handle: value,
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType> From<RgBuffer<T>> for StructuredBuffer<T> {
    fn from(value: RgBuffer<T>) -> Self {
        Self {
            handle: value,
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType> From<RgBuffer<T>> for RWStructuredBuffer<T> {
    fn from(value: RgBuffer<T>) -> Self {
        Self {
            handle: value,
            _marker: PhantomData,
        }
    }
}

#[repr(transparent)]
pub struct DeviceAddress<T: ShaderType> {
    tagged: u64,
    _marker: PhantomData<fn() -> T>,
}

#[repr(transparent)]
pub struct RWDeviceAddress<T: ShaderType> {
    tagged: u64,
    _marker: PhantomData<fn() -> T>,
}

macro_rules! address_copy {
    ($ty:ident) => {
        impl<T: ShaderType> Clone for $ty<T> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<T: ShaderType> Copy for $ty<T> {}
    };
}
address_copy!(DeviceAddress);
address_copy!(RWDeviceAddress);

macro_rules! address_debug {
    ($ty:ident) => {
        impl<T: ShaderType> std::fmt::Debug for $ty<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    concat!(stringify!($ty), "({})"),
                    describe_tagged(self.tagged)
                )
            }
        }
    };
}
address_debug!(DeviceAddress);
address_debug!(RWDeviceAddress);

unsafe impl<T: ShaderType + 'static> bytemuck::Zeroable for DeviceAddress<T> {}
unsafe impl<T: ShaderType + 'static> bytemuck::Pod for DeviceAddress<T> {}
unsafe impl<T: ShaderType + 'static> bytemuck::Zeroable for RWDeviceAddress<T> {}
unsafe impl<T: ShaderType + 'static> bytemuck::Pod for RWDeviceAddress<T> {}

impl<T: ShaderType> From<RgBuffer<T>> for DeviceAddress<T> {
    fn from(value: RgBuffer<T>) -> Self {
        Self {
            tagged: tag_buffer_ref(value),
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType> From<RgBuffer<T>> for RWDeviceAddress<T> {
    fn from(value: RgBuffer<T>) -> Self {
        Self {
            tagged: tag_buffer_ref(value),
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType> ShaderType for DeviceAddress<T> {
    fn type_layout() -> TypeLayout {
        TypeLayout::DeviceAddress {
            pointee: Box::new(T::type_layout()),
            writable: false,
        }
    }
}

impl<T: ShaderType> ShaderType for RWDeviceAddress<T> {
    fn type_layout() -> TypeLayout {
        TypeLayout::DeviceAddress {
            pointee: Box::new(T::type_layout()),
            writable: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShaderParameterType {
    pub name: &'static str,
    pub kind: DescriptorKind,
    pub layout: Option<fn() -> TypeLayout>,
}

#[derive(Clone, Copy, PartialEq)]
pub struct ShaderParameter {
    pub name: &'static str,
    pub kind: DescriptorKind,
    pub buffer: AnyRgBuffer,
}

pub trait ShaderParametersTrait: Sized {
    fn parameter_types() -> Vec<ShaderParameterType>;
    fn parameters(&self) -> Vec<ShaderParameter>;
}

impl ShaderParametersTrait for () {
    fn parameter_types() -> Vec<ShaderParameterType> {
        Vec::new()
    }
    fn parameters(&self) -> Vec<ShaderParameter> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {}
