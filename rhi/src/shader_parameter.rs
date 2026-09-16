use crate::{DeviceBuffer, DeviceBufferInner, ShaderType, TypeLayout, utils::Handle};
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

    fn handle(&self) -> Handle<DeviceBufferInner>;
}

pub struct ConstantBuffer<T: ShaderType> {
    buffer: DeviceBuffer<T>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ShaderType> Descriptor for ConstantBuffer<T> {
    fn kind() -> DescriptorKind {
        DescriptorKind::ConstantBuffer
    }
    fn layout() -> Option<fn() -> TypeLayout> {
        Some(<T as ShaderType>::type_layout)
    }
    fn handle(&self) -> Handle<DeviceBufferInner> {
        self.buffer.handle()
    }
}

pub struct StructuredBuffer<T: ShaderType> {
    buffer: DeviceBuffer<T>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ShaderType> Descriptor for StructuredBuffer<T> {
    fn kind() -> DescriptorKind {
        DescriptorKind::StructuredBuffer
    }
    fn layout() -> Option<fn() -> TypeLayout> {
        Some(<T as ShaderType>::type_layout)
    }
    fn handle(&self) -> Handle<DeviceBufferInner> {
        self.buffer.handle()
    }
}

pub struct RWStructuredBuffer<T: ShaderType> {
    buffer: DeviceBuffer<T>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ShaderType> Descriptor for RWStructuredBuffer<T> {
    fn kind() -> DescriptorKind {
        DescriptorKind::RWStructuredBuffer
    }
    fn layout() -> Option<fn() -> TypeLayout> {
        Some(<T as ShaderType>::type_layout)
    }
    fn handle(&self) -> Handle<DeviceBufferInner> {
        self.buffer.handle()
    }
}

impl<T: ShaderType> From<DeviceBuffer<T>> for ConstantBuffer<T> {
    fn from(buffer: DeviceBuffer<T>) -> Self {
        Self {
            buffer,
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType> From<DeviceBuffer<T>> for StructuredBuffer<T> {
    fn from(buffer: DeviceBuffer<T>) -> Self {
        Self {
            buffer,
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType> From<DeviceBuffer<T>> for RWStructuredBuffer<T> {
    fn from(buffer: DeviceBuffer<T>) -> Self {
        Self {
            buffer,
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

macro_rules! impl_address_copy {
    ($ty:ident) => {
        impl<T: ShaderType> Clone for $ty<T> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<T: ShaderType> Copy for $ty<T> {}
    };
}
impl_address_copy!(DeviceAddress);
impl_address_copy!(RWDeviceAddress);

unsafe impl<T: ShaderType + 'static> bytemuck::Zeroable for DeviceAddress<T> {}
unsafe impl<T: ShaderType + 'static> bytemuck::Pod for DeviceAddress<T> {}
unsafe impl<T: ShaderType + 'static> bytemuck::Zeroable for RWDeviceAddress<T> {}
unsafe impl<T: ShaderType + 'static> bytemuck::Pod for RWDeviceAddress<T> {}

impl<T: ShaderType> From<&DeviceBuffer<T>> for DeviceAddress<T> {
    fn from(value: &DeviceBuffer<T>) -> Self {
        Self {
            tagged: value.handle().as_u64(),
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType> From<&DeviceBuffer<T>> for RWDeviceAddress<T> {
    fn from(value: &DeviceBuffer<T>) -> Self {
        Self {
            tagged: value.handle().as_u64(),
            _marker: PhantomData,
        }
    }
}

/// Cast a DeviceBuffer to DeviceAddress/RWDeviceAddress
#[macro_export]
macro_rules! address {
    ($buffer:expr) => {
        (&$buffer).into()
    };
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
    pub handle: Handle<DeviceBufferInner>,
}

pub trait DynShaderParameters: Sized {
    fn parameter_types() -> Vec<ShaderParameterType>;
    fn parameters(&self) -> Vec<ShaderParameter>;
}

impl DynShaderParameters for () {
    fn parameter_types() -> Vec<ShaderParameterType> {
        Vec::new()
    }
    fn parameters(&self) -> Vec<ShaderParameter> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {}
