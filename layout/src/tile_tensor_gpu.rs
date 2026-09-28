use core::mem::{offset_of, size_of};

use gpu::{
    DeviceAddress, DeviceBuffer, FieldLayout, RWDeviceAddress, ShaderType, TypeLayout, bytemuck,
};

use crate::tile_tensor::{DevicePointerEngine, Mutable, TensorAccess, TensorLayout, TileTensor};

macro_rules! impl_device_tensor {
    ($name:ident, $address:ident, $to_address:ident, $doc:literal) => {
        #[doc = $doc]
        #[repr(C)]
        pub struct $name<DType: ShaderType, LayoutType> {
            storage: $address<DType>,
            layout: LayoutType,
        }

        impl<DType: ShaderType, LayoutType> $name<DType, LayoutType> {
            pub fn new(buffer: &DeviceBuffer<DType>, layout: LayoutType) -> Self {
                Self {
                    storage: buffer.$to_address(),
                    layout,
                }
            }

            /// The address the shader will dereference.
            pub fn storage(&self) -> $address<DType>
            where
                $address<DType>: Copy,
            {
                self.storage
            }

            /// The layout the shader will index with.
            pub fn layout(&self) -> LayoutType
            where
                LayoutType: Copy,
            {
                self.layout
            }
        }

        impl<DType: ShaderType, LayoutType: Clone> Clone for $name<DType, LayoutType> {
            fn clone(&self) -> Self {
                Self {
                    storage: self.storage,
                    layout: self.layout.clone(),
                }
            }
        }

        impl<DType: ShaderType, LayoutType: Copy> Copy for $name<DType, LayoutType> {}

        unsafe impl<DType, LayoutType> bytemuck::Zeroable for $name<DType, LayoutType>
        where
            DType: ShaderType,
            LayoutType: bytemuck::Zeroable,
        {
        }

        unsafe impl<DType, LayoutType> bytemuck::Pod for $name<DType, LayoutType>
        where
            DType: ShaderType + 'static,
            LayoutType: bytemuck::Pod,
        {
        }

        impl<DType: ShaderType, LayoutType: ShaderType> ShaderType for $name<DType, LayoutType> {
            fn type_layout() -> TypeLayout {
                TypeLayout::Struct {
                    name: stringify!($name),
                    size: size_of::<Self>() as u32,
                    fields: vec![
                        FieldLayout {
                            name: "storage",
                            offset: offset_of!(Self, storage) as u32,
                            ty: <$address<DType> as ShaderType>::type_layout(),
                        },
                        FieldLayout {
                            name: "layout",
                            offset: offset_of!(Self, layout) as u32,
                            ty: LayoutType::type_layout(),
                        },
                    ],
                }
            }
        }
    };
}

impl_device_tensor!(
    DeviceTensor,
    DeviceAddress,
    read_only,
    "A tensor a shader may read."
);
impl_device_tensor!(
    RWDeviceTensor,
    RWDeviceAddress,
    read_write,
    "A tensor a shader may read and write."
);

impl<DType, LayoutType, Access> TileTensor<'_, DType, LayoutType, DevicePointerEngine, Access>
where
    DType: ShaderType,
    LayoutType: TensorLayout + Copy,
    Access: TensorAccess,
{
    /// This tensor as a shader argument the shader only reads.
    pub fn read_only(&self) -> DeviceTensor<DType, LayoutType> {
        DeviceTensor::new(self.storage(), self.layout())
    }
}

impl<DType, LayoutType> TileTensor<'_, DType, LayoutType, DevicePointerEngine, Mutable>
where
    DType: ShaderType,
    LayoutType: TensorLayout + Copy,
{
    /// This tensor as a shader argument the shader reads and writes.
    pub fn read_write(&self) -> RWDeviceTensor<DType, LayoutType> {
        RWDeviceTensor::new(self.storage(), self.layout())
    }
}
