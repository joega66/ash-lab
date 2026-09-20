use core::mem::{offset_of, size_of};

use gpu::{
    DeviceAddress, DeviceBuffer, FieldLayout, RWDeviceAddress, ShaderType, TypeLayout, bytemuck,
};

use crate::tile_tensor::{DevicePointerEngine, TensorLayout, TileTensor};

macro_rules! impl_device_tensor {
    ($name:ident, $address:ident, $doc:literal) => {
        #[doc = $doc]
        #[repr(C)]
        pub struct $name<DType: ShaderType, LayoutType> {
            storage: $address<DType>,
            layout: LayoutType,
        }

        impl<DType: ShaderType, LayoutType> $name<DType, LayoutType> {
            pub fn new(buffer: &DeviceBuffer<DType>, layout: LayoutType) -> Self {
                Self {
                    storage: buffer.into(),
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

        impl<DType, LayoutType> From<&TileTensor<DType, LayoutType, DevicePointerEngine>>
            for $name<DType, LayoutType>
        where
            DType: ShaderType,
            LayoutType: TensorLayout + Copy,
        {
            fn from(value: &TileTensor<DType, LayoutType, DevicePointerEngine>) -> Self {
                Self::new(value.storage(), value.layout())
            }
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

impl_device_tensor!(DeviceTensor, DeviceAddress, "A tensor a shader may read.");
impl_device_tensor!(
    RWDeviceTensor,
    RWDeviceAddress,
    "A tensor a shader may read and write."
);

