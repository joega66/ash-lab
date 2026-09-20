use core::ops::{Index, IndexMut};

use gpu::DeviceBuffer;

use crate::{DimAt, Shape, layout::Layout};

/// `[usize; RANK]` for the layout's shape: the coordinate a tensor over it is indexed by.
pub type Coords<LayoutType> = <<LayoutType as TensorLayout>::ShapeType as Shape>::Coords;

pub trait TensorLayout {
    type ShapeType: Shape;
    type StrideType: Shape<Coords = <Self::ShapeType as Shape>::Coords>;

    fn shape_coord(&self) -> Self::ShapeType;

    fn crd2idx(&self, crd: <Self::ShapeType as Shape>::Coords) -> usize;
}

impl<S, D> TensorLayout for Layout<S, D>
where
    S: Shape,
    D: Shape<Coords = S::Coords>,
{
    type ShapeType = S;
    type StrideType = D;

    #[inline]
    fn shape_coord(&self) -> S {
        self.shape
    }

    #[inline]
    fn crd2idx(&self, crd: S::Coords) -> usize {
        Layout::crd2idx(self, crd)
    }
}

pub trait TensorEngine {
    type StorageType<DType>;
}

pub struct DevicePointerEngine;

impl TensorEngine for DevicePointerEngine {
    type StorageType<DType> = DeviceBuffer<DType>;
}

pub struct DefaultEngine;

impl TensorEngine for DefaultEngine {
    type StorageType<DType> = Vec<DType>;
}

pub trait TensorStorage {
    type DType;
    type Engine: TensorEngine;

    fn into_storage(self) -> <Self::Engine as TensorEngine>::StorageType<Self::DType>;
}

impl<T> TensorStorage for DeviceBuffer<T> {
    type DType = T;
    type Engine = DevicePointerEngine;

    fn into_storage(self) -> DeviceBuffer<T> {
        self
    }
}

impl<T> TensorStorage for Vec<T> {
    type DType = T;
    type Engine = DefaultEngine;

    fn into_storage(self) -> Vec<T> {
        self
    }
}

pub struct TileTensor<DType, LayoutType: TensorLayout, Engine: TensorEngine = DevicePointerEngine> {
    storage: Engine::StorageType<DType>,
    layout: LayoutType,
}

impl<DType, LayoutType: TensorLayout, Engine: TensorEngine> TileTensor<DType, LayoutType, Engine> {
    pub fn new<S>(storage: S, layout: LayoutType) -> Self
    where
        S: TensorStorage<DType = DType, Engine = Engine>,
    {
        Self {
            storage: storage.into_storage(),
            layout,
        }
    }

    pub fn as_ref(&self) -> &Self {
        self
    }

    pub(crate) fn storage(&self) -> &Engine::StorageType<DType> {
        &self.storage
    }

    pub(crate) fn layout(&self) -> LayoutType
    where
        LayoutType: Copy,
    {
        self.layout
    }

    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        self.layout.shape_coord().get(i)
    }

    #[inline]
    pub fn dim_at<const I: usize>(&self) -> <LayoutType::ShapeType as DimAt<I>>::Dim
    where
        LayoutType::ShapeType: DimAt<I>,
    {
        self.layout.shape_coord().dim_at()
    }
}

/// Indexing by coordinate, for storage the host can reach: `t[[y, x]]` for a rank-2 tensor.
///
/// The coordinate is `[usize; RANK]`, so a wrong number of indices does not compile. The bound is
/// on the storage rather than on a particular engine, which is what leaves out
/// [`DevicePointerEngine`]: a [`DeviceBuffer`] is GPU memory and needs
/// [`map_to_host`](DeviceBuffer::map_to_host) and a device context before the host may read it.
impl<DType, LayoutType, Engine> Index<Coords<LayoutType>> for TileTensor<DType, LayoutType, Engine>
where
    LayoutType: TensorLayout,
    Engine: TensorEngine,
    Engine::StorageType<DType>: Index<usize, Output = DType>,
{
    type Output = DType;

    #[inline]
    fn index(&self, crd: Coords<LayoutType>) -> &DType {
        &self.storage[self.layout.crd2idx(crd)]
    }
}

impl<DType, LayoutType, Engine> IndexMut<Coords<LayoutType>>
    for TileTensor<DType, LayoutType, Engine>
where
    LayoutType: TensorLayout,
    Engine: TensorEngine,
    Engine::StorageType<DType>: IndexMut<usize, Output = DType>,
{
    #[inline]
    fn index_mut(&mut self, crd: Coords<LayoutType>) -> &mut DType {
        let i = self.layout.crd2idx(crd);
        &mut self.storage[i]
    }
}
