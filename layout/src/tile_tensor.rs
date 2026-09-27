use core::ops::{Deref, Index, IndexMut};

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

    fn as_storage(&self) -> &<Self::Engine as TensorEngine>::StorageType<Self::DType>;

    fn as_storage_mut(&mut self) -> &mut <Self::Engine as TensorEngine>::StorageType<Self::DType>;
}

impl<T> TensorStorage for DeviceBuffer<T> {
    type DType = T;
    type Engine = DevicePointerEngine;

    fn as_storage(&self) -> &DeviceBuffer<T> {
        self
    }

    fn as_storage_mut(&mut self) -> &mut DeviceBuffer<T> {
        self
    }
}

impl<T> TensorStorage for Vec<T> {
    type DType = T;
    type Engine = DefaultEngine;

    fn as_storage(&self) -> &Vec<T> {
        self
    }

    fn as_storage_mut(&mut self) -> &mut Vec<T> {
        self
    }
}

/// Whether a [`TileTensor`] borrows its storage shared ([`Immutable`]) or exclusively ([`Mutable`]).
pub trait TensorAccess {
    type Ref<'a, T: 'a>: Deref<Target = T>;
}

pub struct Immutable;

impl TensorAccess for Immutable {
    type Ref<'a, T: 'a> = &'a T;
}

pub struct Mutable;

impl TensorAccess for Mutable {
    type Ref<'a, T: 'a> = &'a mut T;
}

/// A borrow of storage a [`TileTensor`] can be built over: `&S` or `&mut S`.
pub trait StorageRef<'a> {
    type DType;
    type Engine: TensorEngine;
    type Access: TensorAccess;

    fn into_ref(self) -> TensorRef<'a, Self::DType, Self::Engine, Self::Access>;
}

type TensorRef<'a, DType, Engine, Access> =
    <Access as TensorAccess>::Ref<'a, <Engine as TensorEngine>::StorageType<DType>>;

impl<'a, S: TensorStorage> StorageRef<'a> for &'a S {
    type DType = S::DType;
    type Engine = S::Engine;
    type Access = Immutable;

    fn into_ref(self) -> &'a <S::Engine as TensorEngine>::StorageType<S::DType> {
        self.as_storage()
    }
}

impl<'a, S: TensorStorage> StorageRef<'a> for &'a mut S {
    type DType = S::DType;
    type Engine = S::Engine;
    type Access = Mutable;

    fn into_ref(self) -> &'a mut <S::Engine as TensorEngine>::StorageType<S::DType> {
        self.as_storage_mut()
    }
}

pub struct TileTensor<
    'a,
    DType,
    LayoutType: TensorLayout,
    Engine: TensorEngine = DevicePointerEngine,
    Access: TensorAccess = Immutable,
> where
    Engine::StorageType<DType>: 'a,
{
    storage: TensorRef<'a, DType, Engine, Access>,
    layout: LayoutType,
}

impl<'a, DType, LayoutType, Engine, Access> TileTensor<'a, DType, LayoutType, Engine, Access>
where
    LayoutType: TensorLayout,
    Engine: TensorEngine,
    Access: TensorAccess,
    Engine::StorageType<DType>: 'a,
{
    pub fn new<S>(storage: S, layout: LayoutType) -> Self
    where
        S: StorageRef<'a, DType = DType, Engine = Engine, Access = Access>,
    {
        Self {
            storage: storage.into_ref(),
            layout,
        }
    }

    pub fn as_ref(&self) -> &Self {
        self
    }

    pub(crate) fn storage(&self) -> &Engine::StorageType<DType> {
        &*self.storage
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
impl<'a, DType, LayoutType, Engine, Access> Index<Coords<LayoutType>>
    for TileTensor<'a, DType, LayoutType, Engine, Access>
where
    LayoutType: TensorLayout,
    Engine: TensorEngine,
    Access: TensorAccess,
    Engine::StorageType<DType>: Index<usize, Output = DType> + 'a,
{
    type Output = DType;

    #[inline]
    fn index(&self, crd: Coords<LayoutType>) -> &DType {
        &self.storage[self.layout.crd2idx(crd)]
    }
}

impl<'a, DType, LayoutType, Engine> IndexMut<Coords<LayoutType>>
    for TileTensor<'a, DType, LayoutType, Engine, Mutable>
where
    LayoutType: TensorLayout,
    Engine: TensorEngine,
    Engine::StorageType<DType>: IndexMut<usize, Output = DType> + 'a,
{
    #[inline]
    fn index_mut(&mut self, crd: Coords<LayoutType>) -> &mut DType {
        let i = self.layout.crd2idx(crd);
        &mut self.storage[i]
    }
}
