#ifndef TILE_TENSOR_SLANG
#define TILE_TENSOR_SLANG

#include "layout.slang"

interface ITensorEngine<DType> {
    associatedtype StorageType;

    static DType load(StorageType storage, uint64_t index);
    static void store(StorageType storage, uint64_t index, DType value);
}

struct DevicePointerEngine<DType> : ITensorEngine<DType> {
    typealias StorageType = Ptr<DType>;

    static DType load(Ptr<DType> storage, uint64_t index) {
        return storage[index];
    }
    static void store(Ptr<DType> storage, uint64_t index, DType value) {
        storage[index] = value;
    }
}

struct TileTensor<DType, L : ITensorLayout, Engine : ITensorEngine<DType> = DevicePointerEngine<DType>> {
    Engine.StorageType storage;
    L layout;

    typealias Coords = uint64_t[L.ShapeType.RANK];

    static const int RANK = L.ShapeType.RANK;

    __init(Engine.StorageType storage, L layout) {
        this.storage = storage;
        this.layout = layout;
    }

    uint64_t dim(int i) {
        return layout.shape_coord().dims()[i];
    }

    uint64_t size() {
        return layout.size();
    }

    uint64_t cosize() {
        return layout.cosize();
    }

    Coords coords(uint64_t i0) {
        static_assert(RANK == 1, "a rank-1 tensor is indexed with 1 coordinate");
        Coords c;
        c[0] = i0;
        return c;
    }
    Coords coords(uint64_t i0, uint64_t i1) {
        static_assert(RANK == 2, "a rank-2 tensor is indexed with 2 coordinates");
        Coords c;
        c[0] = i0;
        c[1] = i1;
        return c;
    }
    Coords coords(uint64_t i0, uint64_t i1, uint64_t i2) {
        static_assert(RANK == 3, "a rank-3 tensor is indexed with 3 coordinates");
        Coords c;
        c[0] = i0;
        c[1] = i1;
        c[2] = i2;
        return c;
    }
    Coords coords(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
        static_assert(RANK == 4, "a rank-4 tensor is indexed with 4 coordinates");
        Coords c;
        c[0] = i0;
        c[1] = i1;
        c[2] = i2;
        c[3] = i3;
        return c;
    }
    Coords coords(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3, uint64_t i4) {
        static_assert(RANK == 5, "a rank-5 tensor is indexed with 5 coordinates");
        Coords c;
        c[0] = i0;
        c[1] = i1;
        c[2] = i2;
        c[3] = i3;
        c[4] = i4;
        return c;
    }

    /// Whether `crd` is inside this tensor on every axis.
    bool in_bounds(Coords crd) {
        return shape_in_bounds(layout.shape_coord(), crd);
    }

    bool in_bounds(uint64_t i0) {
        return in_bounds(coords(i0));
    }
    bool in_bounds(uint64_t i0, uint64_t i1) {
        return in_bounds(coords(i0, i1));
    }
    bool in_bounds(uint64_t i0, uint64_t i1, uint64_t i2) {
        return in_bounds(coords(i0, i1, i2));
    }
    bool in_bounds(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
        return in_bounds(coords(i0, i1, i2, i3));
    }
    bool in_bounds(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3, uint64_t i4) {
        return in_bounds(coords(i0, i1, i2, i3, i4));
    }

    __subscript(Coords crd)->DType {
        get { return Engine.load(storage, layout.crd2idx(crd)); }
        set { Engine.store(storage, layout.crd2idx(crd), newValue); }
    }

    __subscript(uint64_t i0)->DType {
        get { return Engine.load(storage, layout.crd2idx(coords(i0))); }
        set { Engine.store(storage, layout.crd2idx(coords(i0)), newValue); }
    }
    __subscript(uint64_t i0, uint64_t i1)->DType {
        get { return Engine.load(storage, layout.crd2idx(coords(i0, i1))); }
        set { Engine.store(storage, layout.crd2idx(coords(i0, i1)), newValue); }
    }
    __subscript(uint64_t i0, uint64_t i1, uint64_t i2)->DType {
        get { return Engine.load(storage, layout.crd2idx(coords(i0, i1, i2))); }
        set { Engine.store(storage, layout.crd2idx(coords(i0, i1, i2)), newValue); }
    }
    __subscript(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3)->DType {
        get { return Engine.load(storage, layout.crd2idx(coords(i0, i1, i2, i3))); }
        set { Engine.store(storage, layout.crd2idx(coords(i0, i1, i2, i3)), newValue); }
    }
    __subscript(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3, uint64_t i4)->DType {
        get { return Engine.load(storage, layout.crd2idx(coords(i0, i1, i2, i3, i4))); }
        set { Engine.store(storage, layout.crd2idx(coords(i0, i1, i2, i3, i4)), newValue); }
    }

    DType load_linear(uint64_t index) {
        return Engine.load(storage, index);
    }
    void store_linear(uint64_t index, DType value) {
        Engine.store(storage, index, value);
    }
}

TileTensor<DType, L, DevicePointerEngine<DType>> tile_tensor<DType, L : ITensorLayout>(
    Ptr<DType> storage,
    L layout) {
    return TileTensor<DType, L, DevicePointerEngine<DType>>(storage, layout);
}

#endif // TILE_TENSOR_SLANG
