#ifndef LAYOUT_SLANG
#define LAYOUT_SLANG

/// What a compile-time query reports when the value is not known until run time.
static const int64_t UNKNOWN_VALUE = -1;

interface IDim {
    /// Whether this dimension is known at compile time.
    static const bool STATIC_KNOWN;
    /// This dimension's compile-time extent; meaningless unless `STATIC_KNOWN`.
    static const uint64_t STATIC_VALUE;

    /// This dimension's extent.
    uint64_t value();
}

/// A dimension whose extent is known at compile time.
struct Const<let N : uint64_t> : IDim {
    static const bool STATIC_KNOWN = true;
    static const uint64_t STATIC_VALUE = N;

    uint64_t value() {
        return N;
    }
}

/// A dimension supplied at run time.
extension uint64_t : IDim {
    static const bool STATIC_KNOWN = false;
    static const uint64_t STATIC_VALUE = 0;

    uint64_t value() {
        return this;
    }
}

/// The product of two dimensions, static whenever both of its operands are.
struct Prod<A : IDim, B : IDim> : IDim {
    A a;
    B b;

    static const bool STATIC_KNOWN = A.STATIC_KNOWN && B.STATIC_KNOWN;
    static const uint64_t STATIC_VALUE = A.STATIC_VALUE * B.STATIC_VALUE;

    uint64_t value() {
        // Specializes to a constant when both halves are static.
        if (STATIC_KNOWN) {
            return STATIC_VALUE;
        }
        return a.value() * b.value();
    }
}

/// Builds the `Prod` of `a` and `b`.
Prod<A, B> prod<A : IDim, B : IDim>(A a, B b) {
    Prod<A, B> p;
    p.a = a;
    p.b = b;
    return p;
}

/// `D`'s compile-time extent, or `UNKNOWN_VALUE` if it is a runtime value.
int64_t dim_static<D : IDim>() {
    if (!D.STATIC_KNOWN) {
        return UNKNOWN_VALUE;
    }
    return int64_t(D.STATIC_VALUE);
}

/// `D`'s compile-time extent, or a compile-time error.
uint64_t dim_static_value<D : IDim>() {
    static_assert(D.STATIC_KNOWN, "dimension is not known at compile time");
    return D.STATIC_VALUE;
}

/// A rank-N tuple of dimensions, e.g. `Shape2<Const<4>, uint64_t>` for a 4 x N shape.
interface IShape {
    /// Number of dimensions.
    static const int RANK;

    /// Whether every dimension is known at compile time.
    static const bool ALL_KNOWN;

    /// Product of all extents, or `UNKNOWN_VALUE` if any dimension is dynamic.
    static const int64_t STATIC_PRODUCT;

    /// This shape with the dimension order reversed.
    associatedtype Reversed : IShape;

    /// This shape with every dimension turned into a runtime `uint64_t`.
    associatedtype Dynamic : IShape;

    /// The extent of every dimension.
    uint64_t[RANK] dims();

    /// An all-zero coordinate.
    static uint64_t[RANK] zero_coords();

    /// Reverses the dimension order.
    Reversed reversed();

    /// Materializes every dimension as a runtime value.
    Dynamic to_dynamic();

    /// The compile-time extent of dimension `i`, or `UNKNOWN_VALUE` if it is a runtime value.
    /// 
    /// Prefer `static_dim_at<I>` wherever the index is known, which reports a 
    /// dynamic dimension as a compile error instead of a value to test.
    static int64_t static_at(int i);

    /// The compile-time extent of dimension `I`, as a constant.
    ///
    /// A dynamic dimension at `I`, or an `I` outside the rank, is a compile error.
    static uint64_t static_dim_at<let I : int>();
}

/// Shapes that can be given row-major (C-order) strides: the rightmost dimension has stride 1 and
/// each preceding dimension has stride equal to the product of all following dimensions.
interface IRowMajor : IShape {
    /// The row-major stride tuple for this shape.
    associatedtype RowMajorStride : IShape;

    /// Computes the row-major strides.
    RowMajorStride row_major_stride();
}

/// Shapes that can be given column-major (glm-order) strides: the leftmost dimension has
/// stride 1 and each following dimension has stride equal to the product of all preceding ones.
interface IColMajor : IShape {
    /// The column-major stride tuple for this shape.
    associatedtype ColMajorStride : IShape;

    /// Computes the column-major strides.
    ColMajorStride col_major_stride();
}

/// The extent of dimension `i`.
uint64_t shape_get<S : IShape>(S shape, int i) {
    return shape.dims()[i];
}

/// Whether `crd` lies inside `shape` on every axis.
bool shape_in_bounds<S : IShape>(S shape, uint64_t[S.RANK] crd) {
    let d = shape.dims();
    for (int i = 0; i < S.RANK; ++i) {
        if (crd[i] >= d[i]) {
            return false;
        }
    }
    return true;
}

/// Product of all of `shape`'s extents.
uint64_t shape_product<S : IShape>(S shape) {
    if (S.STATIC_PRODUCT != UNKNOWN_VALUE) {
        return uint64_t(S.STATIC_PRODUCT);
    }
    let d = shape.dims();
    uint64_t acc = 1;
    for (int i = 0; i < S.RANK; ++i) {
        acc *= d[i];
    }
    return acc;
}

/// The compile-time size of the memory region spanned by the layout, or `UNKNOWN_VALUE` if any
/// dimension is dynamic.
int64_t layout_static_cosize<S : IShape, D : IShape>() {
    int64_t acc = 1;
    for (int i = 0; i < S.RANK; ++i) {
        let s = S.static_at(i);
        let d = D.static_at(i);
        if (s == UNKNOWN_VALUE || d == UNKNOWN_VALUE) {
            return UNKNOWN_VALUE;
        }
        if (s == 0) {
            return 0;
        }
        acc += (s - 1) * d;
    }
    return acc;
}

/// A rank-1 shape or stride.
struct Shape1<D0 : IDim> : IShape, IRowMajor, IColMajor {
    D0 d0;

    static const int RANK = 1;
    static const bool ALL_KNOWN = D0.STATIC_KNOWN;
    static const int64_t STATIC_PRODUCT = ALL_KNOWN ? int64_t(D0.STATIC_VALUE) : UNKNOWN_VALUE;

    typealias Coords = uint64_t[1];
    typealias Reversed = Shape1<D0>;
    typealias Dynamic = Shape1<uint64_t>;
    typealias RowMajorStride = Shape1<Const<1>>;
    typealias ColMajorStride = Shape1<Const<1>>;

    Coords dims() {
        return { d0.value() };
    }
    static Coords zero_coords() {
        return { 0 };
    }

    Reversed reversed() {
        Reversed r;
        r.d0 = d0;
        return r;
    }

    Dynamic to_dynamic() {
        Dynamic r;
        r.d0 = d0.value();
        return r;
    }

    static int64_t static_at(int i) {
        int64_t s[1] = { dim_static<D0>() };
        return s[i];
    }

    static uint64_t static_dim_at<let I : int>() {
        static_assert(I >= 0 && I < RANK, "dimension index out of range");
        return dim_static_value<D0>();
    }

    RowMajorStride row_major_stride() {
        RowMajorStride s;
        s.d0 = Const<1>();
        return s;
    }

    ColMajorStride col_major_stride() {
        ColMajorStride s;
        s.d0 = Const<1>();
        return s;
    }
}

/// A rank-2 shape or stride.
struct Shape2<D0 : IDim, D1 : IDim> : IShape, IRowMajor, IColMajor {
    D0 d0;
    D1 d1;

    static const int RANK = 2;
    static const bool ALL_KNOWN = D0.STATIC_KNOWN && D1.STATIC_KNOWN;
    static const int64_t STATIC_PRODUCT = ALL_KNOWN ? int64_t(D0.STATIC_VALUE * D1.STATIC_VALUE) : UNKNOWN_VALUE;

    typealias Coords = uint64_t[2];
    typealias Reversed = Shape2<D1, D0>;
    typealias Dynamic = Shape2<uint64_t, uint64_t>;
    typealias RowMajorStride = Shape2<D1, Const<1>>;
    typealias ColMajorStride = Shape2<Const<1>, D0>;

    Coords dims() {
        return { d0.value(), d1.value() };
    }
    static Coords zero_coords() {
        return { 0, 0 };
    }

    Reversed reversed() {
        Reversed r;
        r.d0 = d1;
        r.d1 = d0;
        return r;
    }

    Dynamic to_dynamic() {
        Dynamic r;
        r.d0 = d0.value();
        r.d1 = d1.value();
        return r;
    }

    static int64_t static_at(int i) {
        int64_t s[2] = { dim_static<D0>(), dim_static<D1>() };
        return s[i];
    }

    static uint64_t static_dim_at<let I : int>() {
        static_assert(I >= 0 && I < RANK, "dimension index out of range");
        if (I == 0) {
            return dim_static_value<D0>();
        }
        return dim_static_value<D1>();
    }

    RowMajorStride row_major_stride() {
        RowMajorStride s;
        s.d0 = d1;
        s.d1 = Const<1>();
        return s;
    }

    ColMajorStride col_major_stride() {
        ColMajorStride s;
        s.d0 = Const<1>();
        s.d1 = d0;
        return s;
    }
}

/// A rank-3 shape or stride.
struct Shape3<D0 : IDim, D1 : IDim, D2 : IDim> : IShape, IRowMajor, IColMajor {
    D0 d0;
    D1 d1;
    D2 d2;

    static const int RANK = 3;
    static const bool ALL_KNOWN = D0.STATIC_KNOWN && D1.STATIC_KNOWN && D2.STATIC_KNOWN;
    static const int64_t STATIC_PRODUCT = ALL_KNOWN ? int64_t(D0.STATIC_VALUE * D1.STATIC_VALUE * D2.STATIC_VALUE) : UNKNOWN_VALUE;

    typealias Coords = uint64_t[3];
    typealias Reversed = Shape3<D2, D1, D0>;
    typealias Dynamic = Shape3<uint64_t, uint64_t, uint64_t>;
    typealias RowMajorStride = Shape3<Prod<D1, D2>, D2, Const<1>>;
    typealias ColMajorStride = Shape3<Const<1>, D0, Prod<D0, D1>>;

    Coords dims() {
        return { d0.value(), d1.value(), d2.value() };
    }
    static Coords zero_coords() {
        return { 0, 0, 0 };
    }

    Reversed reversed() {
        Reversed r;
        r.d0 = d2;
        r.d1 = d1;
        r.d2 = d0;
        return r;
    }

    Dynamic to_dynamic() {
        Dynamic r;
        r.d0 = d0.value();
        r.d1 = d1.value();
        r.d2 = d2.value();
        return r;
    }

    static int64_t static_at(int i) {
        int64_t s[3] = { dim_static<D0>(), dim_static<D1>(), dim_static<D2>() };
        return s[i];
    }

    static uint64_t static_dim_at<let I : int>() {
        static_assert(I >= 0 && I < RANK, "dimension index out of range");
        if (I == 0) {
            return dim_static_value<D0>();
        }
        if (I == 1) {
            return dim_static_value<D1>();
        }
        return dim_static_value<D2>();
    }

    RowMajorStride row_major_stride() {
        RowMajorStride s;
        s.d0 = prod(d1, d2);
        s.d1 = d2;
        s.d2 = Const<1>();
        return s;
    }

    ColMajorStride col_major_stride() {
        ColMajorStride s;
        s.d0 = Const<1>();
        s.d1 = d0;
        s.d2 = prod(d0, d1);
        return s;
    }
}

/// A rank-4 shape or stride.
struct Shape4<D0 : IDim, D1 : IDim, D2 : IDim, D3 : IDim> : IShape, IRowMajor, IColMajor {
    D0 d0;
    D1 d1;
    D2 d2;
    D3 d3;

    static const int RANK = 4;
    static const bool ALL_KNOWN = D0.STATIC_KNOWN && D1.STATIC_KNOWN && D2.STATIC_KNOWN && D3.STATIC_KNOWN;
    static const int64_t STATIC_PRODUCT = ALL_KNOWN ? int64_t(D0.STATIC_VALUE * D1.STATIC_VALUE * D2.STATIC_VALUE * D3.STATIC_VALUE) : UNKNOWN_VALUE;

    typealias Coords = uint64_t[4];
    typealias Reversed = Shape4<D3, D2, D1, D0>;
    typealias Dynamic = Shape4<uint64_t, uint64_t, uint64_t, uint64_t>;
    typealias RowMajorStride = Shape4<Prod<D1, Prod<D2, D3>>, Prod<D2, D3>, D3, Const<1>>;
    typealias ColMajorStride = Shape4<Const<1>, D0, Prod<D0, D1>, Prod<D0, Prod<D1, D2>>>;

    Coords dims() {
        return { d0.value(), d1.value(), d2.value(), d3.value() };
    }
    static Coords zero_coords() {
        return { 0, 0, 0, 0 };
    }

    Reversed reversed() {
        Reversed r;
        r.d0 = d3;
        r.d1 = d2;
        r.d2 = d1;
        r.d3 = d0;
        return r;
    }

    Dynamic to_dynamic() {
        Dynamic r;
        r.d0 = d0.value();
        r.d1 = d1.value();
        r.d2 = d2.value();
        r.d3 = d3.value();
        return r;
    }

    static int64_t static_at(int i) {
        int64_t s[4] = { dim_static<D0>(), dim_static<D1>(), dim_static<D2>(), dim_static<D3>() };
        return s[i];
    }

    static uint64_t static_dim_at<let I : int>() {
        static_assert(I >= 0 && I < RANK, "dimension index out of range");
        if (I == 0) {
            return dim_static_value<D0>();
        }
        if (I == 1) {
            return dim_static_value<D1>();
        }
        if (I == 2) {
            return dim_static_value<D2>();
        }
        return dim_static_value<D3>();
    }

    RowMajorStride row_major_stride() {
        RowMajorStride s;
        s.d0 = prod(d1, prod(d2, d3));
        s.d1 = prod(d2, d3);
        s.d2 = d3;
        s.d3 = Const<1>();
        return s;
    }

    ColMajorStride col_major_stride() {
        ColMajorStride s;
        s.d0 = Const<1>();
        s.d1 = d0;
        s.d2 = prod(d0, d1);
        s.d3 = prod(d0, prod(d1, d2));
        return s;
    }
}

/// A rank-5 shape or stride.
struct Shape5<D0 : IDim, D1 : IDim, D2 : IDim, D3 : IDim, D4 : IDim> : IShape, IRowMajor, IColMajor {
    D0 d0;
    D1 d1;
    D2 d2;
    D3 d3;
    D4 d4;

    static const int RANK = 5;
    static const bool ALL_KNOWN = D0.STATIC_KNOWN && D1.STATIC_KNOWN && D2.STATIC_KNOWN && D3.STATIC_KNOWN && D4.STATIC_KNOWN;
    static const int64_t STATIC_PRODUCT = ALL_KNOWN ? int64_t(D0.STATIC_VALUE * D1.STATIC_VALUE * D2.STATIC_VALUE * D3.STATIC_VALUE * D4.STATIC_VALUE) : UNKNOWN_VALUE;

    typealias Coords = uint64_t[5];
    typealias Reversed = Shape5<D4, D3, D2, D1, D0>;
    typealias Dynamic = Shape5<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>;
    typealias RowMajorStride = Shape5<Prod<D1, Prod<D2, Prod<D3, D4>>>, Prod<D2, Prod<D3, D4>>, Prod<D3, D4>, D4, Const<1>>;
    typealias ColMajorStride = Shape5<Const<1>, D0, Prod<D0, D1>, Prod<D0, Prod<D1, D2>>, Prod<D0, Prod<D1, Prod<D2, D3>>>>;

    Coords dims() {
        return { d0.value(), d1.value(), d2.value(), d3.value(), d4.value() };
    }
    static Coords zero_coords() {
        return { 0, 0, 0, 0, 0 };
    }

    Reversed reversed() {
        Reversed r;
        r.d0 = d4;
        r.d1 = d3;
        r.d2 = d2;
        r.d3 = d1;
        r.d4 = d0;
        return r;
    }

    Dynamic to_dynamic() {
        Dynamic r;
        r.d0 = d0.value();
        r.d1 = d1.value();
        r.d2 = d2.value();
        r.d3 = d3.value();
        r.d4 = d4.value();
        return r;
    }

    static int64_t static_at(int i) {
        int64_t s[5] = { dim_static<D0>(), dim_static<D1>(), dim_static<D2>(), dim_static<D3>(), dim_static<D4>() };
        return s[i];
    }

    static uint64_t static_dim_at<let I : int>() {
        static_assert(I >= 0 && I < RANK, "dimension index out of range");
        if (I == 0) {
            return dim_static_value<D0>();
        }
        if (I == 1) {
            return dim_static_value<D1>();
        }
        if (I == 2) {
            return dim_static_value<D2>();
        }
        if (I == 3) {
            return dim_static_value<D3>();
        }
        return dim_static_value<D4>();
    }

    RowMajorStride row_major_stride() {
        RowMajorStride s;
        s.d0 = prod(d1, prod(d2, prod(d3, d4)));
        s.d1 = prod(d2, prod(d3, d4));
        s.d2 = prod(d3, d4);
        s.d3 = d4;
        s.d4 = Const<1>();
        return s;
    }

    ColMajorStride col_major_stride() {
        ColMajorStride s;
        s.d0 = Const<1>();
        s.d1 = d0;
        s.d2 = prod(d0, d1);
        s.d3 = prod(d0, prod(d1, d2));
        s.d4 = prod(d0, prod(d1, prod(d2, d3)));
        return s;
    }
}

/// A layout a tensor can be built over: `Layout<S, D>`.
interface ITensorLayout {
    associatedtype ShapeType : IShape;
    associatedtype StrideType : IShape;

    /// The full shape, which is how a tensor reaches its own extents.
    ShapeType shape_coord();

    /// Maps logical coordinates to a linear element index.
    uint64_t crd2idx(uint64_t[ShapeType.RANK] crd);

    /// Total number of elements in the layout's domain.
    uint64_t size();

    /// Size of the memory region the layout spans.
    uint64_t cosize();
}

/// A layout that supports mixed compile-time and runtime dimensions.
///
/// This layout provides a unified interface for layouts where some dimensions
//  are known at compile time and others are determined at runtime. It enables
/// more ergonomic layout definitions while maintaining performance.
///
/// A Layout's shape and strides must be non-negative.
struct Layout<S : IShape, D : IShape> : ITensorLayout {
    /// The logical coordinate space.
    S shape;
    /// The memory step taken per unit of each coordinate.
    D stride;

    typealias ShapeType = S;
    typealias StrideType = D;

    /// Number of dimensions.
    static const int RANK = S.RANK;

    /// Number of dimensions after flattening nested coordinates.
    static const int FLAT_RANK = S.RANK;

    /// Whether all shape dimensions are known at compile time.
    static const bool SHAPE_KNOWN = S.ALL_KNOWN;

    /// Whether all stride dimensions are known at compile time.
    static const bool STRIDE_KNOWN = D.ALL_KNOWN;

    /// Whether all shape *and* stride dimensions are known at compile time.
    static const bool ALL_DIMS_KNOWN = S.ALL_KNOWN && D.ALL_KNOWN;

    /// Compile-time product of all shape dimensions, or `UNKNOWN_VALUE` if any of them is dynamic.
    static const int64_t STATIC_PRODUCT = S.STATIC_PRODUCT;

    /// `uint64_t[RANK]`.
    typealias Coords = uint64_t[S.RANK];

    /// Builds a layout from an explicit shape and stride.
    __init(S shape, D stride) {
        static_assert(S.RANK == D.RANK, "shape and stride must have the same rank");
        this.shape = shape;
        this.stride = stride;
    }

    /// Size of the memory region spanned by a shape/stride pair, or `UNKNOWN_VALUE` if any
    /// dimension is dynamic.
    static int64_t static_cosize() {
        return layout_static_cosize<S, D>();
    }

    /// Returns the compile-time value of the i-th flattened shape dimension.
    static uint64_t static_shape<let I : int>() {
        return S.static_dim_at<I>();
    }

    /// Returns the compile-time value of the i-th flattened stride dimension.
    static uint64_t static_stride<let I : int>() {
        return D.static_dim_at<I>();
    }

    /// The number of dimensions in the layout.
    int rank() {
        return S.RANK;
    }

    /// Returns the full shape as a Coord.
    S shape_coord() {
        return shape;
    }

    /// Returns the full stride as a Coord.
    D stride_coord() {
        return stride;
    }

    /// Returns the total number of elements in the layout's domain.
    ///
    /// For a layout with shape(m, n), this returns m*n, representing the total
    /// number of valid coordinates in the layout.
    uint64_t product() {
        return shape_product(shape);
    }

    /// Returns the total number of elements in the layout's domain.
    /// Alias for `product()`.
    uint64_t size() {
        return product();
    }

    /// Returns the size of the memory region spanned by the layout.
    ///
    /// For a layout with `shape(m, n)` and `stride(r, s)`, this returns `(m - 1) * r + (n - 1) * s + 1`,
    /// representing the memory footprint.
    uint64_t cosize() {
        let known = static_cosize();
        if (known != UNKNOWN_VALUE) {
            return uint64_t(known);
        }
        let sh = shape.dims();
        let st = stride.dims();
        uint64_t acc = 1;
        for (int i = 0; i < S.RANK; ++i) {
            if (sh[i] == 0) {
                return 0;
            }
            acc += (sh[i] - 1) * st[i];
        }
        return acc;
    }

    /// Converts multi-dimensional coordinates to a linear index.
    /// 
    /// This function is the inverse of `idx2crd` , transforming a set of coordinates into a flat index
    /// based on the provided shape and stride information. This is essential for mapping multi-
    /// dimensional tensor elements to linear memory.
    uint64_t crd2idx(Coords crd) {
        let st = stride.dims();
        uint64_t idx = 0;
        for (int i = 0; i < S.RANK; ++i) {
            idx += crd[i] * st[i];
        }
        return idx;
    }

    /// Converts a linear index to multi-dimensional coordinates. This function transforms a flat
    /// index into coordinate values based on the provided shape and stride information. This is
    /// essential for mapping linear memory accesses to multi-dimensional tensor elements.
    Coords idx2crd(uint64_t idx) {
        let sh = shape.dims();
        let st = stride.dims();
        Coords crd = S.zero_coords();
        for (int i = 0; i < S.RANK; ++i) {
            crd[i] = sh[i] == 0 ? 0 : (idx / max(st[i], 1)) % sh[i];
        }
        return crd;
    }

    /// Reverse the order of dimensions in the layout.
    /// Turns row-major into column-major ordering where the stride-1 dimension comes first,
    /// enabling coalesced scalar iteration.
    Layout<S.Reversed, D.Reversed> reverse() {
        return Layout<S.Reversed, D.Reversed>(shape.reversed(), stride.reversed());
    }

    /// Transposes the layout by reversing the order of dimensions.
    /// For an n-dimensional layout, this reverses the order of both shapes and strides. For 2D layouts,
    /// this swaps rows and columns, converting row-major to column-major and vice versa.
    Layout<S.Reversed, D.Reversed> transpose() {
        return reverse();
    }

    /// Convert all elements in shape and stride to Scalar<DType>.
    Layout<S.Dynamic, D.Dynamic> make_dynamic() {
        return Layout<S.Dynamic, D.Dynamic>(shape.to_dynamic(), stride.to_dynamic());
    }
}

/// A row-major layout over `S`.
typealias RowMajorLayout<S : IRowMajor> = Layout<S, S.RowMajorStride>;

/// A column-major layout over `S`.
typealias ColMajorLayout<S : IColMajor> = Layout<S, S.ColMajorStride>;

/// Creates a row-major layout from a shape Coord.
/// Row-major means the rightmost dimension has stride 1, and each preceding dimension has
/// stride equal to the product of all following dimensions.
///
/// For shape(M, N, K):
///     row_major strides : (N*K, K, 1)
///     col_major strides : (1, M, M*N)
RowMajorLayout<S> row_major<S : IRowMajor>(S shape) {
    return RowMajorLayout<S>(shape, shape.row_major_stride());
}

/// Create a column-major layout from variadic arguments.
/// Column-major means the first dimension has stride 1, and each subsequent dimension has
/// stride equal to the product of all previous dimensions.
ColMajorLayout<S> col_major<S : IColMajor>(S shape) {
    return ColMajorLayout<S>(shape, shape.col_major_stride());
}

#endif // LAYOUT_SLANG
