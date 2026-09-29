#include "../tile_tensor.slang"
#include "../dtype.slang"

struct Add102dPush<let SIZE : uint32_t> {
    typealias LayoutType = RowMajorLayout<Shape2<Const<SIZE>, Const<SIZE>>>;
    TileTensor<float, LayoutType> output;
    TileTensor<float, LayoutType> a;
};

[numthreads(X, Y, 1)]
void main<let SIZE : uint32_t, let X : uint32_t, let Y : uint32_t>(
    uint2 thread_id: SV_GroupThreadID,
    uniform Add102dPush<SIZE> push, ) {
    let row = thread_id.y;
    let col = thread_id.x;
    if (!push.output.in_bounds(row, col)) {
        return;
    }
    push.output[row, col] = push.a[row, col] + 10.0f;
}
