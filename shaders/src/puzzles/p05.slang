#include "../tile_tensor.slang"

struct BroadcastAddPush<let SIZE : uint32_t> {
    typealias OutLayout = RowMajorLayout<Shape2<Const<SIZE>, Const<SIZE>>>;
    typealias ALayout = RowMajorLayout<Shape2<Const<1>, Const<SIZE>>>;
    typealias BLayout = RowMajorLayout<Shape2<Const<SIZE>, Const<1>>>;

    TileTensor<float, OutLayout> output;
    TileTensor<float, ALayout> a;
    TileTensor<float, BLayout> b;
};

[numthreads(X, Y, 1)]
void main<let SIZE : uint32_t, let X : uint32_t, let Y : uint32_t>(
    uint2 thread_id: SV_DispatchThreadID,
    uniform BroadcastAddPush<SIZE> push) {
    let row = thread_id.y;
    let col = thread_id.x;
    if (!push.output.in_bounds(row, col)) {
        return;
    }
    push.output[row, col] = push.a[0, col] + push.b[row, 0];
}
