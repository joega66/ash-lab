#include "../tile_tensor.slang"

static const uint SIZE = 2;
static const uint THREADS_PER_BLOCK_X = 3;
static const uint THREADS_PER_BLOCK_Y = 3;
typealias DType = float;
typealias OutLayout = RowMajorLayout<Shape2<Const<SIZE>, Const<SIZE>>>;
typealias ALayout = RowMajorLayout<Shape2<Const<1>, Const<SIZE>>>;
typealias BLayout = RowMajorLayout<Shape2<Const<SIZE>, Const<1>>>;

struct BroadcastAddPush {
    TileTensor<DType, OutLayout> output;
    TileTensor<DType, ALayout> a;
    TileTensor<DType, BLayout> b;
};

[numthreads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1)]
void main(
    uint2 thread_id: SV_DispatchThreadID,
    uniform BroadcastAddPush push) {
    let row = thread_id.y;
    let col = thread_id.x;
    if (!push.output.in_bounds(row, col)) {
        return;
    }
    push.output[row, col] = push.a[0, col] + push.b[row, 0];
}
