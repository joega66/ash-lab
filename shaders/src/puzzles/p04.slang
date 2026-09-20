#include "../tile_tensor.slang"

static const uint SIZE = 2;
static const uint BLOCKS_PER_GRID = 1;
static const uint THREADS_PER_BLOCK_X = 3;
static const uint THREADS_PER_BLOCK_Y = 3;
typealias DType = float;
typealias LayoutType = RowMajorLayout<Shape2<Const<SIZE>, Const<SIZE>>>;

struct Add102dPush {
    TileTensor<DType, LayoutType> output;
    TileTensor<DType, LayoutType> a;
};
[vk::push_constant]
Add102dPush pc;

[numthreads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1)]
void main(uint2 thread_id: SV_GroupThreadID) {
    let row = thread_id.y;
    let col = thread_id.x;
    if (!pc.output.in_bounds(row, col)) {
        return;
    }
    var output = pc.output;
    output[row, col] = pc.a[row, col] + 10;
}
