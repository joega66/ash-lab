#include "../tile_tensor.slang"
#include "../dtype.slang"

static const uint SIZE = 2;
static const uint BLOCKS_PER_GRID = 1;
static const uint THREADS_PER_BLOCK_X = 3;
static const uint THREADS_PER_BLOCK_Y = 3;
typealias LayoutType = RowMajorLayout<Shape2<Const<SIZE>, Const<SIZE>>>;

struct Add102dPush<T : IDType> {
    TileTensor<T, LayoutType> output;
    TileTensor<T, LayoutType> a;
};

[numthreads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1)]
void main<T : IDType>(
    uint2 thread_id: SV_GroupThreadID,
    uniform Add102dPush<T> push,
) {
    let row = thread_id.y;
    let col = thread_id.x;
    if (!push.output.in_bounds(row, col)) {
        return;
    }
    push.output[row, col] = push.a[row, col] + T(10);
}
