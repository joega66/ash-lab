#include "../layout.slang"
 
struct Add10BlocksPush {
    float *output;
    float *a;
    uint64_t size;
};

[numthreads(X, 1, 1)]
void main<let X : uint32_t>(
    uint idx: SV_DispatchThreadID,
    uniform Add10BlocksPush push) {
    if (idx >= push.size) {
        return;
    }
    push.output[idx] = push.a[idx] + 10.0f;
}
