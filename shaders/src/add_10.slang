#include "common.slang"

struct Add10PushConstant {
    float *output;
    float *a;
    uint len;
    uint _padding;
};
[vk::push_constant]
Add10PushConstant pc;

[numthreads(DEFAULT_NUMTHREADS, 1, 1)]
void main(uint tid: SV_DispatchThreadID) {
    if (tid >= pc.len) {
        return;
    }
    pc.output[tid] = pc.a[tid] + 10;
}
