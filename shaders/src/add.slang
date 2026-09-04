#include "common.slang"

struct AddPushConstant {
    float *output;
    float *a;
    float *b;
    uint len;
    uint _padding;
};
[vk::push_constant]
AddPushConstant pc;

[numthreads(DEFAULT_NUMTHREADS, 1, 1)]
void main(uint tid: SV_DispatchThreadID) {
    if (tid >= pc.len) {
        return;
    }
    pc.output[tid] = pc.a[tid] + pc.b[tid];
}