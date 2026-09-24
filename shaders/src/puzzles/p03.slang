[SpecializationConstant]
const uint block_dim_x = 1;

struct Add10GuardPush {
    float *output;
    float *a;
    uint64_t len;
};

[numthreads(block_dim_x, 1, 1)]
void main(uint thread_id: SV_GroupThreadID, uniform Add10GuardPush push) {
    if (thread_id >= push.len) {
        return;
    }
    push.output[thread_id] = push.a[thread_id] + 10;
}
