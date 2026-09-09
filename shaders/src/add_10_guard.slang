[SpecializationConstant]
const uint grid_size_x = 1;

struct Add10GuardPush {
    float *output;
    float *a;
    uint64_t len;
};
[vk::push_constant]
Add10GuardPush pc;

[numthreads(grid_size_x, 1, 1)]
void main(uint thread_id: SV_GroupThreadID) {
    if (thread_id >= pc.len) {
        return;
    }
    pc.output[thread_id] = pc.a[thread_id] + 10;
}
