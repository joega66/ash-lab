[SpecializationConstant]
const uint block_dim_x = 1;

struct Add10Push {
    float *output;
    float *a;
};
[vk::push_constant]
Add10Push pc;

[numthreads(block_dim_x, 1, 1)]
void main(uint thread_id: SV_GroupThreadID) {
    pc.output[thread_id] = pc.a[thread_id] + 10.0;
}
