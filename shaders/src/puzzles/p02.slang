[SpecializationConstant]
const uint block_dim_x = 1;

struct AddPush {
    float *output;
    float *a;
    float *b;
};
[vk::push_constant]
AddPush pc;

[numthreads(block_dim_x, 1, 1)]
void main(uint thread_id: SV_GroupThreadID) {
    pc.output[thread_id] = pc.a[thread_id] + pc.b[thread_id];
}
