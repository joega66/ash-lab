[SpecializationConstant]
const uint grid_size_x = 1;

struct AddPushConstant {
    float *output;
    float *a;
    float *b;
};
[vk::push_constant]
AddPushConstant pc;

[numthreads(grid_size_x, 1, 1)]
void main(uint thread_id: SV_GroupThreadID) {
    pc.output[thread_id] = pc.a[thread_id] + pc.b[thread_id];
}
