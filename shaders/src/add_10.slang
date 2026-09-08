[SpecializationConstant]
const uint grid_size_x = 1;

struct Add10PushConstant {
    float *output;
    float *a;
};
[vk::push_constant]
Add10PushConstant pc;

[numthreads(grid_size_x, 1, 1)]
void main(uint thread_id: SV_GroupThreadID) {
    pc.output[thread_id] = pc.a[thread_id] + 10;
}
