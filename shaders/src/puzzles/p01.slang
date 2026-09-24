[SpecializationConstant]
const uint block_dim_x = 1;

struct Add10Push {
    float *output;
    float *a;
};

[numthreads(block_dim_x, 1, 1)]
void main(uint thread_id: SV_GroupThreadID, uniform Add10Push push) {
    push.output[thread_id] = push.a[thread_id] + 10.0;
}
