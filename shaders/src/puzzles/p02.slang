[SpecializationConstant]
const uint block_dim_x = 1;

struct AddPush {
    float *output;
    float *a;
    float *b;
};

[numthreads(block_dim_x, 1, 1)]
void main(uint thread_id: SV_GroupThreadID, uniform AddPush push) {
    push.output[thread_id] = push.a[thread_id] + push.b[thread_id];
}
