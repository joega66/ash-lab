struct Add10PushConstant
{
    float *output;
    float *a;
};
[vk::push_constant]
Add10PushConstant pc;

[numthreads(4, 1, 1)]
void main(uint gtid: SV_GroupThreadID)
{
    pc.output[gtid] = pc.a[gtid] + 10;
}
