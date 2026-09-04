static const float3 positions[] = {
    float3(0.0, +0.6667, 0),
    float3(-0.5, -0.3333, 0),
    float3(+0.5, -0.3333, 0)
};

static const float3 colors[] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
};

struct CameraUniform {
    float4x4 view_proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<CameraUniform> camera;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

[shader("vertex")]
void vs_main(in uint vertex_id: SV_VertexID, out VertexOutput output) {
    static const float3 TRANSLATION = float3(0, 0, -1);

    let position_world = float4(positions[vertex_id] + TRANSLATION, 1);
    let position = mul(camera.view_proj, position_world);
    output.position = position;
    output.color = colors[vertex_id];
}

[shader("pixel")]
void ps_main(in VertexOutput input, out float4 output) {
    output = float4(input.color, 1);
}
