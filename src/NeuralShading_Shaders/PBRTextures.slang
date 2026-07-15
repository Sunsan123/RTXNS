/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

struct PbrMaterialSample
{
    float3 baseColor;
    float roughness;
    float metallic;
    float3 normal;
}

float LengthSquared(float3 v)
{
    return dot(v, v);
}

float3 SafeNormalize(float3 v, float3 fallback)
{
    return LengthSquared(v) > 1e-12f ? normalize(v) : fallback;
}

float3 GetFallbackTangent(float3 normal)
{
    float3 axis = abs(normal.y) < 0.9f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
    return SafeNormalize(cross(axis, normal), float3(1.f, 0.f, 0.f));
}

float3 OrthogonalizeTangent(float3 tangent, float3 normal)
{
    float3 N = SafeNormalize(normal, float3(0.f, 1.f, 0.f));
    return SafeNormalize(tangent - N * dot(tangent, N), GetFallbackTangent(N));
}

float3 ApplyNormalMap(float3 vertexNormal, float3 vertexTangent, float3 tangentSpaceNormal)
{
    float3 N = SafeNormalize(vertexNormal, float3(0.f, 1.f, 0.f));
    float3 T = OrthogonalizeTangent(vertexTangent, N);
    float3 B = SafeNormalize(cross(N, T), GetFallbackTangent(N));

    return SafeNormalize(tangentSpaceNormal.x * T + tangentSpaceNormal.y * B + tangentSpaceNormal.z * N, N);
}

PbrMaterialSample SamplePbrMaterial(
    Texture2D<float4> baseColorMap,
    Texture2D<float4> roughnessMap,
    Texture2D<float4> metallicMap,
    Texture2D<float4> normalMap,
    SamplerState samplerState,
    float2 uv,
    float3 vertexNormal,
    float3 vertexTangent,
    float4 baseColorMultiplier,
    float roughnessScale,
    float metallicScale)
{
    PbrMaterialSample material;

    float3 tangentSpaceNormal = normalMap.Sample(samplerState, uv).xyz * 2.f - 1.f;
    material.baseColor = saturate(baseColorMap.Sample(samplerState, uv).rgb * baseColorMultiplier.rgb);
    material.roughness = clamp(roughnessMap.Sample(samplerState, uv).r * roughnessScale, 0.3f, 1.f);
    material.metallic = saturate(metallicMap.Sample(samplerState, uv).r * metallicScale);
    material.normal = ApplyNormalMap(vertexNormal, vertexTangent, tangentSpaceNormal);

    return material;
}
