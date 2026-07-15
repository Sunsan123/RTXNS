/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

static const float IBL_PI = 3.14159265358979323846f;

float2 DirectionToEquirectUV(float3 direction, float rotation)
{
    float s, c;
    sincos(rotation, s, c);

    float3 d = normalize(direction);
    d = float3(d.x * c - d.z * s, d.y, d.x * s + d.z * c);

    float u = atan2(d.x, -d.z) / (2.f * IBL_PI) + 0.5f;
    float v = acos(clamp(d.y, -1.f, 1.f)) / IBL_PI;
    return float2(u, v);
}

float3 SampleEquirectEnvironment(
    Texture2D<float4> environmentMap,
    SamplerState environmentSampler,
    float3 direction,
    float lod,
    float rotation)
{
    return max(environmentMap.SampleLevel(environmentSampler, DirectionToEquirectUV(direction, rotation), lod).rgb, float3(0.f));
}

float3 AddImageBasedLighting(
    Texture2D<float4> environmentMap,
    SamplerState environmentSampler,
    float3 directColor,
    float3 diffuseAlbedo,
    float3 specularF0,
    float metallic,
    float roughness,
    float3 normal,
    float3 view,
    float4 iblSettings)
{
    if (iblSettings.w <= 0.f || iblSettings.x <= 0.f)
    {
        return directColor;
    }

    float mipCount = max(iblSettings.z, 1.f);
    float diffuseLod = max(mipCount - 2.f, 0.f);
    float specularLod = roughness * max(mipCount - 1.f, 0.f);

    float3 N = normalize(normal);
    float3 V = normalize(view);
    float3 R = reflect(-V, N);

    float3 diffuseEnvironment = SampleEquirectEnvironment(environmentMap, environmentSampler, N, diffuseLod, iblSettings.y);
    float3 specularEnvironment = SampleEquirectEnvironment(environmentMap, environmentSampler, R, specularLod, iblSettings.y);

    float NdotV = max(0.f, dot(N, V));
    float3 fresnel = specularF0 + (float3(1.f) - specularF0) * pow(max(1.f - NdotV, 0.f), 5.f);

    float3 diffuseIBL = diffuseEnvironment * diffuseAlbedo * (1.f - metallic) * 0.35f;
    float3 specularIBL = specularEnvironment * fresnel * lerp(1.f, 0.5f, roughness);

    return directColor + iblSettings.x * (diffuseIBL + specularIBL);
}
