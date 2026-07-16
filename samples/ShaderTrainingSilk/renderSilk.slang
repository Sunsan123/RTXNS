/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

DECLARE_CBUFFER(DirectConstantBufferEntry, gConst, 0, 0);
Texture2D<float4> gEnvironmentMap : REGISTER_SRV(1, 0);
Texture2D<float4> gBaseColorMap : REGISTER_SRV(2, 0);
Texture2D<float4> gRoughnessMap : REGISTER_SRV(3, 0);
Texture2D<float4> gMetallicMap : REGISTER_SRV(4, 0);
Texture2D<float4> gNormalMap : REGISTER_SRV(5, 0);
SamplerState gEnvironmentSampler : REGISTER_SAMPLER(0, 0);

struct PS_INPUT
{
    float4 pos : SV_Position;
    float3 norm : NORMAL;
    float3 view : VIEW;
    float3 tan : TANGENT;
    float2 uv : TEXCOORD;
}

[shader("vertex")]
void main_vs(
    float3 i_pos: POSITION,
    float3 i_norm: NORMAL,
    float3 i_tan: TANGENT,
    float2 i_uv: TEXCOORD,
    out PS_INPUT output)
{
    output.pos = mul(float4(i_pos, 1), gConst.viewProject);
    output.norm = i_norm;
    output.tan = i_tan;
    output.uv = i_uv;
    output.view = gConst.cameraPos.xyz - i_pos;
}

import Silk;
import IBL;
import PBRTextures;

float3 ToTangentSpace(float3 direction, float3 normal, float3 tangent)
{
    float3 bitangent = SafeNormalize(cross(normal, tangent), GetFallbackTangent(normal));
    return float3(dot(direction, tangent), dot(direction, bitangent), max(0.0f, dot(direction, normal)));
}

[shader("fragment")]
void main_ps(
    PS_INPUT input,
    out float4 o_color : SV_Target0)
{
    float3 view = normalize(input.view);
    float3 light = normalize(-gConst.lightDir.xyz);
    PbrMaterialSample material = SamplePbrMaterial(
        gBaseColorMap,
        gRoughnessMap,
        gMetallicMap,
        gNormalMap,
        gEnvironmentSampler,
        input.uv,
        input.norm,
        input.tan,
        gConst.baseColor,
        gConst.roughness,
        gConst.metallic);

    float3 norm = material.normal;
    float3 tan = OrthogonalizeTangent(input.tan, norm);
    float3 L = ToTangentSpace(light, norm, tan);
    float3 V = ToTangentSpace(view, norm, tan);
    float NdotL = max(0.0f, dot(norm, light));
    float anisotropy = clamp(gConst.anisotropy, -0.99f, 0.99f);

    float4 outParams = Silk(L, V, material.roughness, anisotropy, gConst.specularShift);
    float3 colorh = EvaluateSilkBrdf(outParams, material.baseColor, material.metallic, gConst.specular) * NdotL * gConst.lightIntensity.rgb;
    colorh = AddImageBasedLighting(
        gEnvironmentMap,
        gEnvironmentSampler,
        colorh,
        material.baseColor,
        GetSilkIBLSpecularColor(material.baseColor, material.metallic, gConst.specular),
        material.metallic,
        material.roughness,
        norm,
        view,
        gConst.iblSettings);

    o_color = float4(colorh, 1.0f);
}
