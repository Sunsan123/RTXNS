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

import Silk;
import IBL;
import PBRTextures;

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
    float3 h = normalize(light + view);

    float NdotL = max(0.0f, dot(norm, light));
    float NdotV = max(0.0f, dot(norm, view));
    float NdotH = max(0.0f, dot(norm, h));
    float LdotH = max(0.0f, dot(light, h));

    float4 outParams = SilkOriginal(NdotL, NdotV, NdotH, LdotH, material.roughness);
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
