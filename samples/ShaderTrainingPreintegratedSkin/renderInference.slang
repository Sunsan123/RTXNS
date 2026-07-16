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

import CooperativeVectorAutoDiff;
import CooperativeVectorFunctions;
import Utils;
import Activation;
import MLP;
import PreintegratedSkin;
import PreintegratedSkinMLP;
import IBL;
import PBRTextures;

DECLARE_CBUFFER(InferenceConstantBufferEntry, gConst, 0, 0);
ByteAddressBuffer gMLPParams : REGISTER_SRV(0, 0);
Texture2D<float4> gEnvironmentMap : REGISTER_SRV(1, 0);
Texture2D<float4> gBaseColorMap : REGISTER_SRV(2, 0);
Texture2D<float4> gRoughnessMap : REGISTER_SRV(3, 0);
Texture2D<float4> gMetallicMap : REGISTER_SRV(4, 0);
Texture2D<float4> gNormalMap : REGISTER_SRV(5, 0);
Texture2D<float4> gSubsurfaceColorMap : REGISTER_SRV(6, 0);
Texture2D<float4> gPreintegratedSkinLut : REGISTER_SRV(7, 0);
SamplerState gSampler : REGISTER_SAMPLER(0, 0);

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
    float3 i_pos : POSITION,
    float3 i_norm : NORMAL,
    float3 i_tan : TANGENT,
    float2 i_uv : TEXCOORD,
    out PS_INPUT output)
{
    output.pos = mul(float4(i_pos, 1), gConst.viewProject);
    output.norm = i_norm;
    output.view = gConst.cameraPos.xyz - i_pos;
    output.tan = i_tan;
    output.uv = i_uv;
}

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
        gSampler,
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
    float opacity = saturate(gConst.anisotropy);
    float3 subsurfaceColor = saturate(gSubsurfaceColorMap.Sample(gSampler, input.uv).rgb * gConst.specularShift);
    float blurNoL = saturate(dot(normalize(input.norm), light));

    float4 outParams = PreintegratedSkinMLP<NUM_HIDDEN_LAYERS, HIDDEN_NEURONS>(
        L,
        V,
        material.roughness,
        opacity,
        gMLPParams,
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets)
    );

    if (gConst.onlyNeuralDebug != 0)
    {
        o_color = float4(saturate(outParams.rgb), 1.0f);
        return;
    }

    float3 colorh = EvaluatePreintegratedSkinBrdf(outParams, material.baseColor, subsurfaceColor, gConst.specular, opacity, blurNoL, L, V) * gConst.lightIntensity.rgb;
    colorh = AddImageBasedLighting(
        gEnvironmentMap,
        gSampler,
        colorh,
        material.baseColor * subsurfaceColor,
        GetPreintegratedSkinIBLSpecularColor(gConst.specular),
        material.metallic,
        material.roughness,
        norm,
        view,
        gConst.iblSettings);

    o_color = float4(colorh, 1.0f);
}
