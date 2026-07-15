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
import DisneyMLP;
import IBL;
import PBRTextures;

DECLARE_CBUFFER(InferenceConstantBufferEntry, gConst, 0, 0);
ByteAddressBuffer gMLPParams : REGISTER_SRV(0, 0);
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

[shader("fragment")]
void main_ps(
    PS_INPUT input,
    out float4 o_color : SV_Target0)
{
    // Prepare input parameters
    float3 view = normalize(input.view);
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
    float3 h = normalize(-gConst.lightDir.xyz + view);

    float NdotL = max(0.f, dot(norm, -gConst.lightDir.xyz));
    float NdotV = max(0.f, dot(norm, view));
    float NdotH = max(0.f, dot(norm, h));
    float LdotH = max(0.f, dot(h, -gConst.lightDir.xyz));

    // Calculate approximated core shader
    float4 outParams = DisneyMLPOriginal<NUM_HIDDEN_LAYERS, ORIGINAL_HIDDEN_NEURONS>(
        NdotL, NdotV, NdotH, LdotH, material.roughness,
        gMLPParams,
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets)
    );

    // Calculate final color
    float3 Cdlin = material.baseColor;
    float3 Cspec0 = lerp(gConst.specular * float3(.08f), Cdlin, material.metallic);
    float3 brdfn = outParams.x * Cdlin * (1 - material.metallic) + outParams.y * lerp(Cspec0, float3(1), outParams.z) + outParams.w;
    float3 colorh = brdfn * NdotL * gConst.lightIntensity.rgb;
    colorh = AddImageBasedLighting(gEnvironmentMap, gEnvironmentSampler, colorh, Cdlin, Cspec0, material.metallic, material.roughness, norm, view, gConst.iblSettings);

    o_color = float4(colorh, 1.f);
}
