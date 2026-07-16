/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Standalone Slang port of the PreintegratedSkin direct-light BxDF used by
// J:/P_S1/EngineSource/Engine/Shaders/Private/ShadingModels.ush.

const static float SKIN_PI = 3.14159265358979323846f;
const static float SKIN_INV_PI = 1.0f / SKIN_PI;

float SkinPow4(float x)
{
    float x2 = x * x;
    return x2 * x2;
}

float SkinPow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}

float SkinLengthSquared(float3 v)
{
    return dot(v, v);
}

float3 SkinSafeNormalize(float3 v, float3 fallback)
{
    return SkinLengthSquared(v) > 1e-12f ? normalize(v) : fallback;
}

float SkinD_GGX(float a2, float NoH)
{
    float d = (NoH * a2 - NoH) * NoH + 1.0f;
    return min(a2 / (SKIN_PI * d * d), 255.0f);
}

float SkinVis_SmithJointApprox(float a2, float NoV, float NoL)
{
    float a = sqrt(a2);
    float visSmithV = NoL * (NoV * (1.0f - a) + a);
    float visSmithL = NoV * (NoL * (1.0f - a) + a);
    return 0.5f / max(visSmithV + visSmithL, 1e-4f);
}

float3 SkinF_Schlick(float3 specularColor, float VoH)
{
    float Fc = SkinPow5(1.0f - VoH);
    return saturate(50.0f * specularColor.g) * Fc + (1.0f - Fc) * specularColor;
}

float SkinVoH(float3 L, float3 V)
{
    float3 H = SkinSafeNormalize(L + V, float3(0.0f, 0.0f, 1.0f));
    return saturate(dot(V, H));
}

float SkinSpecularDVis(float3 L, float3 V, float roughness)
{
    L = SkinSafeNormalize(L, float3(0.0f, 0.0f, 1.0f));
    V = SkinSafeNormalize(V, float3(0.0f, 0.0f, 1.0f));
    float3 H = SkinSafeNormalize(L + V, float3(0.0f, 0.0f, 1.0f));

    float NoL = saturate(L.z);
    float NoV = saturate(abs(V.z) + 1e-5f);
    float NoH = saturate(H.z);
    float a2 = max(SkinPow4(clamp(roughness, 0.3f, 1.0f)), 1e-4f);

    return SkinD_GGX(a2, NoH) * SkinVis_SmithJointApprox(a2, NoV, NoL);
}

float3 SamplePreintegratedSkinBRDF(
    Texture2D<float4> preintegratedSkinLut,
    SamplerState samplerState,
    float opacity,
    float NoL)
{
    float2 uv = float2(saturate(NoL * 0.5f + 0.5f), saturate(1.0f - opacity));
    float3 preintegratedBRDF = preintegratedSkinLut.SampleLevel(samplerState, uv, 0).rgb;
    return pow(max(preintegratedBRDF, float3(0.0f)), float3(0.45f));
}

float4 PreintegratedSkin(
    Texture2D<float4> preintegratedSkinLut,
    SamplerState samplerState,
    float3 L,
    float3 V,
    float roughness,
    float opacity)
{
    L = SkinSafeNormalize(L, float3(0.0f, 0.0f, 1.0f));
    V = SkinSafeNormalize(V, float3(0.0f, 0.0f, 1.0f));

    float3 preintegratedBRDF = SamplePreintegratedSkinBRDF(preintegratedSkinLut, samplerState, opacity, L.z);
    float specularDVis = SkinSpecularDVis(L, V, roughness);
    return float4(preintegratedBRDF, specularDVis);
}

float3 EvaluatePreintegratedSkinBrdf(
    float4 skinParams,
    float3 baseColor,
    float3 subsurfaceColor,
    float specularControl,
    float opacity,
    float blurNoL,
    float3 L,
    float3 V)
{
    float NoL = saturate(L.z);
    float VoH = SkinVoH(L, V);
    float3 specularColor = float3(0.08f * specularControl);
    float3 resolvedSubsurfaceColor = lerp(subsurfaceColor * opacity, float3(1.0f), saturate(blurNoL));

    float3 diffuse = SKIN_INV_PI * baseColor * skinParams.rgb * resolvedSubsurfaceColor;
    float3 specular = NoL * skinParams.a * SkinF_Schlick(specularColor, VoH);
    return diffuse + specular;
}

float3 GetPreintegratedSkinIBLSpecularColor(float specularControl)
{
    return float3(0.08f * specularControl);
}
