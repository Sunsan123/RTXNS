/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Standalone Slang port of the Cloth Silk direct-light BRDF from
// J:/P_S1/EngineSource/Engine/Shaders/Private/ShadingModels.ush and
// MobileShadingModels.ush.

const static float SILK_PI = 3.14159265358979323846f;
const static float SILK_INV_PI = 1.0f / SILK_PI;
const static float SILK_MASK = 1.0f;
const static float SILK_DEFAULT_SPECULAR_SHIFT = 0.1f;

float SilkSquare(float x)
{
    return x * x;
}

float SilkPow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}

float3 ComputeSilkF0(float3 specular, float3 baseColor, float metallic)
{
    return lerp(0.08f * specular, baseColor, metallic);
}

float SilkFresnelFactor(float VoH)
{
    return SilkPow5(saturate(1.0f - VoH));
}

float3 SilkFresnelFromFactor(float3 specularColor, float fresnelFactor)
{
    return saturate(50.0f * specularColor.g) * fresnelFactor + (1.0f - fresnelFactor) * specularColor;
}

float SilkD_MobileGGX(float a, float NoH, float3 H, float3 N)
{
    float3 NxH = cross(N, H);
    float oneMinusNoHSqr = NoH > 0.01f ? dot(NxH, NxH) : 1.0f;
    float n = NoH * a;
    float p = 1.0f / max(oneMinusNoHSqr / a + n * NoH, 1e-2f);
    float d = p * p * SILK_INV_PI;
    return min(d, 65504.0f);
}

float SilkVis_MobileSmithJointApprox(float a, float NoV, float NoL)
{
    float visSmithV = NoL * (NoV * (1.0f - a) + a);
    float visSmithL = NoV * (NoL * (1.0f - a) + a);
    return 0.5f / (visSmithV + visSmithL + 1e-3f);
}

float SilkUltimateDVis(float roughness, float3 N, float3 H, float3 V, float NoL)
{
    float NoH = max(1e-3f, dot(N, H));
    float a = max(roughness * roughness, 1e-3f);
    float D = SilkD_MobileGGX(a, NoH, H, N);
    float NoV = max(1e-3f, saturate(dot(N, V)));
    float Vis = saturate(SilkVis_MobileSmithJointApprox(a, NoV, NoL));
    return saturate(D * Vis);
}

void SilkConvertAnisotropyToRoughness(float roughness, float anisotropy, out float roughnessT, out float roughnessB)
{
    float anisoAspect = sqrt(max(0.001f, 1.0f - 0.9f * anisotropy));
    roughnessT = max(0.001f, roughness / anisoAspect);
    roughnessB = max(0.001f, roughness * anisoAspect);
}

float SilkModifyD_GGXAnisoRoughnessFast(float aTT, float aTB, float aBB, float NoH, float ToH, float BoH)
{
    float d = ToH * ToH / aTT + BoH * BoH / aBB + NoH * NoH;
    return 1.0f / (aTB * d * d * SILK_PI + 1e-3f);
}

float SilkV_SmithJointGGXAnisoRoughnessFast(
    float ToV, float BoV, float NoV,
    float ToL, float BoL, float NoL,
    float aT2, float aB2)
{
    float lambdaV = NoL * sqrt(aT2 * ToV * ToV + aB2 * BoV * BoV + NoV * NoV);
    float lambdaL = NoV * sqrt(aT2 * ToL * ToL + aB2 * BoL * BoL + NoL * NoL);
    return 0.5f / (lambdaV + lambdaL + 1e-3f);
}

float SilkAnisotropicDVis(float roughness, float anisotropy, float3 T, float3 N, float3 L, float3 V)
{
    float3 B = normalize(cross(N, T));
    float3 H = normalize(L + V);

    float NoL = saturate(dot(N, L));
    float NoV = max(1e-3f, abs(dot(N, V)));
    float NoH = saturate(dot(N, H));

    float roughnessT = 0.0f;
    float roughnessB = 0.0f;
    SilkConvertAnisotropyToRoughness(roughness * roughness, anisotropy, roughnessT, roughnessB);

    float roughnessTT = roughnessT * roughnessT;
    float roughnessTB = roughnessT * roughnessB;
    float roughnessBB = roughnessB * roughnessB;

    float D = saturate(SilkModifyD_GGXAnisoRoughnessFast(roughnessTT, roughnessTB, roughnessBB, NoH, dot(T, H), dot(B, H)));
    float Vis = saturate(SilkV_SmithJointGGXAnisoRoughnessFast(
        dot(T, V), dot(B, V), NoV,
        dot(T, L), dot(B, L), NoL,
        roughnessTT, roughnessBB));

    return saturate(D * Vis);
}

float SilkDiffuseTerm(float roughness)
{
    return SILK_INV_PI * lerp(1.0f, lerp(1.0f, 0.5f, roughness), SILK_MASK);
}

float4 SilkOriginal(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{
    float3 N = float3(0.0f, 0.0f, 1.0f);
    float3 L = normalize(float3(sqrt(max(0.0f, 1.0f - NdotL * NdotL)), 0.0f, NdotL));
    float viewSinTheta = sqrt(max(0.0f, 1.0f - NdotV * NdotV));
    float viewCosPhi = 1.0f;
    if (viewSinTheta > 1e-4f && LdotH < 0.999f)
    {
        float3 H = normalize(float3(sqrt(max(0.0f, 1.0f - NdotH * NdotH)), 0.0f, NdotH));
        viewCosPhi = saturate((2.0f * LdotH * H.x - L.x) / viewSinTheta);
    }
    float3 V = normalize(float3(viewSinTheta * viewCosPhi, 0.0f, NdotV));
    float3 H = normalize(L + V);

    float NoL = saturate(L.z);
    float VoH = saturate(dot(V, H));
    float baseDVis = SilkUltimateDVis(roughness, N, H, V, NoL);
    float fresnelFactor = SilkFresnelFactor(VoH);

    return float4(SilkDiffuseTerm(roughness), baseDVis, baseDVis, fresnelFactor);
}

float4 Silk(float3 L, float3 V, float roughness, float anisotropy, float specularShift)
{
    float3 N = float3(0.0f, 0.0f, 1.0f);
    L = normalize(L);
    V = normalize(V);
    float3 H = normalize(L + V);

    float NoL = saturate(L.z);
    float VoH = saturate(dot(V, H));
    float actualAnisotropy = clamp(anisotropy, -0.99f, 0.99f) * SILK_MASK;
    float actualSpecularShift = saturate(specularShift);

    float3 baseTangent = float3(1.0f, 0.0f, 0.0f);
    float3 T1 = normalize(N * -actualSpecularShift + baseTangent);
    float3 T2 = normalize(N *  actualSpecularShift + baseTangent);

    float baseDVis = SilkUltimateDVis(roughness, N, H, V, NoL);
    float anisoDVis =
        SilkAnisotropicDVis(roughness, actualAnisotropy, T1, N, L, V) +
        SilkAnisotropicDVis(roughness, actualAnisotropy, T2, N, L, V);
    float fresnelFactor = SilkFresnelFactor(VoH);

    return float4(SilkDiffuseTerm(roughness), baseDVis, anisoDVis, fresnelFactor);
}

float4 Silk(float3 L, float3 V, float roughness, float anisotropy)
{
    return Silk(L, V, roughness, anisotropy, SILK_DEFAULT_SPECULAR_SHIFT);
}

float4 Silk(float Lx, float Ly, float Lz, float Vx, float Vy, float Vz, float roughness, float anisotropy, float specularShift)
{
    return Silk(float3(Lx, Ly, Lz), float3(Vx, Vy, Vz), roughness, anisotropy, specularShift);
}

float3 EvaluateSilkBrdf(float4 silkParams, float3 baseColor, float metallic, float specularControl)
{
    float3 diffuseColor = baseColor * (1.0f - metallic);
    float3 baseSpecularColor = ComputeSilkF0(float3(specularControl), baseColor, metallic);
    float3 surfaceSpecularColor = ComputeSilkF0(float3(1.0f), baseColor, metallic);

    float3 baseSpecular = silkParams.y * SilkFresnelFromFactor(baseSpecularColor, silkParams.w);
    float3 anisotropicSpecular = silkParams.z * SilkFresnelFromFactor(surfaceSpecularColor, silkParams.w);

    return silkParams.x * diffuseColor + lerp(baseSpecular, anisotropicSpecular, SILK_MASK);
}

float3 GetSilkIBLSpecularColor(float3 baseColor, float metallic, float specularControl)
{
    float3 baseSpecularColor = ComputeSilkF0(float3(specularControl), baseColor, metallic);
    float3 surfaceSpecularColor = ComputeSilkF0(float3(1.0f), baseColor, metallic);
    return lerp(baseSpecularColor, surfaceSpecularColor, SILK_MASK);
}
