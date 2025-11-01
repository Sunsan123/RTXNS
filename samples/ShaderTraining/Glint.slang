/*
 * Glint BRDF (PG 2020) 移植版
 * 原始 GLSL 代码: "Procedural Physically based BRDF for Real-Time Rendering of Glints"
 * 作者: Xavier Chermain, Basile Sauvage, Jean-Michel Dishler, Carsten Dachsbacher
 * 原始 ShaderToy: https://www.shadertoy.com/view/WlGXRt
 *
 * 此文件将原始逻辑重构为 Slang/HLSL 库格式，类似于 Disney.slang 的结构。
 */

//=========================================================================================================================
//================================================ 数学常量 =================================================
//=========================================================================================================================
static const float PI = 3.141592f;
static const float IPI = 0.318309f;
static const float ISQRT2 = 0.707106f;

//=========================================================================================================================
//================================================== 材质参数 ===================================================
//=========================================================================================================================
// 闪点材质的粗糙度 [0.1, 1.]
static const float ALPHA_X = 0.5f;
static const float ALPHA_Y = 0.5f;

// 微表面相对面积 [0.01, 1.]
static const float MICROFACETRELATIVEAREA = 1.0f;

// 对数微表面密度 [5., 25.]
static const float LOGMICROFACETDENSITY = 5.0f;

// 像素足迹的最大各向异性
static const float MAXANISOTROPY = 8.0f;

// 清漆材质 (添加一个粗糙度较小的镜面波瓣)
static const bool VARNISHED = true;

//=========================================================================================================================
//============================================== 字典参数 =============================================
//=========================================================================================================================
// 字典生成期间使用的粗糙度
static const float ALPHA_DIC = 0.5f;
// 分布数量 (在shadertoy中设为极大值以模拟)
static const int N = 999999;
// LOD 层级
static const int NLEVELS = 8;
// 边缘分布的分辨率 (未使用，因为 P_procedural 是即时生成的)
// static const int DISTRESOLUTION = 32;

//=========================================================================================================================
//=============================================== Beckmann 各向异性 NDF ================================================
//=========================================================================================================================
float p22_beckmann_anisotropic(float x, float y, float alpha_x, float alpha_y)
{
    float x_sqr = x * x;
    float y_sqr = y * y;
    float sigma_x = alpha_x * ISQRT2;
    float sigma_y = alpha_y * ISQRT2;
    float sigma_x_sqr = sigma_x * sigma_x;
    float sigma_y_sqr = sigma_y * sigma_y;
    return (
        exp(-0.5f * ((x_sqr / sigma_x_sqr) + (y_sqr / sigma_y_sqr)))
        / //-------------------------------------------------------------------
        (2.0f * PI * sigma_x * sigma_y)
        );
}

float ndf_beckmann_anisotropic(float3 omega_h, float alpha_x, float alpha_y)
{
    float slope_x = -(omega_h.x / omega_h.z);
    float slope_y = -(omega_h.y / omega_h.z);
    float cos_theta = omega_h.z;
    float cos_2_theta = omega_h.z * omega_h.z;
    float cos_4_theta = cos_2_theta * cos_2_theta;
    float beckmann_p22 = p22_beckmann_anisotropic(slope_x, slope_y, alpha_x, alpha_y);
    return (
        beckmann_p22
        / //---------------------------
        cos_4_theta
        );
}

//=========================================================================================================================
//======================================== Schlick 菲涅尔近似 ===============================================
//=========================================================================================================================
float3 fresnel_schlick(in float wo_dot_wh, in float3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - wo_dot_wh, 5.0f);
}

//=========================================================================================================================
//===================================== 标准微表面 BRDF (Cook-Torrance 1982) =========================================
//=========================================================================================================================
// (这是 Glint.glsl 中的 f_specular，用于清漆层)
// 注意：它返回 f_r * NdotL (假设 Li 在外部乘以)
float3 f_specular(float3 wo, float3 wi)
{
    if (wo.z <= 0.0f) return float3(0.0f, 0.0f, 0.0f);
    if (wi.z <= 0.0f) return float3(0.0f, 0.0f, 0.0f);
    float3 wh = normalize(wo + wi);
    if (wh.z <= 0.0f) return float3(0.0f, 0.0f, 0.0f);
    // 局部遮挡
    if (dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) return float3(0.0f);
    float wi_dot_wh = clamp(dot(wi, wh), 0.0f, 1.0f);

    // 使用 0.1 的固定粗糙度
    float D = ndf_beckmann_anisotropic(wh, 0.1f, 0.1f);
    // V-cavity 遮挡
    float G1wowh = min(1.0f, 2.0f * wh.z * wo.z / dot(wo, wh));
    float G1wiwh = min(1.0f, 2.0f * wh.z * wi.z / dot(wi, wh));
    float G = G1wowh * G1wiwh;

    // Glint.glsl 中 F0 硬编码为 1.0
    float3 F = fresnel_schlick(wi_dot_wh, float3(1.0f, 1.0f, 1.0f));

    // 返回 (D * F * G) / ( 4 * wo.z ) == (D * F * G) / ( 4 * NdotV )
    // 这等于 BRDF * NdotL
    return (D * F * G) / (4.0f * wo.z);
}

//=========================================================================================================================
//=============================================== 漫反射 Lambertian BRDF =================================================
//=========================================================================================================================
// (这是 Glint.glsl 中的 f_diffuse)
// 注意：它返回 f_r * NdotL
float3 f_diffuse(float3 wo, float3 wi)
{
    if (wo.z <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);
    if (wi.z <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);

    // albedo * (1/PI) * NdotL
    // Glint.glsl 中硬编码了红色 albedo
    return float3(0.8f, 0.0f, 0.0f) * IPI * wi.z;
}

//=========================================================================================================================
//=============================================== 反误差函数 ==================================================
//=========================================================================================================================
float erfinv(float x)
{
    float w, p;
    w = -log((1.0f - x) * (1.0f + x));
    if (w < 5.000000f)
    {
        w = w - 2.500000f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    }
    else
    {
        w = sqrt(w) - 3.000000f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

//=========================================================================================================================
//================================================== Hash 函数 (Inigo Quilez) ========================================================
//=========================================================================================================================
float hashIQ(uint n)
{
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float(n & 0x7fffffffU) / float(0x7fffffff);
}

//=========================================================================================================================
//=============================================== 金字塔 LOD 层级大小 ===============================================
//=========================================================================================================================
int pyramidSize(int level)
{
    return (int)(pow(2.0f, (float)(NLEVELS - 1 - level)));
}

float normalDistribution1D(float x, float mean, float std_dev) {
    float xMinusMean = x - mean;
    float xMinusMeanSqr = xMinusMean * xMinusMean;
    return exp(-xMinusMeanSqr / (2.0f * std_dev * std_dev)) /
        (std_dev * 2.506628f); // 2.506628 \approx sqrt(2 * \pi)
}

//=========================================================================================================================
//========================================= 从正态分布采样 ===========================================
//=========================================================================================================================
float sampleNormalDistribution(float U, float mu, float sigma)
{
    float x = sigma * 1.414213f * erfinv(2.0f * U - 1.0f) + mu;
    return x;
}

//=========================================================================================================================
//==================================== 边缘分布 P 的程序化生成 ====================================
//=========================================================================================================================
float P_procedural(float x, int i, int level) {
    // 偶函数
    x = abs(x);
    // 4 倍标准差之外视为 0
    float sigma_dist_4 = 4.0f * ALPHA_DIC / 1.414214f;
    if (x >= sigma_dist_4) return 0.0f;

    int nMicrofacetsCurrentLevel = (int)(pow(2.0f, (float)(level)));
    float density = 0.0f;
    
    // Shadertoy 中即时生成，限制波瓣数量
    nMicrofacetsCurrentLevel = min(32, nMicrofacetsCurrentLevel);

    for (int n = 0; n < nMicrofacetsCurrentLevel; ++n) {
        float U_n = hashIQ(uint(i * 7333 + n * 5741));
        // std_dev = ALPHA_DIC / sqrt(2)
        float currentMean = sampleNormalDistribution(U_n, 0.0f, ALPHA_DIC / 1.414214f);
        density += normalDistribution1D(x, currentMean, 0.05f) +
            normalDistribution1D(-x, currentMean, 0.05f);
    }
    // 0.5 是因为循环中加了两次 PDF
    return density / (float)(nMicrofacetsCurrentLevel) * 0.5f;
}

//=========================================================================================================================
//=================== 多尺度、旋转和缩放的坡度分布函数 (Eq. 11, Alg. 3) ======================
//=========================================================================================================================
float P22_theta_alpha(float2 slope_h, int l, int s0, int t0)
{
    // Coherent index (Eq. 18, Alg. 3, line 1)
    s0 *= (1 << l);
    t0 *= (1 << l);

    // Seed (Alg. 3, line 2)
    int rngSeed = s0 + 1549 * t0;

    // (Alg.3, line 3)
    float uMicrofacetRelativeArea = hashIQ(uint(rngSeed) * 13U);
    // (Alg.3, line 4)
    if (uMicrofacetRelativeArea > MICROFACETRELATIVEAREA)
        return 0.0f;

    // (Alg. 3, line 5)
    float n = pow(2.0f, (float)(2 * l - (2 * (NLEVELS - 1))));
    n *= exp(LOGMICROFACETDENSITY);

    // (Alg. 3, line 6)
    float l_dist = log(n) / 1.38629f; // 2. * log(2)

    // (Alg. 3, line 7)
    float uDensityRandomisation = hashIQ(uint(rngSeed) * 2171U);

    // (Notation: \zeta)
    float densityRandomisation = 2.0f;

    // (Alg. 3, line 8)
    l_dist = sampleNormalDistribution(uDensityRandomisation, l_dist, densityRandomisation);

    // (Alg. 3, line 9)
    int l_disti = clamp((int)(round(l_dist)), 0, NLEVELS);

    // (Alg. 3, line 10)
    if (l_disti == NLEVELS)
        return p22_beckmann_anisotropic(slope_h.x, slope_h.y, ALPHA_X, ALPHA_Y);

    // (Alg. 3, line 13)
    float uTheta = hashIQ(uint(rngSeed));
    float theta = 2.0f * PI * uTheta;
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);

    float2 scaleFactor = float2(ALPHA_X / ALPHA_DIC,
                            ALPHA_Y / ALPHA_DIC);

    // 旋转和缩放坡度 (Alg. 3, line 16)
    slope_h = float2(slope_h.x * cosTheta / scaleFactor.x + slope_h.y * sinTheta / scaleFactor.y,
                   -slope_h.x * sinTheta / scaleFactor.x + slope_h.y * cosTheta / scaleFactor.y);

    // (Alg. 3, line 17)
    float u1 = hashIQ(uint(rngSeed) * 16807U);
    float u2 = hashIQ(uint(rngSeed) * 48271U);

    // (Alg. 3, line 18)
    int i = (int)(u1 * (float)(N));
    int j = (int)(u2 * (float)(N));

    float P_i = P_procedural(slope_h.x, i, l_disti);
    float P_j = P_procedural(slope_h.y, j, l_disti);

    // (Alg. 3, line 19)
    return P_i * P_j / (scaleFactor.x * scaleFactor.y);
}

//=========================================================================================================================
//========================================= Alg. 2, P-SDF (离散 LOD) ==============================================
//=========================================================================================================================
// EWA (Elliptically Weighted Average) 滤波
float P22_floorP(int l, float2 slope_h, float2 st, float2 dst0, float2 dst1)
{
    // 转换表面坐标到LOD层级
    float pyrSize = (float)(pyramidSize(l));
    st[0] = st[0] * pyrSize - 0.5f;
    st[1] = st[1] * pyrSize - 0.5f;
    dst0[0] *= pyrSize;
    dst0[1] *= pyrSize;
    dst1[0] *= pyrSize;
    dst1[1] *= pyrSize;

    // 计算椭圆系数
    float A = dst0[1] * dst0[1] + dst1[1] * dst1[1] + 1.0f;
    float B = -2.0f * (dst0[0] * dst0[1] + dst1[0] * dst1[1]);
    float C = dst0[0] * dst0[0] + dst1[0] * dst1[0] + 1.0f;
    float invF = 1.0f / (A * C - B * B * 0.25f);
    A *= invF;
    B *= invF;
    C *= invF;

    // 计算椭圆包围盒
    float det = -B * B + 4.0f * A * C;
    float invDet = 1.0f / det;
    float uSqrt = sqrt(det * C), vSqrt = sqrt(A * det);
    int s0 = (int)(ceil(st[0] - 2.0f * invDet * uSqrt));
    int s1 = (int)(floor(st[0] + 2.0f * invDet * uSqrt));
    int t0 = (int)(ceil(st[1] - 2.0f * invDet * vSqrt));
    int t1 = (int)(floor(st[1] + 2.0f * invDet * vSqrt));

    float sum = 0.0f;
    float sumWts = 0.0f;
    int nbrOfIter = 0;

    for (int it = t0; it <= t1; ++it)
    {
        float tt = (float)(it) - st[1];
        for (int is = s0; is <= s1; ++is)
        {
            float ss = (float)(is) - st[0];
            float r2 = A * ss * ss + B * ss * tt + C * tt * tt;
            if (r2 < 1.0f)
            {
                float alpha = 2.0f;
                float W_P = exp(-alpha * r2) - exp(-alpha);
                // (Alg. 2, line 3)
                sum += P22_theta_alpha(slope_h, l, is, it) * W_P;
                sumWts += W_P;
            }
            nbrOfIter++;
            if (nbrOfIter > 100) break; // 循环保护
        }
        if (nbrOfIter > 100) break; // 循环保护
    }
    
    // 避免除以零
    if (sumWts == 0.0f) return 0.0f;
    
    return sum / sumWts;
}

//=========================================================================================================================
//=============================== 程序化物理闪点 BRDF (Alg. 1, Eq. 14) ===============================
//=========================================================================================================================
// 注意：它返回 f_r * NdotL (假设 Li 在外部乘以)
float3 f_P(float3 wo, float3 wi, float2 uv)
{
    if (wo.z <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);
    if (wi.z <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);

    // (Alg. 1, line 1)
    float3 wh = normalize(wo + wi);
    if (wh.z <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);

    // 局部遮挡
    if (dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f)
        return float3(0.0f);

    // (Eq. 1, Alg. 1, line 2)
    float2 slope_h = float2(-wh.x / wh.z, -wh.y / wh.z);

    float2 texCoord = uv;
    float D_P = 0.0f;
    float P22_P = 0.0f;

    // (Alg. 1, line 3)
    // 关键：这里需要屏幕空间导数
    float2 dst0 = dFdx(texCoord);
    float2 dst1 = dFdy(texCoord);

    float dst0Length = length(dst0);
    float dst1Length = length(dst1);

    if (dst0Length < dst1Length)
    {
        float2 tmp = dst0;
        dst0 = dst1;
        dst1 = tmp;
    }
    float majorLength = length(dst0);
    // (Alg. 1, line 5)
    float minorLength = length(dst1);

    // (Alg. 1, line 4)
    if (minorLength * MAXANISOTROPY < majorLength && minorLength > 0.0f)
    {
        float scale = majorLength / (minorLength * MAXANISOTROPY);
        dst1 *= scale;
        minorLength *= scale;
    }

    if (minorLength == 0.0f)
    {
        // 没有足迹，回退到标准 Beckmann
        D_P = ndf_beckmann_anisotropic(wh, ALPHA_X, ALPHA_Y);
    }
    else
    {
        // (Alg. 1, line 6)
        float l = max(0.0f, (float)(NLEVELS)-1.0f + log2(minorLength));
        int il = (int)(floor(l));

        // (Alg. 1, line 7)
        float w = l - (float)(il);

        // (Alg. 1, line 8)
        // 确保 il+1 不越界
        int il_next = min(il + 1, NLEVELS - 1);
        
        P22_P = lerp(P22_floorP(il, slope_h, texCoord, dst0, dst1),
                     P22_floorP(il_next, slope_h, texCoord, dst0, dst1),
                     w);

        // (Eq. 13, Alg. 1, line 10)
        D_P = P22_P / (wh.z * wh.z * wh.z * wh.z);
    }

    // V-cavity 遮挡
    float G1wowh = min(1.0f, 2.0f * wh.z * wo.z / dot(wo, wh));
    float G1wiwh = min(1.0f, 2.0f * wh.z * wi.z / dot(wi, wh));
    float G = G1wowh * G1wiwh;

    // Glint.glsl 中 F 硬编码为 1.0
    float3 F = float3(1.0f, 1.0f, 1.0f);

    // (Eq. 14, Alg. 1, line 11)
    // 返回 (F * G * D_P) / (4 * wo.z) == (F * G * D_P) / (4 * NdotV)
    // 这等于 BRDF * NdotL
    return (F * G * D_P) / (4.0f * wo.z);
}


//=========================================================================================================================
//=========================================== 主 BRDF 结构体和函数 ===========================================
//=========================================================================================================================

// 定义一个结构体来返回所有 BRDF 组件
// 这些都是 f_r * NdotL 的结果，可直接乘以 Li (光照强度)
struct GlintBRDFResult
{
    float3 diffuse; // 漫反射项 (f_diffuse)
    float3 glint;   // 闪点项 (f_P)
    float3 varnish; // 清漆/镜面项 (f_specular)
};

/**
 * @brief 计算 Glint BRDF (PG 2020) 的所有分量。
 * * 此函数是 Glint.glsl 逻辑的核心封装，用于替换 Disney.slang。
 * 它返回的每个分量都已经是 f_r * NdotL 的形式，
 * 调用者只需将其乘以光源颜色/强度 (Li) 即可。
 * * @param wo 局部切线空间中的观察向量 (View Vector)
 * @param wi 局部切线空间中的光照向量 (Light Vector)
 * @param uv 用于程序化闪点图案的 UV 坐标。
 * **重要**: 此 UV 必须附加了屏幕空间导数 (dFdx/dFdy)，
 * 否则 f_P 函数将无法正确计算 LOD。
 * @return GlintBRDFResult 结构体，包含 diffuse, glint 和 varnish 分量。
 */
GlintBRDFResult CalculateGlintBRDF(float3 wo, float3 wi, float2 uv)
{
    GlintBRDFResult result;

    // 1. 计算漫反射 (来自 Glint.glsl 的硬编码红色)
    result.diffuse = f_diffuse(wo, wi);

    // 2. 计算核心的程序化闪点 (Glint)
    result.glint = f_P(wo, wi, uv);

    // 3. 计算清漆 (Varnish) 层 (来自 Glint.glsl 的标准镜面)
    result.varnish = float3(0.0f, 0.0f, 0.0f);
    if (VARNISHED)
    {
        result.varnish = f_specular(wo, wi);
    }

    return result;
}
