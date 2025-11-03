/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

//----------- 核心辅助函数

const static float PI = 3.14159265358979323846;

// 辅助函数: Sqr
float Sqr(float x) { return x * x; }

// 菲涅尔
float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

// --- 用于 透明涂层 的函数 (来自你的原始代码) ---

// 用于 Clearcoat D
float Gtr1(float NdotH, float a)
{
    if (a >= 1)
    {
        return 1 / PI;
    }
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

// 用于 Clearcoat G
float SmithGGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

// --- Kajiya-Kay 特定的辅助函数 ---

/**
 * @brief 计算 Kajiya-Kay 的 'shininess' (光泽度) 指数
 * 我们从 GGX Aniso 的逻辑中派生它，以保持参数一致性。
 */
float KajiyaKayExponent(float roughness, float anisotropy)
{
    float alpha = max(.001f, roughness * roughness); 
    // aspect 范围 [~0.3, 1]
    float aspect = sqrt(1.0 - (anisotropy * 0.9)); 
    
    // ay 是沿副切线(B)的粗糙度，即 "横跨" 纤维/划痕的粗糙度
    float ay = max(0.001, alpha / aspect); 

    // Kajiya-Kay 的指数 'm' 控制高光在 sin(T,H) 上的聚集程度
    // 我们将 GGX alpha (ay) 映射到一个类似 Blinn-Phong 的指数 'm'
    // 这是一个常见的映射: m = 2 / (alpha^2) - 2
    // 我们使用 m = 2 / ay - 2 (ay 已经是 alpha)
    float shininess = 2.0 / ay - 2.0;
    
    return max(2.0, shininess); // 确保指数至少为 2
}


// --- 修改后的 Disney BRDF (使用 Kajiya-Kay 高光) ---

/**
 * @brief Disney-style BRDF, 但高光项被替换为 Kajiya-Kay 模型
 * @param NdotL, NdotV, NdotH, LdotH 基础点积
 * @param roughness 材质粗糙度
 * @param TdotH (必需) 半向量 H 与 切线 T 的点积
 * @param anisotropy (必需) 各向异性程度 [0, 1]
 */
float4 DisneyKajiyaKay(
    // 原始参数 (保留了大部分)
    float NdotL, float NdotV, float NdotH, float LdotH, float roughness,
    // Kajiya-Kay 需要的参数 (已减少)
    float TdotH,
    float anisotropy)
{
    anisotropy = clamp(anisotropy, -0.99f, 0.99f);

    // --- 1. 漫反射 & 次表面散射 (不变) ---
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - .5f) + .5f);

    // --- 2. 高光 (替换为 Kajiya-Kay) ---

    // a) 计算 Kajiya-Kay 指数 (shininess)
    float m = KajiyaKayExponent(roughness, anisotropy);

    // b) 计算 Kajiya-Kay D 项 (分布)
    // D = pow(sin(T, H), m)
    // sin(T, H) = sqrt(1 - cos(T,H)^2)
    // cos(T, H) = TdotH
    float sinTH_Sqr = 1.0 - Sqr(TdotH);
    float sinTH = sqrt(max(0.0, sinTH_Sqr)); // 确保为正
    float D_kajiya = pow(sinTH, m);

    // c) 归一化 (使用 Blinn-Phong 的归一化因子)
    // 这是一个经验模型，不是物理精确的，但总比没有好
    float KajiyaNorm = (m + 2.0) / (2.0 * PI);
    float Ds = KajiyaNorm * D_kajiya;

    // d) 菲涅尔 (F)
    float FH = SchlickFresnel(LdotH);
    
    // e) 几何遮蔽 (G)
    // 纯粹的 Kajiya-Kay 模型 *没有* 几何项。
    // 我们将其设为 1.0 (即没有遮蔽)。
    float Gs = 1.0; 

    // --- 3. 透明涂层 (不变) ---
    // (注意: 涂层仍使用 GGX Gtr1 和 SmithGGX)
    float Dr = Gtr1(NdotH, .01f);
    float Fr = lerp(.04f, 1.0f, FH);
    float Gr = SmithGGX(NdotL, .25f) * SmithGGX(NdotV, .25f);

    // --- 4. 组合 ---
    // 返回值结构与原始代码保持一致:
    // x: 漫反射项 (ss / PI)
    // y: 高光 D*G 项 (Kajiya D * 1.0)
    // z: 高光 F 项
    // w: 透明涂层项
    return float4((1 / PI) * ss, Gs * Ds, FH, .25 * Gr * Fr * Dr);
}