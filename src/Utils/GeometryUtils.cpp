/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "GeometryUtils.h"

using namespace dm;

inline float lengthSq(const float3& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> GenerateSphere(float radius, uint32_t segmentsU, uint32_t segmentsV)
{
    std::vector<Vertex> vs;
    std::vector<uint32_t> indices;

    // Create vertices.
    for (uint32_t v = 0; v <= segmentsV; ++v)
    {
        for (uint32_t u = 0; u <= segmentsU; ++u)
        {
            float2 uv = float2(u / float(segmentsU), v / float(segmentsV));
            float theta = uv.x * 2.f * PI_f;
            float phi = uv.y * PI_f;
            float sinPhi = std::sin(phi);
            float cosPhi = std::cos(phi);
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);

            float3 dir = float3(cosTheta * sinPhi, cosPhi, sinTheta * sinPhi);

            float3 tangent = float3(-sinTheta * sinPhi, 0.f, cosTheta * sinPhi);
            if (lengthSq(tangent) < 1e-6f)
            {
                // At the poles sin(phi) -> 0 and the derivative degenerates. Use an arbitrary tangent.
                tangent = float3(1.f, 0.f, 0.f);
            }
            tangent = normalize(tangent);

            vs.push_back({ dir * radius, dir, tangent });
        }
    }

    // Create indices.
    for (uint32_t v = 0; v < segmentsV; ++v)
    {
        for (uint32_t u = 0; u < segmentsU; ++u)
        {
            uint32_t i0 = v * (segmentsU + 1) + u;
            uint32_t i1 = v * (segmentsU + 1) + (u + 1) % (segmentsU + 1);
            uint32_t i2 = (v + 1) * (segmentsU + 1) + u;
            uint32_t i3 = (v + 1) * (segmentsU + 1) + (u + 1) % (segmentsU + 1);

            indices.emplace_back(i0);
            indices.emplace_back(i1);
            indices.emplace_back(i2);

            indices.emplace_back(i2);
            indices.emplace_back(i1);
            indices.emplace_back(i3);
        }
    }

    return { vs, indices };
}
