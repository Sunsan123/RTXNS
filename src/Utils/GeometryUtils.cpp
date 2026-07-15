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

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

using namespace dm;

inline float lengthSq(const float3& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

namespace
{

struct ObjVertexRef
{
    int position = 0;
    int texcoord = 0;
    int normal = 0;
};

bool ParseObjIndex(std::string_view text, int& out)
{
    if (text.empty())
    {
        return false;
    }

    int value = 0;
    const auto* begin = text.data();
    const auto* end = begin + text.size();
    auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc() || result.ptr != end)
    {
        return false;
    }

    out = value;
    return true;
}

int ResolveObjIndex(int index, size_t count)
{
    if (index > 0)
    {
        return index - 1;
    }

    if (index < 0)
    {
        return int(count) + index;
    }

    return -1;
}

ObjVertexRef ParseObjVertexRef(std::string_view token)
{
    ObjVertexRef ref;

    const size_t firstSlash = token.find('/');
    if (firstSlash == std::string_view::npos)
    {
        ParseObjIndex(token, ref.position);
        return ref;
    }

    ParseObjIndex(token.substr(0, firstSlash), ref.position);

    const size_t secondSlash = token.find('/', firstSlash + 1);
    if (secondSlash == std::string_view::npos)
    {
        ParseObjIndex(token.substr(firstSlash + 1), ref.texcoord);
        return ref;
    }

    ParseObjIndex(token.substr(firstSlash + 1, secondSlash - firstSlash - 1), ref.texcoord);
    ParseObjIndex(token.substr(secondSlash + 1), ref.normal);
    return ref;
}

float3 SafeNormalize(const float3& v, const float3& fallback)
{
    if (lengthSq(v) < 1e-12f)
    {
        return fallback;
    }

    return normalize(v);
}

float3 GetFallbackTangent(const float3& normal)
{
    const float3 axis = std::fabs(normal.y) < 0.9f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
    return SafeNormalize(cross(axis, normal), float3(1.f, 0.f, 0.f));
}

float3 OrthogonalizeTangent(const float3& tangent, const float3& normal)
{
    const float3 projected = tangent - normal * dot(normal, tangent);
    return SafeNormalize(projected, GetFallbackTangent(normal));
}

} // namespace

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

            vs.push_back({ dir * radius, dir, tangent, uv });
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

std::pair<std::vector<Vertex>, std::vector<uint32_t>> LoadObjModel(const std::filesystem::path& fileName, float targetMaxExtent)
{
    std::ifstream file(fileName);
    if (!file)
    {
        return {};
    }

    std::vector<float3> positions;
    std::vector<float2> texcoords;
    std::vector<float3> normals;
    std::vector<std::vector<ObjVertexRef>> faces;

    std::string line;
    while (std::getline(file, line))
    {
        if (line.rfind("v ", 0) == 0)
        {
            std::istringstream stream(line.substr(2));
            float x = 0.f;
            float y = 0.f;
            float z = 0.f;
            stream >> x >> y >> z;
            positions.emplace_back(x, y, z);
        }
        else if (line.rfind("vt ", 0) == 0)
        {
            std::istringstream stream(line.substr(3));
            float u = 0.f;
            float v = 0.f;
            stream >> u >> v;
            texcoords.emplace_back(u, v);
        }
        else if (line.rfind("vn ", 0) == 0)
        {
            std::istringstream stream(line.substr(3));
            float x = 0.f;
            float y = 0.f;
            float z = 0.f;
            stream >> x >> y >> z;
            normals.emplace_back(SafeNormalize(float3(x, y, z), float3(0.f, 1.f, 0.f)));
        }
        else if (line.rfind("f ", 0) == 0)
        {
            std::istringstream stream(line.substr(2));
            std::string token;
            std::vector<ObjVertexRef> face;
            while (stream >> token)
            {
                if (!token.empty() && token[0] == '#')
                {
                    break;
                }

                ObjVertexRef ref = ParseObjVertexRef(token);
                ref.position = ResolveObjIndex(ref.position, positions.size());
                ref.texcoord = ResolveObjIndex(ref.texcoord, texcoords.size());
                ref.normal = ResolveObjIndex(ref.normal, normals.size());
                face.push_back(ref);
            }

            if (face.size() >= 3)
            {
                faces.push_back(std::move(face));
            }
        }
    }

    if (positions.empty() || faces.empty())
    {
        return {};
    }

    float3 minBounds(std::numeric_limits<float>::max());
    float3 maxBounds(std::numeric_limits<float>::lowest());
    for (const float3& position : positions)
    {
        minBounds.x = std::min(minBounds.x, position.x);
        minBounds.y = std::min(minBounds.y, position.y);
        minBounds.z = std::min(minBounds.z, position.z);
        maxBounds.x = std::max(maxBounds.x, position.x);
        maxBounds.y = std::max(maxBounds.y, position.y);
        maxBounds.z = std::max(maxBounds.z, position.z);
    }

    const float3 center = (minBounds + maxBounds) * 0.5f;
    const float3 extents = maxBounds - minBounds;
    const float maxExtent = std::max(extents.x, std::max(extents.y, extents.z));
    const float scale = maxExtent > 0.f ? targetMaxExtent / maxExtent : 1.f;

    auto transformPosition = [&](const float3& position) {
        return (position - center) * scale;
    };

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    size_t triangleCount = 0;
    for (const auto& face : faces)
    {
        triangleCount += face.size() - 2;
    }
    vertices.reserve(triangleCount * 3);
    indices.reserve(triangleCount * 3);

    for (const auto& face : faces)
    {
        for (size_t i = 1; i + 1 < face.size(); ++i)
        {
            const std::array<ObjVertexRef, 3> triangle = { face[0], face[i], face[i + 1] };
            std::array<float3, 3> trianglePositions = {};
            std::array<float2, 3> triangleTexcoords = {};
            std::array<float3, 3> triangleNormals = {};

            bool validTriangle = true;
            bool hasTexcoords = true;
            bool hasNormals = true;

            for (size_t vertexIndex = 0; vertexIndex < triangle.size(); ++vertexIndex)
            {
                const ObjVertexRef& ref = triangle[vertexIndex];
                if (ref.position < 0 || size_t(ref.position) >= positions.size())
                {
                    validTriangle = false;
                    break;
                }

                trianglePositions[vertexIndex] = transformPosition(positions[ref.position]);

                if (ref.texcoord >= 0 && size_t(ref.texcoord) < texcoords.size())
                {
                    triangleTexcoords[vertexIndex] = texcoords[ref.texcoord];
                }
                else
                {
                    hasTexcoords = false;
                }

                if (ref.normal >= 0 && size_t(ref.normal) < normals.size())
                {
                    triangleNormals[vertexIndex] = normals[ref.normal];
                }
                else
                {
                    hasNormals = false;
                }
            }

            if (!validTriangle)
            {
                continue;
            }

            const float3 faceNormal = SafeNormalize(cross(trianglePositions[1] - trianglePositions[0], trianglePositions[2] - trianglePositions[0]), float3(0.f, 1.f, 0.f));
            if (!hasNormals)
            {
                triangleNormals = { faceNormal, faceNormal, faceNormal };
            }

            float3 tangent = GetFallbackTangent(faceNormal);
            if (hasTexcoords)
            {
                const float3 edge1 = trianglePositions[1] - trianglePositions[0];
                const float3 edge2 = trianglePositions[2] - trianglePositions[0];
                const float2 duv1 = triangleTexcoords[1] - triangleTexcoords[0];
                const float2 duv2 = triangleTexcoords[2] - triangleTexcoords[0];
                const float denominator = duv1.x * duv2.y - duv2.x * duv1.y;
                if (std::fabs(denominator) > 1e-8f)
                {
                    tangent = SafeNormalize((edge1 * duv2.y - edge2 * duv1.y) / denominator, tangent);
                }
            }

            for (size_t vertexIndex = 0; vertexIndex < triangle.size(); ++vertexIndex)
            {
                const uint32_t index = uint32_t(vertices.size());
                const float3 normal = SafeNormalize(triangleNormals[vertexIndex], faceNormal);
                const float2 texcoord = hasTexcoords ? triangleTexcoords[vertexIndex] : float2(0.f);
                vertices.push_back({ trianglePositions[vertexIndex], normal, OrthogonalizeTangent(tangent, normal), texcoord });
                indices.push_back(index);
            }
        }
    }

    if (vertices.empty() || indices.empty())
    {
        return {};
    }

    return { std::move(vertices), std::move(indices) };
}
