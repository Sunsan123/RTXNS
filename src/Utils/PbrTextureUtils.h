/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <nvrhi/nvrhi.h>

namespace donut::engine
{
class CommonRenderPasses;
class TextureCache;
} // namespace donut::engine

struct PbrTextureSet
{
    nvrhi::TextureHandle baseColor;
    nvrhi::TextureHandle roughness;
    nvrhi::TextureHandle metallic;
    nvrhi::TextureHandle normal;

    bool IsComplete() const
    {
        return baseColor && roughness && metallic && normal;
    }
};

PbrTextureSet LoadDefaultPbrTextures(donut::engine::TextureCache& textureCache, donut::engine::CommonRenderPasses* commonPasses, nvrhi::ICommandList* commandList);
