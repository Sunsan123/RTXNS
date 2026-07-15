/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "PbrTextureUtils.h"

#include "DirectoryHelper.h"

#include <donut/core/log.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>

#include <filesystem>
#include <memory>

namespace
{

nvrhi::TextureHandle LoadPbrTexture(donut::engine::TextureCache& textureCache,
                                    donut::engine::CommonRenderPasses* commonPasses,
                                    nvrhi::ICommandList* commandList,
                                    const std::filesystem::path& fileName,
                                    bool sRGB)
{
    std::shared_ptr<donut::engine::LoadedTexture> texture = textureCache.LoadTextureFromFile(fileName, sRGB, commonPasses, commandList);
    if (!texture || texture->texture == nullptr)
    {
        donut::log::error("Failed to load PBR texture: %s", fileName.string().c_str());
        return nvrhi::TextureHandle();
    }

    return texture->texture;
}

} // namespace

PbrTextureSet LoadDefaultPbrTextures(donut::engine::TextureCache& textureCache, donut::engine::CommonRenderPasses* commonPasses, nvrhi::ICommandList* commandList)
{
    const std::filesystem::path texturePath = GetLocalPath("assets/data/Texture");

    PbrTextureSet textures;
    textures.baseColor = LoadPbrTexture(textureCache, commonPasses, commandList, texturePath / "Metal053B_1K-JPG_Color.jpg", true);
    textures.roughness = LoadPbrTexture(textureCache, commonPasses, commandList, texturePath / "Metal053B_1K-JPG_Roughness.jpg", false);
    textures.metallic = LoadPbrTexture(textureCache, commonPasses, commandList, texturePath / "Metal053B_1K-JPG_Metalness.jpg", false);
    textures.normal = LoadPbrTexture(textureCache, commonPasses, commandList, texturePath / "Metal053B_1K-JPG_NormalDX.jpg", false);

    return textures;
}
