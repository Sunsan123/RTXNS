/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/app/ApplicationBase.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/UserInterfaceUtils.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/json.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/BindingCache.h>
#include <nvrhi/utils.h>

#include "DeviceUtils.h"
#include "GraphicsResources.h"
#include "GeometryUtils.h"
#include "NeuralNetwork.h"
#include "Float16.h"
#include "DirectoryHelper.h"
#include "PbrTextureUtils.h"

#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <format>
#include <cstddef>
#include <utility>
#include <vector>

using namespace donut;
using namespace donut::math;

#include "NetworkConfig.h"

static_assert(offsetof(DirectConstantBufferEntry, anisotropy) == 220);
static_assert(offsetof(DirectConstantBufferEntry, useAnisotropy) == 224);
static_assert(offsetof(DirectConstantBufferEntry, specularShift) == 228);
static_assert(offsetof(DirectConstantBufferEntry, onlyNeuralDebug) == 236);
static_assert(sizeof(DirectConstantBufferEntry) == 240);
static_assert(offsetof(InferenceConstantBufferEntry, weightOffsets) == 240);
static_assert(offsetof(InferenceConstantBufferEntry, biasOffsets) == 256);
static_assert(offsetof(TrainingConstantBufferEntry, seed) == 48);
static_assert(offsetof(TrainingConstantBufferEntry, useAnisotropy) == 56);
static_assert(sizeof(TrainingConstantBufferEntry) == 64);

static const char* g_windowTitle = "RTX Neural Shading Example: PreintegratedSkin Shader Training (Ground Truth | Training | Loss )";
constexpr int g_viewsNum = 3;
constexpr int g_statisticsPerFrames = 100;

static std::random_device rd;

struct UIData
{
    float lightIntensity = 1.f;
    float iblIntensity = 0.35f;
    float iblRotation = 0.f;
    float specular = 0.5f;
    float roughness = 0.9f;
    float metallic = 0.f;
    float anisotropy = 0.45f;
    float specularShift = 1.0f;

    float trainingTime = 0.0f;
    int defaultPbrRenderTimeUs = 0;
    int neuralRenderTimeUs = 0;
    int trainingPassTimeUs = 0;
    int optimizerPassTimeUs = 0;
    uint32_t epochs = 0;

    bool useAnisotropy = true;
    bool useIBL = true;
    bool onlyNeuralDebug = false;
    bool reset = false;
    bool training = true;
    bool load = false;
    std::string fileName;
};

class SimpleShading : public app::IRenderPass
{

public:
    SimpleShading(app::DeviceManager* deviceManager, UIData* uiParams) : IRenderPass(deviceManager), m_userInterfaceParameters(uiParams)
    {
    }

    bool Init()
    {
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();

        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/ShaderTrainingPreintegratedSkin" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        rootFS->mount("/shaders/app", appShaderPath);

        m_shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_commonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_shaderFactory);
        m_bindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        ////////////////////
        //
        // Create the Neural network class and initialize it the hyper parameters from NetworkConfig.h.
        //
        ////////////////////
        m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
        m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
        m_activeUseAnisotropy = m_userInterfaceParameters->useAnisotropy;
        m_netArch = GetNetworkArchitecture(m_activeUseAnisotropy);
        if (!m_neuralNetwork->Initialise(m_netArch))
        {
            log::error("Failed to create a network.");
            return false;
        }

        ////////////////////
        //
        // Create the shaders/buffers for the Neural Training
        //
        ////////////////////
        m_trainingShaderAnisotropic = m_shaderFactory->CreateShader("app/computeTraining", "main_cs", nullptr, nvrhi::ShaderType::Compute);
        m_trainingShaderOriginal = m_trainingShaderAnisotropic;
        m_optimizerPass.computeShader = m_shaderFactory->CreateShader("app/computeOptimizer", "adam_cs", nullptr, nvrhi::ShaderType::Compute);
        m_convertWeightsPass.computeShader = m_shaderFactory->CreateShader("app/computeOptimizer", "convert_weights_cs", nullptr, nvrhi::ShaderType::Compute);
        assert(m_trainingShaderAnisotropic && m_trainingShaderOriginal && m_optimizerPass.computeShader && m_convertWeightsPass.computeShader);

        m_trainingConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(TrainingConstantBufferEntry), "TrainingConstantBuffer")
                                                                 .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                 .setKeepInitialState(true));

        ////////////////////
        //
        // Continue to load the render data and create the required structures
        //
        ////////////////////
        const std::filesystem::path previewModelFileName = GetLocalPath("assets/data/Skin/LeePerrySmith") / "head.obj";
        auto [vertices, indices] = LoadObjModel(previewModelFileName, 2.f, true);
        if (vertices.empty() || indices.empty())
        {
            log::error("Failed to load preview model: %s", previewModelFileName.string().c_str());
            return false;
        }
        m_indicesNum = (int)indices.size();

        nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc().setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(0).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(1).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("TANGENT").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(2).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("TEXCOORD").setFormat(nvrhi::Format::RG32_FLOAT).setOffset(0).setBufferIndex(3).setElementStride(sizeof(Vertex)),
        };

        // Initialize direct pass
        {
            m_directPass.constantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(DirectConstantBufferEntry), "DirectConstantBuffer")
                                                                        .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                        .setKeepInitialState(true));
            m_directPass.vertexShader = m_shaderFactory->CreateShader("app/renderPreintegratedSkin", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_directPixelShaderAnisotropic = m_shaderFactory->CreateShader("app/renderPreintegratedSkin", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
            m_directPixelShaderOriginal = m_directPixelShaderAnisotropic;
            assert(m_directPass.vertexShader && m_directPixelShaderAnisotropic && m_directPixelShaderOriginal);

            m_directPass.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_directPass.vertexShader);
        }

        // Initialize neural pass
        {
            m_inferencePass.constantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(InferenceConstantBufferEntry), "NeuralConstantBuffer")
                                                                           .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
                                                                           .setKeepInitialState(true));
            m_inferencePass.vertexShader = m_shaderFactory->CreateShader("app/renderInference", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_inferencePixelShaderAnisotropic = m_shaderFactory->CreateShader("app/renderInference", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
            m_inferencePixelShaderOriginal = m_inferencePixelShaderAnisotropic;
            assert(m_inferencePass.vertexShader && m_inferencePixelShaderAnisotropic && m_inferencePixelShaderOriginal);

            m_inferencePass.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_inferencePass.vertexShader);
        }

        // Initialize difference pass
        {
            m_differencePass.constantBuffer = m_inferencePass.constantBuffer;
            m_differencePass.vertexShader = m_shaderFactory->CreateShader("app/renderDifference", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_differencePixelShaderAnisotropic = m_shaderFactory->CreateShader("app/renderDifference", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
            m_differencePixelShaderOriginal = m_differencePixelShaderAnisotropic;
            assert(m_differencePass.vertexShader && m_differencePixelShaderAnisotropic && m_differencePixelShaderOriginal);

            m_differencePass.inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_differencePass.vertexShader);
        }

        // Create and fill render buffers
        {
            m_commandList = GetDevice()->createCommandList();
            m_commandList->open();

            engine::TextureCache textureCache(GetDevice(), nativeFS, nullptr);
            const std::filesystem::path environmentMapFileName = GetLocalPath("assets/data/HDR") / "IndoorEnvironmentHDRI013_1K_HDR.exr";
            std::shared_ptr<engine::LoadedTexture> environmentMap = textureCache.LoadTextureFromFile(environmentMapFileName, false, m_commonPasses.get(), m_commandList);
            if (!environmentMap || environmentMap->texture == nullptr)
            {
                log::error("Failed to load IBL environment map: %s", environmentMapFileName.string().c_str());
                m_commandList->close();
                return false;
            }
            m_environmentMap = environmentMap->texture;
            m_environmentMipCount = float(m_environmentMap->getDesc().mipLevels);

            const std::filesystem::path skinTexturePath = GetLocalPath("assets/data/Skin/LeePerrySmith");
            auto loadTexture = [&](const char* fileName, bool sRGB) -> nvrhi::TextureHandle {
                const std::filesystem::path fullPath = skinTexturePath / fileName;
                std::shared_ptr<engine::LoadedTexture> loadedTexture = textureCache.LoadTextureFromFile(fullPath, sRGB, m_commonPasses.get(), m_commandList);
                if (!loadedTexture || loadedTexture->texture == nullptr)
                {
                    log::error("Failed to load PreintegratedSkin texture: %s", fullPath.string().c_str());
                    return nvrhi::TextureHandle();
                }

                return loadedTexture->texture;
            };

            m_pbrTextures.baseColor = loadTexture("lambertian.jpg", true);
            m_pbrTextures.roughness = loadTexture("roughness.png", false);
            m_pbrTextures.metallic = loadTexture("metallic.png", false);
            m_pbrTextures.normal = loadTexture("normal.png", false);
            m_subsurfaceColorMap = loadTexture("subsurface_color.png", true);
            m_preintegratedSkinLut = loadTexture("PreintegratedSkinBRDF_Color.png", false);
            if (!m_pbrTextures.IsComplete() || !m_subsurfaceColorMap || !m_preintegratedSkinLut)
            {
                m_commandList->close();
                return false;
            }

            nvrhi::BufferDesc vertexBufferDesc;
            vertexBufferDesc.byteSize = vertices.size() * sizeof(vertices[0]);
            vertexBufferDesc.isVertexBuffer = true;
            vertexBufferDesc.debugName = "VertexBuffer";
            vertexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            m_vertexBuffer = GetDevice()->createBuffer(vertexBufferDesc);

            m_commandList->beginTrackingBufferState(m_vertexBuffer, nvrhi::ResourceStates::CopyDest);
            m_commandList->writeBuffer(m_vertexBuffer, vertices.data(), vertices.size() * sizeof(vertices[0]));
            m_commandList->setPermanentBufferState(m_vertexBuffer, nvrhi::ResourceStates::VertexBuffer);

            nvrhi::BufferDesc indexBufferDesc;
            indexBufferDesc.byteSize = indices.size() * sizeof(indices[0]);
            indexBufferDesc.isIndexBuffer = true;
            indexBufferDesc.debugName = "IndexBuffer";
            indexBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            m_indexBuffer = GetDevice()->createBuffer(indexBufferDesc);

            m_commandList->beginTrackingBufferState(m_indexBuffer, nvrhi::ResourceStates::CopyDest);
            m_commandList->writeBuffer(m_indexBuffer, indices.data(), indices.size() * sizeof(indices[0]));
            m_commandList->setPermanentBufferState(m_indexBuffer, nvrhi::ResourceStates::IndexBuffer);

            m_commandList->close();
            GetDevice()->executeCommandList(m_commandList);
        }

        // Direct binding
        {
            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_directPass.constantBuffer),
                nvrhi::BindingSetItem::Texture_SRV(1, m_environmentMap),
                nvrhi::BindingSetItem::Texture_SRV(2, m_pbrTextures.baseColor),
                nvrhi::BindingSetItem::Texture_SRV(3, m_pbrTextures.roughness),
                nvrhi::BindingSetItem::Texture_SRV(4, m_pbrTextures.metallic),
                nvrhi::BindingSetItem::Texture_SRV(5, m_pbrTextures.normal),
                nvrhi::BindingSetItem::Texture_SRV(6, m_subsurfaceColorMap),
                nvrhi::BindingSetItem::Texture_SRV(7, m_preintegratedSkinLut),
                nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_LinearClampSampler),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_directPass.bindingLayout, m_directPass.bindingSet);
        }

        SetActiveModeShaders(m_activeUseAnisotropy);
        CreateMLPBuffers();

        m_disneyTimer = GetDevice()->createTimerQuery();
        m_neuralTimer = GetDevice()->createTimerQuery();
        m_trainingTimer = GetDevice()->createTimerQuery();
        m_optimizerTimer = GetDevice()->createTimerQuery();

        return true;
    }

    void CreateMLPBuffers()
    {
        GetDevice()->waitForIdle();
        GetDevice()->runGarbageCollection();

        const auto& params = m_neuralNetwork->GetNetworkParams();

        // Get a device optimized layout
        m_deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(m_neuralNetwork->GetNetworkLayout(), rtxns::MatrixLayout::TrainingOptimal);

        for (int i = 0; i < NUM_TRANSITIONS; ++i)
        {
            m_weightOffsets[i / 4][i % 4] = m_deviceNetworkLayout.networkLayers[i].weightOffset;
            m_biasOffsets[i / 4][i % 4] = m_deviceNetworkLayout.networkLayers[i].biasOffset;
        }

        m_totalParameterCount = uint(m_deviceNetworkLayout.networkSize / sizeof(uint16_t));
        m_batchSize = BATCH_SIZE;

        // Create and fill buffers
        {
            nvrhi::CommandListHandle uploadCommandList = GetDevice()->createCommandList();
            uploadCommandList->open();

            nvrhi::BufferDesc paramsBufferDesc;

            paramsBufferDesc.debugName = "MLPParamsDeviceBuffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
            paramsBufferDesc.byteSize = params.size();
            paramsBufferDesc.keepInitialState = true;
            paramsBufferDesc.canHaveUAVs = true;
            m_mlpHostBuffer = GetDevice()->createBuffer(paramsBufferDesc);

            paramsBufferDesc.debugName = "MLPParamsDeviceBuffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = m_deviceNetworkLayout.networkSize;
            paramsBufferDesc.canHaveRawViews = true;
            paramsBufferDesc.canHaveTypedViews = true;
            paramsBufferDesc.canHaveUAVs = true;
            paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
            m_mlpDeviceBuffer = GetDevice()->createBuffer(paramsBufferDesc);

            // Upload the parameters
            UpdateDeviceNetworkParameters(uploadCommandList);

            paramsBufferDesc.debugName = "MLPParamsBuffer32";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = m_totalParameterCount * sizeof(float);
            paramsBufferDesc.format = nvrhi::Format::R32_FLOAT;
            m_mlpParamsBuffer32 = GetDevice()->createBuffer(paramsBufferDesc);

            uploadCommandList->beginTrackingBufferState(m_mlpParamsBuffer32, nvrhi::ResourceStates::UnorderedAccess);
            uploadCommandList->clearBufferUInt(m_mlpParamsBuffer32, 0);

            paramsBufferDesc.debugName = "MLPGradientsBuffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = (m_totalParameterCount * sizeof(uint16_t) + 3) & ~3; // Round up to nearest multiple of 4
            paramsBufferDesc.structStride = sizeof(uint16_t);
            paramsBufferDesc.format = nvrhi::Format::R16_FLOAT;
            m_mlpGradientsBuffer = GetDevice()->createBuffer(paramsBufferDesc);

            uploadCommandList->beginTrackingBufferState(m_mlpGradientsBuffer, nvrhi::ResourceStates::UnorderedAccess);
            uploadCommandList->clearBufferUInt(m_mlpGradientsBuffer, 0);

            paramsBufferDesc.debugName = "MLPMoments1Buffer";
            paramsBufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            paramsBufferDesc.byteSize = m_totalParameterCount * sizeof(float);
            paramsBufferDesc.format = nvrhi::Format::R32_FLOAT;
            paramsBufferDesc.canHaveRawViews = false;
            m_mlpMoments1Buffer = GetDevice()->createBuffer(paramsBufferDesc);

            uploadCommandList->beginTrackingBufferState(m_mlpMoments1Buffer, nvrhi::ResourceStates::UnorderedAccess);
            uploadCommandList->clearBufferUInt(m_mlpMoments1Buffer, 0);

            paramsBufferDesc.debugName = "MLPMoments2Buffer";
            m_mlpMoments2Buffer = GetDevice()->createBuffer(paramsBufferDesc);

            uploadCommandList->beginTrackingBufferState(m_mlpMoments2Buffer, nvrhi::ResourceStates::UnorderedAccess);
            uploadCommandList->clearBufferUInt(m_mlpMoments2Buffer, 0);

            uploadCommandList->close();
            GetDevice()->executeCommandList(uploadCommandList);
            GetDevice()->waitForIdle();
            GetDevice()->runGarbageCollection();
        }

        nvrhi::BindingSetDesc bindingSetDesc = {};
        // Training binding
        {
            m_trainingPass.bindingSet = nullptr;
            m_trainingPass.bindingLayout = nullptr;

            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_trainingConstantBuffer),
                nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer),
                nvrhi::BindingSetItem::Texture_SRV(1, m_preintegratedSkinLut),
                nvrhi::BindingSetItem::RawBuffer_UAV(0, m_mlpGradientsBuffer),
                nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_LinearClampSampler),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_trainingPass.bindingLayout, m_trainingPass.bindingSet);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.bindingLayouts = { m_trainingPass.bindingLayout };
            pipelineDesc.CS = m_trainingPass.computeShader;
            m_trainingPass.pipeline = GetDevice()->createComputePipeline(pipelineDesc);
        }

        // Optimization binding
        {
            m_optimizerPass.bindingSet = nullptr;
            m_optimizerPass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_trainingConstantBuffer), nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_mlpDeviceBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_mlpParamsBuffer32),     nvrhi::BindingSetItem::TypedBuffer_UAV(2, m_mlpGradientsBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(3, m_mlpMoments1Buffer),     nvrhi::BindingSetItem::TypedBuffer_UAV(4, m_mlpMoments2Buffer),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_optimizerPass.bindingLayout, m_optimizerPass.bindingSet);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.bindingLayouts = { m_optimizerPass.bindingLayout };
            pipelineDesc.CS = m_optimizerPass.computeShader;
            m_optimizerPass.pipeline = GetDevice()->createComputePipeline(pipelineDesc);
        }

        // Convert pass used before the first optimizer step after weights are uploaded.
        {
            m_convertWeightsPass.bindingSet = nullptr;
            m_convertWeightsPass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_trainingConstantBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_mlpDeviceBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_mlpParamsBuffer32),
                nvrhi::BindingSetItem::TypedBuffer_UAV(2, m_mlpGradientsBuffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(3, m_mlpMoments1Buffer),
                nvrhi::BindingSetItem::TypedBuffer_UAV(4, m_mlpMoments2Buffer),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_convertWeightsPass.bindingLayout, m_convertWeightsPass.bindingSet);

            nvrhi::ComputePipelineDesc pipelineDesc;
            pipelineDesc.bindingLayouts = { m_convertWeightsPass.bindingLayout };
            pipelineDesc.CS = m_convertWeightsPass.computeShader;
            m_convertWeightsPass.pipeline = GetDevice()->createComputePipeline(pipelineDesc);
        }

        // Inference binding
        {
            m_inferencePass.pipeline = nullptr;
            m_inferencePass.bindingSet = nullptr;
            m_inferencePass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_inferencePass.constantBuffer),
                nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer),
                nvrhi::BindingSetItem::Texture_SRV(1, m_environmentMap),
                nvrhi::BindingSetItem::Texture_SRV(2, m_pbrTextures.baseColor),
                nvrhi::BindingSetItem::Texture_SRV(3, m_pbrTextures.roughness),
                nvrhi::BindingSetItem::Texture_SRV(4, m_pbrTextures.metallic),
                nvrhi::BindingSetItem::Texture_SRV(5, m_pbrTextures.normal),
                nvrhi::BindingSetItem::Texture_SRV(6, m_subsurfaceColorMap),
                nvrhi::BindingSetItem::Texture_SRV(7, m_preintegratedSkinLut),
                nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_LinearClampSampler),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_inferencePass.bindingLayout, m_inferencePass.bindingSet);
        }

        // Difference binding
        {
            m_differencePass.pipeline = nullptr;
            m_differencePass.bindingSet = nullptr;
            m_differencePass.bindingLayout = nullptr;

            bindingSetDesc = {};
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_differencePass.constantBuffer),
                nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer),
                nvrhi::BindingSetItem::Texture_SRV(1, m_environmentMap),
                nvrhi::BindingSetItem::Texture_SRV(2, m_pbrTextures.baseColor),
                nvrhi::BindingSetItem::Texture_SRV(3, m_pbrTextures.roughness),
                nvrhi::BindingSetItem::Texture_SRV(4, m_pbrTextures.metallic),
                nvrhi::BindingSetItem::Texture_SRV(5, m_pbrTextures.normal),
                nvrhi::BindingSetItem::Texture_SRV(6, m_subsurfaceColorMap),
                nvrhi::BindingSetItem::Texture_SRV(7, m_preintegratedSkinLut),
                nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_LinearClampSampler),
            };
            nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_differencePass.bindingLayout, m_differencePass.bindingSet);
        }

        // Reset training parameters
        m_currentOptimizationStep = 0;
        m_userInterfaceParameters->epochs = 0;
        m_userInterfaceParameters->trainingTime = 0.0f;
    }

    // expects an open command list
    void UpdateDeviceNetworkParameters(nvrhi::CommandListHandle commandList)
    {
        // Upload the host side parameters
        commandList->setBufferState(m_mlpHostBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->commitBarriers();
        commandList->writeBuffer(m_mlpHostBuffer, m_neuralNetwork->GetNetworkParams().data(), m_neuralNetwork->GetNetworkParams().size());

        // Convert to GPU optimized layout
        m_networkUtils->ConvertWeights(m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), commandList);

        // Update barriers for use
        commandList->setBufferState(m_mlpDeviceBuffer, nvrhi::ResourceStates::ShaderResource);
        commandList->commitBarriers();
        m_convertWeights = true;
    }


    std::shared_ptr<engine::ShaderFactory> GetShaderFactory() const
    {
        return m_shaderFactory;
    }

    bool MousePosUpdate(double xpos, double ypos) override
    {
        const float2 mousePos{ float(xpos), float(ypos) };
        const float2 delta = mousePos - m_currentXY;

        if (m_lightDragActive)
        {
            float a, e, d;
            cartesianToSpherical(m_lightDir, a, e, d);
            a += delta.x * 0.01f;
            e += delta.y * 0.01f;
            m_lightDir = sphericalToCartesian(a, e, d);
        }

        if (m_modelDragActive)
        {
            m_modelYaw += delta.x * 0.01f;
            m_modelPitch = std::clamp(m_modelPitch + delta.y * 0.01f, -1.45f, 1.45f);
        }

        m_currentXY = mousePos;
        return true;
    }

    bool MouseButtonUpdate(int button, int action, int mods) override
    {
        constexpr int mouseButtonLeft = 0;
        constexpr int mouseButtonRight = 1;
        const bool pressed = action == 1;

        if (button == mouseButtonLeft)
        {
            m_lightDragActive = pressed;
        }
        else if (button == mouseButtonRight)
        {
            m_modelDragActive = pressed;
        }

        return true;
    }

    void Animate(float seconds) override
    {
        if (m_userInterfaceParameters->training)
        {
            m_userInterfaceParameters->trainingTime += seconds;
        }

        auto toMicroSeconds = [&](const auto& timer) { return int(GetDevice()->getTimerQueryTime(timer) * 1000000); };

        const int defaultPbrRenderTimeUs = toMicroSeconds(m_disneyTimer);
        if (defaultPbrRenderTimeUs != 0)
        {
            const int neuralRenderTimeUs = toMicroSeconds(m_neuralTimer);
            const int trainingPassTimeUs = toMicroSeconds(m_trainingTimer);
            const int optimizerPassTimeUs = toMicroSeconds(m_optimizerTimer);

            m_userInterfaceParameters->defaultPbrRenderTimeUs = defaultPbrRenderTimeUs;
            m_userInterfaceParameters->neuralRenderTimeUs = neuralRenderTimeUs;
            m_userInterfaceParameters->trainingPassTimeUs = trainingPassTimeUs;
            m_userInterfaceParameters->optimizerPassTimeUs = optimizerPassTimeUs;

            m_extraStatus = std::format(" - PreintegratedSkin - {:3d}us, Neural - {:3d}us, Training - {:3d}us, Optimization - {:3d}us", defaultPbrRenderTimeUs, neuralRenderTimeUs,
                                        trainingPassTimeUs, optimizerPassTimeUs);
        }
        GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, true, m_extraStatus.c_str());

        ////////////////////
        //
        // Reset/Load/Save the Neural network if required
        //
        ////////////////////
        if (m_userInterfaceParameters->reset)
        {
            const bool requestedUseAnisotropy = m_userInterfaceParameters->useAnisotropy;
            const auto requestedNetArch = GetNetworkArchitecture(requestedUseAnisotropy);
            m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
            if (m_neuralNetwork->Initialise(requestedNetArch))
            {
                m_activeUseAnisotropy = requestedUseAnisotropy;
                m_netArch = requestedNetArch;
                SetActiveModeShaders(m_activeUseAnisotropy);
                CreateMLPBuffers();
            }
            else
            {
                log::error("Failed to create a network.");
                m_userInterfaceParameters->useAnisotropy = m_activeUseAnisotropy;
            }

            m_userInterfaceParameters->reset = false;
        }

        if (!m_userInterfaceParameters->fileName.empty())
        {
            if (m_userInterfaceParameters->load)
            {
                m_neuralNetwork = std::make_unique<rtxns::HostNetwork>(m_networkUtils);
                if (m_neuralNetwork->InitialiseFromFile(m_userInterfaceParameters->fileName))
                {
                    const auto& loadedNetArch = m_neuralNetwork->GetNetworkArchitecture();
                    if (IsKnownNetworkArchitecture(loadedNetArch))
                    {
                        m_activeUseAnisotropy = loadedNetArch.inputNeurons == ANISOTROPIC_INPUT_NEURONS;
                        m_userInterfaceParameters->useAnisotropy = m_activeUseAnisotropy;
                        m_netArch = loadedNetArch;
                        SetActiveModeShaders(m_activeUseAnisotropy);
                        CreateMLPBuffers();
                    }
                    else
                    {
                        log::error("Loaded network architecture does not match ShaderTrainingPreintegratedSkin modes.");
                        m_userInterfaceParameters->useAnisotropy = m_activeUseAnisotropy;
                    }
                }
            }
            else
            {
                m_neuralNetwork->UpdateFromBufferToFile(m_mlpHostBuffer, m_mlpDeviceBuffer, m_neuralNetwork->GetNetworkLayout(), m_deviceNetworkLayout,
                                                        m_userInterfaceParameters->fileName, GetDevice(), m_commandList);
            }
            m_userInterfaceParameters->fileName = "";
        }
    }

    void BackBufferResizing() override
    {
        m_directPass.pipeline = nullptr;
        m_inferencePass.pipeline = nullptr;
        m_differencePass.pipeline = nullptr;
        m_depthBuffer = nullptr;
        m_depthFramebuffers.clear();
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        std::uniform_int_distribution<uint64_t> ldist;
        uint64_t seed = ldist(rd);

        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
        const float height = float(fbinfo.height);
        const float width = height;

        // Update statistics every g_statisticsPerFrames frames
        bool updateStat = GetDeviceManager()->GetCurrentBackBufferIndex() % g_statisticsPerFrames == 0;
        nvrhi::IFramebuffer* renderFramebuffer = GetDepthFramebuffer(framebuffer);

        // Orbit camera around the preview model. Right mouse drag changes yaw/pitch.
        const float3 cameraPos = GetCameraPosition();
        const float3 viewDir = -normalize(cameraPos);
        float3 cameraUp(0, 1, 0);

        // Fill out the constant buffer slices for multiple views of the model.
        bool useAnisotropy = m_activeUseAnisotropy;
        float clampedAnisotropy = std::clamp(m_userInterfaceParameters->anisotropy, 0.f, 1.f);

        DirectConstantBufferEntry directModelConstant{ {},
                                                       {},
                                                       float4(cameraPos, 0.f),
                                                       float4(m_lightDir, 1.f),
                                                       float4(m_userInterfaceParameters->lightIntensity),
                                                       float4(m_userInterfaceParameters->iblIntensity, m_userInterfaceParameters->iblRotation, m_environmentMipCount,
                                                              m_userInterfaceParameters->useIBL ? 1.f : 0.f),
                                                       float4(1.f, 1.f, 1.f, 1.f),
                                                       m_userInterfaceParameters->specular,
                                                       m_userInterfaceParameters->roughness,
                                                       m_userInterfaceParameters->metallic,
                                                       clampedAnisotropy,
                                                       useAnisotropy ? 1u : 0u,
                                                       std::clamp(m_userInterfaceParameters->specularShift, 0.f, 2.f),
                                                       0.f,
                                                       m_userInterfaceParameters->onlyNeuralDebug ? 1u : 0u };
        directModelConstant.view = affineToHomogeneous(translation(-cameraPos) * lookatZ(-viewDir, cameraUp));
        directModelConstant.viewProject = directModelConstant.view * perspProjD3DStyle(radians(67.4f), float(width) / float(height), 0.1f, 10.f);

        ////////////////////
        //
        // Fill out the inference constant buffer including the neural weight/bias offsets.
        //
        ////////////////////
        InferenceConstantBufferEntry inferenceModelConstant;
        static_cast<DirectConstantBufferEntry&>(inferenceModelConstant) = directModelConstant;
        std::ranges::copy(m_weightOffsets, inferenceModelConstant.weightOffsets);
        std::ranges::copy(m_biasOffsets, inferenceModelConstant.biasOffsets);

        m_commandList->open();

        ////////////////////
        //
        // Start the training loop
        //
        ////////////////////
        if (m_userInterfaceParameters->training)
        {
            if (m_convertWeights)
            {
                TrainingConstantBufferEntry convertConstants = {
                    .maxParamSize = m_totalParameterCount,
                    .learningRate = m_learningRate,
                    .currentStep = float(m_currentOptimizationStep),
                    .batchSize = m_batchSize,
                    .seed = seed,
                    .useAnisotropy = useAnisotropy ? 1u : 0u,
                    .pad0 = 0.f
                };
                std::ranges::copy(m_weightOffsets, convertConstants.weightOffsets);
                std::ranges::copy(m_biasOffsets, convertConstants.biasOffsets);
                m_commandList->writeBuffer(m_trainingConstantBuffer, &convertConstants, sizeof(convertConstants));

                nvrhi::ComputeState state;
                state.bindings = { m_convertWeightsPass.bindingSet };
                state.pipeline = m_convertWeightsPass.pipeline;
                m_commandList->beginMarker("Convert Weights");
                m_commandList->setComputeState(state);
                m_commandList->dispatch(div_ceil(m_totalParameterCount, 32), 1, 1);
                m_commandList->endMarker();
                m_convertWeights = false;
            }

            for (int i = 0; i < BATCH_COUNT; ++i)
            {
                TrainingConstantBufferEntry trainingModelConstant = {
                    .maxParamSize = m_totalParameterCount,
                    .learningRate = m_learningRate,
                    .currentStep = float(++m_currentOptimizationStep),
                    .batchSize = m_batchSize,
                    .seed = seed,
                    .useAnisotropy = useAnisotropy ? 1u : 0u,
                    .pad0 = 0.f
                };
                std::ranges::copy(m_weightOffsets, trainingModelConstant.weightOffsets);
                std::ranges::copy(m_biasOffsets, trainingModelConstant.biasOffsets);

                m_commandList->writeBuffer(m_trainingConstantBuffer, &trainingModelConstant, sizeof(trainingModelConstant));

                nvrhi::ComputeState state;

                // Training pass
                state.bindings = { m_trainingPass.bindingSet };
                state.pipeline = m_trainingPass.pipeline;
                m_commandList->beginMarker("Training");

                if (updateStat && i == 0)
                {
                    GetDevice()->resetTimerQuery(m_trainingTimer);
                    m_commandList->beginTimerQuery(m_trainingTimer);
                }

                m_commandList->setComputeState(state);
                m_commandList->dispatch(m_batchSize / 64, 1, 1);

                if (updateStat && i == 0)
                {
                    m_commandList->endTimerQuery(m_trainingTimer);
                }
                m_commandList->endMarker();

                // Optimizer pass
                state.bindings = { m_optimizerPass.bindingSet };
                state.pipeline = m_optimizerPass.pipeline;
                m_commandList->beginMarker("Update Weights");

                if (updateStat && i == 0)
                {
                    GetDevice()->resetTimerQuery(m_optimizerTimer);
                    m_commandList->beginTimerQuery(m_optimizerTimer);
                }

                m_commandList->setComputeState(state);
                m_commandList->dispatch(div_ceil(m_totalParameterCount, 32), 1, 1);

                if (updateStat && i == 0)
                {
                    m_commandList->endTimerQuery(m_optimizerTimer);
                }
                m_commandList->endMarker();
            }

            ++m_userInterfaceParameters->epochs;
        }

        nvrhi::utils::ClearColorAttachment(m_commandList, renderFramebuffer, 0, nvrhi::Color(0.f));
        nvrhi::utils::ClearDepthStencilAttachment(m_commandList, renderFramebuffer, 1.f, 0);

        RenderPass* passes[] = { &m_directPass, &m_inferencePass, &m_differencePass };
        for (int viewIndex = 0; viewIndex < g_viewsNum; ++viewIndex)
        {
            nvrhi::TimerQueryHandle timer;
            if (viewIndex < 2 && updateStat)
            {
                timer = viewIndex == 0 ? m_disneyTimer.Get() : m_neuralTimer.Get();
                GetDevice()->resetTimerQuery(timer);
                m_commandList->beginTimerQuery(timer);
            }

            auto& pass = *passes[viewIndex];

            if (!pass.pipeline)
            {
                nvrhi::GraphicsPipelineDesc psoDesc;
                psoDesc.VS = pass.vertexShader;
                psoDesc.PS = pass.pixelShader;
                psoDesc.inputLayout = pass.inputLayout;
                psoDesc.bindingLayouts = { pass.bindingLayout };
                psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
                psoDesc.renderState.depthStencilState.depthTestEnable = true;
                psoDesc.renderState.depthStencilState.depthWriteEnable = true;
                psoDesc.renderState.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;

                pass.pipeline = GetDevice()->createGraphicsPipeline(psoDesc, renderFramebuffer);
            }

            if (viewIndex == 0)
            {
                m_commandList->writeBuffer(pass.constantBuffer, &directModelConstant, sizeof(directModelConstant));
            }
            else
            {
                m_commandList->writeBuffer(pass.constantBuffer, &inferenceModelConstant, sizeof(inferenceModelConstant));
            }

            nvrhi::GraphicsState state;
            state.bindings = { pass.bindingSet };
            state.indexBuffer = { m_indexBuffer, nvrhi::Format::R32_UINT, 0 };

            state.vertexBuffers = {
                { m_vertexBuffer, 0, offsetof(Vertex, position) },
                { m_vertexBuffer, 1, offsetof(Vertex, normal) },
                { m_vertexBuffer, 2, offsetof(Vertex, tangent) },
                { m_vertexBuffer, 3, offsetof(Vertex, texcoord) },
            };
            state.pipeline = pass.pipeline;
            state.framebuffer = renderFramebuffer;

            // Construct the viewport so that all viewports form a grid.
            const float left = width * viewIndex;
            const float top = 0;

            const nvrhi::Viewport viewport = nvrhi::Viewport(left, left + width, 0, height, 0.f, 1.f);
            state.viewport.addViewportAndScissorRect(viewport);

            // Update the pipeline, bindings, and other state.
            m_commandList->setGraphicsState(state);

            // Draw the model.
            nvrhi::DrawArguments args;
            args.vertexCount = m_indicesNum;
            m_commandList->drawIndexed(args);

            if (viewIndex < 2 && updateStat)
            {
                m_commandList->endTimerQuery(timer);
            }
        }

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
    }

    rtxns::NetworkArchitecture GetNetworkArchitecture(bool useAnisotropy) const
    {
        return {
            .numHiddenLayers = NUM_HIDDEN_LAYERS,
            .inputNeurons = uint32_t(useAnisotropy ? ANISOTROPIC_INPUT_NEURONS : ORIGINAL_INPUT_NEURONS),
            .hiddenNeurons = uint32_t(useAnisotropy ? ANISOTROPIC_HIDDEN_NEURONS : ORIGINAL_HIDDEN_NEURONS),
            .outputNeurons = OUTPUT_NEURONS,
            .weightPrecision = rtxns::Precision::F16,
            .biasPrecision = rtxns::Precision::F16,
        };
    }

    bool IsKnownNetworkArchitecture(const rtxns::NetworkArchitecture& netArch) const
    {
        const bool commonSettingsMatch =
            netArch.numHiddenLayers == NUM_HIDDEN_LAYERS &&
            netArch.outputNeurons == OUTPUT_NEURONS &&
            netArch.weightPrecision == rtxns::Precision::F16 &&
            netArch.biasPrecision == rtxns::Precision::F16;

        const bool originalSettingsMatch =
            netArch.inputNeurons == ORIGINAL_INPUT_NEURONS &&
            netArch.hiddenNeurons == ORIGINAL_HIDDEN_NEURONS;

        const bool anisotropicSettingsMatch =
            netArch.inputNeurons == ANISOTROPIC_INPUT_NEURONS &&
            netArch.hiddenNeurons == ANISOTROPIC_HIDDEN_NEURONS;

        return commonSettingsMatch && (originalSettingsMatch || anisotropicSettingsMatch);
    }

    void SetActiveModeShaders(bool useAnisotropy)
    {
        m_trainingPass.computeShader = useAnisotropy ? m_trainingShaderAnisotropic : m_trainingShaderOriginal;
        m_directPass.pixelShader = useAnisotropy ? m_directPixelShaderAnisotropic : m_directPixelShaderOriginal;
        m_inferencePass.pixelShader = useAnisotropy ? m_inferencePixelShaderAnisotropic : m_inferencePixelShaderOriginal;
        m_differencePass.pixelShader = useAnisotropy ? m_differencePixelShaderAnisotropic : m_differencePixelShaderOriginal;

        m_trainingPass.pipeline = nullptr;
        m_directPass.pipeline = nullptr;
        m_inferencePass.pipeline = nullptr;
        m_differencePass.pipeline = nullptr;
    }

private:
    nvrhi::IFramebuffer* GetDepthFramebuffer(nvrhi::IFramebuffer* framebuffer)
    {
        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
        const nvrhi::TextureDesc* depthDesc = m_depthBuffer ? &m_depthBuffer->getDesc() : nullptr;
        if (!depthDesc || depthDesc->width != fbinfo.width || depthDesc->height != fbinfo.height || depthDesc->sampleCount != fbinfo.sampleCount ||
            depthDesc->sampleQuality != fbinfo.sampleQuality)
        {
            const nvrhi::Format depthFormats[] = { nvrhi::Format::D24S8, nvrhi::Format::D32, nvrhi::Format::D16 };
            const nvrhi::Format depthFormat = nvrhi::utils::ChooseFormat(GetDevice(), nvrhi::FormatSupport::Texture | nvrhi::FormatSupport::DepthStencil, depthFormats,
                                                                         uint32_t(sizeof(depthFormats) / sizeof(depthFormats[0])));

            nvrhi::TextureDesc newDepthDesc;
            newDepthDesc.width = fbinfo.width;
            newDepthDesc.height = fbinfo.height;
            newDepthDesc.sampleCount = fbinfo.sampleCount;
            newDepthDesc.sampleQuality = fbinfo.sampleQuality;
            newDepthDesc.format = depthFormat;
            newDepthDesc.isRenderTarget = true;
            newDepthDesc.isShaderResource = false;
            newDepthDesc.initialState = nvrhi::ResourceStates::DepthWrite;
            newDepthDesc.keepInitialState = true;
            newDepthDesc.clearValue = nvrhi::Color(1.f);
            newDepthDesc.useClearValue = true;
            newDepthDesc.debugName = "ShaderTrainingPreintegratedSkinDepth";

            m_depthBuffer = GetDevice()->createTexture(newDepthDesc);
            m_depthFramebuffers.clear();
        }

        for (auto& item : m_depthFramebuffers)
        {
            if (item.first == framebuffer)
            {
                return item.second;
            }
        }

        nvrhi::FramebufferDesc desc;
        for (const nvrhi::FramebufferAttachment& attachment : framebuffer->getDesc().colorAttachments)
        {
            desc.addColorAttachment(attachment);
        }
        desc.setDepthAttachment(m_depthBuffer);

        nvrhi::FramebufferHandle depthFramebuffer = GetDevice()->createFramebuffer(desc);
        nvrhi::IFramebuffer* result = depthFramebuffer;
        m_depthFramebuffers.emplace_back(framebuffer, std::move(depthFramebuffer));
        return result;
    }

    float3 GetCameraPosition() const
    {
        const float cosPitch = std::cos(m_modelPitch);
        return float3(std::sin(m_modelYaw) * cosPitch, std::sin(m_modelPitch), std::cos(m_modelYaw) * cosPitch) * m_cameraDistance;
    }

    std::string m_extraStatus;
    nvrhi::TimerQueryHandle m_disneyTimer;
    nvrhi::TimerQueryHandle m_neuralTimer;
    nvrhi::TimerQueryHandle m_trainingTimer;
    nvrhi::TimerQueryHandle m_optimizerTimer;

    std::shared_ptr<engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<engine::CommonRenderPasses> m_commonPasses;
    std::unique_ptr<engine::BindingCache> m_bindingCache;

    nvrhi::ShaderHandle m_trainingShaderOriginal;
    nvrhi::ShaderHandle m_trainingShaderAnisotropic;
    nvrhi::ShaderHandle m_directPixelShaderOriginal;
    nvrhi::ShaderHandle m_directPixelShaderAnisotropic;
    nvrhi::ShaderHandle m_inferencePixelShaderOriginal;
    nvrhi::ShaderHandle m_inferencePixelShaderAnisotropic;
    nvrhi::ShaderHandle m_differencePixelShaderOriginal;
    nvrhi::ShaderHandle m_differencePixelShaderAnisotropic;

    struct RenderPass
    {
        nvrhi::ShaderHandle vertexShader;
        nvrhi::ShaderHandle pixelShader;
        nvrhi::BufferHandle constantBuffer;
        nvrhi::InputLayoutHandle inputLayout;
        nvrhi::BindingLayoutHandle bindingLayout;
        nvrhi::BindingSetHandle bindingSet;
        nvrhi::GraphicsPipelineHandle pipeline;
    };

    RenderPass m_directPass;
    RenderPass m_inferencePass;
    RenderPass m_differencePass;

    float3 m_lightDir{ -0.761f, -0.467f, -0.450f };
    float m_environmentMipCount = 1.f;
    float m_modelYaw = 0.f;
    float m_modelPitch = 0.f;
    float m_cameraDistance = 2.f;
    float2 m_currentXY;
    bool m_lightDragActive = false;
    bool m_modelDragActive = false;

    nvrhi::BufferHandle m_vertexBuffer;
    nvrhi::BufferHandle m_indexBuffer;

    nvrhi::BufferHandle m_trainingConstantBuffer;
    nvrhi::BufferHandle m_mlpHostBuffer;
    nvrhi::BufferHandle m_mlpDeviceBuffer;
    nvrhi::BufferHandle m_mlpParamsBuffer32;
    nvrhi::BufferHandle m_mlpGradientsBuffer;
    nvrhi::BufferHandle m_mlpMoments1Buffer;
    nvrhi::BufferHandle m_mlpMoments2Buffer;
    nvrhi::TextureHandle m_environmentMap;
    PbrTextureSet m_pbrTextures;
    nvrhi::TextureHandle m_subsurfaceColorMap;
    nvrhi::TextureHandle m_preintegratedSkinLut;
    nvrhi::TextureHandle m_depthBuffer;
    std::vector<std::pair<nvrhi::IFramebuffer*, nvrhi::FramebufferHandle>> m_depthFramebuffers;

    uint m_totalParameterCount = 0;
    uint m_batchSize = BATCH_SIZE;
    uint m_currentOptimizationStep = 0;
    float m_learningRate = LEARNING_RATE;

    nvrhi::CommandListHandle m_commandList;

    int m_indicesNum = 0;

    struct NeuralPass
    {
        nvrhi::ShaderHandle computeShader;
        nvrhi::BindingLayoutHandle bindingLayout;
        nvrhi::BindingSetHandle bindingSet;
        nvrhi::ComputePipelineHandle pipeline;
    };

    NeuralPass m_trainingPass;
    NeuralPass m_optimizerPass;
    NeuralPass m_convertWeightsPass;

    uint4 m_weightOffsets[NUM_TRANSITIONS_ALIGN4];
    uint4 m_biasOffsets[NUM_TRANSITIONS_ALIGN4];

    UIData* m_userInterfaceParameters;
    bool m_activeUseAnisotropy = false;
    bool m_convertWeights = true;

    std::shared_ptr<rtxns::NetworkUtilities> m_networkUtils;
    std::unique_ptr<rtxns::HostNetwork> m_neuralNetwork;
    rtxns::NetworkLayout m_deviceNetworkLayout;

    rtxns::NetworkArchitecture m_netArch = {
        .numHiddenLayers = NUM_HIDDEN_LAYERS,
        .inputNeurons = ORIGINAL_INPUT_NEURONS,
        .hiddenNeurons = ORIGINAL_HIDDEN_NEURONS,
        .outputNeurons = OUTPUT_NEURONS,
        .weightPrecision = rtxns::Precision::F16,
        .biasPrecision = rtxns::Precision::F16,
    };
};

class UserInterface : public app::ImGui_Renderer
{
public:
    UserInterface(app::DeviceManager* deviceManager, UIData* ui) : ImGui_Renderer(deviceManager), m_userInterfaceParameters(ui)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void buildUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), 0);
        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::SetNextItemOpen(true, ImGuiCond_Always);
        if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("PreintegratedSkin/PBR Render : %d us (%.3f ms)", m_userInterfaceParameters->defaultPbrRenderTimeUs,
                        float(m_userInterfaceParameters->defaultPbrRenderTimeUs) / 1000.f);
            ImGui::Text("Neural Render : %d us (%.3f ms)", m_userInterfaceParameters->neuralRenderTimeUs,
                        float(m_userInterfaceParameters->neuralRenderTimeUs) / 1000.f);

            if (m_userInterfaceParameters->defaultPbrRenderTimeUs > 0 && m_userInterfaceParameters->neuralRenderTimeUs > 0)
            {
                const int neuralDeltaUs = m_userInterfaceParameters->neuralRenderTimeUs - m_userInterfaceParameters->defaultPbrRenderTimeUs;
                const float neuralRatio = float(m_userInterfaceParameters->neuralRenderTimeUs) / float(m_userInterfaceParameters->defaultPbrRenderTimeUs);
                const float pbrToNeuralSpeedup = float(m_userInterfaceParameters->defaultPbrRenderTimeUs) / float(m_userInterfaceParameters->neuralRenderTimeUs);
                ImGui::Text("Neural vs PreintegratedSkin/PBR Delta : %+d us", neuralDeltaUs);
                ImGui::Text("PreintegratedSkin/PBR / Neural Speedup : %.2fx", pbrToNeuralSpeedup);
                ImGui::Text("Neural / PreintegratedSkin/PBR Cost : %.2fx", neuralRatio);
            }

            ImGui::Text("Training : %d us (%.3f ms)", m_userInterfaceParameters->trainingPassTimeUs, float(m_userInterfaceParameters->trainingPassTimeUs) / 1000.f);
            ImGui::Text("Optimization : %d us (%.3f ms)", m_userInterfaceParameters->optimizerPassTimeUs,
                        float(m_userInterfaceParameters->optimizerPassTimeUs) / 1000.f);
        }
        ImGui::Separator();

        ImGui::SliderFloat("Light Intensity", &m_userInterfaceParameters->lightIntensity, 0.f, 20.f);
        ImGui::Checkbox("Enable IBL", &m_userInterfaceParameters->useIBL);
        ImGui::SliderFloat("IBL Intensity", &m_userInterfaceParameters->iblIntensity, 0.f, 5.f);
        ImGui::SliderFloat("IBL Rotation", &m_userInterfaceParameters->iblRotation, -3.14159f, 3.14159f);
        ImGui::SliderFloat("Specular Base", &m_userInterfaceParameters->specular, 0.f, 1.f);
        ImGui::SliderFloat("Roughness Scale", &m_userInterfaceParameters->roughness, 0.1f, 2.f);
        ImGui::SliderFloat("Metallic Scale", &m_userInterfaceParameters->metallic, 0.f, 1.f);
        ImGui::SliderFloat("Skin Opacity", &m_userInterfaceParameters->anisotropy, 0.f, 1.f);
        ImGui::SliderFloat("Subsurface Scale", &m_userInterfaceParameters->specularShift, 0.f, 2.f);
        ImGui::Checkbox("Only Neural Debug", &m_userInterfaceParameters->onlyNeuralDebug);

        ImGui::Text("Epochs : %d", m_userInterfaceParameters->epochs);
        ImGui::Text("Training Time : %.2f s", m_userInterfaceParameters->trainingTime);

        if (ImGui::Button(m_userInterfaceParameters->training ? "Disable Training" : "Enable Training"))
        {
            m_userInterfaceParameters->training = !m_userInterfaceParameters->training;
        }

        if (ImGui::Button("Reset Training"))
        {
            m_userInterfaceParameters->reset = true;
        }

        if (ImGui::Button("Load Model"))
        {
            std::string fileName;
            if (app::FileDialog(true, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
            {
                m_userInterfaceParameters->fileName = fileName;
                m_userInterfaceParameters->load = true;
            }
        }

        if (ImGui::Button("Save Model"))
        {
            std::string fileName;
            if (app::FileDialog(false, "BIN files\0*.bin\0All files\0*.*\0\0", fileName))
            {
                m_userInterfaceParameters->fileName = fileName;
                m_userInterfaceParameters->load = false;
            }
        }

        ImGui::End();
    }

private:
    UIData* m_userInterfaceParameters;
};

#ifdef WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI graphicsApi = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    if (graphicsApi == nvrhi::GraphicsAPI::D3D11)
    {
        log::error("This sample does not support D3D11.");
        return 1;
    }
    std::unique_ptr<app::DeviceManager> deviceManager(app::DeviceManager::Create(graphicsApi));

    app::DeviceCreationParameters deviceParams;
    deviceParams.backBufferWidth = deviceParams.backBufferHeight * g_viewsNum;

#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    ////////////////////
    //
    // Setup the CoopVector extensions.
    //
    ////////////////////
    SetCoopVectorExtensionParameters(deviceParams, graphicsApi, true, g_windowTitle);

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_windowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters. Please try a NVIDIA driver version greater than 570");
        return 1;
    }

    auto graphicsResources = std::make_unique<rtxns::GraphicsResources>(deviceManager->GetDevice());
    if (!graphicsResources->GetCoopVectorFeatures().inferenceSupported && !graphicsResources->GetCoopVectorFeatures().trainingSupported &&
        !graphicsResources->GetCoopVectorFeatures().fp16InferencingSupported && !graphicsResources->GetCoopVectorFeatures().fp16TrainingSupported)
    {
        log::fatal("Not all required Coop Vector features are available");
        return 1;
    }

    {
        UIData uiData;
        SimpleShading example(deviceManager.get(), &uiData);
        UserInterface gui(deviceManager.get(), &uiData);

        if (example.Init() && gui.Init(example.GetShaderFactory()))
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->AddRenderPassToBack(&gui);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&gui);
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();

    return 0;
}
