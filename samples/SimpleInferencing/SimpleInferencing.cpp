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
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>

#include "DeviceUtils.h"
#include "GraphicsResources.h"
#include "GeometryUtils.h"
#include "NeuralNetwork.h"
#include "DirectoryHelper.h"
#include "PbrTextureUtils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <format>
#include <utility>
#include <vector>

using namespace donut;
using namespace donut::math;

#include "NetworkConfig.h"

static const char* g_windowTitle = "RTX Neural Shading Example: Simple Inferencing";

struct UIData
{
    float lightIntensity = 1.f;
    float iblIntensity = 0.35f;
    float iblRotation = 0.f;
    float specular = 0.5f;
    float roughness = 1.f;
    float metallic = 1.f;
    bool useIBL = true;
};

class SimpleInferencing : public app::IRenderPass
{
public:
    SimpleInferencing(app::DeviceManager* deviceManager, UIData* uiParams) : IRenderPass(deviceManager), m_userInterfaceParameters(uiParams)
    {
    }

    bool Init()
    {
        ////////////////////
        //
        // Create the Neural network class and initialise it from a file.
        //
        ////////////////////
        m_networkUtils = std::make_shared<rtxns::NetworkUtilities>(GetDevice());
        rtxns::HostNetwork net(m_networkUtils);
        if (!net.InitialiseFromFile(GetLocalPath("assets/data").string() + std::string("/disney.ns.bin")))
        {
            log::debug("Loaded Neural Shading Network from file failed.");
            return false;
        }

        // We are expecting 4 layers, validate
        assert(net.GetNetworkLayout().networkLayers.size() == 4);

        // Get a device optimized layout
        rtxns::NetworkLayout deviceNetworkLayout = m_networkUtils->GetNewMatrixLayout(net.GetNetworkLayout(), rtxns::MatrixLayout::InferencingOptimal);

        // Store the weight and bias offsets into a uint4.
        m_weightOffsets = dm::uint4(deviceNetworkLayout.networkLayers[0].weightOffset, deviceNetworkLayout.networkLayers[1].weightOffset,
                                    deviceNetworkLayout.networkLayers[2].weightOffset, deviceNetworkLayout.networkLayers[3].weightOffset);

        m_biasOffsets = dm::uint4(deviceNetworkLayout.networkLayers[0].biasOffset, deviceNetworkLayout.networkLayers[1].biasOffset, deviceNetworkLayout.networkLayers[2].biasOffset,
                                  deviceNetworkLayout.networkLayers[3].biasOffset);

        ////////////////////
        //
        // Continue to load the render data and create the required structures
        //
        ////////////////////
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/SimpleInferencing" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        rootFS->mount("/shaders/app", appShaderPath);

        m_shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_vertexShader = m_shaderFactory->CreateShader("app/SimpleInferencing", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
        m_pixelShader = m_shaderFactory->CreateShader("app/SimpleInferencing", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

        if (!m_vertexShader || !m_pixelShader)
        {
            return false;
        }

        const std::filesystem::path previewModelFileName = GetLocalPath("assets/data/Model") / "Meet_MAT.obj";
        auto [vertices, indices] = LoadObjModel(previewModelFileName);
        if (vertices.empty() || indices.empty())
        {
            log::error("Failed to load preview model: %s", previewModelFileName.string().c_str());
            return false;
        }
        m_indicesNum = (int)indices.size();

        m_constantBuffer = GetDevice()->createBuffer(
            nvrhi::utils::CreateStaticConstantBufferDesc(sizeof(NeuralConstants), "ConstantBuffer").setInitialState(nvrhi::ResourceStates::ConstantBuffer).setKeepInitialState(true));

        nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc().setName("POSITION").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(0).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("NORMAL").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(1).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("TANGENT").setFormat(nvrhi::Format::RGB32_FLOAT).setOffset(0).setBufferIndex(2).setElementStride(sizeof(Vertex)),
            nvrhi::VertexAttributeDesc().setName("TEXCOORD").setFormat(nvrhi::Format::RG32_FLOAT).setOffset(0).setBufferIndex(3).setElementStride(sizeof(Vertex)),
        };
        m_inputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), m_vertexShader);

        m_commonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_shaderFactory);

        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        engine::TextureCache textureCache(GetDevice(), nativeFS, nullptr);

        m_commandList = GetDevice()->createCommandList();
        m_commandList->open();

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

        m_pbrTextures = LoadDefaultPbrTextures(textureCache, m_commonPasses.get(), m_commandList);
        if (!m_pbrTextures.IsComplete())
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

        ////////////////////
        //
        // Create buffers for storing the neural parameters/weights and biases
        //
        ////////////////////
        const auto& params = net.GetNetworkParams();

        // Create a buffer for the host side weight and bias parameters
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = params.size();
        bufferDesc.debugName = "MLPParamsUploadBuffer";
        bufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
        bufferDesc.keepInitialState = true;
        m_mlpHostBuffer = GetDevice()->createBuffer(bufferDesc);

        // Create a buffer for a device optimized parameters layout
        bufferDesc.byteSize = deviceNetworkLayout.networkSize;
        bufferDesc.canHaveRawViews = true;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.debugName = "MLPParamsByteAddressBuffer";
        bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        m_mlpDeviceBuffer = GetDevice()->createBuffer(bufferDesc);

        // Upload the parameters
        m_commandList->writeBuffer(m_mlpHostBuffer, params.data(), params.size());

        // Convert to GPU optimized layout
        m_networkUtils->ConvertWeights(net.GetNetworkLayout(), deviceNetworkLayout, m_mlpHostBuffer, 0, m_mlpDeviceBuffer, 0, GetDevice(), m_commandList);

        m_commandList->setBufferState(m_mlpDeviceBuffer, nvrhi::ResourceStates::ShaderResource);
        m_commandList->commitBarriers();

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);

        ////////////////////
        //
        // Create the binding set
        //
        ////////////////////
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = { // Note: using viewIndex to construct a buffer range.
                                    nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
                                    // Parameters buffer
                                    nvrhi::BindingSetItem::RawBuffer_SRV(0, m_mlpDeviceBuffer),
                                    nvrhi::BindingSetItem::Texture_SRV(1, m_environmentMap),
                                    nvrhi::BindingSetItem::Texture_SRV(2, m_pbrTextures.baseColor),
                                    nvrhi::BindingSetItem::Texture_SRV(3, m_pbrTextures.roughness),
                                    nvrhi::BindingSetItem::Texture_SRV(4, m_pbrTextures.metallic),
                                    nvrhi::BindingSetItem::Texture_SRV(5, m_pbrTextures.normal),
                                    nvrhi::BindingSetItem::Sampler(0, m_commonPasses->m_LinearWrapSampler)
        };

        // Create the binding layout (if it's empty -- so, on the first iteration) and the binding set.
        if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_bindingLayout, m_bindingSet))
        {
            log::error("Couldn't create the binding set or layout");
            return false;
        }

        m_neuralTimer = GetDevice()->createTimerQuery();

        return true;
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
        auto t = int(GetDevice()->getTimerQueryTime(m_neuralTimer) * 1000000);
        if (t != 0)
        {
            m_extraStatus = std::format(" - Neural - {:3d}us", t);
        }

        GetDeviceManager()->SetInformativeWindowTitle(g_windowTitle, true, m_extraStatus.c_str());
    }

    void BackBufferResizing() override
    {
        m_pipeline = nullptr;
        m_depthBuffer = nullptr;
        m_depthFramebuffers.clear();
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();
        const float width = float(fbinfo.width);
        const float height = float(fbinfo.height);
        const float left = 0;
        const float top = 0;

        bool updateStat = GetDeviceManager()->GetCurrentBackBufferIndex() % 100 == 0;

        nvrhi::IFramebuffer* renderFramebuffer = GetDepthFramebuffer(framebuffer);

        if (!m_pipeline)
        {
            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_vertexShader;
            psoDesc.PS = m_pixelShader;
            psoDesc.inputLayout = m_inputLayout;
            psoDesc.bindingLayouts = { m_bindingLayout };
            psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
            psoDesc.renderState.depthStencilState.depthTestEnable = true;
            psoDesc.renderState.depthStencilState.depthWriteEnable = true;
            psoDesc.renderState.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;

            m_pipeline = GetDevice()->createGraphicsPipeline(psoDesc, renderFramebuffer);
        }

        m_commandList->open();

        nvrhi::utils::ClearColorAttachment(m_commandList, renderFramebuffer, 0, nvrhi::Color(0.f));
        nvrhi::utils::ClearDepthStencilAttachment(m_commandList, renderFramebuffer, 1.f, 0);

        // Orbit camera around the preview model. Right mouse drag changes yaw/pitch.
        const float3 cameraPos = GetCameraPosition();
        const float3 viewDir = -normalize(cameraPos);
        float3 cameraUp(0, 1, 0);

        ////////////////////
        //
        // Fill out the constant buffer including the neural weight/bias offsets.
        //
        ////////////////////
        NeuralConstants modelConstant{ {},
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
                                       0.f,
                                       m_weightOffsets,
                                       m_biasOffsets };
        modelConstant.view = affineToHomogeneous(translation(-cameraPos) * lookatZ(-viewDir, cameraUp));
        modelConstant.viewProject = modelConstant.view * perspProjD3DStyle(radians(67.4f), float(width) / float(height), 0.1f, 10.f);

        // Upload the constant buffer.
        m_commandList->writeBuffer(m_constantBuffer, &modelConstant, sizeof(modelConstant));

        nvrhi::GraphicsState state;
        state.bindings = { m_bindingSet };
        state.indexBuffer = { m_indexBuffer, nvrhi::Format::R32_UINT, 0 };

        state.vertexBuffers = {
            { m_vertexBuffer, 0, offsetof(Vertex, position) },
            { m_vertexBuffer, 1, offsetof(Vertex, normal) },
            { m_vertexBuffer, 2, offsetof(Vertex, tangent) },
            { m_vertexBuffer, 3, offsetof(Vertex, texcoord) },
        };
        state.pipeline = m_pipeline;
        state.framebuffer = renderFramebuffer;

        const nvrhi::Viewport viewport = nvrhi::Viewport(left, left + width, top, top + height, 0.f, 1.f);
        state.viewport.addViewportAndScissorRect(viewport);

        if (updateStat)
        {
            GetDevice()->resetTimerQuery(m_neuralTimer);
            m_commandList->beginTimerQuery(m_neuralTimer);
        }

        // Update the pipeline, bindings, and other state.
        m_commandList->setGraphicsState(state);

        // Draw the model.
        nvrhi::DrawArguments args;
        args.vertexCount = m_indicesNum;
        m_commandList->drawIndexed(args);

        if (updateStat)
        {
            m_commandList->endTimerQuery(m_neuralTimer);
        }

        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);
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
            newDepthDesc.debugName = "SimpleInferencingDepth";

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
    nvrhi::TimerQueryHandle m_neuralTimer;
    nvrhi::ShaderHandle m_vertexShader;
    nvrhi::ShaderHandle m_pixelShader;
    nvrhi::BufferHandle m_constantBuffer;
    nvrhi::BufferHandle m_mlpHostBuffer;
    nvrhi::BufferHandle m_mlpDeviceBuffer;
    nvrhi::BufferHandle m_vertexBuffer;
    nvrhi::BufferHandle m_indexBuffer;
    nvrhi::InputLayoutHandle m_inputLayout;
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingSetHandle m_bindingSet;
    nvrhi::GraphicsPipelineHandle m_pipeline;
    nvrhi::TextureHandle m_depthBuffer;
    std::vector<std::pair<nvrhi::IFramebuffer*, nvrhi::FramebufferHandle>> m_depthFramebuffers;
    nvrhi::CommandListHandle m_commandList;

    std::shared_ptr<engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<rtxns::NetworkUtilities> m_networkUtils;
    nvrhi::TextureHandle m_environmentMap;
    PbrTextureSet m_pbrTextures;

    float3 m_lightDir{ -0.761f, -0.467f, -0.450f };
    float m_environmentMipCount = 1.f;
    float m_modelYaw = 0.f;
    float m_modelPitch = 0.f;
    float m_cameraDistance = 2.f;
    float2 m_currentXY;
    bool m_lightDragActive = false;
    bool m_modelDragActive = false;

    int m_indicesNum = 0;

    dm::uint4 m_weightOffsets; // Offsets to weight matrices in bytes.
    dm::uint4 m_biasOffsets; // Offsets to bias vectors in bytes.

    UIData* m_userInterfaceParameters;
};

class UserInterface : public app::ImGui_Renderer
{
private:
    UIData* m_userInterfaceParameters;

public:
    UserInterface(app::DeviceManager* deviceManager, UIData* uiParams) : ImGui_Renderer(deviceManager), m_userInterfaceParameters(uiParams)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void buildUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), 0);
        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::SliderFloat("Light Intensity", &m_userInterfaceParameters->lightIntensity, 0.f, 20.f);
        ImGui::Checkbox("Enable IBL", &m_userInterfaceParameters->useIBL);
        ImGui::SliderFloat("IBL Intensity", &m_userInterfaceParameters->iblIntensity, 0.f, 5.f);
        ImGui::SliderFloat("IBL Rotation", &m_userInterfaceParameters->iblRotation, -3.14159f, 3.14159f);
        ImGui::SliderFloat("Specular", &m_userInterfaceParameters->specular, 0.f, 1.f);
        ImGui::SliderFloat("Roughness Scale", &m_userInterfaceParameters->roughness, 0.1f, 2.f);
        ImGui::SliderFloat("Metallic Scale", &m_userInterfaceParameters->metallic, 0.f, 1.f);

        ImGui::End();
    }
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
        log::error("This sample does not support D3D11");
        return 1;
    }
    std::unique_ptr<app::DeviceManager> deviceManager(app::DeviceManager::Create(graphicsApi));

    app::DeviceCreationParameters deviceParams;
    deviceParams.backBufferWidth = deviceParams.backBufferHeight;

#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableGPUValidation = false;
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
    if (!graphicsResources->GetCoopVectorFeatures().inferenceSupported && !graphicsResources->GetCoopVectorFeatures().fp16InferencingSupported)
    {
        log::fatal("Not all required Coop Vector features are available");
        return 1;
    }

    {
        UIData uiData;
        SimpleInferencing example(deviceManager.get(), &uiData);
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
