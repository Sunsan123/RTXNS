/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import CooperativeVectorAutoDiff;
import CooperativeVectorFunctions;
import Utils;
import Activation;
import MLP;
import Loss;
import PCG32;
import PreintegratedSkin;

DECLARE_CBUFFER(TrainingConstantBufferEntry, gConst, 0, 0);
ByteAddressBuffer gMLPParams               : REGISTER_SRV(0, 0);
Texture2D<float4> gPreintegratedSkinLut    : REGISTER_SRV(1, 0);
RWByteAddressBuffer gMLPParamsGradients    : REGISTER_UAV(0, 0);
SamplerState gSampler                      : REGISTER_SAMPLER(0, 0);

[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    PCG32 rng = PCG32(gConst.seed, idx);

    float lightAzimuth = 2.0f * SKIN_PI * rng.nextFloat();
    float lightCosTheta = rng.nextFloat();
    float lightSinTheta = sqrt(max(0.0f, 1.0f - lightCosTheta * lightCosTheta));
    float sinLightAzimuth, cosLightAzimuth;
    sincos(lightAzimuth, sinLightAzimuth, cosLightAzimuth);
    float3 L = float3(lightSinTheta * cosLightAzimuth, lightSinTheta * sinLightAzimuth, lightCosTheta);

    float viewAzimuth = 2.0f * SKIN_PI * rng.nextFloat();
    float viewCosTheta = rng.nextFloat();
    float viewSinTheta = sqrt(max(0.0f, 1.0f - viewCosTheta * viewCosTheta));
    float sinViewAzimuth, cosViewAzimuth;
    sincos(viewAzimuth, sinViewAzimuth, cosViewAzimuth);
    float3 V = float3(viewSinTheta * cosViewAzimuth, viewSinTheta * sinViewAzimuth, viewCosTheta);

    float roughness = rng.nextFloat() * 0.6f + 0.35f;
    float opacity = rng.nextFloat() * 0.9f + 0.05f;

    float4 actualSkin = PreintegratedSkin(gPreintegratedSkinLut, gSampler, L, V, roughness, opacity);

    float params[INPUT_FEATURES] = {
        L.x, L.y, L.z,
        V.x, V.y, V.z,
        roughness,
        opacity
    };
    var inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

    var model = rtxns::mlp::TrainingMLP<half,
        NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS,
        CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
        gMLPParams,
        gMLPParamsGradients,
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

    var hiddenActivation = rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>();
    var finalActivation = rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>();

    var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);

    float4 predictedSkin = { outputParams[0], outputParams[1], outputParams[2], outputParams[3] };
    float4 lossGradient = rtxns::mlp::L2Relative<float, 4>.deriv(actualSkin, predictedSkin, float4(LOSS_SCALE / (gConst.batchSize * 4)) * COMPONENT_WEIGHTS);

    model.backward(inputParams, hiddenActivation, finalActivation, rtxns::HCoopVec<OUTPUT_NEURONS>(lossGradient[0], lossGradient[1], lossGradient[2], lossGradient[3]));
}
