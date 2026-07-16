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
import Silk;

DECLARE_CBUFFER(TrainingConstantBufferEntry, gConst, 0, 0);
ByteAddressBuffer gMLPParams               : REGISTER_SRV(0, 0);
RWByteAddressBuffer gMLPParamsGradients    : REGISTER_UAV(0, 0);

[shader("compute")]
[numthreads(64, 1, 1)]
void main_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    PCG32 rng = PCG32(gConst.seed, idx);

    float3 L;
    L.y = 0.0f;
    sincos(rng.nextFloat() * SILK_PI * 0.5f, L.z, L.x);

    float sa, ca;
    sincos(-SILK_PI + 2.0f * SILK_PI * rng.nextFloat(), sa, ca);
    float se, ce;
    sincos(SILK_PI * 0.5f * rng.nextFloat(), se, ce);
    float3 V = float3(ce * ca, ce * sa, se);

    float NdotL = L.z;
    float NdotV = V.z;
    float3 H = normalize(L + V);
    float NdotH = H.z;
    float LdotH = dot(L, H);
    float roughness = rng.nextFloat() * 0.7f + 0.3f;

    float4 actualSilk = SilkOriginal(NdotL, NdotV, NdotH, LdotH, roughness);

    float params[ORIGINAL_INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };
    var inputParams = rtxns::EncodeFrequency<half, ORIGINAL_INPUT_FEATURES>(params);

    var model = rtxns::mlp::TrainingMLP<half,
        NUM_HIDDEN_LAYERS, ORIGINAL_INPUT_NEURONS, ORIGINAL_HIDDEN_NEURONS, OUTPUT_NEURONS,
        CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
        gMLPParams,
        gMLPParamsGradients,
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets),
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

    var hiddenActivation = rtxns::mlp::ReLUAct<half, ORIGINAL_HIDDEN_NEURONS>();
    var finalActivation = rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>();

    var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);

    float4 predictedSilk = { outputParams[0], outputParams[1], outputParams[2], outputParams[3] };
    float4 lossGradient = rtxns::mlp::L2Relative<float, 4>.deriv(actualSilk, predictedSilk, float4(LOSS_SCALE / (gConst.batchSize * 4)) * COMPONENT_WEIGHTS);

    model.backward(inputParams, hiddenActivation, finalActivation, rtxns::HCoopVec<OUTPUT_NEURONS>(lossGradient[0], lossGradient[1], lossGradient[2], lossGradient[3]));
}
