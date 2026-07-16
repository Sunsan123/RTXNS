/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

import MLP;
import CooperativeVectorFunctions;
import Activation;
import Utils;

#define SILK_ORIGINAL_INPUT_FEATURES 5
#define SILK_INPUT_FEATURES 9
#define SILK_OUTPUT_NEURONS 4
#define SILK_FREQUENCY_EXPANSION 6

float4 SilkMLPOriginal<let HIDDEN_LAYERS : int, let HIDDEN_NEURONS : int>(
    float NdotL, float NdotV, float NdotH, float LdotH, float roughness, ByteAddressBuffer mlpBuffer,
    uint weightOffsets[HIDDEN_LAYERS+1], uint biasOffsets[HIDDEN_LAYERS+1])
{
    float params[SILK_ORIGINAL_INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };
    var inputParams = rtxns::EncodeFrequency<half, SILK_ORIGINAL_INPUT_FEATURES>(params);

    var model = rtxns::mlp::InferenceMLP<half,
        HIDDEN_LAYERS,
        SILK_ORIGINAL_INPUT_FEATURES * SILK_FREQUENCY_EXPANSION,
        HIDDEN_NEURONS,
        SILK_OUTPUT_NEURONS,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16>
        (mlpBuffer, weightOffsets, biasOffsets);

    var outputParams = model.forward(inputParams, rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>(), rtxns::mlp::ExponentialAct<half, SILK_OUTPUT_NEURONS>());
    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}

float4 SilkMLP<let HIDDEN_LAYERS : int, let HIDDEN_NEURONS : int>(
    float3 L, float3 V, float roughness, float anisotropy, float specularShift, ByteAddressBuffer mlpBuffer,
    uint weightOffsets[HIDDEN_LAYERS+1], uint biasOffsets[HIDDEN_LAYERS+1])
{
    L = normalize(L);
    V = normalize(V);

    float params[SILK_INPUT_FEATURES] = {
        L.x, L.y, saturate(L.z),
        V.x, V.y, saturate(V.z),
        roughness,
        clamp(anisotropy, -0.99f, 0.99f),
        saturate(specularShift)
    };

    var inputParams = rtxns::EncodeFrequency<half, SILK_INPUT_FEATURES>(params);

    var model = rtxns::mlp::InferenceMLP<half,
        HIDDEN_LAYERS,
        SILK_INPUT_FEATURES * SILK_FREQUENCY_EXPANSION,
        HIDDEN_NEURONS,
        SILK_OUTPUT_NEURONS,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16>
        (mlpBuffer, weightOffsets, biasOffsets);

    var outputParams = model.forward(inputParams, rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>(), rtxns::mlp::ExponentialAct<half, SILK_OUTPUT_NEURONS>());
    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}
