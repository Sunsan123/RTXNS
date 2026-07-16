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

#define SKIN_INPUT_FEATURES 8
#define SKIN_OUTPUT_NEURONS 4
#define SKIN_FREQUENCY_EXPANSION 6

float4 PreintegratedSkinMLP<let HIDDEN_LAYERS : int, let HIDDEN_NEURONS : int>(
    float3 L,
    float3 V,
    float roughness,
    float opacity,
    ByteAddressBuffer mlpBuffer,
    uint weightOffsets[HIDDEN_LAYERS + 1],
    uint biasOffsets[HIDDEN_LAYERS + 1])
{
    L = normalize(L);
    V = normalize(V);

    float params[SKIN_INPUT_FEATURES] = {
        L.x, L.y, saturate(L.z),
        V.x, V.y, saturate(V.z),
        clamp(roughness, 0.3f, 1.0f),
        saturate(opacity)
    };

    var inputParams = rtxns::EncodeFrequency<half, SKIN_INPUT_FEATURES>(params);

    var model = rtxns::mlp::InferenceMLP<half,
        HIDDEN_LAYERS,
        SKIN_INPUT_FEATURES * SKIN_FREQUENCY_EXPANSION,
        HIDDEN_NEURONS,
        SKIN_OUTPUT_NEURONS,
        CoopVecMatrixLayout::TrainingOptimal,
        CoopVecComponentType::Float16>
        (mlpBuffer, weightOffsets, biasOffsets);

    var outputParams = model.forward(inputParams, rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>(), rtxns::mlp::ExponentialAct<half, SKIN_OUTPUT_NEURONS>());
    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}
