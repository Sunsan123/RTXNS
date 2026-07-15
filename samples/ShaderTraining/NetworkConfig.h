/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define ORIGINAL_INPUT_FEATURES 5
#define ANISOTROPIC_INPUT_FEATURES 7
#define ORIGINAL_INPUT_NEURONS (ORIGINAL_INPUT_FEATURES * 6) // 6* from Frequency Encoding
#define ANISOTROPIC_INPUT_NEURONS (ANISOTROPIC_INPUT_FEATURES * 6)
#define INPUT_FEATURES ANISOTROPIC_INPUT_FEATURES
#define INPUT_NEURONS ANISOTROPIC_INPUT_NEURONS
#define OUTPUT_NEURONS 4

#define ORIGINAL_HIDDEN_NEURONS 32
#define ANISOTROPIC_HIDDEN_NEURONS 32
#define HIDDEN_NEURONS ANISOTROPIC_HIDDEN_NEURONS
#define NUM_HIDDEN_LAYERS 3
#define BATCH_SIZE (1 << 16)
#define BATCH_COUNT 100

#define LEARNING_RATE 1e-2f
#define COMPONENT_WEIGHTS float4(1.f, 10.f, 1.f, 5.f)

#define NUM_TRANSITIONS (NUM_HIDDEN_LAYERS + 1)
#define NUM_TRANSITIONS_ALIGN4 ((NUM_TRANSITIONS + 3) / 4)
#define LOSS_SCALE 128.0

struct DirectConstantBufferEntry
{
    // Scene setup
    float4x4 viewProject;
    float4x4 view;
    float4 cameraPos;

    // Light setup
    float4 lightDir;
    float4 lightIntensity;
    float4 iblSettings;

    // Material props
    float4 baseColor;
    float specular = 0;
    float roughness = 0;
    float metallic = 0;
    float anisotropy = 0;
    uint32_t useAnisotropy = 0;
    float pad0 = 0;
    float pad1 = 0;
    float pad2 = 0;
};

struct InferenceConstantBufferEntry : DirectConstantBufferEntry
{
    // Neural weight & bias offsets
    uint4 weightOffsets[NUM_TRANSITIONS_ALIGN4];
    uint4 biasOffsets[NUM_TRANSITIONS_ALIGN4];
};

struct TrainingConstantBufferEntry
{
    uint4 weightOffsets[NUM_TRANSITIONS_ALIGN4];
    uint4 biasOffsets[NUM_TRANSITIONS_ALIGN4];
    uint32_t maxParamSize;
    float learningRate;
    float currentStep;
    uint32_t batchSize;
    uint64_t seed;
    uint32_t useAnisotropy;
    float pad0;
};
