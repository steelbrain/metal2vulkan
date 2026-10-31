// Generated from the macOS 27.2 SDK Metal.framework headers (names and values).
// Regenerate with scripts/regen-metal-enums/regen_metal_enums.py; do not edit.

metal_enum! {
    /// `MTLArgumentBuffersTier`.
    #[non_exhaustive]
    ArgumentBuffersTier = "MTLArgumentBuffersTier" {
        Tier1 = 0, "MTLArgumentBuffersTier1";
        Tier2 = 1, "MTLArgumentBuffersTier2";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLAttributeFormat`.
    #[non_exhaustive]
    AttributeFormat = "MTLAttributeFormat" {
        Invalid = 0, "MTLAttributeFormatInvalid";
        UChar2 = 1, "MTLAttributeFormatUChar2";
        UChar3 = 2, "MTLAttributeFormatUChar3";
        UChar4 = 3, "MTLAttributeFormatUChar4";
        Char2 = 4, "MTLAttributeFormatChar2";
        Char3 = 5, "MTLAttributeFormatChar3";
        Char4 = 6, "MTLAttributeFormatChar4";
        UChar2Normalized = 7, "MTLAttributeFormatUChar2Normalized";
        UChar3Normalized = 8, "MTLAttributeFormatUChar3Normalized";
        UChar4Normalized = 9, "MTLAttributeFormatUChar4Normalized";
        Char2Normalized = 10, "MTLAttributeFormatChar2Normalized";
        Char3Normalized = 11, "MTLAttributeFormatChar3Normalized";
        Char4Normalized = 12, "MTLAttributeFormatChar4Normalized";
        UShort2 = 13, "MTLAttributeFormatUShort2";
        UShort3 = 14, "MTLAttributeFormatUShort3";
        UShort4 = 15, "MTLAttributeFormatUShort4";
        Short2 = 16, "MTLAttributeFormatShort2";
        Short3 = 17, "MTLAttributeFormatShort3";
        Short4 = 18, "MTLAttributeFormatShort4";
        UShort2Normalized = 19, "MTLAttributeFormatUShort2Normalized";
        UShort3Normalized = 20, "MTLAttributeFormatUShort3Normalized";
        UShort4Normalized = 21, "MTLAttributeFormatUShort4Normalized";
        Short2Normalized = 22, "MTLAttributeFormatShort2Normalized";
        Short3Normalized = 23, "MTLAttributeFormatShort3Normalized";
        Short4Normalized = 24, "MTLAttributeFormatShort4Normalized";
        Half2 = 25, "MTLAttributeFormatHalf2";
        Half3 = 26, "MTLAttributeFormatHalf3";
        Half4 = 27, "MTLAttributeFormatHalf4";
        Float = 28, "MTLAttributeFormatFloat";
        Float2 = 29, "MTLAttributeFormatFloat2";
        Float3 = 30, "MTLAttributeFormatFloat3";
        Float4 = 31, "MTLAttributeFormatFloat4";
        Int = 32, "MTLAttributeFormatInt";
        Int2 = 33, "MTLAttributeFormatInt2";
        Int3 = 34, "MTLAttributeFormatInt3";
        Int4 = 35, "MTLAttributeFormatInt4";
        UInt = 36, "MTLAttributeFormatUInt";
        UInt2 = 37, "MTLAttributeFormatUInt2";
        UInt3 = 38, "MTLAttributeFormatUInt3";
        UInt4 = 39, "MTLAttributeFormatUInt4";
        Int1010102Normalized = 40, "MTLAttributeFormatInt1010102Normalized";
        UInt1010102Normalized = 41, "MTLAttributeFormatUInt1010102Normalized";
        UChar4NormalizedBGRA = 42, "MTLAttributeFormatUChar4Normalized_BGRA";
        UChar = 45, "MTLAttributeFormatUChar";
        Char = 46, "MTLAttributeFormatChar";
        UCharNormalized = 47, "MTLAttributeFormatUCharNormalized";
        CharNormalized = 48, "MTLAttributeFormatCharNormalized";
        UShort = 49, "MTLAttributeFormatUShort";
        Short = 50, "MTLAttributeFormatShort";
        UShortNormalized = 51, "MTLAttributeFormatUShortNormalized";
        ShortNormalized = 52, "MTLAttributeFormatShortNormalized";
        Half = 53, "MTLAttributeFormatHalf";
        FloatRG11B10 = 54, "MTLAttributeFormatFloatRG11B10";
        FloatRGB9E5 = 55, "MTLAttributeFormatFloatRGB9E5";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLBindingAccess`.
    #[non_exhaustive]
    BindingAccess = "MTLBindingAccess" {
        ReadOnly = 0, "MTLBindingAccessReadOnly";
        ReadWrite = 1, "MTLBindingAccessReadWrite";
        WriteOnly = 2, "MTLBindingAccessWriteOnly";
    }
    aliases {
        "MTLArgumentAccessReadOnly" => ReadOnly;
        "MTLArgumentAccessReadWrite" => ReadWrite;
        "MTLArgumentAccessWriteOnly" => WriteOnly;
    }
}

metal_enum! {
    /// `MTLBindingType`.
    #[non_exhaustive]
    BindingType = "MTLBindingType" {
        Buffer = 0, "MTLBindingTypeBuffer";
        ThreadgroupMemory = 1, "MTLBindingTypeThreadgroupMemory";
        Texture = 2, "MTLBindingTypeTexture";
        Sampler = 3, "MTLBindingTypeSampler";
        ImageblockData = 16, "MTLBindingTypeImageblockData";
        Imageblock = 17, "MTLBindingTypeImageblock";
        VisibleFunctionTable = 24, "MTLBindingTypeVisibleFunctionTable";
        PrimitiveAccelerationStructure = 25, "MTLBindingTypePrimitiveAccelerationStructure";
        InstanceAccelerationStructure = 26, "MTLBindingTypeInstanceAccelerationStructure";
        IntersectionFunctionTable = 27, "MTLBindingTypeIntersectionFunctionTable";
        ObjectPayload = 34, "MTLBindingTypeObjectPayload";
        Tensor = 37, "MTLBindingTypeTensor";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLBlendFactor`.
    #[non_exhaustive]
    BlendFactor = "MTLBlendFactor" {
        Zero = 0, "MTLBlendFactorZero";
        One = 1, "MTLBlendFactorOne";
        SourceColor = 2, "MTLBlendFactorSourceColor";
        OneMinusSourceColor = 3, "MTLBlendFactorOneMinusSourceColor";
        SourceAlpha = 4, "MTLBlendFactorSourceAlpha";
        OneMinusSourceAlpha = 5, "MTLBlendFactorOneMinusSourceAlpha";
        DestinationColor = 6, "MTLBlendFactorDestinationColor";
        OneMinusDestinationColor = 7, "MTLBlendFactorOneMinusDestinationColor";
        DestinationAlpha = 8, "MTLBlendFactorDestinationAlpha";
        OneMinusDestinationAlpha = 9, "MTLBlendFactorOneMinusDestinationAlpha";
        SourceAlphaSaturated = 10, "MTLBlendFactorSourceAlphaSaturated";
        BlendColor = 11, "MTLBlendFactorBlendColor";
        OneMinusBlendColor = 12, "MTLBlendFactorOneMinusBlendColor";
        BlendAlpha = 13, "MTLBlendFactorBlendAlpha";
        OneMinusBlendAlpha = 14, "MTLBlendFactorOneMinusBlendAlpha";
        Source1Color = 15, "MTLBlendFactorSource1Color";
        OneMinusSource1Color = 16, "MTLBlendFactorOneMinusSource1Color";
        Source1Alpha = 17, "MTLBlendFactorSource1Alpha";
        OneMinusSource1Alpha = 18, "MTLBlendFactorOneMinusSource1Alpha";
        Unspecialized = 19, "MTLBlendFactorUnspecialized";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLBlendOperation`.
    #[non_exhaustive]
    BlendOperation = "MTLBlendOperation" {
        Add = 0, "MTLBlendOperationAdd";
        Subtract = 1, "MTLBlendOperationSubtract";
        ReverseSubtract = 2, "MTLBlendOperationReverseSubtract";
        Min = 3, "MTLBlendOperationMin";
        Max = 4, "MTLBlendOperationMax";
        Unspecialized = 5, "MTLBlendOperationUnspecialized";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLCompareFunction`.
    #[non_exhaustive]
    CompareFunction = "MTLCompareFunction" {
        Never = 0, "MTLCompareFunctionNever";
        Less = 1, "MTLCompareFunctionLess";
        Equal = 2, "MTLCompareFunctionEqual";
        LessEqual = 3, "MTLCompareFunctionLessEqual";
        Greater = 4, "MTLCompareFunctionGreater";
        NotEqual = 5, "MTLCompareFunctionNotEqual";
        GreaterEqual = 6, "MTLCompareFunctionGreaterEqual";
        Always = 7, "MTLCompareFunctionAlways";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLCullMode`.
    CullMode = "MTLCullMode" {
        None = 0, "MTLCullModeNone";
        Front = 1, "MTLCullModeFront";
        Back = 2, "MTLCullModeBack";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLDepthClipMode`.
    #[non_exhaustive]
    DepthClipMode = "MTLDepthClipMode" {
        Clip = 0, "MTLDepthClipModeClip";
        Clamp = 1, "MTLDepthClipModeClamp";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLDispatchType`.
    #[non_exhaustive]
    DispatchType = "MTLDispatchType" {
        Serial = 0, "MTLDispatchTypeSerial";
        Concurrent = 1, "MTLDispatchTypeConcurrent";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLFunctionType`.
    #[non_exhaustive]
    FunctionType = "MTLFunctionType" {
        Vertex = 1, "MTLFunctionTypeVertex";
        Fragment = 2, "MTLFunctionTypeFragment";
        Kernel = 3, "MTLFunctionTypeKernel";
        Visible = 5, "MTLFunctionTypeVisible";
        Intersection = 6, "MTLFunctionTypeIntersection";
        Mesh = 7, "MTLFunctionTypeMesh";
        Object = 8, "MTLFunctionTypeObject";
        Extern = 4, "MTLFunctionTypeExtern";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLIndexType`.
    IndexType = "MTLIndexType" {
        UInt16 = 0, "MTLIndexTypeUInt16";
        UInt32 = 1, "MTLIndexTypeUInt32";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLMutability`.
    #[non_exhaustive]
    Mutability = "MTLMutability" {
        Default = 0, "MTLMutabilityDefault";
        Mutable = 1, "MTLMutabilityMutable";
        Immutable = 2, "MTLMutabilityImmutable";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLPatchType`.
    #[non_exhaustive]
    PatchType = "MTLPatchType" {
        None = 0, "MTLPatchTypeNone";
        Triangle = 1, "MTLPatchTypeTriangle";
        Quad = 2, "MTLPatchTypeQuad";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLPixelFormat`.
    #[non_exhaustive]
    PixelFormat = "MTLPixelFormat" {
        Invalid = 0, "MTLPixelFormatInvalid";
        A8Unorm = 1, "MTLPixelFormatA8Unorm";
        R8Unorm = 10, "MTLPixelFormatR8Unorm";
        R8UnormSRGB = 11, "MTLPixelFormatR8Unorm_sRGB";
        R8Snorm = 12, "MTLPixelFormatR8Snorm";
        R8Uint = 13, "MTLPixelFormatR8Uint";
        R8Sint = 14, "MTLPixelFormatR8Sint";
        R16Unorm = 20, "MTLPixelFormatR16Unorm";
        R16Snorm = 22, "MTLPixelFormatR16Snorm";
        R16Uint = 23, "MTLPixelFormatR16Uint";
        R16Sint = 24, "MTLPixelFormatR16Sint";
        R16Float = 25, "MTLPixelFormatR16Float";
        RG8Unorm = 30, "MTLPixelFormatRG8Unorm";
        RG8UnormSRGB = 31, "MTLPixelFormatRG8Unorm_sRGB";
        RG8Snorm = 32, "MTLPixelFormatRG8Snorm";
        RG8Uint = 33, "MTLPixelFormatRG8Uint";
        RG8Sint = 34, "MTLPixelFormatRG8Sint";
        B5G6R5Unorm = 40, "MTLPixelFormatB5G6R5Unorm";
        A1BGR5Unorm = 41, "MTLPixelFormatA1BGR5Unorm";
        ABGR4Unorm = 42, "MTLPixelFormatABGR4Unorm";
        BGR5A1Unorm = 43, "MTLPixelFormatBGR5A1Unorm";
        RGB8Unorm = 45, "MTLPixelFormatRGB8Unorm";
        RGB8Snorm = 46, "MTLPixelFormatRGB8Snorm";
        RGB8Uint = 47, "MTLPixelFormatRGB8Uint";
        RGB8Sint = 48, "MTLPixelFormatRGB8Sint";
        R32Uint = 53, "MTLPixelFormatR32Uint";
        R32Sint = 54, "MTLPixelFormatR32Sint";
        R32Float = 55, "MTLPixelFormatR32Float";
        RG16Unorm = 60, "MTLPixelFormatRG16Unorm";
        RG16Snorm = 62, "MTLPixelFormatRG16Snorm";
        RG16Uint = 63, "MTLPixelFormatRG16Uint";
        RG16Sint = 64, "MTLPixelFormatRG16Sint";
        RG16Float = 65, "MTLPixelFormatRG16Float";
        RGBA8Unorm = 70, "MTLPixelFormatRGBA8Unorm";
        RGBA8UnormSRGB = 71, "MTLPixelFormatRGBA8Unorm_sRGB";
        RGBA8Snorm = 72, "MTLPixelFormatRGBA8Snorm";
        RGBA8Uint = 73, "MTLPixelFormatRGBA8Uint";
        RGBA8Sint = 74, "MTLPixelFormatRGBA8Sint";
        BGRA8Unorm = 80, "MTLPixelFormatBGRA8Unorm";
        BGRA8UnormSRGB = 81, "MTLPixelFormatBGRA8Unorm_sRGB";
        RGB10A2Unorm = 90, "MTLPixelFormatRGB10A2Unorm";
        RGB10A2Uint = 91, "MTLPixelFormatRGB10A2Uint";
        RG11B10Float = 92, "MTLPixelFormatRG11B10Float";
        RGB9E5Float = 93, "MTLPixelFormatRGB9E5Float";
        BGR10A2Unorm = 94, "MTLPixelFormatBGR10A2Unorm";
        BGR10XR = 554, "MTLPixelFormatBGR10_XR";
        BGR10XRSRGB = 555, "MTLPixelFormatBGR10_XR_sRGB";
        RGB16Unorm = 95, "MTLPixelFormatRGB16Unorm";
        RGB16Snorm = 96, "MTLPixelFormatRGB16Snorm";
        RGB16Uint = 97, "MTLPixelFormatRGB16Uint";
        RGB16Sint = 98, "MTLPixelFormatRGB16Sint";
        RGB16Float = 99, "MTLPixelFormatRGB16Float";
        RG32Uint = 103, "MTLPixelFormatRG32Uint";
        RG32Sint = 104, "MTLPixelFormatRG32Sint";
        RG32Float = 105, "MTLPixelFormatRG32Float";
        RGBA16Unorm = 110, "MTLPixelFormatRGBA16Unorm";
        RGBA16Snorm = 112, "MTLPixelFormatRGBA16Snorm";
        RGBA16Uint = 113, "MTLPixelFormatRGBA16Uint";
        RGBA16Sint = 114, "MTLPixelFormatRGBA16Sint";
        RGBA16Float = 115, "MTLPixelFormatRGBA16Float";
        BGRA10XR = 552, "MTLPixelFormatBGRA10_XR";
        BGRA10XRSRGB = 553, "MTLPixelFormatBGRA10_XR_sRGB";
        RGB32Uint = 120, "MTLPixelFormatRGB32Uint";
        RGB32Sint = 121, "MTLPixelFormatRGB32Sint";
        RGB32Float = 122, "MTLPixelFormatRGB32Float";
        RGBA32Uint = 123, "MTLPixelFormatRGBA32Uint";
        RGBA32Sint = 124, "MTLPixelFormatRGBA32Sint";
        RGBA32Float = 125, "MTLPixelFormatRGBA32Float";
        BC1RGBA = 130, "MTLPixelFormatBC1_RGBA";
        BC1RGBASRGB = 131, "MTLPixelFormatBC1_RGBA_sRGB";
        BC2RGBA = 132, "MTLPixelFormatBC2_RGBA";
        BC2RGBASRGB = 133, "MTLPixelFormatBC2_RGBA_sRGB";
        BC3RGBA = 134, "MTLPixelFormatBC3_RGBA";
        BC3RGBASRGB = 135, "MTLPixelFormatBC3_RGBA_sRGB";
        BC4RUnorm = 140, "MTLPixelFormatBC4_RUnorm";
        BC4RSnorm = 141, "MTLPixelFormatBC4_RSnorm";
        BC5RGUnorm = 142, "MTLPixelFormatBC5_RGUnorm";
        BC5RGSnorm = 143, "MTLPixelFormatBC5_RGSnorm";
        BC6HRGBFloat = 150, "MTLPixelFormatBC6H_RGBFloat";
        BC6HRGBUfloat = 151, "MTLPixelFormatBC6H_RGBUfloat";
        BC7RGBAUnorm = 152, "MTLPixelFormatBC7_RGBAUnorm";
        BC7RGBAUnormSRGB = 153, "MTLPixelFormatBC7_RGBAUnorm_sRGB";
        PVRTCRGB2BPP = 160, "MTLPixelFormatPVRTC_RGB_2BPP";
        PVRTCRGB2BPPSRGB = 161, "MTLPixelFormatPVRTC_RGB_2BPP_sRGB";
        PVRTCRGB4BPP = 162, "MTLPixelFormatPVRTC_RGB_4BPP";
        PVRTCRGB4BPPSRGB = 163, "MTLPixelFormatPVRTC_RGB_4BPP_sRGB";
        PVRTCRGBA2BPP = 164, "MTLPixelFormatPVRTC_RGBA_2BPP";
        PVRTCRGBA2BPPSRGB = 165, "MTLPixelFormatPVRTC_RGBA_2BPP_sRGB";
        PVRTCRGBA4BPP = 166, "MTLPixelFormatPVRTC_RGBA_4BPP";
        PVRTCRGBA4BPPSRGB = 167, "MTLPixelFormatPVRTC_RGBA_4BPP_sRGB";
        EACR11Unorm = 170, "MTLPixelFormatEAC_R11Unorm";
        EACR11Snorm = 172, "MTLPixelFormatEAC_R11Snorm";
        EACRG11Unorm = 174, "MTLPixelFormatEAC_RG11Unorm";
        EACRG11Snorm = 176, "MTLPixelFormatEAC_RG11Snorm";
        EACRGBA8 = 178, "MTLPixelFormatEAC_RGBA8";
        EACRGBA8SRGB = 179, "MTLPixelFormatEAC_RGBA8_sRGB";
        ETC2RGB8 = 180, "MTLPixelFormatETC2_RGB8";
        ETC2RGB8SRGB = 181, "MTLPixelFormatETC2_RGB8_sRGB";
        ETC2RGB8A1 = 182, "MTLPixelFormatETC2_RGB8A1";
        ETC2RGB8A1SRGB = 183, "MTLPixelFormatETC2_RGB8A1_sRGB";
        ASTC4x4SRGB = 186, "MTLPixelFormatASTC_4x4_sRGB";
        ASTC5x4SRGB = 187, "MTLPixelFormatASTC_5x4_sRGB";
        ASTC5x5SRGB = 188, "MTLPixelFormatASTC_5x5_sRGB";
        ASTC6x5SRGB = 189, "MTLPixelFormatASTC_6x5_sRGB";
        ASTC6x6SRGB = 190, "MTLPixelFormatASTC_6x6_sRGB";
        ASTC8x5SRGB = 192, "MTLPixelFormatASTC_8x5_sRGB";
        ASTC8x6SRGB = 193, "MTLPixelFormatASTC_8x6_sRGB";
        ASTC8x8SRGB = 194, "MTLPixelFormatASTC_8x8_sRGB";
        ASTC10x5SRGB = 195, "MTLPixelFormatASTC_10x5_sRGB";
        ASTC10x6SRGB = 196, "MTLPixelFormatASTC_10x6_sRGB";
        ASTC10x8SRGB = 197, "MTLPixelFormatASTC_10x8_sRGB";
        ASTC10x10SRGB = 198, "MTLPixelFormatASTC_10x10_sRGB";
        ASTC12x10SRGB = 199, "MTLPixelFormatASTC_12x10_sRGB";
        ASTC12x12SRGB = 200, "MTLPixelFormatASTC_12x12_sRGB";
        ASTC4x4LDR = 204, "MTLPixelFormatASTC_4x4_LDR";
        ASTC5x4LDR = 205, "MTLPixelFormatASTC_5x4_LDR";
        ASTC5x5LDR = 206, "MTLPixelFormatASTC_5x5_LDR";
        ASTC6x5LDR = 207, "MTLPixelFormatASTC_6x5_LDR";
        ASTC6x6LDR = 208, "MTLPixelFormatASTC_6x6_LDR";
        ASTC8x5LDR = 210, "MTLPixelFormatASTC_8x5_LDR";
        ASTC8x6LDR = 211, "MTLPixelFormatASTC_8x6_LDR";
        ASTC8x8LDR = 212, "MTLPixelFormatASTC_8x8_LDR";
        ASTC10x5LDR = 213, "MTLPixelFormatASTC_10x5_LDR";
        ASTC10x6LDR = 214, "MTLPixelFormatASTC_10x6_LDR";
        ASTC10x8LDR = 215, "MTLPixelFormatASTC_10x8_LDR";
        ASTC10x10LDR = 216, "MTLPixelFormatASTC_10x10_LDR";
        ASTC12x10LDR = 217, "MTLPixelFormatASTC_12x10_LDR";
        ASTC12x12LDR = 218, "MTLPixelFormatASTC_12x12_LDR";
        ASTC4x4HDR = 222, "MTLPixelFormatASTC_4x4_HDR";
        ASTC5x4HDR = 223, "MTLPixelFormatASTC_5x4_HDR";
        ASTC5x5HDR = 224, "MTLPixelFormatASTC_5x5_HDR";
        ASTC6x5HDR = 225, "MTLPixelFormatASTC_6x5_HDR";
        ASTC6x6HDR = 226, "MTLPixelFormatASTC_6x6_HDR";
        ASTC8x5HDR = 228, "MTLPixelFormatASTC_8x5_HDR";
        ASTC8x6HDR = 229, "MTLPixelFormatASTC_8x6_HDR";
        ASTC8x8HDR = 230, "MTLPixelFormatASTC_8x8_HDR";
        ASTC10x5HDR = 231, "MTLPixelFormatASTC_10x5_HDR";
        ASTC10x6HDR = 232, "MTLPixelFormatASTC_10x6_HDR";
        ASTC10x8HDR = 233, "MTLPixelFormatASTC_10x8_HDR";
        ASTC10x10HDR = 234, "MTLPixelFormatASTC_10x10_HDR";
        ASTC12x10HDR = 235, "MTLPixelFormatASTC_12x10_HDR";
        ASTC12x12HDR = 236, "MTLPixelFormatASTC_12x12_HDR";
        GBGR422 = 240, "MTLPixelFormatGBGR422";
        BGRG422 = 241, "MTLPixelFormatBGRG422";
        Depth16Unorm = 250, "MTLPixelFormatDepth16Unorm";
        Depth32Float = 252, "MTLPixelFormatDepth32Float";
        Stencil8 = 253, "MTLPixelFormatStencil8";
        Depth24UnormStencil8 = 255, "MTLPixelFormatDepth24Unorm_Stencil8";
        Depth32FloatStencil8 = 260, "MTLPixelFormatDepth32Float_Stencil8";
        X32Stencil8 = 261, "MTLPixelFormatX32_Stencil8";
        X24Stencil8 = 262, "MTLPixelFormatX24_Stencil8";
        Unspecialized = 263, "MTLPixelFormatUnspecialized";
        ARGB8Unorm = 84, "MTLPixelFormatARGB8Unorm";
        RGB10A82PXR10 = 550, "MTLPixelFormatRGB10A8_2P_XR10";
        RGB10A82PXR10SRGB = 551, "MTLPixelFormatRGB10A8_2P_XR10_sRGB";
        YCBCR84202P = 500, "MTLPixelFormatYCBCR8_420_2P";
        YCBCR84222P = 502, "MTLPixelFormatYCBCR8_422_2P";
        YCBCR84442P = 503, "MTLPixelFormatYCBCR8_444_2P";
        YCBCR104202P = 505, "MTLPixelFormatYCBCR10_420_2P";
        YCBCR104222P = 506, "MTLPixelFormatYCBCR10_422_2P";
        YCBCR104442P = 507, "MTLPixelFormatYCBCR10_444_2P";
        YCBCR104202PPACKED = 508, "MTLPixelFormatYCBCR10_420_2P_PACKED";
        YCBCR104222PPACKED = 509, "MTLPixelFormatYCBCR10_422_2P_PACKED";
        YCBCR104442PPACKED = 510, "MTLPixelFormatYCBCR10_444_2P_PACKED";
        YCBCR84202PSRGB = 520, "MTLPixelFormatYCBCR8_420_2P_sRGB";
        YCBCR84222PSRGB = 522, "MTLPixelFormatYCBCR8_422_2P_sRGB";
        YCBCR84442PSRGB = 523, "MTLPixelFormatYCBCR8_444_2P_sRGB";
        YCBCR104202PSRGB = 525, "MTLPixelFormatYCBCR10_420_2P_sRGB";
        YCBCR104222PSRGB = 526, "MTLPixelFormatYCBCR10_422_2P_sRGB";
        YCBCR104442PSRGB = 527, "MTLPixelFormatYCBCR10_444_2P_sRGB";
        YCBCR104202PPACKEDSRGB = 528, "MTLPixelFormatYCBCR10_420_2P_PACKED_sRGB";
        YCBCR104222PPACKEDSRGB = 529, "MTLPixelFormatYCBCR10_422_2P_PACKED_sRGB";
        YCBCR104442PPACKEDSRGB = 530, "MTLPixelFormatYCBCR10_444_2P_PACKED_sRGB";
        RGB84202P = 540, "MTLPixelFormatRGB8_420_2P";
        RGB84222P = 541, "MTLPixelFormatRGB8_422_2P";
        RGB84442P = 542, "MTLPixelFormatRGB8_444_2P";
        RGB104202P = 543, "MTLPixelFormatRGB10_420_2P";
        RGB104222P = 544, "MTLPixelFormatRGB10_422_2P";
        RGB104442P = 545, "MTLPixelFormatRGB10_444_2P";
        RGB104202PPACKED = 546, "MTLPixelFormatRGB10_420_2P_PACKED";
        RGB104222PPACKED = 547, "MTLPixelFormatRGB10_422_2P_PACKED";
        RGB104442PPACKED = 548, "MTLPixelFormatRGB10_444_2P_PACKED";
        YCBCR104202PPQ = 564, "MTLPixelFormatYCBCR10_420_2P_PQ";
        YCBCR104222PPQ = 565, "MTLPixelFormatYCBCR10_422_2P_PQ";
        YCBCR104442PPQ = 566, "MTLPixelFormatYCBCR10_444_2P_PQ";
        YCBCR104202PPACKEDPQ = 567, "MTLPixelFormatYCBCR10_420_2P_PACKED_PQ";
        YCBCR104222PPACKEDPQ = 568, "MTLPixelFormatYCBCR10_422_2P_PACKED_PQ";
        YCBCR104442PPACKEDPQ = 569, "MTLPixelFormatYCBCR10_444_2P_PACKED_PQ";
        YCBCR124202P = 570, "MTLPixelFormatYCBCR12_420_2P";
        YCBCR124222P = 571, "MTLPixelFormatYCBCR12_422_2P";
        YCBCR124442P = 572, "MTLPixelFormatYCBCR12_444_2P";
        YCBCR124202PPQ = 573, "MTLPixelFormatYCBCR12_420_2P_PQ";
        YCBCR124222PPQ = 574, "MTLPixelFormatYCBCR12_422_2P_PQ";
        YCBCR124442PPQ = 575, "MTLPixelFormatYCBCR12_444_2P_PQ";
        YCBCR124202PPACKED = 580, "MTLPixelFormatYCBCR12_420_2P_PACKED";
        YCBCR124222PPACKED = 581, "MTLPixelFormatYCBCR12_422_2P_PACKED";
        YCBCR124442PPACKED = 582, "MTLPixelFormatYCBCR12_444_2P_PACKED";
        YCBCR124202PPACKEDPQ = 583, "MTLPixelFormatYCBCR12_420_2P_PACKED_PQ";
        YCBCR124222PPACKEDPQ = 584, "MTLPixelFormatYCBCR12_422_2P_PACKED_PQ";
        YCBCR124442PPACKEDPQ = 585, "MTLPixelFormatYCBCR12_444_2P_PACKED_PQ";
        YCBCR104202PXR = 591, "MTLPixelFormatYCBCR10_420_2P_XR";
        YCBCR104222PXR = 592, "MTLPixelFormatYCBCR10_422_2P_XR";
        YCBCR104442PXR = 593, "MTLPixelFormatYCBCR10_444_2P_XR";
        YCBCR104202PPACKEDXR = 594, "MTLPixelFormatYCBCR10_420_2P_PACKED_XR";
        YCBCR104222PPACKEDXR = 595, "MTLPixelFormatYCBCR10_422_2P_PACKED_XR";
        YCBCR104442PPACKEDXR = 596, "MTLPixelFormatYCBCR10_444_2P_PACKED_XR";
        YCBCR124202PXR = 597, "MTLPixelFormatYCBCR12_420_2P_XR";
        YCBCR124222PXR = 598, "MTLPixelFormatYCBCR12_422_2P_XR";
        YCBCR124442PXR = 599, "MTLPixelFormatYCBCR12_444_2P_XR";
        YCBCR124202PPACKEDXR = 600, "MTLPixelFormatYCBCR12_420_2P_PACKED_XR";
        YCBCR124222PPACKEDXR = 601, "MTLPixelFormatYCBCR12_422_2P_PACKED_XR";
        YCBCR124442PPACKEDXR = 602, "MTLPixelFormatYCBCR12_444_2P_PACKED_XR";
        YCBCR104202PHLG = 608, "MTLPixelFormatYCBCR10_420_2P_HLG";
        YCBCR104222PHLG = 609, "MTLPixelFormatYCBCR10_422_2P_HLG";
        YCBCR104442PHLG = 610, "MTLPixelFormatYCBCR10_444_2P_HLG";
        YCBCR104202PPACKEDHLG = 611, "MTLPixelFormatYCBCR10_420_2P_PACKED_HLG";
        YCBCR104222PPACKEDHLG = 612, "MTLPixelFormatYCBCR10_422_2P_PACKED_HLG";
        YCBCR104442PPACKEDHLG = 613, "MTLPixelFormatYCBCR10_444_2P_PACKED_HLG";
        YCBCR124202PHLG = 615, "MTLPixelFormatYCBCR12_420_2P_HLG";
        YCBCR124222PHLG = 616, "MTLPixelFormatYCBCR12_422_2P_HLG";
        YCBCR124442PHLG = 617, "MTLPixelFormatYCBCR12_444_2P_HLG";
        YCBCR124202PPACKEDHLG = 618, "MTLPixelFormatYCBCR12_420_2P_PACKED_HLG";
        YCBCR124222PPACKEDHLG = 619, "MTLPixelFormatYCBCR12_422_2P_PACKED_HLG";
        YCBCR124442PPACKEDHLG = 620, "MTLPixelFormatYCBCR12_444_2P_PACKED_HLG";
        RGBA16FloatXR = 556, "MTLPixelFormatRGBA16Float_XR";
        AnyDepth = 651, "MTLPixelFormatAnyDepth";
        AnyStencil = 652, "MTLPixelFormatAnyStencil";
        ASTC4x2SRGB = 185, "MTLPixelFormatASTC_4x2_sRGB";
        ASTC8x4SRGB = 191, "MTLPixelFormatASTC_8x4_sRGB";
        ASTC4x2LDR = 203, "MTLPixelFormatASTC_4x2_LDR";
        ASTC8x4LDR = 209, "MTLPixelFormatASTC_8x4_LDR";
        ASTC4x2HDR = 221, "MTLPixelFormatASTC_4x2_HDR";
        ASTC8x4HDR = 227, "MTLPixelFormatASTC_8x4_HDR";
        R16Bfloat = 26, "MTLPixelFormatR16Bfloat";
        BGRA4Unorm = 44, "MTLPixelFormatBGRA4Unorm";
        RG16Bfloat = 66, "MTLPixelFormatRG16Bfloat";
        ARGB8UnormSRGB = 85, "MTLPixelFormatARGB8Unorm_sRGB";
        RGB16Bfloat = 100, "MTLPixelFormatRGB16Bfloat";
        RGBA16Bfloat = 116, "MTLPixelFormatRGBA16Bfloat";
        RGBX8Unorm = 300, "MTLPixelFormatRGBX8Unorm";
        RGBX8UnormSRGB = 301, "MTLPixelFormatRGBX8Unorm_sRGB";
        RGBX8Snorm = 302, "MTLPixelFormatRGBX8Snorm";
        RGBX8Uint = 303, "MTLPixelFormatRGBX8Uint";
        RGBX8Sint = 304, "MTLPixelFormatRGBX8Sint";
        RGBX16Uint = 313, "MTLPixelFormatRGBX16Uint";
        RGBX16Sint = 314, "MTLPixelFormatRGBX16Sint";
        RGBX16Float = 315, "MTLPixelFormatRGBX16Float";
        RGBX32Uint = 323, "MTLPixelFormatRGBX32Uint";
        RGBX32Sint = 324, "MTLPixelFormatRGBX32Sint";
        RGBX32Float = 325, "MTLPixelFormatRGBX32Float";
        YCBCR84221P = 501, "MTLPixelFormatYCBCR8_422_1P";
        YCBCR104441P = 504, "MTLPixelFormatYCBCR10_444_1P";
        YCBCR84221PSRGB = 521, "MTLPixelFormatYCBCR8_422_1P_sRGB";
        YCBCR104441PSRGB = 524, "MTLPixelFormatYCBCR10_444_1P_sRGB";
        YCBCRA84441P = 560, "MTLPixelFormatYCBCRA8_444_1P";
        YCBCR84221PREV = 562, "MTLPixelFormatYCBCR8_422_1P_REV";
        YCBCR104441PPQ = 563, "MTLPixelFormatYCBCR10_444_1P_PQ";
        R10UnormX6 = 576, "MTLPixelFormatR10Unorm_X6";
        R10UnormX6SRGB = 577, "MTLPixelFormatR10Unorm_X6_sRGB";
        RG10UnormX12 = 578, "MTLPixelFormatRG10Unorm_X12";
        RG10UnormX12SRGB = 579, "MTLPixelFormatRG10Unorm_X12_sRGB";
        RGB10A2UnormSRGB = 586, "MTLPixelFormatRGB10A2Unorm_sRGB";
        RGB10A2UnormPQ = 587, "MTLPixelFormatRGB10A2Unorm_PQ";
        R10UnormPACKED = 588, "MTLPixelFormatR10Unorm_PACKED";
        RG10UnormPACKED = 589, "MTLPixelFormatRG10Unorm_PACKED";
        YCBCR104441PXR = 590, "MTLPixelFormatYCBCR10_444_1P_XR";
        R12UnormX4 = 603, "MTLPixelFormatR12Unorm_X4";
        R12UnormX4PQ = 604, "MTLPixelFormatR12Unorm_X4_PQ";
        RG12UnormX8 = 605, "MTLPixelFormatRG12Unorm_X8";
        R10UnormX6PQ = 606, "MTLPixelFormatR10Unorm_X6_PQ";
        YCBCR104441PHLG = 607, "MTLPixelFormatYCBCR10_444_1P_HLG";
        R10UnormX6HLG = 621, "MTLPixelFormatR10Unorm_X6_HLG";
        R12UnormX4HLG = 622, "MTLPixelFormatR12Unorm_X4_HLG";
        RGB10A2UnormHLG = 623, "MTLPixelFormatRGB10A2Unorm_HLG";
        RGB10A82P = 624, "MTLPixelFormatRGB10A8_2P";
        RGB10A82PSRGB = 625, "MTLPixelFormatRGB10A8_2P_sRGB";
        RGB10A82PPQ = 626, "MTLPixelFormatRGB10A8_2P_PQ";
        RGB10A82PHLG = 627, "MTLPixelFormatRGB10A8_2P_HLG";
        BGR10A2UnormSRGB = 628, "MTLPixelFormatBGR10A2Unorm_sRGB";
        BGR10A2UnormPQ = 629, "MTLPixelFormatBGR10A2Unorm_PQ";
        BGRA10UintPACKED = 630, "MTLPixelFormatBGRA10Uint_PACKED";
        BGRA10XRPACKED = 631, "MTLPixelFormatBGRA10_XR_PACKED";
        BGRA10XRPACKEDSRGB = 632, "MTLPixelFormatBGRA10_XR_PACKED_sRGB";
        BGRA10UnormPACKED = 633, "MTLPixelFormatBGRA10Unorm_PACKED";
        BGRA10UnormPACKEDSRGB = 634, "MTLPixelFormatBGRA10Unorm_PACKED_sRGB";
        BGRA10UnormPACKEDHLG = 635, "MTLPixelFormatBGRA10Unorm_PACKED_HLG";
        BGRA10UnormPACKEDPQ = 636, "MTLPixelFormatBGRA10Unorm_PACKED_PQ";
        BGRA10Uint = 637, "MTLPixelFormatBGRA10Uint";
        BGRA10Unorm = 638, "MTLPixelFormatBGRA10Unorm";
        BGRA10UnormSRGB = 639, "MTLPixelFormatBGRA10Unorm_sRGB";
        BGRA10UnormHLG = 640, "MTLPixelFormatBGRA10Unorm_HLG";
        BGRA10UnormPQ = 641, "MTLPixelFormatBGRA10Unorm_PQ";
        R10UintX6 = 642, "MTLPixelFormatR10Uint_X6";
        R10UintPACKED = 643, "MTLPixelFormatR10Uint_PACKED";
        RG10UintX12 = 644, "MTLPixelFormatRG10Uint_X12";
        RG10UintPACKED = 645, "MTLPixelFormatRG10Uint_PACKED";
        R12UintX4 = 646, "MTLPixelFormatR12Uint_X4";
        R12UintPACKED = 647, "MTLPixelFormatR12Uint_PACKED";
        RG12UintX8 = 648, "MTLPixelFormatRG12Uint_X8";
        RG12UintPACKED = 649, "MTLPixelFormatRG12Uint_PACKED";
        RGB10A8Uint2P = 650, "MTLPixelFormatRGB10A8Uint_2P";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLPrimitiveTopologyClass`.
    #[non_exhaustive]
    PrimitiveTopologyClass = "MTLPrimitiveTopologyClass" {
        Unspecified = 0, "MTLPrimitiveTopologyClassUnspecified";
        Point = 1, "MTLPrimitiveTopologyClassPoint";
        Line = 2, "MTLPrimitiveTopologyClassLine";
        Triangle = 3, "MTLPrimitiveTopologyClassTriangle";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLPrimitiveType`.
    PrimitiveType = "MTLPrimitiveType" {
        Point = 0, "MTLPrimitiveTypePoint";
        Line = 1, "MTLPrimitiveTypeLine";
        LineStrip = 2, "MTLPrimitiveTypeLineStrip";
        Triangle = 3, "MTLPrimitiveTypeTriangle";
        TriangleStrip = 4, "MTLPrimitiveTypeTriangleStrip";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLReadWriteTextureTier`.
    #[non_exhaustive]
    ReadWriteTextureTier = "MTLReadWriteTextureTier" {
        None = 0, "MTLReadWriteTextureTierNone";
        Tier1 = 1, "MTLReadWriteTextureTier1";
        Tier2 = 2, "MTLReadWriteTextureTier2";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLSamplerAddressMode`.
    #[non_exhaustive]
    SamplerAddressMode = "MTLSamplerAddressMode" {
        ClampToEdge = 0, "MTLSamplerAddressModeClampToEdge";
        MirrorClampToEdge = 1, "MTLSamplerAddressModeMirrorClampToEdge";
        Repeat = 2, "MTLSamplerAddressModeRepeat";
        MirrorRepeat = 3, "MTLSamplerAddressModeMirrorRepeat";
        ClampToZero = 4, "MTLSamplerAddressModeClampToZero";
        ClampToBorderColor = 5, "MTLSamplerAddressModeClampToBorderColor";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLSamplerBorderColor`.
    #[non_exhaustive]
    SamplerBorderColor = "MTLSamplerBorderColor" {
        TransparentBlack = 0, "MTLSamplerBorderColorTransparentBlack";
        OpaqueBlack = 1, "MTLSamplerBorderColorOpaqueBlack";
        OpaqueWhite = 2, "MTLSamplerBorderColorOpaqueWhite";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLSamplerMinMagFilter`.
    #[non_exhaustive]
    SamplerMinMagFilter = "MTLSamplerMinMagFilter" {
        Nearest = 0, "MTLSamplerMinMagFilterNearest";
        Linear = 1, "MTLSamplerMinMagFilterLinear";
        Cubic = 2, "MTLSamplerMinMagFilterCubic";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLSamplerMipFilter`.
    #[non_exhaustive]
    SamplerMipFilter = "MTLSamplerMipFilter" {
        NotMipmapped = 0, "MTLSamplerMipFilterNotMipmapped";
        Nearest = 1, "MTLSamplerMipFilterNearest";
        Linear = 2, "MTLSamplerMipFilterLinear";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLSamplerReductionMode`.
    #[non_exhaustive]
    SamplerReductionMode = "MTLSamplerReductionMode" {
        WeightedAverage = 0, "MTLSamplerReductionModeWeightedAverage";
        Minimum = 1, "MTLSamplerReductionModeMinimum";
        Maximum = 2, "MTLSamplerReductionModeMaximum";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLStencilOperation`.
    #[non_exhaustive]
    StencilOperation = "MTLStencilOperation" {
        Keep = 0, "MTLStencilOperationKeep";
        Zero = 1, "MTLStencilOperationZero";
        Replace = 2, "MTLStencilOperationReplace";
        IncrementClamp = 3, "MTLStencilOperationIncrementClamp";
        DecrementClamp = 4, "MTLStencilOperationDecrementClamp";
        Invert = 5, "MTLStencilOperationInvert";
        IncrementWrap = 6, "MTLStencilOperationIncrementWrap";
        DecrementWrap = 7, "MTLStencilOperationDecrementWrap";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLStepFunction`.
    #[non_exhaustive]
    StepFunction = "MTLStepFunction" {
        Constant = 0, "MTLStepFunctionConstant";
        PerVertex = 1, "MTLStepFunctionPerVertex";
        PerInstance = 2, "MTLStepFunctionPerInstance";
        PerPatch = 3, "MTLStepFunctionPerPatch";
        PerPatchControlPoint = 4, "MTLStepFunctionPerPatchControlPoint";
        ThreadPositionInGridX = 5, "MTLStepFunctionThreadPositionInGridX";
        ThreadPositionInGridY = 6, "MTLStepFunctionThreadPositionInGridY";
        ThreadPositionInGridXIndexed = 7, "MTLStepFunctionThreadPositionInGridXIndexed";
        ThreadPositionInGridYIndexed = 8, "MTLStepFunctionThreadPositionInGridYIndexed";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLTessellationControlPointIndexType`.
    #[non_exhaustive]
    TessellationControlPointIndexType = "MTLTessellationControlPointIndexType" {
        None = 0, "MTLTessellationControlPointIndexTypeNone";
        UInt16 = 1, "MTLTessellationControlPointIndexTypeUInt16";
        UInt32 = 2, "MTLTessellationControlPointIndexTypeUInt32";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLTessellationFactorFormat`.
    #[non_exhaustive]
    TessellationFactorFormat = "MTLTessellationFactorFormat" {
        Half = 0, "MTLTessellationFactorFormatHalf";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLTessellationFactorStepFunction`.
    #[non_exhaustive]
    TessellationFactorStepFunction = "MTLTessellationFactorStepFunction" {
        Constant = 0, "MTLTessellationFactorStepFunctionConstant";
        PerPatch = 1, "MTLTessellationFactorStepFunctionPerPatch";
        PerInstance = 2, "MTLTessellationFactorStepFunctionPerInstance";
        PerPatchAndPerInstance = 3, "MTLTessellationFactorStepFunctionPerPatchAndPerInstance";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLTessellationPartitionMode`.
    #[non_exhaustive]
    TessellationPartitionMode = "MTLTessellationPartitionMode" {
        Pow2 = 0, "MTLTessellationPartitionModePow2";
        Integer = 1, "MTLTessellationPartitionModeInteger";
        FractionalOdd = 2, "MTLTessellationPartitionModeFractionalOdd";
        FractionalEven = 3, "MTLTessellationPartitionModeFractionalEven";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLTextureSwizzle`.
    #[non_exhaustive]
    TextureSwizzle = "MTLTextureSwizzle" {
        Zero = 0, "MTLTextureSwizzleZero";
        One = 1, "MTLTextureSwizzleOne";
        Red = 2, "MTLTextureSwizzleRed";
        Green = 3, "MTLTextureSwizzleGreen";
        Blue = 4, "MTLTextureSwizzleBlue";
        Alpha = 5, "MTLTextureSwizzleAlpha";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLTextureType`.
    TextureType = "MTLTextureType" {
        Type1D = 0, "MTLTextureType1D";
        Type1DArray = 1, "MTLTextureType1DArray";
        Type2D = 2, "MTLTextureType2D";
        Type2DArray = 3, "MTLTextureType2DArray";
        Type2DMultisample = 4, "MTLTextureType2DMultisample";
        Cube = 5, "MTLTextureTypeCube";
        CubeArray = 6, "MTLTextureTypeCubeArray";
        Type3D = 7, "MTLTextureType3D";
        Type2DMultisampleArray = 8, "MTLTextureType2DMultisampleArray";
        TextureBuffer = 9, "MTLTextureTypeTextureBuffer";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLTriangleFillMode`.
    #[non_exhaustive]
    TriangleFillMode = "MTLTriangleFillMode" {
        Fill = 0, "MTLTriangleFillModeFill";
        Lines = 1, "MTLTriangleFillModeLines";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLVertexFormat`.
    #[non_exhaustive]
    VertexFormat = "MTLVertexFormat" {
        Invalid = 0, "MTLVertexFormatInvalid";
        UChar2 = 1, "MTLVertexFormatUChar2";
        UChar3 = 2, "MTLVertexFormatUChar3";
        UChar4 = 3, "MTLVertexFormatUChar4";
        Char2 = 4, "MTLVertexFormatChar2";
        Char3 = 5, "MTLVertexFormatChar3";
        Char4 = 6, "MTLVertexFormatChar4";
        UChar2Normalized = 7, "MTLVertexFormatUChar2Normalized";
        UChar3Normalized = 8, "MTLVertexFormatUChar3Normalized";
        UChar4Normalized = 9, "MTLVertexFormatUChar4Normalized";
        Char2Normalized = 10, "MTLVertexFormatChar2Normalized";
        Char3Normalized = 11, "MTLVertexFormatChar3Normalized";
        Char4Normalized = 12, "MTLVertexFormatChar4Normalized";
        UShort2 = 13, "MTLVertexFormatUShort2";
        UShort3 = 14, "MTLVertexFormatUShort3";
        UShort4 = 15, "MTLVertexFormatUShort4";
        Short2 = 16, "MTLVertexFormatShort2";
        Short3 = 17, "MTLVertexFormatShort3";
        Short4 = 18, "MTLVertexFormatShort4";
        UShort2Normalized = 19, "MTLVertexFormatUShort2Normalized";
        UShort3Normalized = 20, "MTLVertexFormatUShort3Normalized";
        UShort4Normalized = 21, "MTLVertexFormatUShort4Normalized";
        Short2Normalized = 22, "MTLVertexFormatShort2Normalized";
        Short3Normalized = 23, "MTLVertexFormatShort3Normalized";
        Short4Normalized = 24, "MTLVertexFormatShort4Normalized";
        Half2 = 25, "MTLVertexFormatHalf2";
        Half3 = 26, "MTLVertexFormatHalf3";
        Half4 = 27, "MTLVertexFormatHalf4";
        Float = 28, "MTLVertexFormatFloat";
        Float2 = 29, "MTLVertexFormatFloat2";
        Float3 = 30, "MTLVertexFormatFloat3";
        Float4 = 31, "MTLVertexFormatFloat4";
        Int = 32, "MTLVertexFormatInt";
        Int2 = 33, "MTLVertexFormatInt2";
        Int3 = 34, "MTLVertexFormatInt3";
        Int4 = 35, "MTLVertexFormatInt4";
        UInt = 36, "MTLVertexFormatUInt";
        UInt2 = 37, "MTLVertexFormatUInt2";
        UInt3 = 38, "MTLVertexFormatUInt3";
        UInt4 = 39, "MTLVertexFormatUInt4";
        Int1010102Normalized = 40, "MTLVertexFormatInt1010102Normalized";
        UInt1010102Normalized = 41, "MTLVertexFormatUInt1010102Normalized";
        UChar4NormalizedBGRA = 42, "MTLVertexFormatUChar4Normalized_BGRA";
        UChar = 45, "MTLVertexFormatUChar";
        Char = 46, "MTLVertexFormatChar";
        UCharNormalized = 47, "MTLVertexFormatUCharNormalized";
        CharNormalized = 48, "MTLVertexFormatCharNormalized";
        UShort = 49, "MTLVertexFormatUShort";
        Short = 50, "MTLVertexFormatShort";
        UShortNormalized = 51, "MTLVertexFormatUShortNormalized";
        ShortNormalized = 52, "MTLVertexFormatShortNormalized";
        Half = 53, "MTLVertexFormatHalf";
        FloatRG11B10 = 54, "MTLVertexFormatFloatRG11B10";
        FloatRGB9E5 = 55, "MTLVertexFormatFloatRGB9E5";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLVertexStepFunction`.
    #[non_exhaustive]
    VertexStepFunction = "MTLVertexStepFunction" {
        Constant = 0, "MTLVertexStepFunctionConstant";
        PerVertex = 1, "MTLVertexStepFunctionPerVertex";
        PerInstance = 2, "MTLVertexStepFunctionPerInstance";
        PerPatch = 3, "MTLVertexStepFunctionPerPatch";
        PerPatchControlPoint = 4, "MTLVertexStepFunctionPerPatchControlPoint";
    }
    aliases {
    }
}

metal_enum! {
    /// `MTLWinding`.
    Winding = "MTLWinding" {
        Clockwise = 0, "MTLWindingClockwise";
        CounterClockwise = 1, "MTLWindingCounterClockwise";
    }
    aliases {
    }
}

metal_flags! {
    /// `MTLBarrierScope`.
    BarrierScope = "MTLBarrierScope" {
        BUFFERS = 0x1, "MTLBarrierScopeBuffers";
        TEXTURES = 0x2, "MTLBarrierScopeTextures";
        RENDER_TARGETS = 0x4, "MTLBarrierScopeRenderTargets";
    }
    aliases {
    }
    dropped {
    }
}

metal_flags! {
    /// `MTLColorWriteMask`.
    ColorWriteMask = "MTLColorWriteMask" {
        NONE = 0x0, "MTLColorWriteMaskNone";
        RED = 0x8, "MTLColorWriteMaskRed";
        GREEN = 0x4, "MTLColorWriteMaskGreen";
        BLUE = 0x2, "MTLColorWriteMaskBlue";
        ALPHA = 0x1, "MTLColorWriteMaskAlpha";
        ALL = 0xf, "MTLColorWriteMaskAll";
        UNSPECIALIZED = 0x10, "MTLColorWriteMaskUnspecialized";
    }
    aliases {
    }
    dropped {
    }
}

metal_flags! {
    /// `MTLFunctionOptions`.
    FunctionOptions = "MTLFunctionOptions" {
        NONE = 0x0, "MTLFunctionOptionNone";
        COMPILE_TO_BINARY = 0x1, "MTLFunctionOptionCompileToBinary";
        STORE_FUNCTION_IN_METAL_PIPELINES_SCRIPT = 0x2, "MTLFunctionOptionStoreFunctionInMetalPipelinesScript";
        FAIL_ON_BINARY_ARCHIVE_MISS = 0x4, "MTLFunctionOptionFailOnBinaryArchiveMiss";
        PIPELINE_INDEPENDENT = 0x8, "MTLFunctionOptionPipelineIndependent";
    }
    aliases {
        "MTLFunctionOptionStoreFunctionInMetalScript" => STORE_FUNCTION_IN_METAL_PIPELINES_SCRIPT;
    }
    dropped {
    }
}

metal_flags! {
    /// `MTLIndirectCommandType`.
    IndirectCommandType = "MTLIndirectCommandType" {
        DRAW = 0x1, "MTLIndirectCommandTypeDraw";
        DRAW_INDEXED = 0x2, "MTLIndirectCommandTypeDrawIndexed";
        DRAW_PATCHES = 0x4, "MTLIndirectCommandTypeDrawPatches";
        DRAW_INDEXED_PATCHES = 0x8, "MTLIndirectCommandTypeDrawIndexedPatches";
        CONCURRENT_DISPATCH = 0x20, "MTLIndirectCommandTypeConcurrentDispatch";
        CONCURRENT_DISPATCH_THREADS = 0x40, "MTLIndirectCommandTypeConcurrentDispatchThreads";
        DRAW_MESH_THREADGROUPS = 0x80, "MTLIndirectCommandTypeDrawMeshThreadgroups";
        DRAW_MESH_THREADS = 0x100, "MTLIndirectCommandTypeDrawMeshThreads";
    }
    aliases {
    }
    dropped {
    }
}

metal_flags! {
    /// `MTLRenderStages`.
    RenderStages = "MTLRenderStages" {
        VERTEX = 0x1, "MTLRenderStageVertex";
        FRAGMENT = 0x2, "MTLRenderStageFragment";
        TILE = 0x4, "MTLRenderStageTile";
        OBJECT = 0x8, "MTLRenderStageObject";
        MESH = 0x10, "MTLRenderStageMesh";
    }
    aliases {
    }
    dropped {
    }
}

metal_flags! {
    /// `MTLResourceUsage`.
    ResourceUsage = "MTLResourceUsage" {
        READ = 0x1, "MTLResourceUsageRead";
        WRITE = 0x2, "MTLResourceUsageWrite";
        SAMPLE = 0x4, "MTLResourceUsageSample";
    }
    aliases {
    }
    dropped {
    }
}

metal_flags! {
    /// `MTLTextureUsage`.
    TextureUsage = "MTLTextureUsage" {
        UNKNOWN = 0x0, "MTLTextureUsageUnknown";
        SHADER_READ = 0x1, "MTLTextureUsageShaderRead";
        SHADER_WRITE = 0x2, "MTLTextureUsageShaderWrite";
        RENDER_TARGET = 0x4, "MTLTextureUsageRenderTarget";
        PIXEL_FORMAT_VIEW = 0x10, "MTLTextureUsagePixelFormatView";
        SHADER_ATOMIC = 0x20, "MTLTextureUsageShaderAtomic";
        NO_CONCURRENT_ACCESS = 0x4000, "MTLTextureUsageNoConcurrentAccess";
        VIDEO = 0x8000, "MTLTextureUsageVideo";
        BLOCK_WRITES_ONLY = 0x10000, "MTLTextureUsageBlockWritesOnly";
    }
    aliases {
    }
    dropped {
        0x8;
    }
}
