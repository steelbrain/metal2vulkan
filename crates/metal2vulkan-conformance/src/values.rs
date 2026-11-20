//! The Metal enumerations and option words the case vocabulary states, by Metal's own names
//! and raw values: what a case asks of a load and store action, a resolve filter, a visibility
//! result, a heap, a resource's purgeable state and a sparse page, spelt once for both sides.

/// `MTLPurgeableState`, with Metal's raw values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PurgeableState {
    /// `MTLPurgeableStateKeepCurrent`: a query; the state does not change.
    KeepCurrent,
    /// `MTLPurgeableStateNonVolatile`: the contents may not be discarded.
    NonVolatile,
    /// `MTLPurgeableStateVolatile`: the contents may be discarded.
    Volatile,
    /// `MTLPurgeableStateEmpty`: the contents are discarded.
    Empty,
}

impl PurgeableState {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            PurgeableState::KeepCurrent => 1,
            PurgeableState::NonVolatile => 2,
            PurgeableState::Volatile => 3,
            PurgeableState::Empty => 4,
        }
    }
}

/// `MTLHeapType`, with Metal's raw values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HeapType {
    /// `MTLHeapTypeAutomatic`: the heap places each resource.
    #[default]
    Automatic,
    /// `MTLHeapTypePlacement`: the program places each resource at an offset.
    Placement,
    /// `MTLHeapTypeSparse`: sparse textures' tiles.
    Sparse,
}

impl HeapType {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            HeapType::Automatic => 0,
            HeapType::Placement => 1,
            HeapType::Sparse => 2,
        }
    }
}

/// `MTLLoadAction`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LoadAction {
    /// `MTLLoadActionDontCare`: the pass starts from texels it does not read.
    #[default]
    DontCare,
    /// `MTLLoadActionLoad`: the pass starts from the attachment's texels.
    Load,
    /// `MTLLoadActionClear`: the pass starts from the attachment's clear value.
    Clear,
}

impl LoadAction {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            LoadAction::DontCare => 0,
            LoadAction::Load => 1,
            LoadAction::Clear => 2,
        }
    }

    /// The value whose raw value is `raw`; another raw value is not one of `MTLLoadAction`'s.
    pub fn from_raw(raw: u64) -> Result<LoadAction, String> {
        [LoadAction::DontCare, LoadAction::Load, LoadAction::Clear]
            .into_iter()
            .find(|value| value.raw() == raw)
            .ok_or_else(|| format!("MTLLoadAction has no raw value {raw}"))
    }
}

/// `MTLStoreAction`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StoreAction {
    /// `MTLStoreActionDontCare`: the pass's texels are not kept. Apple's GPU leaves the
    /// attachment's earlier texels.
    DontCare,
    /// `MTLStoreActionStore`: the pass's texels are kept.
    #[default]
    Store,
    /// `MTLStoreActionMultisampleResolve`: the samples are resolved into the resolve texture
    /// and not kept.
    MultisampleResolve,
    /// `MTLStoreActionStoreAndMultisampleResolve`: kept and resolved.
    StoreAndMultisampleResolve,
    /// `MTLStoreActionUnknown`: decided later, by the encoder's `set_*_store_action` calls,
    /// before `end_encoding`.
    Unknown,
    /// `MTLStoreActionCustomSampleDepthStore`: depth kept for the custom sample positions.
    CustomSampleDepthStore,
}

impl StoreAction {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            StoreAction::DontCare => 0,
            StoreAction::Store => 1,
            StoreAction::MultisampleResolve => 2,
            StoreAction::StoreAndMultisampleResolve => 3,
            StoreAction::Unknown => 4,
            StoreAction::CustomSampleDepthStore => 5,
        }
    }

    /// The value whose raw value is `raw`; another raw value is not one of `MTLStoreAction`'s.
    pub fn from_raw(raw: u64) -> Result<StoreAction, String> {
        [
            StoreAction::DontCare,
            StoreAction::Store,
            StoreAction::MultisampleResolve,
            StoreAction::StoreAndMultisampleResolve,
            StoreAction::Unknown,
            StoreAction::CustomSampleDepthStore,
        ]
        .into_iter()
        .find(|value| value.raw() == raw)
        .ok_or_else(|| format!("MTLStoreAction has no raw value {raw}"))
    }

    /// Whether the action resolves the samples into a resolve texture.
    pub fn resolves(self) -> bool {
        matches!(
            self,
            StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
        )
    }
}

/// `MTLStoreActionOptions`: bit 0 is `MTLStoreActionOptionCustomSamplePositions`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StoreActionOptions(u64);

impl StoreActionOptions {
    /// `MTLStoreActionOptionNone`.
    pub const NONE: StoreActionOptions = StoreActionOptions(0);
    /// `MTLStoreActionOptionCustomSamplePositions`.
    pub const CUSTOM_SAMPLE_POSITIONS: StoreActionOptions = StoreActionOptions(1);

    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Whether every option of `other` is set.
    pub fn contains(self, other: StoreActionOptions) -> bool {
        self.0 & other.0 == other.0
    }
}

/// `MTLSamplePosition`: where a sample lies in its pixel, from its top-left corner, each
/// coordinate in `[0, 1)` (Metal asserts on another, measured).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplePosition {
    /// `x`: from the pixel's left edge.
    pub x: f32,
    /// `y`: from the pixel's top edge.
    pub y: f32,
}

impl SamplePosition {
    /// `MTLSamplePositionMake`.
    pub fn new(x: f32, y: f32) -> SamplePosition {
        SamplePosition { x, y }
    }
}

/// `MTLMultisampleDepthResolveFilter`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MultisampleDepthResolveFilter {
    /// `MTLMultisampleDepthResolveFilterSample0`.
    #[default]
    Sample0,
    /// `MTLMultisampleDepthResolveFilterMin`.
    Min,
    /// `MTLMultisampleDepthResolveFilterMax`.
    Max,
}

impl MultisampleDepthResolveFilter {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            MultisampleDepthResolveFilter::Sample0 => 0,
            MultisampleDepthResolveFilter::Min => 1,
            MultisampleDepthResolveFilter::Max => 2,
        }
    }

    /// The value whose raw value is `raw`; another raw value is not one of `MTLMultisampleDepthResolveFilter`'s.
    pub fn from_raw(raw: u64) -> Result<MultisampleDepthResolveFilter, String> {
        [
            MultisampleDepthResolveFilter::Sample0,
            MultisampleDepthResolveFilter::Min,
            MultisampleDepthResolveFilter::Max,
        ]
        .into_iter()
        .find(|value| value.raw() == raw)
        .ok_or_else(|| format!("MTLMultisampleDepthResolveFilter has no raw value {raw}"))
    }
}

/// `MTLMultisampleStencilResolveFilter`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MultisampleStencilResolveFilter {
    /// `MTLMultisampleStencilResolveFilterSample0`.
    #[default]
    Sample0,
    /// `MTLMultisampleStencilResolveFilterDepthResolvedSample`.
    DepthResolvedSample,
}

impl MultisampleStencilResolveFilter {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            MultisampleStencilResolveFilter::Sample0 => 0,
            MultisampleStencilResolveFilter::DepthResolvedSample => 1,
        }
    }

    /// The value whose raw value is `raw`; another raw value is not one of `MTLMultisampleStencilResolveFilter`'s.
    pub fn from_raw(raw: u64) -> Result<MultisampleStencilResolveFilter, String> {
        [
            MultisampleStencilResolveFilter::Sample0,
            MultisampleStencilResolveFilter::DepthResolvedSample,
        ]
        .into_iter()
        .find(|value| value.raw() == raw)
        .ok_or_else(|| format!("MTLMultisampleStencilResolveFilter has no raw value {raw}"))
    }
}

/// `MTLCPUCacheMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CpuCacheMode {
    /// `MTLCPUCacheModeDefaultCache`.
    #[default]
    DefaultCache,
    /// `MTLCPUCacheModeWriteCombined`: write-combined CPU access.
    WriteCombined,
}

impl CpuCacheMode {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            CpuCacheMode::DefaultCache => 0,
            CpuCacheMode::WriteCombined => 1,
        }
    }
}

/// `MTLHazardTrackingMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HazardTrackingMode {
    /// `MTLHazardTrackingModeDefault`: tracked for a resource made outside a heap.
    #[default]
    Default,
    /// `MTLHazardTrackingModeUntracked`: the program orders the resource's accesses.
    Untracked,
    /// `MTLHazardTrackingModeTracked`.
    Tracked,
}

impl HazardTrackingMode {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            HazardTrackingMode::Default => 0,
            HazardTrackingMode::Untracked => 1,
            HazardTrackingMode::Tracked => 2,
        }
    }
}

/// `MTLVisibilityResultMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VisibilityResultMode {
    /// `MTLVisibilityResultModeDisabled`: the draws write no result.
    #[default]
    Disabled,
    /// `MTLVisibilityResultModeBoolean`: nonzero when any sample passed, else 0. The nonzero value
    /// is the device's occlusion query's, where an Apple M3 Max writes 1
    /// (`Degradation::DeviceBooleanVisibility`).
    Boolean,
    /// `MTLVisibilityResultModeCounting`: the samples that passed.
    Counting,
}

impl VisibilityResultMode {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            VisibilityResultMode::Disabled => 0,
            VisibilityResultMode::Boolean => 1,
            VisibilityResultMode::Counting => 2,
        }
    }
}

/// `MTLVisibilityResultType`: whether a pass's visibility results replace the words at their
/// offsets or are taken into them: the pass descriptor's `visibilityResultType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VisibilityResultType {
    /// `MTLVisibilityResultTypeReset`, Metal's default: each offset the encoder set a mode at
    /// takes the pass's result in place of its word.
    #[default]
    Reset,
    /// `MTLVisibilityResultTypeAccumulate`: each offset the encoder set a mode at keeps its word
    /// and takes the pass's result into it, a `Counting` result added to the word's low 32 bits
    /// (wrapping there, its high 32 bits kept) and a `Boolean` one ORed, so an offset set with no
    /// draw after it keeps its word.
    ///
    /// As an Apple M3 Max answers it (macOS 27.0.1). Metal on the AMD Radeon Pro 5300M and the
    /// Intel UHD Graphics 630 (macOS 26.7.1) answers it as `Reset`.
    Accumulate,
}

impl VisibilityResultType {
    /// Metal's raw value.
    pub fn raw(self) -> i64 {
        match self {
            VisibilityResultType::Reset => 0,
            VisibilityResultType::Accumulate => 1,
        }
    }
}

/// `MTLPipelineOption`: Metal's raw value, every bit kept. Combine with `|`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PipelineOption(u64);

impl PipelineOption {
    /// `MTLPipelineOptionNone`.
    pub const NONE: PipelineOption = PipelineOption(0);
    /// `MTLPipelineOptionBindingInfo` (and the deprecated `MTLPipelineOptionArgumentInfo`):
    /// the `_with_reflection` creation forms return the pipeline's reflection.
    pub const BINDING_INFO: PipelineOption = PipelineOption(1 << 0);
    /// `MTLPipelineOptionBufferTypeInfo`: the same as `BINDING_INFO` (Metal's reflection carries
    /// the buffers' types under either).
    pub const BUFFER_TYPE_INFO: PipelineOption = PipelineOption(1 << 1);
    /// `MTLPipelineOptionFailOnBinaryArchiveMiss`: a creation that lists a binary archive fails
    /// unless an archive holds its pipeline.
    pub const FAIL_ON_BINARY_ARCHIVE_MISS: PipelineOption = PipelineOption(1 << 2);

    /// Metal's raw value.
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Whether every bit of `other` is set.
    pub const fn contains(self, other: PipelineOption) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for PipelineOption {
    type Output = PipelineOption;

    fn bitor(self, other: PipelineOption) -> PipelineOption {
        PipelineOption(self.0 | other.0)
    }
}

/// `MTLSparsePageSize`, with Metal's raw values: the bytes of one tile of a sparse heap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SparsePageSize {
    /// `MTLSparsePageSize16`: 16 KiB.
    #[default]
    Size16,
    /// `MTLSparsePageSize64`: 64 KiB.
    Size64,
    /// `MTLSparsePageSize256`: 256 KiB.
    Size256,
}

impl SparsePageSize {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            SparsePageSize::Size16 => 101,
            SparsePageSize::Size64 => 102,
            SparsePageSize::Size256 => 103,
        }
    }

    /// The page's bytes.
    pub fn bytes(self) -> u64 {
        match self {
            SparsePageSize::Size16 => 16 << 10,
            SparsePageSize::Size64 => 64 << 10,
            SparsePageSize::Size256 => 256 << 10,
        }
    }
}
