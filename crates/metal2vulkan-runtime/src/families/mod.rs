//! Metal's GPU families (`supportsFamily:`), each judged row by row against Apple's Metal feature
//! set tables on the Vulkan device.
//!
//! **The tables.** Apple's "Metal feature set tables", in two editions. The PDFs are not in the
//! repository:
//! - The edition of 2026-05-21, from <https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf>
//!   (SHA-256 `9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8`). It has
//!   columns Apple2 to Apple10. Its feature rows give the first Apple family and the programming
//!   models ("Metal 3 & 4", "Metal 3", "Metal 4") of each feature. It also has the limits and
//!   the pixel-format capabilities.
//! - The edition of 2023-10-31, archived at
//!   <https://web.archive.org/web/20231213102252id_/https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf>
//!   (SHA-256 `618a2f5d4ea92fdf58f8d7cc8daf525e0aad0000f96ccc1302424600e620c243`). Its feature
//!   table has columns Common1 to Common3, Metal3, Apple2 to Apple9, and Mac2. Its limits and
//!   pixel formats cover Apple2 to Apple9 and Mac2.
//!
//! **One row per entry** of the tables:
//! - a feature ([`FeatureRow`]). The two editions' rows are matched by name, renamed ones
//!   included, and 2023's "Argument buffers tier" is read onto 2026's two tier rows by its tier
//!   cells;
//! - a limit ([`LimitRow`]);
//! - a capability of a pixel format's cell ([`FormatRow`] with a [`FormatCapability`]). A
//!   compressed family such as "ASTC pixel formats" is one entry over all its formats.
//!
//! **The columns a family takes** (orchestrator, 2026-10-05):
//! - Apple2 to Apple10 take the 2026 edition's Apple columns: its rows of the Metal 3 programming
//!   model, from their first Apple family. A row only Metal 4's programming model offers is the
//!   `MTL4` API, which the Metal4 family's one row stands for.
//! - Common1 to Common3 take the 2023 edition's feature columns. That edition has no Common
//!   column of limits or formats.
//! - Mac2 takes the 2023 edition's Mac2 columns.
//! - Metal3 takes the following:
//!   - the 2023 edition's Metal3 feature column, for the rows that edition has;
//!   - for the 2026 edition's other rows, those of the Metal 3 programming model whose first
//!     family is at most Apple7;
//!   - the 2026 edition's Apple7 limits and formats. Its footnotes say every Metal 3 and Metal 4
//!     device has at least those.
//! - A limit of a feature (an imageblock's memory, a tessellation factor) is listed only where
//!   the family lists the feature's row: Metal3's Mac devices have no imageblocks.
//!
//! **Families without a column.** Neither edition has columns for these:
//! - Apple1 answers as Apple2, and Mac1 as Mac2. Each family has every feature of the one before
//!   it, so Apple2's rows are a superset of Apple1's (Apple1, the A7, left the tables when its
//!   devices left the OS). The macOS 27 SDK deprecates `MTLGPUFamilyMac1` since macOS 13. Metal
//!   answers Apple1 and Mac1 wherever it answers Apple2 and Mac2: true on an M3 Max (macOS 26.7)
//!   and an M4 (macOS 27.0.1).
//! - Metal4 has one row, [`FamilyRow::Metal4Api`]: the `MTL4` API, which the runtime does not
//!   build. Its features have no column of their own, so none is listed.
//! - MacCatalyst1 and MacCatalyst2 have one row, [`FamilyRow::MacCatalystProcess`]: a Mac
//!   Catalyst process, which the runtime never runs in.
//! - Apple11 (the macOS 27 SDK's 1011) has one row, [`FamilyRow::NoPublishedColumn`]. No
//!   edition describes it, so nothing is claimed.
//!
//! **Cells.** A cell lists its row for its family when it reads yes, a tier at least the row's,
//! a value, or capabilities. These list nothing, because the family does not promise the row:
//! "Varies", "Not available", "No", and a footnote saying only some devices of the family have
//! it. In a format cell, "All" is every capability but Atomic. It also excludes Sparse in the
//! columns of families without sparse textures: Apple2 to Apple5, which the 2026 legend says,
//! and Mac2, by the 2023 feature table.
//!
//! **States** ([`RowState`]):
//! - **Exact**: the feature is offered through Metal's API on the device, and nothing Metal's
//!   validation accepts is refused there. The family answer is about capability. Where results
//!   differ from Apple's GPU in precision, the degrade catalogue says so (docs/API.md §9). Those
//!   degrades are announced where they occur and stay open as near-equivalents.
//! - **Refused** ([`Refusal`]): no exact path exists on the device. Either the device lacks
//!   what the compile layer or the runtime needs, or no device has it yet (`Unsupported`, the
//!   compile layer's own refusal), or one of Metal's device queries answers no there.
//! - **Gap** ([`RuntimeGap`]): the runtime piece is not built yet. The gap names the step that
//!   builds it, or says none is scheduled.
//! - **Not a device feature** ([`Framework`]): MetalKit, Metal Performance Shaders and the MetalFX
//!   rows. These are frameworks layered on Metal. `supportsFamily:` describes the device, and
//!   frameworks belong to the OS or the consumer, so these rows are shown and never block. A row
//!   under a Metal.framework header is the runtime's own. Fast resource loading
//!   (`MTLIOCommandQueue`) is such a row, a gap until it is built.
//!
//! [`Device::supports_family`] answers true only when every row the family lists is exact (or not
//! a device feature). [`Device::family_blockers`] names the rows that keep it false: the burn-down
//! the remaining runtime work is scheduled against.

mod features;
pub(super) mod formats;
mod limits;

pub use features::FeatureRow;
pub(super) use formats::FormatAnswers;
pub use formats::{FormatCapabilities, FormatCapability, FormatRow};
pub use limits::LimitRow;

use super::capabilities::{DeviceProperties, Query};
use super::error::RuntimeGap;
use super::Device;
use metal2vulkan::device::{DeviceProfile, MetalDeviceProperties};
use metal2vulkan::error::Unsupported;
use metal2vulkan::metal::PixelFormat;

/// Metal's GPU families (`MTLGPUFamily`), as of the macOS 27 SDK.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GpuFamily {
    /// `MTLGPUFamilyApple1`, answered as Apple2.
    Apple1,
    /// `MTLGPUFamilyApple2`.
    Apple2,
    /// `MTLGPUFamilyApple3`.
    Apple3,
    /// `MTLGPUFamilyApple4`.
    Apple4,
    /// `MTLGPUFamilyApple5`.
    Apple5,
    /// `MTLGPUFamilyApple6`.
    Apple6,
    /// `MTLGPUFamilyApple7`.
    Apple7,
    /// `MTLGPUFamilyApple8`.
    Apple8,
    /// `MTLGPUFamilyApple9`.
    Apple9,
    /// `MTLGPUFamilyApple10`.
    Apple10,
    /// `MTLGPUFamilyApple11`: no edition of Apple's tables has its column, so it is never claimed.
    Apple11,
    /// `MTLGPUFamilyMac1`, answered as Mac2 (deprecated since macOS 13).
    Mac1,
    /// `MTLGPUFamilyMac2`.
    Mac2,
    /// `MTLGPUFamilyCommon1`.
    Common1,
    /// `MTLGPUFamilyCommon2`.
    Common2,
    /// `MTLGPUFamilyCommon3`.
    Common3,
    /// `MTLGPUFamilyMacCatalyst1`: the runtime never runs in a Mac Catalyst process.
    MacCatalyst1,
    /// `MTLGPUFamilyMacCatalyst2`: the runtime never runs in a Mac Catalyst process.
    MacCatalyst2,
    /// `MTLGPUFamilyMetal3`.
    Metal3,
    /// `MTLGPUFamilyMetal4`: its one row is the `MTL4` API, which the runtime does not build.
    Metal4,
}

impl GpuFamily {
    /// Every family, in declaration order.
    pub const ALL: [GpuFamily; 20] = [
        GpuFamily::Apple1,
        GpuFamily::Apple2,
        GpuFamily::Apple3,
        GpuFamily::Apple4,
        GpuFamily::Apple5,
        GpuFamily::Apple6,
        GpuFamily::Apple7,
        GpuFamily::Apple8,
        GpuFamily::Apple9,
        GpuFamily::Apple10,
        GpuFamily::Apple11,
        GpuFamily::Mac1,
        GpuFamily::Mac2,
        GpuFamily::Common1,
        GpuFamily::Common2,
        GpuFamily::Common3,
        GpuFamily::MacCatalyst1,
        GpuFamily::MacCatalyst2,
        GpuFamily::Metal3,
        GpuFamily::Metal4,
    ];

    /// Metal's value of the family (`MTLGPUFamilyApple7` is 1007).
    pub fn raw(self) -> i64 {
        use GpuFamily as F;
        match self {
            F::Apple1 => 1001,
            F::Apple2 => 1002,
            F::Apple3 => 1003,
            F::Apple4 => 1004,
            F::Apple5 => 1005,
            F::Apple6 => 1006,
            F::Apple7 => 1007,
            F::Apple8 => 1008,
            F::Apple9 => 1009,
            F::Apple10 => 1010,
            F::Apple11 => 1011,
            F::Mac1 => 2001,
            F::Mac2 => 2002,
            F::Common1 => 3001,
            F::Common2 => 3002,
            F::Common3 => 3003,
            F::MacCatalyst1 => 4001,
            F::MacCatalyst2 => 4002,
            F::Metal3 => 5001,
            F::Metal4 => 5002,
        }
    }

    /// The family of Metal's value `raw`, if it names one.
    pub fn from_raw(raw: i64) -> Option<GpuFamily> {
        GpuFamily::ALL.into_iter().find(|f| f.raw() == raw)
    }

    /// The Apple family's number (7 for Apple7), for Apple1 to Apple11.
    fn apple(self) -> Option<u8> {
        use GpuFamily as F;
        Some(match self {
            F::Apple1 => 1,
            F::Apple2 => 2,
            F::Apple3 => 3,
            F::Apple4 => 4,
            F::Apple5 => 5,
            F::Apple6 => 6,
            F::Apple7 => 7,
            F::Apple8 => 8,
            F::Apple9 => 9,
            F::Apple10 => 10,
            F::Apple11 => 11,
            _ => return None,
        })
    }
}

/// The column of the tables a family's rows are read from (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Column {
    /// The 2026 edition's Apple column of this number (2 to 10).
    Apple(u8),
    /// The 2023 edition's Common column of this number (1 to 3).
    Common(u8),
    Mac2,
    Metal3,
}

impl Column {
    /// The column `family` is read from, or `None` for a family of one row of its own.
    fn of(family: GpuFamily) -> Option<Column> {
        use GpuFamily as F;
        Some(match family {
            F::Apple1 => Column::Apple(2),
            F::Mac1 | F::Mac2 => Column::Mac2,
            F::Common1 => Column::Common(1),
            F::Common2 => Column::Common(2),
            F::Common3 => Column::Common(3),
            F::Metal3 => Column::Metal3,
            F::Apple11 | F::MacCatalyst1 | F::MacCatalyst2 | F::Metal4 => return None,
            apple => Column::Apple(apple.apple()?),
        })
    }
}

/// One requirement a family is judged by: an entry of Apple's tables, or the one row of a family
/// they have no column for.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FamilyRow {
    /// A feature row of Apple's tables.
    Feature(FeatureRow),
    /// A limit row of Apple's tables.
    Limit(LimitRow),
    /// One capability of a pixel format's cell.
    Format(FormatRow, FormatCapability),
    /// Metal 4's API (`MTL4CommandQueue`, `MTL4CommandBuffer`, `MTL4ArgumentTable` and the rest):
    /// the Metal4 family's one row.
    Metal4Api,
    /// A Mac Catalyst process: MacCatalyst1's and MacCatalyst2's one row.
    MacCatalystProcess,
    /// A column of Apple's tables for the family: Apple11's one row. No edition has one.
    NoPublishedColumn,
}

impl FamilyRow {
    /// The row's name as Apple's tables print it, for the rows they have.
    pub fn label(self) -> &'static str {
        match self {
            FamilyRow::Feature(row) => row.label(),
            FamilyRow::Limit(row) => row.label(),
            FamilyRow::Format(row, _) => row.label(),
            FamilyRow::Metal4Api => "Metal 4 API",
            FamilyRow::MacCatalystProcess => "Mac Catalyst process",
            FamilyRow::NoPublishedColumn => "Published feature set table column",
        }
    }
}

/// A row's state on a device (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowState {
    /// Offered through Metal's API on the device, refusing nothing Metal's validation accepts.
    Exact,
    /// No exact path exists on the device.
    Refused(Refusal),
    /// The runtime piece is not built yet.
    Gap(RuntimeGap),
    /// A framework layered on Metal: never blocks.
    NotADeviceFeature(Framework),
}

impl RowState {
    /// Whether the state keeps its family from being claimed: refused or a gap.
    pub fn blocks(&self) -> bool {
        matches!(self, RowState::Refused(_) | RowState::Gap(_))
    }
}

/// Why a row has no exact path on a device.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The compile layer's or the runtime's refusal: the capability the device lacks, that no
    /// Vulkan mechanism states the feature, or that no device has it yet.
    Unsupported(Unsupported),
    /// The runtime never runs in a Mac Catalyst process (an iOS app's environment on macOS), which
    /// the MacCatalyst families describe.
    NotACatalystProcess,
    /// No edition of Apple's tables has a column for the family, so none of its rows is known.
    NoPublishedColumn,
    /// Metal's device query the row is decided by answers no on the device (the query's
    /// documentation says what it reads).
    Query(Query),
    /// Metal's largest point (511 pixels): the device clamps a point to its `pointSizeRange`,
    /// whose upper bound lies below it (255.875 on ANV, 256 on lavapipe), so a larger point draws
    /// smaller there than on Metal.
    PointSizeRange,
    /// Metal's `maxArgumentBufferSamplerCount` as the runtime answers it on the device
    /// ([`DeviceProperties::max_argument_buffer_sampler_count`]: what it honours) lies below the
    /// live sampler states the family's cell lists, so a state past it reads the neutral sampler
    /// through an argument buffer there. The answer is one fewer than the smaller of a set-1
    /// sampler array's elements (the handle table's 65536 entries on MoltenVK, RADV and ANV) and
    /// `maxSamplerAllocationCount` (lavapipe's 32768, NVIDIA's 4000); Metal answers 500000 on an
    /// Apple M3 Max, Apple9's and Apple10's cell.
    ArgumentBufferSamplerCount {
        /// The states the family's cell lists.
        required: u64,
        /// The runtime's answer.
        available: u64,
    },
    /// The textures a function can read through argument buffers on the device lie below the
    /// count the family's cell lists: a function reading a texture whose resource ID is past them
    /// reads the neutral texture there, and a command naming it is refused
    /// (`unnamed-textures-past-capacity`). They
    /// are the textures at IDs 1 and up that a set-1 sampled-image array holds, one fewer than
    /// its elements (element 0 is the neutral): the least of the handle table's 65536 entries
    /// (on MoltenVK, RADV, ANV, NVIDIA and lavapipe) and the device's update-after-bind
    /// sampled-image limits, the count texture placements are checked against too. Apple6 to
    /// Apple10 and Mac2 list 1,000,000.
    ArgumentBufferTextureCount {
        /// The textures the family's cell lists.
        required: u64,
        /// The textures the device's sampled-image array holds.
        available: u64,
    },
}

/// A framework layered on Metal that a feature row names.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Framework {
    /// MetalKit.
    MetalKit,
    /// Metal Performance Shaders.
    MetalPerformanceShaders,
    /// MetalFX.
    MetalFx,
}

/// A row of a family and its state on a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowAudit {
    /// The row.
    pub row: FamilyRow,
    /// Its state on the device.
    pub state: RowState,
}

/// What rows are decided from on a device.
pub(super) struct Facts<'a> {
    /// The compile device's profile: the typed facts the translation decides by.
    pub(super) profile: &'a DeviceProfile,
    /// The compile layer's Metal answers for the device.
    pub(super) metal: &'a MetalDeviceProperties,
    /// The runtime's device queries.
    pub(super) properties: &'a DeviceProperties,
    /// The runtime's answers for a pixel format ([`formats`]).
    pub(super) format: &'a dyn Fn(PixelFormat) -> FormatAnswers,
}

/// The rows `family` lists, each with its state under `facts`.
pub(super) fn audit(family: GpuFamily, facts: &Facts<'_>) -> Vec<RowAudit> {
    let one = |row: FamilyRow, state: RowState| vec![RowAudit { row, state }];
    let Some(column) = Column::of(family) else {
        return match family {
            GpuFamily::Metal4 => one(FamilyRow::Metal4Api, RowState::Gap(RuntimeGap::Metal4Api)),
            GpuFamily::MacCatalyst1 | GpuFamily::MacCatalyst2 => one(
                FamilyRow::MacCatalystProcess,
                RowState::Refused(Refusal::NotACatalystProcess),
            ),
            _ => one(
                FamilyRow::NoPublishedColumn,
                RowState::Refused(Refusal::NoPublishedColumn),
            ),
        };
    };
    let mut rows = Vec::new();
    for row in FeatureRow::ALL {
        if row.listed(column) {
            rows.push(RowAudit {
                row: FamilyRow::Feature(row),
                state: row.state(column, facts),
            });
        }
    }
    for row in LimitRow::ALL {
        if let Some(need) = row.need(column) {
            rows.push(RowAudit {
                row: FamilyRow::Limit(row),
                state: row.state(need, facts),
            });
        }
    }
    for row in FormatRow::ALL {
        for capability in row.listed(column).iter() {
            rows.push(RowAudit {
                row: FamilyRow::Format(row, capability),
                state: row.state(capability, facts),
            });
        }
    }
    rows
}

impl Device {
    /// `supportsFamily:`: true only where every row of the family's column in Apple's tables is
    /// exact on this device (or names a framework layered on Metal); false for a value that names
    /// no family. The module docs say how each family's rows are chosen and judged;
    /// [`family_blockers`](Device::family_blockers) names what keeps a family false.
    pub fn supports_family(&self, family: i64) -> bool {
        self.current().supports_family_now(family)
    }

    /// [`Device::supports_family`] on the generation that serves (`device/line.rs`).
    pub(crate) fn supports_family_now(&self, family: i64) -> bool {
        GpuFamily::from_raw(family).is_some_and(|family| {
            self.family_rows_now(family)
                .iter()
                .all(|r| !r.state.blocks())
        })
    }

    /// Every row `family` lists, with its state on this device.
    pub fn family_rows(&self, family: GpuFamily) -> Vec<RowAudit> {
        self.current().family_rows_now(family)
    }

    /// [`Device::family_rows`] on the generation that serves (`device/line.rs`).
    pub(crate) fn family_rows_now(&self, family: GpuFamily) -> Vec<RowAudit> {
        let compile = self.compile_device();
        let format = |pf: PixelFormat| self.format_answers(pf);
        audit(
            family,
            &Facts {
                profile: compile.profile(),
                metal: compile.metal_properties(),
                properties: self.properties(),
                format: &format,
            },
        )
    }

    /// The rows that keep `family` from being claimed on this device (refused or gaps), with
    /// their states: the family's burn-down. Empty exactly where
    /// [`supports_family`](Device::supports_family) is true.
    pub fn family_blockers(&self, family: GpuFamily) -> Vec<RowAudit> {
        self.current().family_blockers_now(family)
    }

    /// [`Device::family_blockers`] on the generation that serves (`device/line.rs`).
    pub(crate) fn family_blockers_now(&self, family: GpuFamily) -> Vec<RowAudit> {
        let mut rows = self.family_rows_now(family);
        rows.retain(|r| r.state.blocks());
        rows
    }

    /// The texture capabilities of Apple's tables the runtime gives `format` on this device:
    /// the answer each format row is judged by ([`FormatCapability`] says how each is decided).
    pub fn pixel_format_capabilities(&self, format: PixelFormat) -> FormatCapabilities {
        self.current().pixel_format_capabilities_now(format)
    }

    /// [`Device::pixel_format_capabilities`] on the generation that serves (`device/line.rs`).
    pub(crate) fn pixel_format_capabilities_now(&self, format: PixelFormat) -> FormatCapabilities {
        self.format_answers(format).capabilities()
    }
}
