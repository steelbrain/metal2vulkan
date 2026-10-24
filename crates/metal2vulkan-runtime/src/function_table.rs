//! Visible-function tables and function handles (`MTLVisibleFunctionTable`,
//! `MTLFunctionHandle`).
//!
//! - **A table is a buffer of words.** The translation reads a `visible_function_table` as a
//!   read-only buffer of `pipeline::table`'s words at its Metal buffer index: 1 for a bound table,
//!   its size, then one word per entry, 0 for a null entry and else a handle's. A table is one
//!   runtime buffer of those words, `Shared`, written through its mapping, so a dispatch reads
//!   the entries as they are when it runs: as on Metal (measured on an M3 Max and an M4), an
//!   entry set after the dispatch was encoded and before the commit is the one it calls. The
//!   encoder binds the table's buffer at the table's index (`set_visible_function_table`); an
//!   unbound index reads the zero-filled neutral, a null table of size 0, Metal's plain answer.
//! - **Handles are the pipeline's.** `function_handle` is the word `pipeline::table` numbers the
//!   function by among the kernel's linked functions (`functions`, then `binary_functions`):
//!   one per function, none for a private function, an unlinked one or the kernel, as Metal
//!   answers. A handle belongs to the state that gave it, and a table to the state that made it.
//!   Metal's validation stops a handle of another state in `set_function`; plain Metal's table
//!   then calls its own state's function at the position the handle has in its state, which a
//!   table of a compute pipeline's kernel takes here (`metal_numbering`). A table of another state
//!   bound to a dispatch is read by that dispatch's numbering; Apple calls tables and handles
//!   specific to their pipeline (WWDC21 "Discover compilation workflows in Metal"), and Metal's own
//!   numbering there differs (it counts private functions first), so nothing defines that run.
//! - **Metal's answers.** A table is a `Shared`, tracked resource (options 512) outside any heap,
//!   with `allocatedSize` 256 bytes and 8 per entry, whole 16-KiB pages past 16 KiB; its
//!   purgeable state is always non-volatile; its resource ID is the lowest free one of a space of
//!   its own, from 1. A table holds as many entries as one buffer of the device's
//!   `max_buffer_length` holds words, and at most `u32::MAX` (its size word). Metal makes larger
//!   ones on an M3 Max and ends the process on an M4 (from 2^31 entries); a larger one is refused
//!   here, naming the limit.
//! - **Render states' tables**. A render pipeline state gives handles and makes tables per stage
//!   (`functionHandleWithFunction:stage:`, `newVisibleFunctionTableWithDescriptor:stage:`): a
//!   drawing state's vertex or fragment function's, numbered among the functions that function
//!   links, and a tile state's tile function's, which are its kernel state's. A table takes the
//!   handles of its own state and stage alone, and the render encoder binds it at a stage's buffer
//!   index (`set_vertex_visible_function_table` and the rest); a draw lists every live table of its
//!   state, whatever the stage, for its argument buffers.
//! - **Tables in argument buffers**. An argument buffer holds a table as its resource ID, and Metal
//!   reads it there without `useResource` (measured on an M3 Max and an M4). The device keeps every
//!   live table by ID, and a dispatch lists the live tables its pipeline state made
//!   (`BindState::function_tables`): the binder searches the ID in its directory and reads the
//!   table's words from the heap's function-table array, at the element of its ID. The command
//!   buffer retains every listed table until it completes, so an ID it wrote names one table
//!   until then. A table of another state, or an ID of none, reads as the null table.
//! - **The snapshot**. A device without the array's features (`runtimeDescriptorArray`,
//!   `descriptorBindingPartiallyBound`, `shaderStorageBufferArrayNonUniformIndexing`,
//!   `descriptorBindingStorageBufferUpdateAfterBind`) reads the listed tables' words from one
//!   snapshot per command (`bind::FunctionTableSnapshot`) in the command buffer's upload arena,
//!   which the command buffer writes again as it commits: an entry set after the command was
//!   encoded and before the commit is the one it calls, as on Metal. An entry set after the commit,
//!   while the GPU may run the command, is read by Metal if the command had not run yet and not
//!   here, a race of the program's on Metal.

use super::address::AddressPlacement;
use super::buffer::{Buffer, Modes};
use super::device::lock;
use super::error::{internal, invalid, nil, Error, Problem, RuntimeGap};
use super::library::Function;
use super::memory::MemoryClass;
use super::pipeline::{ComputePipelineState, GraphicsState, RenderPipelineState, RenderStateKind};
use super::resource_ids::{IdHold, IdPlacement, IdSpace, ResourceId};
use super::upload::Upload;
use super::{
    CpuCacheMode, Device, HazardTrackingMode, PurgeableState, ResourceOptions, StorageMode,
};
use metal2vulkan::bind::{ArgumentFunctionTable, FunctionTableSnapshot, FunctionTableWords};
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::RenderStages;
use metal2vulkan::pipeline::{LinkedFunction, LinkedFunctions, Stage};
use std::sync::{Arc, Mutex};

/// A table's options: `Shared`, the default CPU cache mode, tracked (Metal's 512).
fn table_options() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Shared,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Tracked,
    )
}

/// `MTLVisibleFunctionTableDescriptor`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisibleFunctionTableDescriptor {
    /// `functionCount` (0 by default).
    pub function_count: u64,
}

impl VisibleFunctionTableDescriptor {
    /// A descriptor of `function_count` entries.
    pub fn new(function_count: u64) -> VisibleFunctionTableDescriptor {
        VisibleFunctionTableDescriptor { function_count }
    }
}

/// What a visible-function table call did wrong (`Problem::FunctionTable`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionTableProblem {
    /// `set_function` with a handle of another pipeline state than the table's (Metal's
    /// validation: "The function handle ... was not created from the pipeline state that created
    /// this function table"), where plain Metal's answer was not measured: a compute state's
    /// handle in a render state's table or the reverse, a tile pipeline's, an object or mesh
    /// stage's, another stage's, a state that lists a private function more than once, in a
    /// numbered list too, or in a group, and a position that names a private function of the
    /// table's state; and so too a handle of a function added without the support flag to the
    /// table's own stage where that stage's numbering was not measured or its other functions
    /// are all private. Plain Metal's table
    /// calls its own state's function at the handle's position (its last one past them), as the
    /// runtime's does for two compute pipelines' kernels and two drawing states' vertex or
    /// fragment functions.
    HandleOfAnotherState,
    /// A command whose function writes a buffer, or reads an intersection function table, at
    /// `index`, where a visible-function table is bound: plain Metal's answer was not measured
    /// (ray tracing's tables are out of scope). A visible-function table where a function only
    /// reads a buffer reads as an unbound buffer, zeros, as plain Metal's read where its
    /// validation stops the command ("incorrect Buffer binding").
    TableAtBufferIndex {
        /// The buffer index.
        index: u32,
    },
    /// A render pipeline state's `function_handle` or `new_visible_function_table` for `stage`,
    /// which is not one stage: empty, or more than one. Metal's validation stops each ("stage
    /// invalid"); plain Metal answers the handle nil, and the table of more than one stage (an
    /// [`Error::Nil`](crate::Error::Nil)), and faults on a table of no stage (an
    /// [`Error::Invalid`](crate::Error::Invalid); an M4 under macOS 27.0.1). One
    /// stage the state has no function of answers no handle and makes a table, as plain Metal's.
    StageWithoutFunction {
        /// The stage asked for.
        stage: RenderStages,
    },
}

/// `MTLFunctionHandle`: a function a pipeline state can place in its tables. Two handles are
/// equal when one state gave both for one function, as Metal gives that state's one object.
#[derive(Clone)]
pub struct FunctionHandle {
    state: Maker,
    handle: metal2vulkan::pipeline::FunctionHandle,
    name: String,
}

/// The pipeline state that gives a handle or makes a table, and which of its functions' linked
/// functions number the handles: a compute state's kernel's (a tile state's, its tile
/// function's), or a drawing state's function of one stage; or a render state's stage it has no
/// function of, which numbers none, on its device.
#[derive(Clone)]
enum Maker {
    Compute(ComputePipelineState),
    Render(GraphicsState, Stage),
    Unlinked(Device),
}

impl Maker {
    /// Whether `other` is this state, and for a drawing state this stage.
    fn is(&self, other: &Maker) -> bool {
        match (self, other) {
            (Maker::Compute(a), Maker::Compute(b)) => a == b,
            (Maker::Render(a, x), Maker::Render(b, y)) => a.is(b) && x == y,
            // No handle comes from a stage without a function.
            _ => false,
        }
    }

    /// The state's device.
    fn device(&self) -> &Device {
        match self {
            Maker::Compute(state) => state.device(),
            Maker::Render(state, _) => state.device(),
            Maker::Unlinked(device) => device,
        }
    }

    /// The functions that number its handles: none for a stage whose function links nothing more
    /// (`GraphicsState::linked_functions`).
    fn linked_functions(&self) -> Option<&LinkedFunctions> {
        match self {
            Maker::Compute(state) => Some(state.linked_functions()),
            Maker::Render(state, stage) => state.linked_functions(*stage),
            Maker::Unlinked(_) => None,
        }
    }

    /// The handle of `function`, or `None` where Metal answers nil: a function the stage does not
    /// link in `functions` or `binary_functions`. A function of another device is refused
    /// (`Invalid(ForeignObject)`).
    fn function_handle(self, function: &Function) -> Result<Option<FunctionHandle>, Error> {
        if !function.device().same(self.device()) {
            return Err(invalid("function_handle", Problem::ForeignObject));
        }
        let handle = self
            .linked_functions()
            .and_then(|linked| linked.function_handle(&function.linked_function()));
        Ok(handle.map(|handle| FunctionHandle {
            state: self,
            handle,
            name: function.name().to_owned(),
        }))
    }

    /// How many of its function's last `binary_functions` were added without the stage's
    /// support flag (`ComputePipelineState::added_unsupported`, a tile state's kernel's too, and
    /// `GraphicsState::added_unsupported`).
    fn added_unsupported(&self) -> usize {
        match self {
            Maker::Compute(state) => state.added_unsupported(),
            Maker::Render(state, stage) => state.added_unsupported(*stage),
            Maker::Unlinked(_) => 0,
        }
    }

    /// The entry a table of this state holds for `handle`: its own function for one of this
    /// state's, but one added without the stage's support flag, which calls this state's last
    /// other callable function, or none where it has none, as plain Metal's handle of it does
    /// (for kernels; for drawing and tile states); else
    /// [`handle_at_position_of`](Self::handle_at_position_of)'s. `None` where Metal's answer
    /// was not measured: Metal numbers the added functions past the others
    /// (`metal_numbering`), so where this state links only private functions, its last is one.
    fn entry_of(
        &self,
        handle: &FunctionHandle,
    ) -> Option<Option<metal2vulkan::pipeline::FunctionHandle>> {
        if !handle.state.is(self) {
            return self.handle_at_position_of(handle);
        }
        let unsupported = self.added_unsupported();
        let Some(linked) = self.linked_functions().filter(|_| unsupported > 0) else {
            return Some(Some(handle.handle));
        };
        let binary = &linked.binary_functions;
        let added_without_flag = binary[binary.len().saturating_sub(unsupported)..]
            .iter()
            .any(|f| linked.function_handle(f) == Some(handle.handle));
        if !added_without_flag {
            return Some(Some(handle.handle));
        }
        let (private, callable) = metal_numbering(linked, unsupported)?;
        if callable.is_empty() && private > 0 {
            return None;
        }
        Some(callable.last().copied())
    }

    /// This state's handle at the position Metal numbers `handle`, another state's, by: the one
    /// a table of this state holds for it, as plain Metal's table calls its own state's function
    /// at that position (an M3 Max and an M4: a handle of another kernel's second function called
    /// the table's kernel's second; an M4: the same of two drawing states' vertex and fragment
    /// functions). Metal numbers a state's private functions first, then its callable ones
    /// (`metal_numbering`). A position at or past this state's functions calls its last one, and
    /// none where it links none, an entry that reads null, as plain Metal's (and through functions
    /// added without the support flag, `bf_off_two`, `bf_off_empty`, `bf_off_one_add_two`). `None`
    /// where Metal's answer was not measured: states of two kinds, a tile state's, an object or
    /// mesh stage's, two stages, either numbering unmeasured, or a position that names a private
    /// function of this state.
    ///
    /// This state's functions added without its support flag are not its functions here: a
    /// position past the others calls its last other one.
    fn handle_at_position_of(
        &self,
        handle: &FunctionHandle,
    ) -> Option<Option<metal2vulkan::pipeline::FunctionHandle>> {
        let measured = match (self, &handle.state) {
            (Maker::Compute(own), Maker::Compute(other)) => !own.is_tile() && !other.is_tile(),
            (Maker::Render(_, own), Maker::Render(_, other)) => {
                own == other && matches!(own, Stage::Vertex | Stage::Fragment)
            }
            _ => false,
        };
        if !measured {
            return None;
        }
        let (other_private, other_callable) = metal_numbering(handle.state.linked_functions()?, 0)?;
        let (own_private, own_callable) =
            metal_numbering(self.linked_functions()?, self.added_unsupported())?;
        let at = other_callable.iter().position(|h| *h == handle.handle)?;
        let position = other_private + at;
        let index = position.checked_sub(own_private)?;
        if own_callable.is_empty() && own_private > 0 {
            return None;
        }
        Some(own_callable.get(index).or(own_callable.last()).copied())
    }

    /// `newVisibleFunctionTableWithDescriptor:` of this state: a table of
    /// `descriptor.function_count` null entries (the docs of
    /// [`ComputePipelineState::new_visible_function_table`]).
    fn new_visible_function_table(
        self,
        descriptor: &VisibleFunctionTableDescriptor,
    ) -> Result<VisibleFunctionTable, Error> {
        const CALL: &str = "new_visible_function_table";
        let device = self.device().clone();
        let count = descriptor.function_count;
        let max_buffer_length = device.properties().max_buffer_length;
        let Ok(size) = u32::try_from(count) else {
            return Err(past_limit(count, max_buffer_length));
        };
        let length = FunctionTableWords::byte_len(size);
        if length > max_buffer_length {
            return Err(past_limit(count, max_buffer_length));
        }
        let modes = Modes::of(table_options())?;
        // The host writes the words and never reads them; the GPU reads them at every call, so
        // they lie where a write-combined `Shared` buffer's do (device-local memory without
        // unified memory, where the device maps such).
        let words = device.zeroed_buffer(
            CALL,
            length,
            modes,
            MemoryClass::WriteCombined,
            false,
            AddressPlacement::Any,
        )?;
        let table = VisibleFunctionTable(Arc::new(TableShared {
            state: self,
            words,
            count: size,
            writes: Mutex::new(()),
            id: IdHold::take(
                &device.shared().resource_ids,
                IdSpace::FunctionTable,
                IdPlacement::Any,
            ),
        }));
        for (word, offset) in FunctionTableWords::header(size)
            .into_iter()
            .zip((0..).step_by(4))
        {
            table.write(offset, word);
        }
        // The ID was free, so a table the registry still names under it is gone.
        lock(&device.shared().function_tables)
            .insert(table.gpu_resource_id().0, Arc::downgrade(&table.0));
        Ok(table)
    }
}

/// The live tables `made` (by a state, the predicate) made, in resource-ID order: what an
/// argument buffer can name for a command of the state (`BindState::function_tables`). Gone ones
/// leave the device's registry.
fn live_tables(device: &Device, made: impl Fn(&Maker) -> bool) -> Vec<VisibleFunctionTable> {
    let mut live = lock(&device.shared().function_tables);
    live.retain(|_, table| table.strong_count() > 0);
    live.values()
        .filter_map(|table| table.upgrade().map(VisibleFunctionTable))
        .filter(|table| made(&table.0.state))
        .collect()
}

/// How Metal numbers the functions `linked` lists, where measured: its private functions first,
/// then the callable ones, `functions` then `binary_functions`, one function in both lists once
/// (an M3 Max and an M4: `number_private_unnumbered`, `number_binary_after_functions`,
/// `number_same_function_in_both_lists`). The count of private functions, and the callable
/// functions' handles in order, but the last `added_unsupported` `binary_functions` (added without
/// the support flag). `None` where it was not measured: a private function listed twice or in a
/// numbered list too, or functions in groups.
fn metal_numbering(
    linked: &LinkedFunctions,
    added_unsupported: usize,
) -> Option<(usize, Vec<metal2vulkan::pipeline::FunctionHandle>)> {
    let same = |a: &LinkedFunction, b: &LinkedFunction| {
        let (x, y) = (a.function.info(), b.function.info());
        x.name == y.name && x.air_hash == y.air_hash && a.constants == b.constants
    };
    let private = &linked.private_functions;
    let numbered = || linked.functions.iter().chain(&linked.binary_functions);
    let repeated = private
        .iter()
        .enumerate()
        .any(|(i, f)| private[..i].iter().any(|g| same(f, g)) || numbered().any(|g| same(f, g)));
    if repeated || !linked.groups.is_empty() {
        return None;
    }
    let binary = &linked.binary_functions;
    let supported = &binary[..binary.len().saturating_sub(added_unsupported)];
    let mut callable = Vec::new();
    for handle in linked
        .functions
        .iter()
        .chain(supported)
        .filter_map(|f| linked.function_handle(f))
    {
        if !callable.contains(&handle) {
            callable.push(handle);
        }
    }
    Some((private.len(), callable))
}

impl std::fmt::Debug for FunctionHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionHandle")
            .field("name", &self.name)
            .field("word", &self.handle.word())
            .finish()
    }
}

impl PartialEq for FunctionHandle {
    fn eq(&self, other: &FunctionHandle) -> bool {
        self.state.is(&other.state) && self.handle == other.handle
    }
}

impl Eq for FunctionHandle {}

impl FunctionHandle {
    /// `name`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `device`.
    pub fn device(&self) -> &Device {
        self.state.device()
    }
}

impl ComputePipelineState {
    /// `functionHandleWithFunction:`: the handle of `function` (with the constant values it was
    /// made with), or `None` where Metal answers nil: a function the kernel does not link in
    /// `functions` or `binary_functions` (a private function, the kernel itself, another
    /// function). A function of another device is refused (`Invalid(ForeignObject)`).
    pub fn function_handle(&self, function: &Function) -> Result<Option<FunctionHandle>, Error> {
        Maker::Compute(self.clone()).function_handle(function)
    }

    /// `newVisibleFunctionTableWithDescriptor:`: a table of `descriptor.function_count` null
    /// entries, whatever the kernel links (Metal makes one for a kernel without a table too). A
    /// table whose words one buffer of the device cannot hold is refused
    /// (`NotImplemented(FunctionTablePastMaxBufferLength)`).
    pub fn new_visible_function_table(
        &self,
        descriptor: &VisibleFunctionTableDescriptor,
    ) -> Result<VisibleFunctionTable, Error> {
        Maker::Compute(self.clone()).new_visible_function_table(descriptor)
    }

    /// The live tables this state made, in resource-ID order: what an argument buffer can name
    /// for a dispatch of the state (`BindState::function_tables`).
    pub(super) fn function_tables(&self) -> Vec<VisibleFunctionTable> {
        live_tables(self.device(), |made| match made {
            Maker::Compute(state) => state == self,
            Maker::Render(..) | Maker::Unlinked(_) => false,
        })
    }
}

impl RenderPipelineState {
    /// `functionHandleWithFunction:stage:`: the handle of `function` among the functions the
    /// function of `stage` links (`vertexLinkedFunctions`, `fragmentLinkedFunctions`; a tile
    /// state's tile function's), or `None` where Metal answers nil, as
    /// [`ComputePipelineState::function_handle`] answers for a kernel; `None` too for one stage
    /// the state has no function of, as plain Metal answers. Refused: a function of another device
    /// (`Invalid(ForeignObject)`), and a `stage` that is not one stage (`Nil`,
    /// `FunctionTableProblem::StageWithoutFunction`).
    pub fn function_handle(
        &self,
        function: &Function,
        stage: RenderStages,
    ) -> Result<Option<FunctionHandle>, Error> {
        self.maker(stage)
            .ok_or_else(|| not_one_stage("function_handle", stage, false))?
            .function_handle(function)
    }

    /// `newVisibleFunctionTableWithDescriptor:stage:`: a table of `descriptor.function_count`
    /// null entries, which takes the handles of `stage` alone, as
    /// [`ComputePipelineState::new_visible_function_table`] makes a kernel's. For one stage the
    /// state has no function of it is made as plain Metal makes it, and takes no handle but a
    /// null one. Refused as a kernel's is, and for a `stage` that is not one stage
    /// (`FunctionTableProblem::StageWithoutFunction`): `Nil` for more than one stage, `Invalid` for
    /// none, where plain Metal faults.
    pub fn new_visible_function_table(
        &self,
        descriptor: &VisibleFunctionTableDescriptor,
        stage: RenderStages,
    ) -> Result<VisibleFunctionTable, Error> {
        self.maker(stage)
            .ok_or_else(|| not_one_stage("new_visible_function_table", stage, true))?
            .new_visible_function_table(descriptor)
    }

    /// The state and stage whose linked functions number `stage`'s handles: none for one stage
    /// the state has no function of (`Maker::Unlinked`); `None` for a stage that is not one stage.
    fn maker(&self, stage: RenderStages) -> Option<Maker> {
        let maker = match self.kind() {
            RenderStateKind::Tile(tile) if stage == RenderStages::TILE => {
                Some(Maker::Compute(tile.kernel().clone()))
            }
            RenderStateKind::Graphics(state) => [
                (RenderStages::VERTEX, Stage::Vertex),
                (RenderStages::FRAGMENT, Stage::Fragment),
                (RenderStages::OBJECT, Stage::Object),
                (RenderStages::MESH, Stage::Mesh),
            ]
            .into_iter()
            .find(|&(of, s)| of == stage && state.descriptor().stage(s).is_some())
            .map(|(_, s)| Maker::Render(state.clone(), s)),
            RenderStateKind::Tile(_) => None,
        };
        if let Some(maker) = maker {
            return Some(maker);
        }
        // One stage the state has no function of: plain Metal answers its handles nil and makes
        // its tables (a vertex-only state's fragment stage, a drawing state's
        // tile, object and mesh stages, a tile state's vertex and fragment stages).
        let stages = [
            RenderStages::VERTEX,
            RenderStages::FRAGMENT,
            RenderStages::TILE,
            RenderStages::OBJECT,
            RenderStages::MESH,
        ];
        stages
            .contains(&stage)
            .then(|| Maker::Unlinked(self.device().clone()))
    }
}

/// `call` for `stage`, which is not one stage, a table's where `table`: Metal's validation stops
/// each; plain Metal answers nil but for a table of no stage, where it faults.
fn not_one_stage(call: &'static str, stage: RenderStages, table: bool) -> Error {
    let problem = Problem::FunctionTable(FunctionTableProblem::StageWithoutFunction { stage });
    if table && stage.is_empty() {
        invalid(call, problem)
    } else {
        nil(call, problem)
    }
}

impl GraphicsState {
    /// The live tables this state made, for any stage, in resource-ID order: what an argument
    /// buffer can name for a draw of the state (`BindState::function_tables`).
    pub(crate) fn function_tables(&self) -> Vec<VisibleFunctionTable> {
        live_tables(self.device(), |made| match made {
            Maker::Render(state, _) => state.is(self),
            Maker::Compute(_) | Maker::Unlinked(_) => false,
        })
    }
}

/// The snapshot of the listed `tables` on `device` (`bind::FunctionTableSnapshot`), a command's
/// tables' words on a device without the function-table array's features. Its one binding
/// reaches `maxStorageBufferRange` bytes: past it, [`past_storage_buffer_range`].
pub(super) fn snapshot_layout(
    device: &Device,
    tables: &[VisibleFunctionTable],
) -> Result<FunctionTableSnapshot, Error> {
    let limit = device
        .compile_device()
        .profile()
        .limits()
        .max_storage_buffer_range;
    match FunctionTableSnapshot::of(tables.iter().map(VisibleFunctionTable::function_count)) {
        Some(layout) if layout.byte_len() <= u64::from(limit) => Ok(layout),
        _ => Err(past_storage_buffer_range(device)),
    }
}

/// The refusal of a table's words, or a snapshot, that passes `device`'s
/// `maxStorageBufferRange` where a plan binds it whole. A plan asks for the variant that reads
/// such words at their addresses wherever the device has device addresses, so it binds them
/// whole only on a device without them: that device lacks what the variant needs
/// (`Unsupported{BufferRange}`, naming device addresses), and anywhere else the binding is the
/// runtime's error.
pub(super) fn past_storage_buffer_range(device: &Device) -> Error {
    let needs = device.compile_device().profile().addressed_needs();
    match needs.is_empty() {
        true => internal("a function table past maxStorageBufferRange bound whole"),
        false => Error::Unsupported(Unsupported::needs_any_of(Feature::BufferRange, vec![needs])),
    }
}

/// The bytes of the snapshot `layout` of `tables`, their words as they stand now.
pub(super) fn snapshot_bytes(
    layout: &FunctionTableSnapshot,
    tables: &[VisibleFunctionTable],
) -> Result<Vec<u8>, Error> {
    let words: Vec<Vec<u8>> = tables
        .iter()
        .map(VisibleFunctionTable::current_bytes)
        .collect();
    layout
        .bytes(words.iter().map(Vec::as_slice))
        .ok_or_else(|| internal("a function-table snapshot of other sizes than its tables'"))
}

/// A command's function-table snapshot, written when the command was bound, which the command
/// buffer writes again with the tables' words as they stand when it commits: Metal's command
/// reads a table's entries as they stand when it runs, so an entry set after the command was
/// encoded and before the commit is the one it calls.
pub(super) struct TableSnapshot {
    region: Upload,
    layout: FunctionTableSnapshot,
    tables: Vec<VisibleFunctionTable>,
}

impl TableSnapshot {
    /// The snapshot `layout` of `tables` in `region`.
    pub(super) fn new(
        region: Upload,
        layout: FunctionTableSnapshot,
        tables: Vec<VisibleFunctionTable>,
    ) -> TableSnapshot {
        TableSnapshot {
            region,
            layout,
            tables,
        }
    }

    /// Write the tables' words as they stand now, before any batch that reads them is
    /// submitted.
    pub(super) fn refresh(&self) -> Result<(), Error> {
        self.region
            .write(&snapshot_bytes(&self.layout, &self.tables)?);
        Ok(())
    }
}

/// The refusal of a table of `count` entries past what one buffer holds.
fn past_limit(count: u64, max_buffer_length: u64) -> Error {
    Error::NotImplemented(RuntimeGap::FunctionTablePastMaxBufferLength {
        count,
        max_buffer_length,
    })
}

/// `MTLVisibleFunctionTable`. Cheap to clone; `Send + Sync`; two tables are equal when they are
/// the same table.
#[derive(Clone)]
pub struct VisibleFunctionTable(Arc<TableShared>);

pub(super) struct TableShared {
    /// The state (and a drawing state's stage) that made the table: the only one whose handles
    /// it takes.
    state: Maker,
    /// The table's words: bound, its size, its entries.
    words: Buffer,
    count: u32,
    /// Held while the CPU writes a word.
    writes: Mutex<()>,
    id: IdHold,
}

impl std::fmt::Debug for VisibleFunctionTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisibleFunctionTable")
            .field("count", &self.0.count)
            .finish()
    }
}

impl PartialEq for VisibleFunctionTable {
    fn eq(&self, other: &VisibleFunctionTable) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for VisibleFunctionTable {}

impl VisibleFunctionTable {
    /// Write the word `value` at byte `offset` of the table's buffer.
    fn write(&self, offset: u64, value: u32) {
        let contents = self.0.words.contents();
        let _writing = lock(&self.0.writes);
        if let Some(contents) = contents {
            // SAFETY: the table's buffer is `Shared` and mapped for its life, and holds
            // `FunctionTableWords::byte_len(count)` bytes; every caller writes a word of its
            // header or of an entry below `count`, at a multiple of 4 of the mapping, whose start
            // is aligned past 4. Writes are one at a time (`writes`); a dispatch reading the word
            // as it is written reads the old or the new word, as the program's race on Metal
            // does.
            unsafe {
                contents
                    .as_ptr()
                    .add(offset as usize)
                    .cast::<u32>()
                    .write_volatile(value.to_le());
            }
        }
    }

    /// The buffer of the table's words, which the encoder binds at the table's index.
    pub(super) fn words(&self) -> &Buffer {
        &self.0.words
    }

    /// The bytes of the table's words: its header and entries.
    pub(super) fn words_len(&self) -> u64 {
        FunctionTableWords::byte_len(self.0.count)
    }

    /// The table as a bind state lists it for an argument buffer's reads
    /// (`BindState::function_tables`): its resource ID, its size and, where the device has device
    /// addresses, its words' address, which a pipeline compiled to read tables there reads.
    pub(super) fn listed(&self) -> ArgumentFunctionTable {
        ArgumentFunctionTable::new(self.gpu_resource_id().0, self.function_count())
            .at_address(self.words().device_address().unwrap_or(0))
    }

    /// The table's words as they stand now, as its buffer holds them.
    fn current_bytes(&self) -> Vec<u8> {
        let len = self.words_len() as usize;
        let mut bytes = vec![0; len];
        let _writing = lock(&self.0.writes);
        if let Some(contents) = self.0.words.contents() {
            // SAFETY: the table's buffer is `Shared` and mapped for its life, and holds `len`
            // bytes; the CPU writes no word meanwhile (`writes`), and no command writes them.
            unsafe {
                std::ptr::copy_nonoverlapping(contents.as_ptr(), bytes.as_mut_ptr(), len);
            }
        }
        bytes
    }

    /// `setFunction:atIndex:`: entry `index` calls `function`, or is null (`None`). A handle of
    /// another compute pipeline's kernel than the table's, or of another drawing state's function
    /// of the table's stage (vertex or fragment), makes the entry call the table's state's
    /// function at the handle's position, as plain Metal's does where its validation stops the
    /// call; and a handle of a function added to the table's stage without its support flag calls
    /// the stage's last other callable function, or none. Refused: an index at or past the table's
    /// size (`Invalid(IndexOutOfRange)`), which Metal's validation stops and plain Metal writes
    /// past the table's entries, where its memory decides: the entries kept their functions, and
    /// its process ended past the table's allocation; a handle of another state where plain Metal's
    /// answer was not measured (`Invalid(FunctionTable(HandleOfAnotherState))`, its doc).
    pub fn set_function(&self, function: Option<&FunctionHandle>, index: u32) -> Result<(), Error> {
        self.set("set_function", &[function], index)
    }

    /// `setFunctions:withRange:`: entry `first + i` calls `functions[i]`, or is null. Refused as
    /// [`set_function`](Self::set_function) is; a refusal sets none of them.
    pub fn set_functions(
        &self,
        functions: &[Option<&FunctionHandle>],
        first: u32,
    ) -> Result<(), Error> {
        self.set("set_functions", functions, first)
    }

    /// Entry `first + i` calls `functions[i]`, or is null, for `call`.
    fn set(
        &self,
        call: &'static str,
        functions: &[Option<&FunctionHandle>],
        first: u32,
    ) -> Result<(), Error> {
        super::argument_table::check_range(call, first, functions.len(), self.0.count)?;
        let entries = functions
            .iter()
            .map(|function| match function {
                None => Ok(None),
                Some(function) => self.0.state.entry_of(function).ok_or_else(|| {
                    invalid(
                        call,
                        Problem::FunctionTable(FunctionTableProblem::HandleOfAnotherState),
                    )
                }),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        for (entry, index) in entries.into_iter().zip(first..) {
            let word = FunctionTableWords::entry_word(entry);
            self.write(FunctionTableWords::entry_offset(index), word);
        }
        Ok(())
    }

    /// `device`.
    pub fn device(&self) -> &Device {
        self.0.state.device()
    }

    /// The table's size, the `functionCount` of the descriptor it was made from. Not public:
    /// `MTLVisibleFunctionTable` declares no count, so a consumer reads it from its own
    /// descriptor.
    fn function_count(&self) -> u32 {
        self.0.count
    }

    /// `gpuResourceID`: the lowest ID free when the table was made, of the tables' own space.
    pub fn gpu_resource_id(&self) -> ResourceId {
        self.0.id.id()
    }

    /// `label`: `None` until one is set.
    pub fn label(&self) -> Option<String> {
        self.0.words.attributes().label()
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        self.0.words.attributes().set_label(label);
    }

    /// `storageMode`: `Shared`.
    pub fn storage_mode(&self) -> StorageMode {
        StorageMode::Shared
    }

    /// `cpuCacheMode`: the default.
    pub fn cpu_cache_mode(&self) -> CpuCacheMode {
        CpuCacheMode::DefaultCache
    }

    /// `hazardTrackingMode`: tracked.
    pub fn hazard_tracking_mode(&self) -> HazardTrackingMode {
        HazardTrackingMode::Tracked
    }

    /// `resourceOptions`: `Shared`, the default cache mode, tracked (512).
    pub fn resource_options(&self) -> ResourceOptions {
        table_options()
    }

    /// `allocatedSize`: Metal's, `table_allocated_size` of the table's size.
    pub fn allocated_size(&self) -> u64 {
        table_allocated_size(self.0.count)
    }

    /// `heap`: none (a table is made by its pipeline state, never in a heap).
    pub fn heap(&self) -> Option<&super::Heap> {
        None
    }

    /// `heapOffset`: 0.
    pub fn heap_offset(&self) -> u64 {
        0
    }

    /// `isAliasable`: false.
    pub fn is_aliasable(&self) -> bool {
        false
    }

    /// `setPurgeableState:`: non-volatile, whatever `state` asks, and the entries stay, as
    /// Metal answers for a keep, volatile or non-volatile state (an empty state was not
    /// measured).
    pub fn set_purgeable_state(&self, _state: PurgeableState) -> PurgeableState {
        PurgeableState::NonVolatile
    }
}

/// Metal's `allocatedSize` of a table of `count` entries (measured on an M3 Max and an M4): 256
/// bytes and 8 per entry, rounded up to whole 16-KiB pages past 16 KiB.
pub(super) fn table_allocated_size(count: u32) -> u64 {
    let bytes = 256 + 8 * u64::from(count);
    if bytes <= super::address::PAGE {
        bytes
    } else {
        bytes.next_multiple_of(super::address::PAGE)
    }
}
