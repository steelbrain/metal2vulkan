//! A render encoder's argument buffers: the resources `use_resource(s)` and `use_heap(s)` make
//! resident in render stages, set 1, the heap, of its draws and tile dispatches, and the
//! visible-function tables each stage binds.
//!
//! - **Residency** (`residency.rs`). `useResource:usage:stages:` and `useHeap:stages:` make a
//!   resource resident for the encoder's later draws and tile dispatches in the render stages they
//!   name, joined with any earlier use: a draw's vertex and fragment functions reach it through
//!   an address or a resource ID held in memory in those of their stages alone (the binder's
//!   `UsedResource::stages`), and a tile dispatch's function only where they hold the tile stage.
//!   An indirect command buffer made resident for writing is written by the GPU from then on, as
//!   on a compute encoder. A stage Vulkan has no form of (object, mesh) reaches no function.
//! - **Set 1**, as a compute dispatch's: a draw or a tile dispatch whose variant has a heap lists
//!   the device's sampler states that fit its sampler arrays (`heap_sets::fit_samplers`) and writes
//!   its heap elements, at their resource IDs, into the command buffer's one heap set of the
//!   variant (`heap_sets.rs`), which it binds as set 1. A draw records that set into the pass's
//!   secondary command buffer; the set's arrays are update-after-bind, so a later command's writes
//!   land after it was bound.
//! - **Visible-function tables** (`function_table.rs`): a stage's table binds at its buffer
//!   index as a compute encoder's does, and a draw refuses a buffer where a stage's function
//!   reads a table, or a table where it reads a buffer, as Metal's validation stops them. A draw
//!   lists the live tables its state made, and a tile dispatch those of its tile function's
//!   state, which its argument buffers can name.
//! - **Shadowed textures** (`shadow.rs`) a draw or a tile dispatch reaches through the residency
//!   are gathered with its bound ones and copied around its render pass instance, or around the
//!   dispatch (`render/shadows.rs`).

use super::RenderCommandEncoder;
use super::RenderLowering;
use super::{RenderCall, Table};
use crate::error::Error;
use crate::function_table::VisibleFunctionTable;
use crate::heap::Heap;
use crate::pvrtc::degrade_unreadable;
use crate::replay::Resource;
use crate::resource::ResourceRef;
use metal2vulkan::metal::{RenderStages, ResourceUsage};

impl RenderLowering {
    /// The lowering of [`RenderCommandEncoder::use_resource`].
    pub(crate) fn use_resource(
        &mut self,
        resource: ResourceRef<'_>,
        usage: ResourceUsage,
        stages: RenderStages,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::UseResource(Resource::one(resource), usage, stages));
        self.use_all("use_resource", &[resource], usage, stages)
    }

    /// The lowering of [`RenderCommandEncoder::use_resources`].
    pub(crate) fn use_resources(
        &mut self,
        resources: &[ResourceRef<'_>],
        usage: ResourceUsage,
        stages: RenderStages,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::UseResources(Resource::all(resources), usage, stages));
        self.use_all("use_resources", resources, usage, stages)
    }

    fn use_all(
        &mut self,
        call: &'static str,
        resources: &[ResourceRef<'_>],
        usage: ResourceUsage,
        stages: RenderStages,
    ) -> Result<(), Error> {
        for resource in resources {
            self.core.check_device(call, resource.device())?;
        }
        // A memoryless texture fails the command buffer from the pass on, with no draw after it
        // too (`RenderCommandEncoder::use_resource`).
        if resources.iter().any(ResourceRef::is_memoryless) {
            self.core.fail_memoryless_use();
            return Ok(());
        }
        for &resource in resources {
            if let Some(used) = self.residency.use_resource(resource, usage, stages) {
                self.core.made_resident(used);
            }
            if let ResourceRef::IndirectCommandBuffer(icb) = resource {
                if usage.contains(ResourceUsage::WRITE) {
                    self.writers.made_writable(icb.written());
                    self.core.routes.writable(icb);
                    self.core.retain(icb.gpu_write());
                }
            }
        }
        Ok(())
    }

    /// The lowering of [`RenderCommandEncoder::use_heap`].
    pub(crate) fn use_heap(&mut self, heap: &Heap, stages: RenderStages) -> Result<(), Error> {
        self.log(|_| RenderCall::UseHeap(heap.clone(), stages));
        self.use_heaps_of("use_heap", &[heap], stages)
    }

    /// The lowering of [`RenderCommandEncoder::use_heaps`].
    pub(crate) fn use_heaps(&mut self, heaps: &[&Heap], stages: RenderStages) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::UseHeaps(heaps.iter().map(|&heap| heap.clone()).collect(), stages)
        });
        self.use_heaps_of("use_heaps", heaps, stages)
    }

    fn use_heaps_of(
        &mut self,
        call: &'static str,
        heaps: &[&Heap],
        stages: RenderStages,
    ) -> Result<(), Error> {
        for heap in heaps {
            self.core.check_device(call, heap.device())?;
        }
        for heap in heaps {
            for used in self.residency.use_heap(heap, stages) {
                self.core.made_resident(used);
            }
        }
        Ok(())
    }

    /// Make resident the textures made since the encoder's last command in the heaps it used
    /// (`Residency::sync_heaps`), held until the command buffer completes: a draw, a tile
    /// dispatch and an execution of render commands reach every texture their heaps hold as
    /// they are encoded.
    pub(super) fn sync_heaps(&mut self) {
        for used in self.residency.sync_heaps() {
            self.core.retain(used);
        }
    }

    /// Once a command was encoded, note the heaps the encoder used; and where its functions read
    /// texture handles from memory in render stages `reads`, the textures they reach whose
    /// blocks a carrier holds.
    pub(super) fn command_encoded(&mut self, reads: RenderStages) {
        // The command, encoded, used the heaps of what the encoder made resident: noted once a
        // change of it, as an earlier command noted those it held then.
        let generation = self.residency.generation();
        if self.heaps_noted != Some(generation) {
            self.core.touch_heaps(self.residency.heaps());
            self.heaps_noted = Some(generation);
        }
        if reads.is_empty() {
            return;
        }
        // A texture the functions' handles reach whose blocks a carrier holds reads as the
        // neutral: those made resident in the stages that read handles, and those bound there.
        let tables = [
            (RenderStages::VERTEX, &self.vertex),
            (RenderStages::FRAGMENT, &self.fragment),
            (RenderStages::OBJECT, &self.object),
            (RenderStages::MESH, &self.mesh),
            (RenderStages::TILE, &self.tile),
        ];
        let bound = tables
            .into_iter()
            .filter(|(stage, _)| reads.intersects(*stage))
            .flat_map(|(_, table)| table.textures.values());
        degrade_unreadable(
            self.core.device(),
            self.residency.textures_reached(reads).chain(bound),
            "a draw or tile dispatch",
        );
    }
}

impl RenderCommandEncoder {
    /// `useResource:usage:stages:`: the encoder's later draws and tile dispatches may reach
    /// `resource` in render `stages` through an address or a resource ID held in memory, reading
    /// or writing as `usage` says (the module docs). A memoryless texture, a texture buffer too,
    /// fails the command buffer from the pass on as plain Metal's does, with no draw after it too
    /// (`CommandBufferError::InvalidResource`, an M3 Max under macOS 27.0.1).
    ///
    /// Refused, changing nothing: a resource of another device (`Invalid`).
    pub fn use_resource(
        &mut self,
        resource: ResourceRef<'_>,
        usage: ResourceUsage,
        stages: RenderStages,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.use_resource(resource, usage, stages)),
            false => {
                self.defer(|_| RenderCall::UseResource(Resource::one(resource), usage, stages))
            }
        }
    }

    /// `useResources:count:usage:stages:`: [`use_resource`](Self::use_resource) of each of
    /// `resources`, refused, changing nothing, as it is for any of them.
    pub fn use_resources(
        &mut self,
        resources: &[ResourceRef<'_>],
        usage: ResourceUsage,
        stages: RenderStages,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.use_resources(resources, usage, stages)),
            false => {
                self.defer(|_| RenderCall::UseResources(Resource::all(resources), usage, stages))
            }
        }
    }

    /// `useHeap:stages:`: the encoder's later draws and tile dispatches may read every buffer
    /// placed in `heap` in render `stages` through an address held in memory, and every texture
    /// made in it or over a buffer placed in it through its resource ID, as Metal's contract makes
    /// a heap's resources resident for reading only (`residency.rs`): the textures that live now,
    /// and each made later, from the first command encoded after it is made.
    ///
    /// A sparse heap's live sparse textures are made resident, as on a compute encoder.
    ///
    /// Refused, changing nothing (`Invalid`): a heap of another device.
    pub fn use_heap(&mut self, heap: &Heap, stages: RenderStages) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.use_heap(heap, stages)),
            false => self.defer(|_| RenderCall::UseHeap(heap.clone(), stages)),
        }
    }

    /// `useHeaps:count:stages:`: [`use_heap`](Self::use_heap) of each of `heaps`, refused,
    /// changing nothing, as it is for any of them.
    pub fn use_heaps(&mut self, heaps: &[&Heap], stages: RenderStages) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.use_heaps(heaps, stages)),
            false => self.defer(|_| {
                RenderCall::UseHeaps(heaps.iter().map(|&heap| heap.clone()).collect(), stages)
            }),
        }
    }
}

impl RenderLowering {
    /// The lowering of [`RenderCommandEncoder::set_vertex_visible_function_table`].
    pub(crate) fn set_vertex_visible_function_table(
        &mut self,
        table: Option<&VisibleFunctionTable>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::FunctionTable(Table::Vertex, table.cloned(), index));
        const CALL: &str = "set_vertex_visible_function_table";
        self.vertex
            .set_function_tables(&mut self.core, CALL, &[table], index)
    }

    /// The lowering of [`RenderCommandEncoder::set_vertex_visible_function_tables`].
    pub(crate) fn set_vertex_visible_function_tables(
        &mut self,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::FunctionTables(
                Table::Vertex,
                tables.iter().map(|t| t.cloned()).collect(),
                first,
            )
        });
        const CALL: &str = "set_vertex_visible_function_tables";
        self.vertex
            .set_function_tables(&mut self.core, CALL, tables, first)
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_visible_function_table`].
    pub(crate) fn set_fragment_visible_function_table(
        &mut self,
        table: Option<&VisibleFunctionTable>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::FunctionTable(Table::Fragment, table.cloned(), index));
        const CALL: &str = "set_fragment_visible_function_table";
        self.fragment
            .set_function_tables(&mut self.core, CALL, &[table], index)
    }

    /// The lowering of [`RenderCommandEncoder::set_fragment_visible_function_tables`].
    pub(crate) fn set_fragment_visible_function_tables(
        &mut self,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::FunctionTables(
                Table::Fragment,
                tables.iter().map(|t| t.cloned()).collect(),
                first,
            )
        });
        const CALL: &str = "set_fragment_visible_function_tables";
        self.fragment
            .set_function_tables(&mut self.core, CALL, tables, first)
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_visible_function_table`].
    pub(crate) fn set_tile_visible_function_table(
        &mut self,
        table: Option<&VisibleFunctionTable>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::FunctionTable(Table::Tile, table.cloned(), index));
        const CALL: &str = "set_tile_visible_function_table";
        self.tile
            .set_function_tables(&mut self.core, CALL, &[table], index)
    }

    /// The lowering of [`RenderCommandEncoder::set_tile_visible_function_tables`].
    pub(crate) fn set_tile_visible_function_tables(
        &mut self,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::FunctionTables(
                Table::Tile,
                tables.iter().map(|t| t.cloned()).collect(),
                first,
            )
        });
        const CALL: &str = "set_tile_visible_function_tables";
        self.tile
            .set_function_tables(&mut self.core, CALL, tables, first)
    }
}

impl RenderCommandEncoder {
    /// `setVertexVisibleFunctionTable:atBufferIndex:` (`None` unbinds the index): the table at
    /// the vertex stage's buffer index `index`, in place of any buffer there, as a compute
    /// encoder's `set_visible_function_table` binds one (`function_table.rs`). Refused: an index
    /// past the buffer table, a table of another device. A draw whose vertex function reads a
    /// buffer at the index, or a table where a buffer is bound, is refused
    /// (`Invalid(FunctionTable)`), as Metal's validation stops it.
    pub fn set_vertex_visible_function_table(
        &mut self,
        table: Option<&VisibleFunctionTable>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_vertex_visible_function_table(table, index))
            }
            false => {
                self.defer(|_| RenderCall::FunctionTable(Table::Vertex, table.cloned(), index))
            }
        }
    }

    /// `setVertexVisibleFunctionTables:withBufferRange:`: `tables[i]` at the vertex stage's
    /// buffer index `first + i`, refused as
    /// [`set_vertex_visible_function_table`](Self::set_vertex_visible_function_table) is; a
    /// refusal binds none of them.
    pub fn set_vertex_visible_function_tables(
        &mut self,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_vertex_visible_function_tables(tables, first))
            }
            false => self.defer(|_| {
                RenderCall::FunctionTables(
                    Table::Vertex,
                    tables.iter().map(|t| t.cloned()).collect(),
                    first,
                )
            }),
        }
    }

    /// `setFragmentVisibleFunctionTable:atBufferIndex:`: as
    /// [`set_vertex_visible_function_table`](Self::set_vertex_visible_function_table), at the
    /// fragment stage's buffer index.
    pub fn set_fragment_visible_function_table(
        &mut self,
        table: Option<&VisibleFunctionTable>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_fragment_visible_function_table(table, index))
            }
            false => {
                self.defer(|_| RenderCall::FunctionTable(Table::Fragment, table.cloned(), index))
            }
        }
    }

    /// `setFragmentVisibleFunctionTables:withBufferRange:`: as
    /// [`set_vertex_visible_function_tables`](Self::set_vertex_visible_function_tables), at the
    /// fragment stage's buffer indices.
    pub fn set_fragment_visible_function_tables(
        &mut self,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self
                .at_call(|lowering| lowering.set_fragment_visible_function_tables(tables, first)),
            false => self.defer(|_| {
                RenderCall::FunctionTables(
                    Table::Fragment,
                    tables.iter().map(|t| t.cloned()).collect(),
                    first,
                )
            }),
        }
    }

    /// `setTileVisibleFunctionTable:atBufferIndex:`: as
    /// [`set_vertex_visible_function_table`](Self::set_vertex_visible_function_table), at the
    /// tile stage's buffer index, which tile dispatches read.
    pub fn set_tile_visible_function_table(
        &mut self,
        table: Option<&VisibleFunctionTable>,
        index: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => self.at_call(|lowering| lowering.set_tile_visible_function_table(table, index)),
            false => self.defer(|_| RenderCall::FunctionTable(Table::Tile, table.cloned(), index)),
        }
    }

    /// `setTileVisibleFunctionTables:withBufferRange:`: as
    /// [`set_vertex_visible_function_tables`](Self::set_vertex_visible_function_tables), at the
    /// tile stage's buffer indices.
    pub fn set_tile_visible_function_tables(
        &mut self,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        match self.lowers_at_call() {
            true => {
                self.at_call(|lowering| lowering.set_tile_visible_function_tables(tables, first))
            }
            false => self.defer(|_| {
                RenderCall::FunctionTables(
                    Table::Tile,
                    tables.iter().map(|t| t.cloned()).collect(),
                    first,
                )
            }),
        }
    }
}
