//! The rt side's visible-function-table call: a compute state's handles, its tables, their entries
//! and a dispatch through one, as the runtime's public API makes them. A render state's stage
//! makes no table on the runtime, so a state it makes answers the next call unavailable.

use super::{waited_for, RtSide, WAIT};
use crate::case::{Answer, Failure, Handle, Side};
use crate::transcript::{Negative, Value};
use crate::visible_tables::{
    Asked, Binding, Entry, HandleFacts, LinkedList, Made, Maker, TableFacts, TableOutcome,
    TableSpec, Writes,
};
use metal2vulkan::pipeline::LinkedFunctions;
use metal2vulkan::Budget;
use metal2vulkan_runtime as rt;

/// The buffer index the fixture's kernel takes its table at.
const TABLE_INDEX: u32 = 1;

impl RtSide<'_> {
    /// [`crate::case::Side::visible_function_table`]: the runtime's state of `spec.entry` of
    /// `caller`, linking `spec.linked`'s functions of `functions` in the order the spec states,
    /// made as the case's maker makes it: a compute state, or a drawing state whose vertex
    /// function is the entry, with no fragment function and rasterization off. The runtime's
    /// refusal of the state is the answer. A compute state answers the case
    /// ([`compute_table`](Self::compute_table)); a drawing state answers
    /// `functionHandleWithFunction:stage:` unavailable, and a tile stage the tile state, as the
    /// runtime makes neither.
    pub(super) fn table(
        &mut self,
        caller: Handle,
        functions: Handle,
        spec: &TableSpec<'_>,
    ) -> Answer<TableOutcome> {
        let on = self.on;
        let translated = |error: metal2vulkan::error::Error| on.translation_failure(&error);
        let defining = self.held_library(functions)?.clone();
        let mut linked = LinkedFunctions::new();
        for (name, list) in spec.linked.functions {
            let function = defining.function(name).map_err(translated)?;
            linked = match list {
                LinkedList::Functions => linked.with_function(function),
                LinkedList::PrivateFunctions => linked.with_private_function(function),
            };
        }
        let entry = (self.held_library(caller)?.function(spec.entry)).map_err(translated)?;
        let stage =
            rt::StageSource::Function(rt::FunctionDescriptor::new(entry).with_linked(linked));
        let next = match spec.maker {
            Maker::Compute => {
                let descriptor = rt::ComputePipelineDescriptor::new(stage);
                let state = (on.device)
                    .new_compute_pipeline_state(
                        &descriptor,
                        |samplers| Ok(on.sampler_ids(samplers)),
                    )
                    .map_err(|error| on.failure(&error))?;
                return self.compute_table(&state, &defining, spec);
            }
            Maker::DrawingStage(_) => {
                let mut descriptor = rt::RenderPipelineDescriptor::new(stage);
                descriptor.rasterization_enabled = false;
                (on.device)
                    .new_render_pipeline_state(&descriptor, |samplers| Ok(on.sampler_ids(samplers)))
                    .map_err(|error| on.failure(&error))?;
                "functionHandleWithFunction:stage:"
            }
            Maker::TileStage(_) => "tile_pipeline",
        };
        Err(Failure::unavailable(next))
    }

    /// The handles `state` answers for `spec.handles` of `defining`, the table it makes and the
    /// entries the case writes, and the words a dispatch of `spec.threads` single-thread
    /// threadgroups writes through it. A handle's `name` and `device` are its consumer's answers:
    /// the runtime's handle is the word a table entry holds, and the consumer names the function
    /// it asked of and answers its own device. So is a table's `device`.
    fn compute_table(
        &mut self,
        state: &rt::ComputePipelineState,
        defining: &rt::Library,
        spec: &TableSpec<'_>,
    ) -> Answer<TableOutcome> {
        let on = self.on;
        let translated = |error: metal2vulkan::error::Error| on.translation_failure(&error);
        let budget = Budget::standard();
        let mut handles = Vec::new();
        let mut function_types = Vec::new();
        let mut held = Vec::new();
        for name in spec.handles {
            let function = defining.function(name).map_err(translated)?;
            let properties = function.properties(&budget).map_err(translated)?;
            function_types.push(properties.function_type());
            let function = rt::LinkedFunction::new(function);
            let handle = state.function_handle(&function);
            handles.push(Asked::answered(handle.map(|_| HandleFacts {
                name: function.name().to_owned(),
                device_is_the_cases: true,
            })));
            held.push((*name, handle));
        }
        let handle_of = |name: &str| {
            (held.iter())
                .find(|(of, _)| *of == name)
                .and_then(|(_, handle)| *handle)
        };
        let wanted = rt::VisibleFunctionTableDescriptor::new(u64::from(spec.function_count));
        let descriptor_function_count = wanted.function_count;
        let table = (state.new_visible_function_table(&wanted)).map_err(|error| on.failure(&error));
        let second = (spec.second_table).then(|| state.new_visible_function_table(&wanted));
        let made = match &table {
            Ok(table) => Made::Table(TableFacts {
                resource_id_is_non_zero: table.gpu_resource_id() != 0,
                a_second_tables_id_differs: second.as_ref().map(|second| {
                    (second.as_ref()).is_ok_and(|s| s.gpu_resource_id() != table.gpu_resource_id())
                }),
                device_is_the_cases: true,
            }),
            Err(failure) => Made::Refused(failure.clone()),
        };
        let taken: Vec<Option<rt::FunctionHandle>> = (spec.entries.iter())
            .map(|entry| match entry {
                Entry::Function(name) => handle_of(name),
                Entry::Nil => None,
            })
            .collect();
        let entries_written = match (&table, spec.writes) {
            (Err(_), _) => Ok(()),
            (Ok(table), Writes::OneAtATime) => (0..)
                .zip(&taken)
                .try_for_each(|(at, handle)| table.set_function(*handle, at)),
            (Ok(table), Writes::WholeRange) => table.set_functions(&taken, 0),
        }
        .map_err(|error| on.failure(&error));
        let Some(threads) = spec.threads else {
            return Ok(TableOutcome::read_only(
                handles,
                descriptor_function_count,
                function_types,
                made,
            ));
        };
        let out = self.zeroed_buffer(threads as usize * 4)?;
        let buffer = self.held(out)?.clone();
        let of_rt = |error: rt::Error| Failure::of_rt(&error);
        let command_buffer = self.queue()?.command_buffer().map_err(of_rt)?;
        let mut encoder =
            (command_buffer.compute_command_encoder(rt::DispatchType::Serial)).map_err(of_rt)?;
        encoder.set_compute_pipeline_state(state).map_err(of_rt)?;
        encoder.set_buffer(Some(&buffer), 0, 0).map_err(of_rt)?;
        if let (Binding::Bound, Ok(table)) = (spec.binding, &table) {
            (encoder.set_visible_function_table(Some(table), TABLE_INDEX)).map_err(of_rt)?;
        }
        (encoder.dispatch_threadgroups([threads, 1, 1], [1, 1, 1])).map_err(of_rt)?;
        encoder.end_encoding();
        command_buffer.commit().map_err(of_rt)?;
        let status = match waited_for(&command_buffer, WAIT) {
            Some(word) => Value::unit_variant(word),
            None => Value::Nothing(Negative::NeverReturned),
        };
        let bytes = self.read_buffer(out, 0, threads as usize * 4)?;
        let words = (bytes.as_chunks::<4>().0.iter())
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        Ok(TableOutcome {
            handles,
            descriptor_function_count,
            function_types,
            table: made,
            entries_written,
            words,
            status,
        })
    }
}
