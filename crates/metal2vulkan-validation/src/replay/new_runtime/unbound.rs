//! The draws and dispatches whose pipeline state reads a buffer argument that nothing is bound
//! to when they run: what Metal leaves undefined, counted, never stood in for. A state's reads
//! are the buffer bindings its reflection lists as used (`isUsed`, the translator's
//! [`ComputePipelineDescriptor::reflection`](rt::ComputePipelineDescriptor::reflection) and
//! [`RenderPipelineDescriptor::reflection`](rt::RenderPipelineDescriptor::reflection)); an
//! encoder's bound indices are those its `setBuffer`, `setBuffers` and `setBytes` calls of each
//! stage set to something since it was made, a `nil` buffer unsetting one, until its
//! `endEncoding`. Every call of an
//! encoder counts, a skipped object's included: what the program's GPU passes reach is the
//! stream's.

use super::super::args::range;
use super::super::stream::Call;
use super::super::value::Value;
use metal2vulkan_rt as rt;
use std::collections::{BTreeSet, HashMap};

/// The stages a buffer argument is read in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::replay) enum Stage {
    Kernel,
    Vertex,
    Fragment,
}

/// The first draws and dispatches with a buffer argument unbound that the line names.
const FIRST: usize = 8;

/// One draw or dispatch that reads a buffer argument unbound: its frame and stream position, its
/// pipeline state, and the first argument unbound.
#[derive(Clone, Copy, Debug)]
struct Read {
    frame: u64,
    position: u64,
    state: u64,
    stage: Stage,
    index: u32,
}

/// An open encoder's pipeline state and the buffer indices bound on it.
#[derive(Default)]
struct Encoder {
    state: Option<u64>,
    bound: BTreeSet<(Stage, u32)>,
}

#[derive(Default)]
pub(in crate::replay) struct Unbound {
    /// Each pipeline state's used buffer arguments.
    reads: HashMap<u64, BTreeSet<(Stage, u32)>>,
    /// The pipeline states the translator gave no reflection of, which nothing is counted for.
    unreflected: u64,
    /// The open encoders.
    encoders: HashMap<u64, Encoder>,
    /// The draws and dispatches that read a buffer argument unbound, and the states they ran.
    count: u64,
    states: BTreeSet<u64>,
    first: Vec<Read>,
}

/// A buffer binding of a reflection's that the function uses.
fn used(bindings: &[metal2vulkan::reflection::Binding], stage: Stage) -> Vec<(Stage, u32)> {
    (bindings.iter())
        .filter(|b| b.is_used() && b.buffer().is_some())
        .map(|b| (stage, b.index()))
        .collect()
}

impl Unbound {
    /// Compute pipeline state `state` of `descriptor` made.
    pub(super) fn compute(&mut self, state: u64, descriptor: &rt::ComputePipelineDescriptor) {
        match descriptor.reflection(&metal2vulkan::Budget::default()) {
            Ok(reflection) => {
                let reads = used(reflection.bindings(), Stage::Kernel);
                self.reads.insert(state, reads.into_iter().collect());
            }
            Err(_) => self.unreflected += 1,
        }
    }

    /// Render pipeline state `state` of `descriptor` made.
    pub(super) fn render(&mut self, state: u64, descriptor: &rt::RenderPipelineDescriptor) {
        match descriptor.reflection(&metal2vulkan::Budget::default()) {
            Ok(reflection) => {
                let mut reads = used(reflection.vertex_bindings(), Stage::Vertex);
                reads.extend(used(reflection.fragment_bindings(), Stage::Fragment));
                self.reads.insert(state, reads.into_iter().collect());
            }
            Err(_) => self.unreflected += 1,
        }
    }

    /// Pipeline state `state` made, reading `reads`: what [`compute`](Self::compute) and
    /// [`render`](Self::render) find in a reflection.
    #[cfg(test)]
    pub(in crate::replay) fn made(&mut self, state: u64, reads: &[(Stage, u32)]) {
        self.reads.insert(state, reads.iter().copied().collect());
    }

    /// A call of encoder `call.receiver`, at `frame` and stream `position`.
    pub(in crate::replay) fn encoded(&mut self, call: &Call, frame: u64, position: u64) {
        let selector = &*call.selector;
        if selector == "endEncoding" {
            self.encoders.remove(&call.receiver);
            return;
        }
        let stage = match selector {
            s if s.starts_with("setVertex") => Stage::Vertex,
            s if s.starts_with("setFragment") => Stage::Fragment,
            _ => Stage::Kernel,
        };
        let encoder = self.encoders.entry(call.receiver).or_default();
        let set = |bound: &mut BTreeSet<(Stage, u32)>, buffer: Option<&Value>, at: u64| {
            let Ok(at) = u32::try_from(at) else {
                return;
            };
            match buffer {
                Some(Value::Nil | Value::Object(0)) => bound.remove(&(stage, at)),
                _ => bound.insert((stage, at)),
            };
        };
        let uint = |at: usize| match call.args.get(at) {
            Some(Value::Uint(v)) => Some(*v),
            _ => None,
        };
        match selector {
            "setComputePipelineState:" | "setRenderPipelineState:" => {
                encoder.state = match call.args.first() {
                    Some(Value::Object(state)) if *state != 0 => Some(*state),
                    _ => None,
                };
            }
            "setBuffer:offset:atIndex:"
            | "setVertexBuffer:offset:atIndex:"
            | "setFragmentBuffer:offset:atIndex:" => {
                if let Some(at) = uint(2) {
                    set(&mut encoder.bound, call.args.first(), at);
                }
            }
            "setBytes:length:atIndex:"
            | "setVertexBytes:length:atIndex:"
            | "setFragmentBytes:length:atIndex:" => {
                if let Some(at) = uint(2) {
                    set(&mut encoder.bound, None, at);
                }
            }
            "setBuffers:offsets:withRange:"
            | "setVertexBuffers:offsets:withRange:"
            | "setFragmentBuffers:offsets:withRange:" => {
                let (Some(Value::List(buffers)), Some(span)) =
                    (call.args.first(), call.args.get(2))
                else {
                    return;
                };
                let Ok(span) = range(call, span) else {
                    return;
                };
                for (at, buffer) in span.zip(buffers) {
                    set(&mut encoder.bound, Some(buffer), at);
                }
            }
            s if s.starts_with("draw") || s.starts_with("dispatch") => {
                let Some(state) = encoder.state else {
                    return;
                };
                let Some(reads) = self.reads.get(&state) else {
                    return;
                };
                let Some(&(stage, index)) = reads.difference(&encoder.bound).next() else {
                    return;
                };
                self.count += 1;
                self.states.insert(state);
                if self.first.len() < FIRST {
                    self.first.push(Read {
                        frame,
                        position,
                        state,
                        stage,
                        index,
                    });
                }
            }
            _ => {}
        }
    }

    /// The draws and dispatches that read a buffer argument unbound so far.
    pub(super) fn count(&self) -> u64 {
        self.count
    }

    /// The `unbound buffer reads N: ...` line.
    pub(in crate::replay) fn line(&self) -> String {
        let first: Vec<String> = (self.first.iter())
            .map(|r| {
                format!(
                    "position {} frame {} state #{} {:?} buffer {}",
                    r.position, r.frame, r.state, r.stage, r.index
                )
            })
            .collect();
        format!(
            "unbound buffer reads {}: draws and dispatches whose pipeline state reads a buffer \
             argument nothing is bound to, of {} states; {} states unreflected; first: {}",
            self.count,
            self.states.len(),
            self.unreflected,
            match first.is_empty() {
                true => "none".to_owned(),
                false => first.join(", "),
            }
        )
    }
}
