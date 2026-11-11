//! The pass log: a default-off diagnostic, `METAL2VULKAN_LOG_PASSES` (`metal2vulkan::env_vars`),
//! that names each render pass that renders into a texture it watches, binds one to its fragment
//! stage or makes one resident: what such a pass does with its attachments, and what its draws
//! read.
//!
//! - **What it watches.** The switch's value: a comma-separated list of `FORMAT:WIDTHxHEIGHT`,
//!   for example `RGBA16Float:128x128,BGRA8Unorm:224x128`. `FORMAT` is a pixel format's name as
//!   the notices print it (`RGBA16Float`, `BGRA8UnormSRGB`) or as Metal names it
//!   (`MTLPixelFormatRGBA16Float`, `BGRA8Unorm_sRGB`); `*` stands for any format, and a size of
//!   `*` for any size. Unset, empty or `0`, the log is off; a value it cannot read turns it off
//!   and says why once, on stderr.
//! - **What it prints.** At the end of a render encoder that names a watched texture, one line
//!   through the device's message sink at `Info` severity (id name `metal2vulkan-passes`), else
//!   on stderr: the render area, its Vulkan render pass instances (more than one where the pass
//!   split) and its draw calls; each color attachment (index, format, size, object ID and label,
//!   whether its texture took its storage form, level and slice, load and store actions as the
//!   pass ends, and the stand-in it renders through, with its cause); the depth/stencil
//!   attachment with each aspect's actions; the pipeline states its draws drew with (label and
//!   functions); the textures bound to the fragment stage at a draw (index, format, size, object
//!   ID and label); the count of textures it made resident and the watched ones among them, with
//!   their stages; and what it watched. A line lists at most [`KEPT`] states and fragment
//!   textures, each once.
//! - **Its volume.** A line the same as one the process printed before is not printed again, so
//!   a pass a frame repeats prints once; and the process prints at most [`LINES`] lines, the last
//!   saying so.
//! - **Off, it costs nothing.** The switch is read once; an encoder notes nothing and no line is
//!   formatted. On, every render encoder notes its draws' states and fragment textures, and one
//!   that names a watched texture formats its line.

use super::device::lock;
use super::driver_facts::inform_through;
use super::render_pass::{AspectTarget, PassLayout};
use super::residency::Residency;
use super::{Device, RenderPipelineState, Texture};
use metal2vulkan::metal::{PixelFormat, RenderStages};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Mutex, OnceLock};

/// The id name of the log's messages.
const ID_NAME: &str = "metal2vulkan-passes";

/// The most lines a process prints.
pub(crate) const LINES: usize = 256;

/// The most pipeline states, and the most fragment textures, a pass's line lists.
const KEPT: usize = 32;

/// A texture the log watches: of `format` and `size`, `None` for any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Watched {
    pub(crate) format: Option<PixelFormat>,
    pub(crate) size: Option<[u64; 2]>,
}

impl Watched {
    /// Whether it watches a texture of `format`, `width` by `height`.
    pub(crate) fn watches(&self, format: PixelFormat, [width, height]: [u64; 2]) -> bool {
        self.format.is_none_or(|watched| watched == format)
            && self.size.is_none_or(|size| size == [width, height])
    }
}

/// The textures `value` watches (the module docs), or why it names none.
pub(crate) fn parse(value: &str) -> Result<Vec<Watched>, String> {
    value.split(',').map(str::trim).map(entry).collect()
}

/// The texture one entry of the switch's list watches.
fn entry(entry: &str) -> Result<Watched, String> {
    let (format, size) = entry
        .split_once(':')
        .ok_or_else(|| format!("`{entry}` is not FORMAT:WIDTHxHEIGHT"))?;
    let format = match format {
        "*" => None,
        name => {
            Some(format_named(name).ok_or_else(|| format!("`{entry}`: no pixel format `{name}`"))?)
        }
    };
    let size = match size {
        "*" => None,
        size => {
            let side = |side: &str| side.parse::<u64>().ok().filter(|&side| side > 0);
            let parsed = size
                .split_once('x')
                .and_then(|(width, height)| Some([side(width)?, side(height)?]));
            Some(parsed.ok_or_else(|| format!("`{entry}`: `{size}` is not WIDTHxHEIGHT"))?)
        }
    };
    Ok(Watched { format, size })
}

/// The pixel format named `name`: as the notices print it (its `Debug` name), or as Metal names
/// it, with or without its `MTLPixelFormat` prefix.
fn format_named(name: &str) -> Option<PixelFormat> {
    PixelFormat::ALL
        .iter()
        .copied()
        .find(|&format| {
            format!("{format:?}") == name
                || format.name().strip_prefix("MTLPixelFormat") == Some(name)
        })
        .or_else(|| PixelFormat::from_name(name))
}

/// The textures the log watches, read once: none while it is off.
fn watched() -> &'static [Watched] {
    static WATCHED: OnceLock<Vec<Watched>> = OnceLock::new();
    WATCHED.get_or_init(|| {
        let Some(value) = metal2vulkan::env_vars::log_passes() else {
            return Vec::new();
        };
        let value = value.to_string_lossy();
        if value.is_empty() || value == "0" {
            return Vec::new();
        }
        parse(&value).unwrap_or_else(|why| {
            eprintln!("metal2vulkan: METAL2VULKAN_LOG_PASSES={value}: {why}; the pass log is off");
            Vec::new()
        })
    })
}

/// Whether the log is on.
pub(crate) fn enabled() -> bool {
    !watched().is_empty()
}

/// Whether the log watches `texture`.
fn watches(texture: &Texture) -> bool {
    let size = [texture.width(), texture.height()];
    (watched().iter()).any(|watched| watched.watches(texture.pixel_format(), size))
}

/// What a render encoder's draws named, for its line at its end, while the log is on.
#[derive(Default)]
pub(crate) struct PassNotes {
    /// Its draw calls.
    draws: u64,
    /// Each pipeline state its draws drew with, once, in the order they first did.
    states: Vec<RenderPipelineState>,
    /// Each texture bound to the fragment stage at a draw, once by index and texture.
    fragment: Vec<(u32, Texture)>,
}

impl PassNotes {
    /// Note a draw call with `state`, `fragment` bound to its fragment stage by index.
    pub(crate) fn drew(
        &mut self,
        state: Option<&RenderPipelineState>,
        fragment: &BTreeMap<u32, Texture>,
    ) {
        self.draws += 1;
        if let Some(state) = state {
            if self.states.len() < KEPT && !self.states.contains(state) {
                self.states.push(state.clone());
            }
        }
        for (&index, texture) in fragment {
            let id = texture.object_id();
            let noted = (self.fragment.iter()).any(|(i, t)| *i == index && t.object_id() == id);
            if !noted && self.fragment.len() < KEPT {
                self.fragment.push((index, texture.clone()));
            }
        }
    }
}

/// A texture as a line names it: its format, size, object ID and label, and whether its own
/// image took its storage form (`texture/storage_form.rs`).
fn named(texture: &Texture) -> String {
    let label = texture
        .label()
        .map_or_else(String::new, |label| format!(" {label:?}"));
    let storage = match texture.took_storage_form() {
        true => ", its storage form",
        false => "",
    };
    format!(
        "{:?} {}x{} object {}{label}{storage}",
        texture.pixel_format(),
        texture.width(),
        texture.height(),
        texture.object_id().get(),
    )
}

/// A pipeline state as a line names it: its label and its functions' names.
fn state_named(state: &RenderPipelineState) -> String {
    let label = state
        .label()
        .map_or_else(String::new, |label| format!("{label:?} "));
    format!("{label}({})", state.function_names().join(", "))
}

/// At the end of a render encoder whose draws `notes` noted, rendering into `layout` in
/// `instances` Vulkan render pass instances with `residency` made resident: its line, where it
/// names a watched texture and the process printed no line the same (the module docs).
pub(crate) fn ended(
    device: &Device,
    notes: &PassNotes,
    layout: &PassLayout,
    instances: usize,
    residency: &Residency,
) {
    let mut seen = Vec::new();
    let mut colors = Vec::new();
    for (index, color) in layout.colors.iter().enumerate() {
        let Some(color) = color else {
            continue;
        };
        let target = &color.target;
        if watches(&target.texture) {
            seen.push(format!("color {index}"));
        }
        let stand_in = color
            .stand_in
            .as_ref()
            .map_or_else(String::new, |stand_in| {
                format!(", through a stand-in ({:?})", stand_in.cause)
            });
        colors.push(format!(
            "{index}: {} level {} slice {}, load {:?}, store {:?}{stand_in}",
            named(&target.texture),
            target.level,
            target.layer,
            color.load,
            color.store,
        ));
    }
    let depth_stencil = match layout.depth_stencil.as_ref().filter(|ds| !ds.hidden) {
        Some(ds) => {
            if watches(&ds.target.texture) {
                seen.push("the depth/stencil attachment".to_owned());
            }
            let aspect = |name: &str, aspect: Option<&AspectTarget>| {
                aspect.map_or_else(String::new, |aspect| {
                    format!(", {name} load {:?} store {:?}", aspect.load, aspect.store)
                })
            };
            format!(
                "{}{}{}",
                named(&ds.target.texture),
                aspect("depth", ds.depth.as_ref()),
                aspect("stencil", ds.stencil.as_ref()),
            )
        }
        None => "none".to_owned(),
    };
    let mut fragment = Vec::new();
    for (index, texture) in &notes.fragment {
        if watches(texture) {
            seen.push(format!("fragment texture {index}"));
        }
        fragment.push(format!("{index}: {}", named(texture)));
    }
    let mut resident = 0;
    let mut resident_watched = Vec::new();
    for (texture, stages) in residency.textures_made_in() {
        resident += 1;
        if watches(texture) {
            resident_watched.push(format!("{} in {}", named(texture), stages_named(stages)));
        }
    }
    if !resident_watched.is_empty() {
        seen.push(format!("{} resident texture(s)", resident_watched.len()));
    }
    if seen.is_empty() {
        return;
    }
    let states: Vec<String> = notes.states.iter().map(state_named).collect();
    let [width, height] = layout.extent;
    let mut text = format!(
        "metal2vulkan: pass: {width}x{height} in {instances} Vulkan render pass instance(s), {} \
         draw call(s); colors [{}]; depth/stencil {depth_stencil}; states [{}]; fragment \
         textures [{}]; {resident} texture(s) resident, watched among them [{}]; it watched {}",
        notes.draws,
        colors.join("; "),
        states.join(", "),
        fragment.join("; "),
        resident_watched.join("; "),
        seen.join(", "),
    );
    if fresh(&mut text) {
        inform_through(device.shared().instance.sink(), ID_NAME, text);
    }
}

/// The render stages `stages` holds, by name.
fn stages_named(stages: RenderStages) -> String {
    let names = [
        (RenderStages::VERTEX, "vertex"),
        (RenderStages::FRAGMENT, "fragment"),
        (RenderStages::TILE, "tile"),
        (RenderStages::OBJECT, "object"),
        (RenderStages::MESH, "mesh"),
    ];
    let named: Vec<&str> = (names.iter())
        .filter(|(stage, _)| stages.contains(*stage))
        .map(|&(_, name)| name)
        .collect();
    match named.is_empty() {
        true => "no stage".to_owned(),
        false => named.join(" and "),
    }
}

/// Whether to print `text`: the process printed no line the same and fewer than [`LINES`]; the
/// last line it prints says so (the module docs).
fn fresh(text: &mut String) -> bool {
    static PRINTED: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
    fresh_in(&PRINTED, text)
}

/// [`fresh`] against the lines `printed` holds, by hash.
pub(crate) fn fresh_in(printed: &Mutex<BTreeSet<u64>>, text: &mut String) -> bool {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let mut printed = lock(printed);
    if printed.len() >= LINES || !printed.insert(hasher.finish()) {
        return false;
    }
    if printed.len() == LINES {
        text.push_str(&format!(
            "; the pass log's last line (it prints at most {LINES})"
        ));
    }
    true
}
