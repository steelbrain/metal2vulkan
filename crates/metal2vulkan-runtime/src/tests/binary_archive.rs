//! Binary archives (`binary_archive`): the file's encoding and what is not one, the file errors
//! Metal answers, each render stage's archive key and the compute key against the fields measured
//! on an M3 Max and an M4 as keyed or not, the mesh key's every field, how a lookup combines the
//! stages, and a metallib loaded as an archive holding nothing. Each file test
//! owns an empty directory it removes ([`Scratch`]); the functions are the committed AIR of the
//! owned `ob3_*` fixtures.

use crate::binary_archive::container::{self, Contents, Place};
use crate::binary_archive::{self, metal_key};
use crate::error::{BinaryArchiveError, Error};
use crate::shader_cache::EntryKind;
use metal2vulkan::cache::Hash256;
use metal2vulkan::library::{Function, FunctionConstantValue, FunctionConstantValues, Library};
use metal2vulkan::metal::{
    AttributeFormat, BlendFactor, ColorWriteMask, DataType, IndexType, Mutability, PixelFormat,
    PrimitiveTopologyClass, StepFunction, TessellationControlPointIndexType,
    TessellationFactorStepFunction, TessellationPartitionMode, VertexFormat, Winding,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, FunctionDescriptor, LinkedFunctions,
    RenderPipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource, TessellationDescriptor, TileRenderPipelineDescriptor,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A directory a test owns: made empty, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "m2v-binary-archive-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    /// The names in the directory, sorted.
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(&self.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Contents of two keys and three entries.
fn contents() -> Contents {
    let mut contents = Contents::default();
    contents.keys.insert([2; 32]);
    contents.keys.insert([1; 32]);
    for (kind, key, bytes) in [
        (EntryKind::RenderPipeline, [9; 32], b"render".as_slice()),
        (EntryKind::Function, [3; 32], b"function".as_slice()),
        (EntryKind::ComputePipeline, [3; 32], b"".as_slice()),
    ] {
        contents
            .entries
            .insert(Place::new(kind, key, [7; 32]), Arc::from(bytes));
    }
    contents
}

/// `body` and its checksum.
fn sealed(body: &[u8]) -> Vec<u8> {
    let mut out = body.to_vec();
    let checksum: Hash256 = Sha256::digest(body).into();
    out.extend_from_slice(&checksum);
    out
}

/// The body of `contents`' encoding (without its checksum).
fn body(contents: &Contents) -> Vec<u8> {
    let bytes = container::encode(contents);
    bytes[..bytes.len() - 32].to_vec()
}

#[test]
fn an_archive_round_trips_through_its_encoding() {
    let contents = contents();
    let bytes = container::encode(&contents);
    assert_eq!(&bytes[..8], b"M2VBARCH");
    assert_eq!(container::decode(&bytes), Some(contents));
    assert_eq!(
        container::decode(&container::encode(&Contents::default())),
        Some(Contents::default())
    );
}

#[test]
fn bytes_that_are_not_exactly_an_archive_are_refused() {
    let good = body(&contents());
    let mut refused: Vec<(&str, Vec<u8>)> = vec![
        ("nothing", Vec::new()),
        ("shorter than a checksum", vec![0; 31]),
        ("a checksum alone", sealed(&[])),
        ("cut short", sealed(&good[..good.len() - 1])),
        (
            "bytes after the end",
            sealed(&[good.as_slice(), &[0]].concat()),
        ),
    ];
    let mut flipped = container::encode(&contents());
    flipped[20] ^= 1;
    refused.push(("a checksum that does not hold", flipped));
    let mut magic = good.clone();
    magic[0] = b'X';
    refused.push(("another magic", sealed(&magic)));
    let mut version = good.clone();
    version[8] = 2;
    refused.push(("another version", sealed(&version)));
    // The first key's 32 bytes start at 16; the first entry's kind after both keys and the count.
    let mut keys = good.clone();
    keys[16..48].copy_from_slice(&[2; 32]);
    refused.push(("keys repeated", sealed(&keys)));
    let mut kind = good.clone();
    kind[84..88].copy_from_slice(&3u32.to_le_bytes());
    refused.push(("the driver's kind", sealed(&kind)));
    let mut count = good.clone();
    count[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    refused.push(("a key count past the end", sealed(&count)));
    let mut entries = good.clone();
    entries[80..84].copy_from_slice(&u32::MAX.to_le_bytes());
    refused.push(("an entry count past the end", sealed(&entries)));
    // Entries swapped: the second entry first.
    let mut order = Contents::default();
    order.entries.insert(
        Place::new(EntryKind::Function, [5; 32], [7; 32]),
        Arc::from(&b"a"[..]),
    );
    let one = body(&order);
    let mut swapped = Contents::default();
    swapped.entries.insert(
        Place::new(EntryKind::Function, [4; 32], [7; 32]),
        Arc::from(&b"b"[..]),
    );
    let other = body(&swapped);
    let header = 20;
    let mut out_of_order = one[..header].to_vec();
    out_of_order[16..20].copy_from_slice(&2u32.to_le_bytes());
    out_of_order.extend_from_slice(&one[header..]);
    out_of_order.extend_from_slice(&other[header..]);
    refused.push(("entries out of order", sealed(&out_of_order)));
    for (what, bytes) in refused {
        assert_eq!(container::decode(&bytes), None, "{what}");
    }
}

fn invalid_file(path: &Path) -> Error {
    Error::BinaryArchive(BinaryArchiveError::InvalidFile {
        path: path.to_path_buf(),
    })
}

#[test]
fn a_file_that_is_missing_or_not_an_archive_is_an_invalid_file() {
    let scratch = Scratch::new();
    let dir = scratch.path();
    fs::write(dir.join("empty"), b"").unwrap();
    fs::write(dir.join("text"), b"not an archive\n").unwrap();
    fs::write(dir.join("garbage"), [0xa5; 4096]).unwrap();
    fs::create_dir(dir.join("directory")).unwrap();
    for name in ["missing", "empty", "text", "garbage", "directory"] {
        let path = dir.join(name);
        assert_eq!(container::read(&path), Err(invalid_file(&path)), "{name}");
    }
}

#[test]
fn writing_replaces_the_file_whole_and_leaves_nothing_else() {
    let scratch = Scratch::new();
    let path = scratch.path().join("archive.bin");
    fs::write(&path, b"an older file").unwrap();
    let contents = contents();
    container::write(&container::encode(&contents), &path).unwrap();
    assert_eq!(container::read(&path), Ok(contents));
    assert_eq!(
        scratch.names(),
        ["archive.bin"],
        "no temporary file is left"
    );
}

#[test]
fn writing_into_a_missing_directory_or_onto_a_directory_fails_as_metal_does() {
    let scratch = Scratch::new();
    let bytes = container::encode(&contents());
    let missing = scratch.path().join("missing/archive.bin");
    assert_eq!(
        container::write(&bytes, &missing),
        Err(invalid_file(&missing))
    );
    let directory = scratch.path().join("directory");
    fs::create_dir(&directory).unwrap();
    match container::write(&bytes, &directory) {
        Err(Error::BinaryArchive(BinaryArchiveError::Write { path, kind, .. })) => {
            assert_eq!((path, kind), (directory, std::io::ErrorKind::IsADirectory));
        }
        other => panic!("onto a directory: {other:?}"),
    }
    assert_eq!(scratch.names(), ["directory"], "no temporary file is left");
}

/// The function `name` of the committed AIR of fixture `stem`, its library made anew.
fn function(stem: &str, name: &str) -> Function {
    let air = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{stem}.air"));
    Library::from_air(&fs::read(air).expect("the committed AIR"))
        .expect("library")
        .function(name)
        .expect("function")
}

fn kernel() -> FunctionDescriptor {
    FunctionDescriptor::new(function("ob3_texture_array", "ob3_texture_array"))
}

fn compute(kernel: FunctionDescriptor) -> ComputePipelineDescriptor {
    ComputePipelineDescriptor::new(StageSource::Function(kernel))
}

const PROFILE: Hash256 = [7; 32];

/// Whether each change of `base` keeps its key (`true`) or changes it, as measured.
fn compute_keys(
    base: &ComputePipelineDescriptor,
    cases: Vec<(&str, bool, ComputePipelineDescriptor)>,
) {
    let key = metal_key::compute(&PROFILE, base);
    for (what, kept, changed) in cases {
        assert_eq!(
            metal_key::compute(&PROFILE, &changed) == key,
            kept,
            "{what}"
        );
    }
}

/// A stage-input layout of `stride` stepping by thread position in grid X.
fn grid_x(stride: Option<u32>) -> StageInputLayoutDescriptor {
    let mut layout = StageInputLayoutDescriptor::new(stride);
    layout.step_function = StepFunction::ThreadPositionInGridX;
    layout
}

#[test]
fn the_compute_key_holds_the_fields_metal_keys() {
    let base = compute(kernel());
    let with = |change: &dyn Fn(&mut ComputePipelineDescriptor)| {
        let mut changed = base.clone();
        change(&mut changed);
        changed
    };
    let mut input = StageInputDescriptor::new();
    input.attributes.insert(
        0,
        StageInputAttributeDescriptor::new(AttributeFormat::Float, 0, 1),
    );
    input.layouts.insert(1, grid_x(Some(4)));
    let with_input = with(&|d| d.stage_input = Some(input.clone()));
    compute_keys(
        &base,
        vec![
            ("the library made again", true, compute(kernel())),
            (
                "a threadgroup size multiple of the width",
                true,
                with(&|d| d.threadgroup_size_is_multiple_of_execution_width = true),
            ),
            (
                "an empty stage input descriptor",
                true,
                with(&|d| d.stage_input = Some(StageInputDescriptor::new())),
            ),
            (
                "an undeclared constant's value",
                true,
                with(&|d| {
                    if let StageSource::Function(kernel) = &mut d.kernel {
                        kernel.constants = FunctionConstantValues::defaults().with_index(
                            3,
                            FunctionConstantValue::new(DataType::Int, vec![1, 0, 0, 0]),
                        );
                    }
                }),
            ),
            (
                "empty linked functions",
                true,
                with(&|d| {
                    if let StageSource::Function(kernel) = &mut d.kernel {
                        kernel.linked = LinkedFunctions::new();
                    }
                }),
            ),
            (
                "another kernel",
                false,
                compute(FunctionDescriptor::new(function(
                    "ob3_stage_input",
                    "ob3_stage_input",
                ))),
            ),
            (
                "the threadgroup limit",
                false,
                with(&|d| d.max_total_threads_per_threadgroup = Some(64)),
            ),
            (
                "indirect command buffers",
                false,
                with(&|d| d.support_indirect_command_buffers = true),
            ),
            ("a stage input descriptor", false, with_input.clone()),
            (
                "binary-function support",
                false,
                with(&|d| d.support_adding_binary_functions = true),
            ),
            (
                "a required threadgroup size",
                false,
                with(&|d| d.required_threads_per_threadgroup = Some([8, 8, 1])),
            ),
            (
                "an all-zero required size, which is none",
                true,
                with(&|d| d.required_threads_per_threadgroup = Some([0; 3])),
            ),
            (
                "a buffer set mutable",
                false,
                with(&|d| {
                    d.buffers.insert(5, Mutability::Mutable);
                }),
            ),
            (
                "a buffer set immutable",
                true,
                with(&|d| {
                    d.buffers.insert(5, Mutability::Immutable);
                }),
            ),
            (
                "a buffer set to the default",
                true,
                with(&|d| {
                    d.buffers.insert(5, Mutability::Default);
                }),
            ),
        ],
    );
    let mut index_type = input.clone();
    index_type.index_type = IndexType::UInt32;
    let mut index_buffer = input.clone();
    index_buffer.index_buffer_index = 5;
    let mut unused = input.clone();
    unused.attributes.insert(
        7,
        StageInputAttributeDescriptor::new(AttributeFormat::Int, 8, 1),
    );
    let mut layout = input.clone();
    layout.layouts.insert(1, grid_x(Some(8)));
    let mut dynamic = input.clone();
    dynamic.layouts.insert(1, grid_x(None));
    // Metal's unset layout (stride 0, per vertex, step rate 1) is every layout Metal's
    // descriptor holds unset, so one listed is none; a stride-0 layout stepping by grid X is
    // not, and Metal's serializer writes its step.
    let mut unset = input.clone();
    let mut unset_layout = grid_x(Some(0));
    (unset_layout.step_function, unset_layout.step_rate) = (StepFunction::PerVertex, 1);
    unset.layouts.insert(5, unset_layout);
    let mut stepped = input.clone();
    stepped.layouts.insert(5, grid_x(Some(0)));
    compute_keys(
        &with_input,
        vec![
            (
                "the index type",
                true,
                with(&|d| d.stage_input = Some(index_type.clone())),
            ),
            (
                "the index buffer, unused",
                false,
                with(&|d| d.stage_input = Some(index_buffer.clone())),
            ),
            (
                "an unused attribute",
                false,
                with(&|d| d.stage_input = Some(unused.clone())),
            ),
            (
                "a layout's stride",
                false,
                with(&|d| d.stage_input = Some(layout.clone())),
            ),
            (
                "a layout's dynamic stride",
                false,
                with(&|d| d.stage_input = Some(dynamic.clone())),
            ),
            (
                "Metal's unset layout",
                true,
                with(&|d| d.stage_input = Some(unset.clone())),
            ),
            (
                "a stride-0 layout stepping by grid X",
                false,
                with(&|d| d.stage_input = Some(stepped.clone())),
            ),
        ],
    );
    assert_ne!(
        metal_key::compute(&[8; 32], &base),
        metal_key::compute(&PROFILE, &base),
        "another device's profile"
    );
}

#[test]
fn linked_functions_are_keyed_as_sets_of_each_list() {
    let a = function("ob3_texture_array", "ob3_texture_array");
    let b = function("ob3_stage_input", "ob3_stage_input");
    let linked = |linked: LinkedFunctions| compute(kernel().with_linked(linked));
    let key = |d: &ComputePipelineDescriptor| metal_key::compute(&PROFILE, d);
    let ab = key(&linked(
        LinkedFunctions::new()
            .with_function(a.clone())
            .with_function(b.clone()),
    ));
    let ba = key(&linked(
        LinkedFunctions::new()
            .with_function(b.clone())
            .with_function(a.clone()),
    ));
    assert_eq!(ab, ba, "the order changes nothing");
    let private = key(&linked(
        LinkedFunctions::new()
            .with_function(a.clone())
            .with_private_function(b.clone()),
    ));
    assert_ne!(ab, private, "a private function is not a function");
    let one = key(&linked(LinkedFunctions::new().with_function(a.clone())));
    assert_ne!(ab, one, "an added function");
    let mut grouped = LinkedFunctions::new()
        .with_function(a.clone())
        .with_function(b.clone());
    grouped.groups.insert("g".into(), vec![a.clone().into()]);
    assert_ne!(ab, key(&linked(grouped)), "a group");
    let mut binary = LinkedFunctions::new().with_function(a);
    binary.binary_functions.push(b.into());
    assert_ne!(ab, key(&linked(binary)), "a binary function");
    assert_ne!(ab, key(&compute(kernel())), "no linked functions");
}

/// A render descriptor of the vertex function of fixture `stem` into an `RGBA8Unorm` target,
/// without a fragment function.
fn vertex(stem: &str) -> RenderPipelineDescriptor {
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        function(stem, stem),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    d
}

/// A fragment stage of the function of fixture `stem`: a key does not decode its function, so
/// any function stands for one.
fn fragment(stem: &str) -> StageSource {
    StageSource::Function(FunctionDescriptor::new(function(stem, stem)))
}

/// `ob3_index_buffer`'s vertex function and a fragment stage into an `RGBA8Unorm` target.
fn render() -> RenderPipelineDescriptor {
    let mut d = vertex("ob3_index_buffer");
    d.fragment = Some(fragment("ob3_vertex_input"));
    d
}

/// Whether each change of `base` keeps its fragment keys and its vertex keys (`true`) or
/// changes them, as measured; each key beside either kind of vertex function.
fn stage_keys(
    base: &RenderPipelineDescriptor,
    cases: Vec<(&str, (bool, bool), RenderPipelineDescriptor)>,
) {
    let keys = metal_key::render(&PROFILE, base);
    let fragment = keys.fragment.expect("a fragment stage");
    for (what, (fragment_kept, vertex_kept), changed) in cases {
        let changed = metal_key::render(&PROFILE, &changed);
        let changed_fragment = changed.fragment.expect("a fragment stage");
        assert_eq!(
            changed_fragment.plain == fragment.plain,
            fragment_kept,
            "{what}: the fragment key"
        );
        assert_eq!(
            changed_fragment.patch == fragment.patch,
            fragment_kept,
            "{what}: the fragment key beside a patch function"
        );
        assert_eq!(
            changed.vertex.plain == keys.vertex.plain,
            vertex_kept,
            "{what}: the vertex key"
        );
        assert_eq!(
            changed.vertex.patch == keys.vertex.patch,
            vertex_kept,
            "{what}: the vertex key of a patch function"
        );
    }
}

#[test]
fn each_render_stage_key_holds_the_fields_metal_keys() {
    let base = render();
    let with = |change: &dyn Fn(&mut RenderPipelineDescriptor)| {
        let mut changed = base.clone();
        change(&mut changed);
        changed
    };
    let attachment = |change: &dyn Fn(&mut ColorAttachmentDescriptor)| {
        with(&|d| {
            let mut a = d.color_attachments[0].unwrap();
            change(&mut a);
            d.color_attachments[0] = Some(a);
        })
    };
    let tessellation = |change: &dyn Fn(&mut TessellationDescriptor)| {
        with(&|d| {
            let mut t = TessellationDescriptor::default();
            change(&mut t);
            d.tessellation = Some(t);
        })
    };
    let linked =
        || LinkedFunctions::new().with_function(function("ob3_texture_array", "ob3_texture_array"));
    let mut layout = VertexDescriptor::new();
    layout
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(16)));
    layout.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 0),
    );
    let mut unused = layout.clone();
    unused
        .attributes
        .insert(5, VertexAttributeDescriptor::new(VertexFormat::Float, 4, 0));
    let mut dynamic = layout.clone();
    dynamic
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(None));
    let both = (true, true);
    let fragment_only = (false, true);
    let vertex_only = (true, false);
    let neither = (false, false);
    stage_keys(
        &base,
        vec![
            ("the library made again", both, render()),
            (
                "the depth format",
                both,
                with(&|d| d.depth_format = Some(PixelFormat::Depth32Float)),
            ),
            (
                "the stencil format",
                both,
                with(&|d| d.stencil_format = Some(PixelFormat::Stencil8)),
            ),
            (
                "an empty vertex descriptor",
                both,
                with(&|d| d.vertex_descriptor = Some(VertexDescriptor::new())),
            ),
            (
                "blend factors with blending off",
                both,
                attachment(&|a| a.source_rgb_blend_factor = BlendFactor::Zero),
            ),
            (
                "the maximum tessellation factor",
                both,
                tessellation(&|t| t.max_tessellation_factor = 64),
            ),
            (
                "the winding order",
                both,
                tessellation(&|t| t.output_winding_order = Winding::CounterClockwise),
            ),
            (
                "factor scale",
                both,
                tessellation(&|t| t.factor_scale_enabled = true),
            ),
            (
                "another vertex function",
                vertex_only,
                with(&|d| {
                    d.vertex = StageSource::Function(FunctionDescriptor::new(function(
                        "ob3_control_point",
                        "ob3_control_point",
                    )))
                }),
            ),
            (
                "the vertex function's linked functions",
                vertex_only,
                with(&|d| {
                    if let StageSource::Function(vertex) = &mut d.vertex {
                        vertex.linked = linked();
                    }
                }),
            ),
            (
                "rasterization",
                vertex_only,
                with(&|d| d.rasterization_enabled = false),
            ),
            (
                "the topology class",
                vertex_only,
                with(&|d| d.input_primitive_topology = PrimitiveTopologyClass::Point),
            ),
            (
                "the amplification count",
                vertex_only,
                with(&|d| d.max_vertex_amplification_count = 2),
            ),
            (
                "the partition mode",
                vertex_only,
                tessellation(&|t| t.partition_mode = TessellationPartitionMode::Integer),
            ),
            (
                "the factor step function",
                vertex_only,
                tessellation(&|t| {
                    t.factor_step_function = TessellationFactorStepFunction::PerPatch
                }),
            ),
            (
                "another fragment function",
                fragment_only,
                with(&|d| d.fragment = Some(fragment("ob3_stage_input"))),
            ),
            (
                "the fragment function's linked functions",
                fragment_only,
                with(&|d| {
                    if let Some(StageSource::Function(fragment)) = &mut d.fragment {
                        fragment.linked = linked();
                    }
                }),
            ),
            (
                "the color format",
                fragment_only,
                attachment(&|a| a.pixel_format = PixelFormat::BGRA8Unorm),
            ),
            (
                "a second attachment",
                fragment_only,
                with(&|d| {
                    d.color_attachments[1] =
                        Some(ColorAttachmentDescriptor::new(PixelFormat::R32Float))
                }),
            ),
            (
                "blending",
                fragment_only,
                attachment(&|a| a.blending_enabled = true),
            ),
            (
                "the write mask",
                fragment_only,
                attachment(&|a| a.write_mask = ColorWriteMask::RED),
            ),
            (
                "the sample count",
                fragment_only,
                with(&|d| d.raster_sample_count = 4),
            ),
            (
                "alpha to coverage",
                fragment_only,
                with(&|d| d.alpha_to_coverage = true),
            ),
            (
                "alpha to one",
                fragment_only,
                with(&|d| d.alpha_to_one = true),
            ),
            (
                "indirect command buffers",
                neither,
                with(&|d| d.support_indirect_command_buffers = true),
            ),
            (
                "vertex binary-function support",
                vertex_only,
                with(&|d| d.support_adding_vertex_binary_functions = true),
            ),
            (
                "fragment binary-function support",
                fragment_only,
                with(&|d| d.support_adding_fragment_binary_functions = true),
            ),
            (
                "a vertex buffer set mutable",
                vertex_only,
                with(&|d| {
                    d.vertex_buffers.insert(2, Mutability::Mutable);
                }),
            ),
            (
                "a fragment buffer set mutable",
                fragment_only,
                with(&|d| {
                    d.fragment_buffers.insert(1, Mutability::Mutable);
                }),
            ),
            (
                "a vertex and a fragment buffer set immutable",
                both,
                with(&|d| {
                    d.vertex_buffers.insert(2, Mutability::Immutable);
                    d.fragment_buffers.insert(1, Mutability::Immutable);
                }),
            ),
            (
                "a vertex descriptor",
                neither,
                with(&|d| d.vertex_descriptor = Some(layout.clone())),
            ),
        ],
    );
    let with_layout = with(&|d| d.vertex_descriptor = Some(layout.clone()));
    stage_keys(
        &with_layout,
        vec![
            (
                "an unused attribute",
                neither,
                with(&|d| d.vertex_descriptor = Some(unused.clone())),
            ),
            (
                "a dynamic stride",
                neither,
                with(&|d| d.vertex_descriptor = Some(dynamic.clone())),
            ),
        ],
    );
    let blending = attachment(&|a| a.blending_enabled = true);
    let mut factor = blending.clone();
    let mut a = factor.color_attachments[0].unwrap();
    a.source_rgb_blend_factor = BlendFactor::Zero;
    factor.color_attachments[0] = Some(a);
    stage_keys(
        &blending,
        vec![("blend factors with blending on", fragment_only, factor)],
    );
}

#[test]
fn line_and_triangle_classes_share_a_vertex_key() {
    let class = |class| {
        let mut d = render();
        d.input_primitive_topology = class;
        metal_key::render(&PROFILE, &d).vertex
    };
    assert_eq!(
        class(PrimitiveTopologyClass::Line),
        class(PrimitiveTopologyClass::Triangle)
    );
    let classes = [
        class(PrimitiveTopologyClass::Unspecified),
        class(PrimitiveTopologyClass::Point),
        class(PrimitiveTopologyClass::Line),
    ];
    for (i, a) in classes.iter().enumerate() {
        for b in &classes[i + 1..] {
            assert_ne!(a, b);
        }
    }
}

#[test]
fn without_a_fragment_stage_only_the_vertex_key_is_looked_up() {
    let without = vertex("ob3_index_buffer");
    let keys = metal_key::render(&PROFILE, &without);
    assert_eq!(keys.fragment, None);
    let mut color = without.clone();
    color.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    color.raster_sample_count = 4;
    assert_eq!(
        metal_key::render(&PROFILE, &color).vertex,
        keys.vertex,
        "the fragment stage's fields"
    );
    assert_ne!(
        metal_key::render(&PROFILE, &render()).vertex,
        keys.vertex,
        "a fragment stage that reads the vertex outputs"
    );
    let mut off = without.clone();
    off.rasterization_enabled = false;
    let mut off_with = render();
    off_with.rasterization_enabled = false;
    assert_eq!(
        metal_key::render(&PROFILE, &off_with).vertex,
        metal_key::render(&PROFILE, &off).vertex,
        "a fragment function under disabled rasterization reads nothing"
    );
}

#[test]
fn only_the_patch_keys_hold_the_control_point_index_type() {
    let mut base = vertex("ob3_control_point");
    base.fragment = Some(fragment("ob3_vertex_input"));
    let mut changed = base.clone();
    let mut tessellation = TessellationDescriptor::default();
    tessellation.control_point_index_type = TessellationControlPointIndexType::UInt32;
    changed.tessellation = Some(tessellation);
    let (base, changed) = (
        metal_key::render(&PROFILE, &base),
        metal_key::render(&PROFILE, &changed),
    );
    for (base, changed) in [
        (base.vertex, changed.vertex),
        (base.fragment.unwrap(), changed.fragment.unwrap()),
    ] {
        assert_eq!(base.plain, changed.plain);
        assert_ne!(base.patch, changed.patch);
        assert_ne!(base.plain, base.patch);
    }
}

/// The function `binary_archive::render_missing` names for `descriptor` when the archives hold
/// `keys`.
fn render_missing(descriptor: &RenderPipelineDescriptor, keys: &[Hash256]) -> Option<String> {
    let held = |key: &Hash256| keys.contains(key);
    binary_archive::render_missing(descriptor)(&PROFILE, &held).map(|stage| match stage {
        StageSource::Function(function) => function.function.info().name.to_string(),
        StageSource::Compiled(_) => unreachable!("the descriptors name functions"),
    })
}

#[test]
fn a_render_lookup_combines_stages_and_names_the_fragment_first() {
    let plain = render();
    let mut other = render();
    other.vertex = StageSource::Function(FunctionDescriptor::new(function(
        "ob3_vertex_input",
        "ob3_vertex_input",
    )));
    other.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    let mut combined = plain.clone();
    combined.color_attachments[0] = other.color_attachments[0];
    let (keys, other_keys) = (
        metal_key::render(&PROFILE, &plain),
        metal_key::render(&PROFILE, &other),
    );
    let fragment = |keys: metal_key::RenderKeys| keys.fragment.unwrap();
    let added = [
        keys.vertex.plain,
        fragment(keys).plain,
        other_keys.vertex.plain,
        fragment(other_keys).plain,
    ];
    assert_eq!(render_missing(&plain, &added), None);
    assert_eq!(
        render_missing(&combined, &added),
        None,
        "one add's vertex stage and another's fragment stage"
    );
    assert_eq!(
        render_missing(&combined, &added[..2]).as_deref(),
        Some("ob3_vertex_input"),
        "the fragment function is named first"
    );
    assert_eq!(
        render_missing(&plain, &[]).as_deref(),
        Some("ob3_vertex_input"),
        "nothing held: the fragment function"
    );
    assert_eq!(
        render_missing(&plain, &[fragment(keys).plain]).as_deref(),
        Some("ob3_index_buffer"),
        "the fragment found: the vertex function"
    );
    let without = vertex("ob3_index_buffer");
    let without_keys = metal_key::render(&PROFILE, &without);
    assert_eq!(render_missing(&without, &[without_keys.vertex.plain]), None);
    assert_eq!(
        render_missing(&plain, &[keys.vertex.patch, fragment(keys).plain]).as_deref(),
        Some("ob3_vertex_input"),
        "a patch function's vertex key held asks for the fragment's patch key"
    );
    assert_eq!(
        render_missing(&plain, &[keys.vertex.patch, fragment(keys).patch]),
        None
    );
}

#[test]
fn a_compute_lookup_names_the_kernel() {
    let descriptor = compute(kernel());
    let key = metal_key::compute(&PROFILE, &descriptor);
    let missing = |keys: &[Hash256]| {
        let held = |key: &Hash256| keys.contains(key);
        binary_archive::compute_missing(&descriptor)(&PROFILE, &held).is_some()
    };
    assert!(!missing(&[key]));
    assert!(missing(&[]));
}

/// The tile key holds binary-function support, the required threadgroup size (the table), an
/// all-zero size as none, and the buffers set mutable.
#[test]
fn the_tile_key_holds_binary_function_support_and_the_required_size() {
    let base = TileRenderPipelineDescriptor::new(StageSource::Function(kernel()));
    let key = metal_key::tile(&PROFILE, &base);
    let with = |change: &dyn Fn(&mut TileRenderPipelineDescriptor)| {
        let mut changed = base.clone();
        change(&mut changed);
        metal_key::tile(&PROFILE, &changed)
    };
    assert_ne!(with(&|d| d.support_adding_binary_functions = true), key);
    assert_ne!(
        with(&|d| d.required_threads_per_threadgroup = Some([16, 16, 1])),
        key
    );
    assert_eq!(
        with(&|d| d.required_threads_per_threadgroup = Some([0; 3])),
        key
    );
    assert_ne!(
        with(&|d| {
            d.buffers.insert(0, Mutability::Mutable);
        }),
        key
    );
    assert_eq!(
        with(&|d| {
            d.buffers.insert(0, Mutability::Immutable);
        }),
        key
    );
}

/// A lookup that finds both render stages from one add's keys reads as one add; from two adds',
/// whose pair no add kept, as combined stages (the `combined-archive-stages` degradation), and
/// never without a fragment stage that reads, or where a stage misses.
#[test]
fn a_render_lookup_tells_one_adds_stages_from_two_adds() {
    let plain = render();
    let mut other = render();
    other.vertex = StageSource::Function(FunctionDescriptor::new(function(
        "ob3_vertex_input",
        "ob3_vertex_input",
    )));
    other.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    let mut combined = plain.clone();
    combined.color_attachments[0] = other.color_attachments[0];
    let kept = |descriptor: &RenderPipelineDescriptor| {
        let keys = metal_key::render(&PROFILE, descriptor);
        let plain = metal_key::VertexKind::Plain;
        [
            keys.vertex.of(plain),
            keys.fragment.unwrap().of(plain),
            keys.pair(plain).unwrap(),
        ]
    };
    let added = [kept(&plain), kept(&other)].concat();
    let combines = |descriptor: &RenderPipelineDescriptor, keys: &[Hash256]| {
        let held = |key: &Hash256| keys.contains(key);
        binary_archive::render_combined(descriptor)(&PROFILE, &held)
    };
    assert!(!combines(&plain, &added), "one add's stages");
    assert!(!combines(&other, &added), "the other add's stages");
    assert!(
        combines(&combined, &added),
        "one add's vertex stage and another's fragment stage"
    );
    assert!(
        !combines(&combined, &added[..3]),
        "the fragment stage misses"
    );
    let mut unrasterized = combined.clone();
    unrasterized.rasterization_enabled = false;
    let mut unrasterized_added = added.clone();
    unrasterized_added.push(
        metal_key::render(&PROFILE, &unrasterized)
            .vertex
            .of(metal_key::VertexKind::Plain),
    );
    assert!(
        !combines(&unrasterized, &unrasterized_added),
        "no fragment stage reads under disabled rasterization"
    );
    let without = vertex("ob3_index_buffer");
    let without_vertex = metal_key::render(&PROFILE, &without)
        .vertex
        .of(metal_key::VertexKind::Plain);
    assert!(!combines(&without, &[without_vertex]), "no fragment stage");
}

/// A metallib loads as an archive holding nothing, as Metal takes one as an archive (the
/// conformance case `binary-archive-sources`, an M3 Max under macOS 27.0.1); bare AIR, which is
/// no metallib, is still an invalid file.
#[test]
fn a_metallib_loads_as_an_archive_holding_nothing() {
    use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};
    let scratch = Scratch::new();
    let air = fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/fixtures/public/engine/ob3_texture_array.air"),
    )
    .expect("the committed AIR");
    let bytes = metallib(&[MtlbFunction {
        name: "ob3_texture_array",
        type_code: 2,
        bitcode: wrap(&air),
        constants: Vec::new(),
        public: Vec::new(),
        tags: Vec::new(),
    }]);
    assert!(Library::from_metallib(&bytes).is_ok(), "a metallib");
    let path = scratch.path().join("library.metallib");
    fs::write(&path, &bytes).unwrap();
    assert_eq!(container::read(&path), Ok(Contents::default()));
    let bare = scratch.path().join("bare.air");
    fs::write(&bare, &air).unwrap();
    assert_eq!(container::read(&bare), Err(invalid_file(&bare)));
}

/// The mesh key holds every field of the descriptor (Metal's is not measured, `metal_key`): an
/// equal descriptor keeps it and each field changed changes it, and a mesh lookup names the mesh
/// function.
#[test]
fn the_mesh_key_holds_every_field_and_a_lookup_names_the_mesh_function() {
    use metal2vulkan::metal::TextureWriteRoundingMode;
    use metal2vulkan::pipeline::MeshPipelineDescriptor;
    fn stage(name: &str) -> StageSource {
        StageSource::Function(FunctionDescriptor::new(function("ob3_texture_array", name)))
    }
    let mut base = MeshPipelineDescriptor::new(stage("ob3_texture_array"));
    base.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let key = metal_key::mesh(&PROFILE, &base);
    assert_eq!(metal_key::mesh(&PROFILE, &base.clone()), key);
    assert_ne!(metal_key::mesh(&[8; 32], &base), key, "the profile");
    let with = |change: &dyn Fn(&mut MeshPipelineDescriptor)| {
        let mut changed = base.clone();
        change(&mut changed);
        metal_key::mesh(&PROFILE, &changed)
    };
    let changes: Vec<(&str, &dyn Fn(&mut MeshPipelineDescriptor))> = vec![
        ("object", &|d| d.object = Some(stage("ob3_texture_array"))),
        ("fragment", &|d| {
            d.fragment = Some(stage("ob3_texture_array"))
        }),
        ("color format", &|d| {
            d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm))
        }),
        ("second attachment", &|d| {
            d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm))
        }),
        ("write mask", &|d| {
            if let Some(color) = d.color_attachments[0].as_mut() {
                color.write_mask = ColorWriteMask::RED;
            }
        }),
        ("depth format", &|d| {
            d.depth_format = Some(PixelFormat::Depth32Float)
        }),
        ("stencil format", &|d| {
            d.stencil_format = Some(PixelFormat::Stencil8)
        }),
        ("raster samples", &|d| d.raster_sample_count = 4),
        ("rasterization", &|d| d.rasterization_enabled = false),
        ("alpha to coverage", &|d| d.alpha_to_coverage = true),
        ("alpha to one", &|d| d.alpha_to_one = true),
        ("rounding", &|d| {
            d.texture_write_rounding_mode = TextureWriteRoundingMode::NearestEven
        }),
        ("indirect", &|d| d.support_indirect_command_buffers = true),
        ("object maximum", &|d| {
            d.max_total_threads_per_object_threadgroup = Some(64)
        }),
        ("mesh maximum", &|d| {
            d.max_total_threads_per_mesh_threadgroup = Some(64)
        }),
        ("grid maximum", &|d| {
            d.max_total_threadgroups_per_mesh_grid = Some(8)
        }),
        ("object multiple", &|d| {
            d.object_threadgroup_size_is_multiple_of_thread_execution_width = true
        }),
        ("mesh multiple", &|d| {
            d.mesh_threadgroup_size_is_multiple_of_thread_execution_width = true
        }),
        ("payload", &|d| d.payload_memory_length = Some(64)),
        ("amplification", &|d| d.max_vertex_amplification_count = 2),
        ("object size", &|d| {
            d.required_threads_per_object_threadgroup = Some([4, 1, 1])
        }),
        ("mesh size", &|d| {
            d.required_threads_per_mesh_threadgroup = Some([4, 1, 1])
        }),
    ];
    for (what, change) in changes {
        assert_ne!(with(change), key, "{what}");
    }
    let missing = |keys: &[Hash256]| {
        let held = |key: &Hash256| keys.contains(key);
        binary_archive::mesh_missing(&base)(&PROFILE, &held).map(|stage| match stage {
            StageSource::Function(descriptor) => descriptor.function.info().name.to_string(),
            StageSource::Compiled(_) => "compiled".to_owned(),
        })
    };
    assert_eq!(missing(&[key]), None);
    assert_eq!(missing(&[]), Some("ob3_texture_array".to_owned()));
}
