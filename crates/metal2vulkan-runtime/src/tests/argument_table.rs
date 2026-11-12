//! The debug builds' check that a stage table brought in place is the table read whole
//! (`ArgumentTable::check_reading`, `argument_table.rs`), which compares a sampler as it prints
//! without printing it (`same_as_printed`); the offset a table takes at an index with no buffer
//! bound, which a render stage's table answers as a compute one does; and the stamps threads draw
//! at once, each from a block of its own, which no two calls share.

use super::super::argument_table::{per_bind_stride, same_as_printed, ArgumentTable, Stamp};
use metal2vulkan::bind::{BoundSampler, SamplerDescriptor, StageTable};

/// A stride form's stride is its low 32 bits, and `MTLAttributeStrideStatic`'s (`u64::MAX`) too:
/// a layout of `MTLBufferLayoutStrideDynamic` reads `2^32 - 1` and so the element at the bind
/// offset and zero past it, as plain Metal on an Apple M3 Max under macOS 27.0.1 answered the
/// compute encoder's four stride forms and the render encoder's vertex form
/// (`metal2vulkan-conformance`'s `attribute-stride-*` recordings).
#[test]
fn the_static_constant_is_the_widest_stride() {
    for (stride, held) in [
        (0u64, 0u32),
        (8, 8),
        (1 << 32, 0),
        ((1 << 32) + 8, 8),
        ((1 << 63) + 8, 8),
        (u64::from(u32::MAX), u32::MAX),
        (u64::MAX, u32::MAX),
    ] {
        assert_eq!(per_bind_stride(stride), held, "{stride}");
    }
}

/// `setBufferOffset:atIndex:`, and its stride form, at an index with no buffer bound: a render
/// stage's table answers exactly as a compute one does, and neither keeps anything
/// (`ArgumentTable::take_buffer_offset`). Plain Metal keeps no offset there and the draw reads
/// zeros, byte for byte as it reads an index never bound (an Apple M3 Max under macOS 27.0.1, the
/// method's own doc).
#[test]
fn an_offset_at_an_unbound_index_answers_alike_in_both_tables() {
    const CALL: &str = "set_vertex_buffer_offset";
    let mut render = ArgumentTable::default();
    let mut compute = ArgumentTable::compute();
    // Index 0, one in the middle of the table, and the last index a table has.
    for index in [0u32, 7, 30] {
        for (offset, stride) in [
            (0u64, None),
            (64, None),
            (16, Some(32u32)),
            (u64::MAX, None),
        ] {
            let answers = [&mut render, &mut compute]
                .map(|table| table.take_buffer_offset(CALL, offset, stride, index));
            assert_eq!(answers[0], Ok(()), "render {index} {offset} {stride:?}");
            assert_eq!(
                answers[0], answers[1],
                "render and compute differ at {index} {offset} {stride:?}"
            );
        }
    }
    // Neither bound anything, kept an offset, nor kept a stride: a later `setBuffer:offset:` at
    // the index states its own.
    for table in [&render, &compute] {
        assert!(table.buffers.is_empty());
        assert!(table.attribute_strides.is_empty());
        assert!(table.function_tables.is_empty());
        assert!(table.null_tables.is_empty());
        assert!(table.bytes.is_empty());
    }
}

/// The field names of a value's `{:?}` print, `Name { a: .., b: .. }`, at its top level.
fn field_names(printed: &str) -> Vec<String> {
    let inner = &printed[printed.find('{').unwrap() + 1..printed.rfind('}').unwrap()];
    let mut names = Vec::new();
    let (mut depth, mut start) = (0, 0);
    for (at, c) in inner.char_indices() {
        match c {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                names.push(inner[start..at].trim().to_owned());
                start = at + 1;
            }
            _ => {}
        }
    }
    names.push(inner[start..].trim().to_owned());
    names
        .iter()
        .filter(|field| !field.is_empty())
        .map(|field| field.split(':').next().unwrap().trim().to_owned())
        .collect()
}

/// Two bound samplers compare alike exactly where they print alike: LOD clamps or LOD biases of
/// opposite zero signs differ, NaN clamps or biases of any sign and payload are alike, and every
/// other field differs where its value does.
#[test]
fn samplers_compare_as_they_print() {
    let base = BoundSampler::new(SamplerDescriptor::new());
    let clamped = |min: f32, max: f32| {
        let mut sampler = base;
        sampler.desc.lod_min_clamp = min;
        sampler.desc.lod_max_clamp = max;
        sampler
    };
    let other = |change: fn(&mut BoundSampler)| {
        let mut sampler = base;
        change(&mut sampler);
        sampler
    };
    let pairs = [
        (base, base),
        (clamped(0.0, 1.0), clamped(-0.0, 1.0)),
        (clamped(0.0, 0.0), clamped(0.0, -0.0)),
        (clamped(f32::NAN, 1.0), clamped(-f32::NAN, 1.0)),
        (
            clamped(0.5, f32::from_bits(0x7fc0_0001)),
            clamped(0.5, f32::NAN),
        ),
        (clamped(f32::NAN, 1.0), clamped(0.0, 1.0)),
        (clamped(0.0, f32::MAX), clamped(0.0, f32::INFINITY)),
        (base, other(|s| s.desc.lod_bias = -0.0)),
        (base, other(|s| s.desc.lod_bias = 1.0)),
        (
            other(|s| s.desc.lod_bias = f32::NAN),
            other(|s| s.desc.lod_bias = -f32::NAN),
        ),
        (
            other(|s| s.desc.lod_bias = f32::NAN),
            other(|s| s.desc.lod_bias = f32::from_bits(0x7fc0_0001)),
        ),
        (base, other(|s| s.desc.lod_bias = f32::NAN)),
        (base, other(|s| s.desc.max_anisotropy = 4)),
        (base, other(|s| s.desc.normalized_coordinates = false)),
        (base, other(|s| s.resource_id = Some(7))),
    ];
    for (a, b) in pairs {
        assert_eq!(
            same_as_printed(&a, &b),
            format!("{a:?}") == format!("{b:?}"),
            "{a:?} against {b:?}"
        );
    }
}

/// The fields the check compares: a field added to a stage table, a bound sampler or a sampler
/// descriptor fails this until `check_reading` and `same_as_printed` compare it as it prints
/// (`at_bind_views` as the state's views leave it, which another encoder may narrow since).
#[test]
fn the_check_compares_every_field() {
    assert_eq!(
        field_names(&format!("{:?}", StageTable::default())),
        [
            "buffers",
            "textures",
            "samplers",
            "threadgroup_lengths",
            "acceleration_structures",
            "at_bind_views"
        ]
    );
    let sampler = BoundSampler::new(SamplerDescriptor::new());
    assert_eq!(
        field_names(&format!("{sampler:?}")),
        ["desc", "resource_id"]
    );
    assert_eq!(
        field_names(&format!("{:?}", sampler.desc)),
        [
            "min_filter",
            "mag_filter",
            "mip_filter",
            "s_address_mode",
            "t_address_mode",
            "r_address_mode",
            "border_color",
            "normalized_coordinates",
            "lod_min_clamp",
            "lod_max_clamp",
            "lod_average",
            "max_anisotropy",
            "compare_function",
            "reduction_mode",
            "lod_bias"
        ]
    );
}

/// Threads drawing stamps at once, each past more than one block of its own, draw no stamp
/// another call drew, on their threads or another's (`Stamp::fresh`): the identity a reader of a
/// table compares holds across threads as it did when every stamp came from one counter.
#[test]
fn stamps_drawn_on_threads_at_once_are_all_distinct() {
    const THREADS: usize = 4;
    // Past a block's stamps, so each thread takes a second block while the others draw.
    const EACH: usize = (1 << 16) + 1000;
    let start = std::sync::Barrier::new(THREADS);
    let drawn: Vec<Vec<Stamp>> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..THREADS)
            .map(|_| {
                scope.spawn(|| {
                    start.wait();
                    (0..EACH).map(|_| Stamp::fresh()).collect::<Vec<_>>()
                })
            })
            .collect();
        threads.into_iter().map(|t| t.join().unwrap()).collect()
    });
    let mut all: Vec<u64> = drawn.iter().flatten().map(|stamp| stamp.raw()).collect();
    let total = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), total, "a stamp drawn twice");
}
