//! A placement heap's texture of one of Apple's one-plane YCbCr formats (held for copies alone,
//! `metal2vulkan::metal::format::copies_only`) lies in the heap's bytes as Metal lays it: a 2D
//! texture's rows one packed row apart where the row is a multiple of 16 bytes, and the row
//! rounded down to 16 bytes apart where it is 16 bytes or more off a multiple of 16, so that
//! consecutive rows share bytes; a 1D texture's one row; and an array's slices each at a multiple
//! of 128 bytes; its blits' copies in and out moving the bytes there and no other, each row read
//! whole from its own offset and a shared byte left from the earlier row, one of the rows Metal's
//! blits leave there (an Apple M3 Max under macOS 27.0.1, a `Private` untracked
//! placement heap with a `Private` buffer over all its bytes,
//! `metal2vulkan-conformance/probes/one_plane_ycbcr_heap_rows.m`, every byte of these shapes in
//! each of the nine formats). The texture takes Metal's answer of the heap. While it lay in the
//! carrier's tiles (`R16Uint`, `R32Uint`), its texels lay elsewhere in its bytes.
//!
//! A shape Metal lays no one way lies otherwise, announced `heap-one-plane-layout`: a 2D texture
//! of rows under 16 bytes in packed rows, and a mipmapped one in memory of its own, the heap's
//! bytes untouched.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::WAIT;

use crate::{
    BlitOption, Buffer, CommandQueue, Degradation, Device, DeviceKind, DeviceOptions,
    HazardTrackingMode, HeapDescriptor, HeapType, MessageSink, Origin, ResourceOptions, Size,
    Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureType};
use std::sync::{Arc, Mutex};

/// The heap's bytes: room for every shape below.
const HEAP: u64 = 64 << 10;

/// The byte the heap's bytes hold before a copy in.
const UNTOUCHED: u8 = 0xee;

/// A 4:2:2 format, two bytes a texel, and a 4:4:4 one, four.
const FORMATS: [(PixelFormat, u64); 2] = [
    (PixelFormat::YCBCR84221P, 2),
    (PixelFormat::YCBCR104441P, 4),
];

/// What the device's sink heard: each notice's kind.
type Heard = Arc<Mutex<Vec<Degradation>>>;

/// The device the cfg selects, with a sink that keeps every notice's kind.
fn listening() -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let hearing = Arc::clone(&heard);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                hearing.lock().unwrap().push(notice.degradation);
            }
        })),
        ..DeviceOptions::default()
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

/// A byte unique to `(tag, index)` for any index below 2^24.
fn pattern(tag: u32, index: u64) -> u8 {
    let mut x = (index as u32).wrapping_mul(2654435761) ^ tag.wrapping_mul(40503);
    x ^= x >> 15;
    x = x.wrapping_mul(2246822519);
    x ^= x >> 13;
    (x ^ (x >> 24)) as u8
}

/// A texture's shape: its type, width, height and slices.
#[derive(Clone, Copy, Debug)]
struct Shape {
    ty: TextureType,
    width: u64,
    height: u64,
    slices: u64,
    levels: u64,
}

impl Shape {
    const fn new(ty: TextureType, width: u64, height: u64, slices: u64) -> Shape {
        Shape {
            ty,
            width,
            height,
            slices,
            levels: 1,
        }
    }

    /// Its descriptor in `pf`, of the heap's `options`.
    fn descriptor(self, pf: PixelFormat, options: ResourceOptions) -> TextureDescriptor {
        TextureDescriptor {
            texture_type: self.ty,
            pixel_format: pf,
            width: self.width,
            height: self.height,
            array_length: self.slices,
            mipmap_level_count: self.levels,
            resource_options: options,
            ..TextureDescriptor::default()
        }
    }

    /// The bytes of one packed row of `unit`-byte texels, and of one slice's packed rows.
    fn packed(self, unit: u64) -> (u64, u64) {
        (self.width * unit, self.width * unit * self.height)
    }

    /// Where Metal lays slice `slice`'s row `y` in the texture's bytes: rows one packed row apart,
    /// or the row rounded down to 16 bytes apart where it is 16 bytes or more off a multiple of
    /// 16; an array's slices each at a multiple of 128 bytes.
    fn metal_row(self, unit: u64, slice: u64, y: u64) -> u64 {
        let (row, rows) = self.packed(unit);
        let slice_pitch = match self.slices > 1 {
            true => rows.next_multiple_of(128),
            false => rows,
        };
        let pitch = match self.height > 1 && row >= 16 {
            true => row - row % 16,
            false => row,
        };
        slice * slice_pitch + y * pitch
    }
}

/// Runs `encode` in a command buffer of its own on `queue`, to its completion.
fn run(queue: &CommandQueue, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The bytes of `buffer`, a `Shared` one, of `length`.
fn read(buffer: &Buffer, length: u64) -> Vec<u8> {
    let at = buffer.contents().unwrap();
    // SAFETY: a `Shared` buffer of at least `length` bytes whose commands all completed.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), length as usize) }.to_vec()
}

/// A `Shared` buffer holding `bytes`.
fn holding(device: &Device, bytes: &[u8]) -> Buffer {
    (device.new_buffer_with_bytes(bytes, ResourceOptions::shared())).unwrap()
}

/// What one texture of `shape` in `pf` (`unit`-byte texels) does in a `Private` untracked
/// placement heap with a `Private` buffer over all its bytes, at offset 0: the heap's bytes after
/// a copy into each of its slices at the packed row, from a heap filled with [`UNTOUCHED`]; and
/// each slice's copy out at the packed row, from a heap holding the bytes `pattern(2, offset)`.
/// The texture, the bytes the copies in wrote, and the two readings.
fn copies(device: &Device, pf: PixelFormat, unit: u64, shape: Shape) -> Copies {
    let queue = device.new_command_queue().unwrap();
    let options =
        ResourceOptions::private().with_hazard_tracking_mode(HazardTrackingMode::Untracked);
    let heap = device
        .new_heap(&HeapDescriptor::new(HeapType::Placement, HEAP, options))
        .unwrap();
    let bytes = heap.new_buffer_at_offset(HEAP, options, 0).unwrap();
    let descriptor = shape.descriptor(pf, options);
    let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
    let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
    let (row, rows) = shape.packed(unit);
    let size = Size::new(shape.width, shape.height, 1);
    let written: Vec<Vec<u8>> = (0..shape.slices)
        .map(|slice| (0..rows).map(|i| pattern(slice as u32, i)).collect())
        .collect();
    let sources: Vec<Buffer> = written.iter().map(|w| holding(device, w)).collect();
    let back = device.new_buffer(HEAP, ResourceOptions::shared()).unwrap();
    run(&queue, |blit| {
        blit.fill_buffer(&bytes, 0..HEAP, UNTOUCHED).unwrap()
    });
    run(&queue, |blit| {
        for (slice, source) in sources.iter().enumerate() {
            let at = Origin::new(0, 0, 0);
            (blit.copy_from_buffer_to_texture(
                source,
                0,
                row,
                rows,
                size,
                &texture,
                slice as u64,
                0,
                at,
                BlitOption::None,
            ))
            .unwrap();
        }
    });
    run(&queue, |blit| {
        (blit.copy_from_buffer_to_buffer(&bytes, 0, &back, 0, HEAP)).unwrap()
    });
    let heap_after_in = read(&back, HEAP);
    let laid: Vec<u8> = (0..HEAP).map(|i| pattern(2, i)).collect();
    let laid_buffer = holding(device, &laid);
    run(&queue, |blit| {
        (blit.copy_from_buffer_to_buffer(&laid_buffer, 0, &bytes, 0, HEAP)).unwrap()
    });
    let outs: Vec<Buffer> = (0..shape.slices)
        .map(|_| device.new_buffer(rows, ResourceOptions::shared()).unwrap())
        .collect();
    run(&queue, |blit| {
        for (slice, out) in outs.iter().enumerate() {
            let at = Origin::new(0, 0, 0);
            (blit.copy_from_texture_to_buffer(
                &texture,
                slice as u64,
                0,
                at,
                size,
                out,
                0,
                row,
                rows,
                BlitOption::None,
            ))
            .unwrap();
        }
    });
    let read_out = outs.iter().map(|out| read(out, rows)).collect();
    Copies {
        texture,
        answer: answer.size,
        written,
        heap_after_in,
        read_out,
    }
}

/// [`copies`]' readings.
struct Copies {
    texture: Texture,
    answer: u64,
    written: Vec<Vec<u8>>,
    heap_after_in: Vec<u8>,
    read_out: Vec<Vec<u8>>,
}

/// Each shape Metal lays out one way, in a 4:2:2 and a 4:4:4 format: the heap's bytes after the
/// copies in hold each slice's rows where Metal's do (a byte two rows share the earlier row's)
/// and [`UNTOUCHED`] everywhere else, the copies out read each row whole from there, the texture
/// takes Metal's answer, and nothing is announced.
#[test]
fn a_one_plane_ycbcr_heap_texture_lies_in_the_heap_s_bytes_as_metal_lays_it() {
    use TextureType as T;
    let shapes = [
        Shape::new(T::Type2D, 8, 8, 1),
        Shape::new(T::Type2D, 16, 16, 1),
        Shape::new(T::Type2D, 64, 4, 1),
        Shape::new(T::Type1D, 37, 1, 1),
        Shape::new(T::Type1DArray, 37, 1, 3),
        Shape::new(T::Type2DArray, 16, 16, 2),
        Shape::new(T::Type2D, 13, 7, 1),
        Shape::new(T::Type2D, 33, 3, 1),
        Shape::new(T::Type2DArray, 13, 7, 3),
    ];
    let (device, heard) = listening();
    for (pf, unit) in FORMATS {
        for shape in shapes {
            let what = format!("{pf:?} {shape:?}");
            let c = copies(&device, pf, unit, shape);
            let (row, _) = shape.packed(unit);
            let mut expected = vec![UNTOUCHED; HEAP as usize];
            let mut laid_out = Vec::new();
            for (slice, written) in c.written.iter().enumerate() {
                let mut out = Vec::new();
                for y in 0..shape.height {
                    let at = shape.metal_row(unit, slice as u64, y) as usize;
                    out.extend((at..at + row as usize).map(|i| pattern(2, i as u64)));
                }
                // The later rows first, so a byte two rows share is the earlier row's.
                for y in (0..shape.height).rev() {
                    let at = shape.metal_row(unit, slice as u64, y) as usize;
                    let from = (y * row) as usize;
                    let texels = &written[from..from + row as usize];
                    expected[at..at + row as usize].copy_from_slice(texels);
                }
                laid_out.push(out);
            }
            let last = shape.metal_row(unit, shape.slices - 1, shape.height - 1) + row;
            assert!(
                last <= c.answer,
                "{what}: rows to {last}, answer {}",
                c.answer
            );
            let differing = (0..HEAP as usize)
                .filter(|&i| c.heap_after_in[i] != expected[i])
                .count();
            assert_eq!(differing, 0, "{what}: the heap's bytes after the copies in");
            assert_eq!(c.read_out, laid_out, "{what}: the copies out");
            assert_eq!(c.texture.allocated_size(), Some(c.answer), "{what}");
        }
    }
    let heard = heard.lock().unwrap();
    assert!(
        !heard.contains(&Degradation::HeapOnePlaneLayout),
        "{heard:?}"
    );
}

/// A 2D texture of 6 x 3 texels at 4:2:2, whose rows are under 16 bytes, lies in packed rows, its
/// copies in writing them and its copies out reading them; a mipmapped one in memory of its own,
/// the heap's bytes untouched, its copies out reading what its copies in wrote: each announced
/// `heap-one-plane-layout`.
#[test]
fn a_one_plane_ycbcr_heap_texture_metal_lays_no_one_way_is_announced() {
    let (pf, unit) = FORMATS[0];
    let odd = Shape::new(TextureType::Type2D, 6, 3, 1);
    let (device, heard) = listening();
    let c = copies(&device, pf, unit, odd);
    let (row, rows) = odd.packed(unit);
    let packed: Vec<u8> = c.written.concat();
    assert_eq!(
        &c.heap_after_in[..rows as usize],
        &packed[..],
        "{row}-byte rows"
    );
    let heap_laid: Vec<u8> = (0..rows).map(|i| pattern(2, i)).collect();
    assert_eq!(c.read_out, [heap_laid], "{row}-byte rows: the copies out");
    let announced = heard
        .lock()
        .unwrap()
        .contains(&Degradation::HeapOnePlaneLayout);
    assert!(announced, "rows under 16 bytes: no heap-one-plane-layout");
    let mipmapped = Shape {
        levels: 3,
        ..Shape::new(TextureType::Type2D, 16, 16, 1)
    };
    let (device, heard) = listening();
    let c = copies(&device, pf, unit, mipmapped);
    let untouched = c.heap_after_in.iter().all(|&b| b == UNTOUCHED);
    assert!(
        untouched,
        "a mipmapped texture's copies in reach the heap's bytes"
    );
    assert_eq!(c.read_out, c.written, "a mipmapped texture: the copies out");
    assert_eq!(
        c.answer,
        mipmapped.packed(unit).1,
        "its first level's answer"
    );
    let heard = heard.lock().unwrap();
    assert!(
        heard.contains(&Degradation::HeapOnePlaneLayout),
        "{heard:?}"
    );
}
