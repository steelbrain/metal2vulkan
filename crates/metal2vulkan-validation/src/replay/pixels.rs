//! Where two plays' pictures of one frame part (`docs/REPLAY-STREAM.md`, section 10, two plays
//! compared): the bytes each play kept of a shown texture, read as the texels of its pixel format,
//! and the size of their difference: the texels that differ and their share, the largest
//! difference of each channel in the format's units, and the box the differing texels lie in.

/// What a play knows of a texture it read: its `MTLPixelFormat`'s raw value and its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub pixel_format: u64,
    pub width: u64,
    pub height: u64,
}

/// How a texel of a pixel format lies in its bytes: its size, its channels' names in their
/// order, how a channel's value is read, and the units of those values.
struct Layout {
    texel: usize,
    channels: &'static [&'static str],
    read: fn(&[u8], usize) -> f64,
    units: &'static str,
}

/// The layout a picture of no known format compares by: a texel is a byte.
const BYTES: Layout = Layout {
    texel: 1,
    channels: &["byte"],
    read: byte,
    units: "bytes",
};

fn byte(texel: &[u8], channel: usize) -> f64 {
    f64::from(texel[channel])
}

/// A 32-bit word of three 10-bit channels from its low bits up, then a 2-bit one.
fn packed_10_10_10_2(texel: &[u8], channel: usize) -> f64 {
    let word = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
    let (shift, mask) = match channel {
        3 => (30, 0x3),
        _ => (10 * channel as u32, 0x3ff),
    };
    f64::from((word >> shift) & mask)
}

/// The IEEE half at `channel`'s two bytes.
fn half(texel: &[u8], channel: usize) -> f64 {
    let bits = u16::from_le_bytes([texel[2 * channel], texel[2 * channel + 1]]);
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f64::from(bits & 0x3ff);
    sign * match exponent {
        0 => fraction * 2f64.powi(-24),
        31 if fraction == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15),
    }
}

/// The layout of the pixel formats a shown texture takes; none for another, whose bytes compare
/// as bytes.
fn layout(pixel_format: u64) -> Option<Layout> {
    let eight = |channels: &'static [&'static str]| Layout {
        texel: channels.len(),
        channels,
        read: byte,
        units: "levels of 1/255",
    };
    let ten = |channels: &'static [&'static str]| Layout {
        texel: 4,
        channels,
        read: packed_10_10_10_2,
        units: "levels of 1/1023, A of 1/3",
    };
    Some(match pixel_format {
        10 | 11 => eight(&["R"]),
        30 => eight(&["R", "G"]),
        70 | 71 => eight(&["R", "G", "B", "A"]),
        80 | 81 => eight(&["B", "G", "R", "A"]),
        90 => ten(&["R", "G", "B", "A"]),
        94 => ten(&["B", "G", "R", "A"]),
        115 => Layout {
            texel: 8,
            channels: &["R", "G", "B", "A"],
            read: half,
            units: "half-float values",
        },
        _ => return None,
    })
}

/// The size of the difference between pictures `a` and `b` of one texture of `shape`, each the
/// texel rows a play read, from the top, as many bytes a row as the picture's bytes over its
/// height. A shape that is unknown, or a pixel format with no layout here, compares the bytes.
pub fn difference(shape: Option<Shape>, a: &[u8], b: &[u8]) -> String {
    if a.len() != b.len() {
        return format!("sizes differ: a {} bytes, b {} bytes", a.len(), b.len());
    }
    let known = shape.and_then(|shape| Some((shape, layout(shape.pixel_format)?)));
    let (height, layout) = match &known {
        Some((shape, layout)) => (shape.height.max(1) as usize, layout),
        None => (1, &BYTES),
    };
    let row = a.len() / height;
    let across = row / layout.texel;
    let mut differing = 0u64;
    let mut largest = vec![0f64; layout.channels.len()];
    let mut boxed: Option<(usize, usize, usize, usize)> = None;
    for y in 0..height {
        for x in 0..across {
            let at = y * row + x * layout.texel;
            let (ta, tb) = (&a[at..at + layout.texel], &b[at..at + layout.texel]);
            if ta == tb {
                continue;
            }
            differing += 1;
            for (channel, most) in largest.iter_mut().enumerate() {
                let apart = ((layout.read)(ta, channel) - (layout.read)(tb, channel)).abs();
                if apart > *most || apart.is_nan() {
                    *most = apart;
                }
            }
            boxed = Some(match boxed {
                None => (x, x, y, y),
                Some((x0, x1, y0, y1)) => (x0.min(x), x1.max(x), y0.min(y), y1.max(y)),
            });
        }
    }
    let total = (across * height) as u64;
    let share = match total {
        0 => 0.0,
        total => differing as f64 * 100.0 / total as f64,
    };
    let channels: Vec<String> = (layout.channels.iter().zip(&largest))
        .map(|(name, most)| format!("{name} {most}"))
        .collect();
    let what = match &known {
        Some((shape, _)) => format!(
            "texels of format {} {}x{}",
            shape.pixel_format, shape.width, shape.height
        ),
        None => "bytes".to_owned(),
    };
    let place = match boxed {
        Some((x0, x1, y0, y1)) => format!("; box x {x0}..{x1}, y {y0}..{y1}"),
        None => String::new(),
    };
    format!(
        "{differing} of {total} {what} differ ({share:.2}%); largest difference per channel {} \
         ({}){place}",
        channels.join(", "),
        layout.units
    )
}
