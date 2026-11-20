//! The sampling routines are routines: the texel conversion states `DontInline`, every other
//! function none, and the entry states none.

use super::kernel_on;
use crate::lower::sampler::bicubic::routines;
use crate::lower::texture::half;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::serialize::serialize;

/// Each `OpFunction`'s function control in `words`, in module order.
fn controls(words: &[u32]) -> Vec<u32> {
    let mut i = 5;
    let mut out = Vec::new();
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::Function as u32 {
            out.push(words[i + 3]);
        }
        i += len;
    }
    out
}

/// The texel conversion and the linear and bicubic filters' sums beside a kernel entry: the texel
/// routine, a kept routine, states `DontInline` (0x2, the in-place linear quad's exact sum) and
/// every other function `None`.
#[test]
fn the_texel_routine_alone_is_declared_dont_inline() {
    let mut b = ModuleBuilder::new();
    half::define(&mut b).expect("half texel");
    routines::define_plane(&mut b).expect("plane");
    routines::define_line(&mut b).expect("line");
    routines::define_quad(&mut b).expect("quad");
    let m = kernel_on(b, 3, &|_, p, _| Ok(p)).expect("kernel");
    let words = serialize(m).expect("serialized").words;
    let mut got = controls(&words);
    got.sort_unstable();
    // 4 routines (the plane's, the line's, the quad's, and the texel conversion) and the entry.
    assert!(got.len() >= 5, "{} functions", got.len());
    let mut want = vec![0; got.len() - 1];
    want.push(0x2);
    assert_eq!(got, want, "each routine's control, and none on the entry");
}
