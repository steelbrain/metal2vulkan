//! The sampling routines are routines. For a profile whose driver converts every function to one
//! inlined at each call (`DriverWorkaround:: FunctionsInlinedAtEveryCall`, MoltenVK 1.4.1) each
//! states `DontInline`; for any other profile none states a control, and the entry states none
//! anywhere.

use super::kernel_on;
use crate::device::profile::DriverWorkaround;
use crate::device::DevicePolicy;
use crate::lower::sampler::bicubic::routines;
use crate::lower::texture::half;
use crate::pipeline::testing::profile;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::serialize::serialize_for;

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

/// The texel conversion and the linear and bicubic filters' sums beside a kernel entry: on
/// moltenvk-m3 each routine states `DontInline` (0x2) and the entry `None`; on amd-rdna3-radv the
/// texel routine states `DontInline` (the in-place linear quad's exact sum) and every
/// other function `None`.
#[test]
fn the_sampling_routines_are_declared_dont_inline_where_every_call_is_inlined() {
    let mut b = ModuleBuilder::new();
    half::define(&mut b).expect("half texel");
    routines::define_plane(&mut b).expect("plane");
    routines::define_line(&mut b).expect("line");
    routines::define_quad(&mut b).expect("quad");
    let m = kernel_on(b, 3, &|_, p, _| Ok(p)).expect("kernel");
    for (stem, inlines_every_call) in [("moltenvk-m3", true), ("amd-rdna3-radv", false)] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        assert_eq!(
            p.fields()
                .workarounds
                .contains(&DriverWorkaround::FunctionsInlinedAtEveryCall),
            inlines_every_call,
            "{stem}"
        );
        let words = serialize_for(m.clone(), &p).expect("serialized").words;
        let mut got = controls(&words);
        got.sort_unstable();
        // 4 routines (the plane's, the line's, the quad's, and the texel conversion) and the
        // entry.
        assert!(got.len() >= 5, "{stem}: {} functions", got.len());
        // The texel routine states it on every profile.
        let kept = if inlines_every_call { got.len() - 1 } else { 1 };
        let mut want = vec![0x2; kept];
        want.splice(0..0, vec![0; got.len() - kept]);
        assert_eq!(
            got, want,
            "{stem}: each routine's control, and none on the entry"
        );
    }
}
