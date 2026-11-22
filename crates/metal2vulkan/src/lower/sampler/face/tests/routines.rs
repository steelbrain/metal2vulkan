//! The bicubic filters' sums are plain routines: each states no function control, as the entry
//! does.

use super::kernel_on;
use crate::lower::sampler::bicubic::routines;
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

/// The bicubic filters' sums beside a kernel entry: each routine and the entry state `None`.
#[test]
fn the_bicubic_routines_state_no_function_control() {
    let mut b = ModuleBuilder::new();
    routines::define_plane(&mut b).expect("plane");
    routines::define_line(&mut b).expect("line");
    let m = kernel_on(b, 3, &|_, p, _| Ok(p)).expect("kernel");
    let words = serialize(m).expect("serialized").words;
    let got = controls(&words);
    // 2 routines (the plane's and the line's) and the entry.
    assert!(got.len() >= 3, "{} functions", got.len());
    assert_eq!(got, vec![0; got.len()], "each function's control");
}
