//! The checksum of an artifact's frame (`envelope`): XXH64 with seed 0, as its specification
//! defines it, written little-endian.
//!
//! It finds damage, not tampering: whoever can write an entry can write its checksum too, so a
//! cryptographic digest there buys nothing a decode needs, while a load checks every byte of every
//! artifact it reads, which a creation from the disk cache waits for. Keys stay SHA-256.

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

/// Bytes a stripe takes: four lanes of eight.
const STRIPE: usize = 32;

/// A checksum's state, fed a chunk at a time: [`finish`](Checksum::finish) gives what [`of`]
/// gives for the chunks' bytes together, however they were cut.
pub(crate) struct Checksum {
    lanes: [u64; 4],
    /// The bytes of the stripe not yet whole.
    pending: [u8; STRIPE],
    pending_len: usize,
    total: u64,
}

impl Default for Checksum {
    fn default() -> Checksum {
        Checksum {
            lanes: [P1.wrapping_add(P2), P2, 0, 0u64.wrapping_sub(P1)],
            pending: [0; STRIPE],
            pending_len: 0,
            total: 0,
        }
    }
}

fn round(acc: u64, lane: u64) -> u64 {
    acc.wrapping_add(lane.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

fn merge(h: u64, lane: u64) -> u64 {
    (h ^ round(0, lane)).wrapping_mul(P1).wrapping_add(P4)
}

fn u64_at(bytes: &[u8; 8]) -> u64 {
    u64::from_le_bytes(*bytes)
}

impl Checksum {
    /// Take the whole stripes of `bytes` into the lanes.
    fn stripes(lanes: &mut [u64; 4], bytes: &[[u8; STRIPE]]) {
        for stripe in bytes {
            let (words, _) = stripe.as_chunks::<8>();
            for (lane, word) in lanes.iter_mut().zip(words) {
                *lane = round(*lane, u64_at(word));
            }
        }
    }

    /// Feed `bytes`.
    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        self.total = self.total.wrapping_add(bytes.len() as u64);
        if self.pending_len > 0 {
            let n = (STRIPE - self.pending_len).min(bytes.len());
            self.pending[self.pending_len..self.pending_len + n].copy_from_slice(&bytes[..n]);
            self.pending_len += n;
            bytes = &bytes[n..];
            if self.pending_len < STRIPE {
                return;
            }
            Self::stripes(&mut self.lanes, std::slice::from_ref(&self.pending));
            self.pending_len = 0;
        }
        let (whole, rest) = bytes.as_chunks::<STRIPE>();
        Self::stripes(&mut self.lanes, whole);
        self.pending[..rest.len()].copy_from_slice(rest);
        self.pending_len = rest.len();
    }

    /// The checksum of every byte fed.
    pub(crate) fn finish(&self) -> u64 {
        let [v1, v2, v3, v4] = self.lanes;
        let mut h = match self.total >= STRIPE as u64 {
            true => {
                let h = v1
                    .rotate_left(1)
                    .wrapping_add(v2.rotate_left(7))
                    .wrapping_add(v3.rotate_left(12))
                    .wrapping_add(v4.rotate_left(18));
                [v1, v2, v3, v4].into_iter().fold(h, merge)
            }
            false => P5,
        };
        h = h.wrapping_add(self.total);
        let tail = &self.pending[..self.pending_len];
        let (words, tail) = tail.as_chunks::<8>();
        for word in words {
            h = (h ^ round(0, u64_at(word)))
                .rotate_left(27)
                .wrapping_mul(P1)
                .wrapping_add(P4);
        }
        let (halves, tail) = tail.as_chunks::<4>();
        for half in halves {
            h = (h ^ u64::from(u32::from_le_bytes(*half)).wrapping_mul(P1))
                .rotate_left(23)
                .wrapping_mul(P2)
                .wrapping_add(P3);
        }
        for &byte in tail {
            h = (h ^ u64::from(byte).wrapping_mul(P5))
                .rotate_left(11)
                .wrapping_mul(P1);
        }
        h ^= h >> 33;
        h = h.wrapping_mul(P2);
        h ^= h >> 29;
        h = h.wrapping_mul(P3);
        h ^ (h >> 32)
    }
}

/// The checksum of `bytes`.
pub(crate) fn of(bytes: &[u8]) -> u64 {
    let mut checksum = Checksum::default();
    checksum.update(bytes);
    checksum.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The specification's values, from its reference implementation.
    #[test]
    fn the_checksum_is_xxh64() {
        for (bytes, expected) in [
            (&b""[..], 0xEF46_DB37_51D8_E999),
            (b"a", 0xD24E_C4F1_A98C_6E5B),
            (b"abc", 0x44BC_2CF5_AD77_0999),
            (
                b"Nobody inspects the spammish repetition",
                0xFBCE_A83C_8A37_8BF1,
            ),
        ] {
            assert_eq!(of(bytes), expected, "{:?}", String::from_utf8_lossy(bytes));
        }
    }

    /// Fed in pieces of any size, the checksum is the whole's.
    #[test]
    fn pieces_sum_to_the_whole() {
        let bytes: Vec<u8> = (0..1000u32)
            .map(|i| (i.wrapping_mul(131) % 251) as u8)
            .collect();
        for len in [0, 1, 7, 31, 32, 33, 63, 64, 65, 100, 1000] {
            let whole = of(&bytes[..len]);
            for piece in [1, 3, 8, 31, 32, 33, 200] {
                let mut checksum = Checksum::default();
                for chunk in bytes[..len].chunks(piece) {
                    checksum.update(chunk);
                }
                assert_eq!(checksum.finish(), whole, "{len} bytes in pieces of {piece}");
            }
        }
    }
}
