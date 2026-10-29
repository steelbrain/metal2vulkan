//! Metal's twelve-bit 8-bit sRGB decode steps, which A1 read in full: the step of every byte.
//! The texture decoders that answer Metal's decoded texels read them (`bind::astc_decode`,
//! `bind::pvrtc_decode`): the linear value of byte `b` is `k / 4095` for its step `k`.
//! `air.pack`/`air.unpack` of `unorm4x8_srgb`, a storage texture's alias, a UNORM twin and the
//! mipmap filter take the MSL transfer through the device's `Pow` instead
//! (`lower::arith::pack`).

/// The decode step of every byte `b`: Metal's `unpack_unorm4x8_srgb_to_float` answers `k / 4095`
/// (correctly rounded) for `k = SRGB_DECODE_STEPS[b]`, which is `round(decode(b / 255) * 4095)`
/// for the MSL transfer `decode(c) = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4`
/// (every margin from a half step is at least `1.6e-4` step). Every color word of A1's
/// `srgb-decode-of-every-byte` (bcb3682b, 768 of 768) and the old engine's
/// `kernel_srgb_unpack_all_bytes` read these steps.
pub(crate) const SRGB_DECODE_STEPS: [u16; 256] = [
    0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 40,
    42, 45, 48, 50, 53, 56, 59, 62, 66, 69, 72, 76, 79, 83, 87, 91, 95, 99, 103, 107, 112, 116,
    121, 126, 131, 136, 141, 146, 151, 156, 162, 168, 173, 179, 185, 191, 197, 204, 210, 216, 223,
    230, 237, 244, 251, 258, 265, 273, 280, 288, 296, 304, 312, 320, 329, 337, 346, 354, 363, 372,
    381, 390, 400, 409, 419, 428, 438, 448, 458, 469, 479, 490, 500, 511, 522, 533, 544, 555, 567,
    578, 590, 602, 614, 626, 639, 651, 664, 676, 689, 702, 715, 728, 742, 755, 769, 783, 797, 811,
    825, 840, 854, 869, 884, 899, 914, 929, 945, 960, 976, 992, 1008, 1024, 1041, 1057, 1074, 1091,
    1108, 1125, 1142, 1159, 1177, 1195, 1213, 1231, 1249, 1267, 1286, 1304, 1323, 1342, 1361, 1381,
    1400, 1420, 1440, 1459, 1480, 1500, 1520, 1541, 1562, 1582, 1603, 1625, 1646, 1668, 1689, 1711,
    1733, 1755, 1778, 1800, 1823, 1846, 1869, 1892, 1916, 1939, 1963, 1987, 2011, 2035, 2059, 2084,
    2109, 2133, 2159, 2184, 2209, 2235, 2260, 2286, 2312, 2339, 2365, 2392, 2419, 2446, 2473, 2500,
    2527, 2555, 2583, 2611, 2639, 2668, 2696, 2725, 2754, 2783, 2812, 2841, 2871, 2901, 2931, 2961,
    2991, 3022, 3052, 3083, 3114, 3146, 3177, 3209, 3240, 3272, 3304, 3337, 3369, 3402, 3435, 3468,
    3501, 3535, 3568, 3602, 3636, 3670, 3705, 3739, 3774, 3809, 3844, 3879, 3915, 3950, 3986, 4022,
    4059, 4095,
];
