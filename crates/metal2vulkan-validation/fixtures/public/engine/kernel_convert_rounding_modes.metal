// Owned fixture: `air.convert` with a rounding after its types (`.rte`, `.rtz`, `.rtp`, `.rtn`)
// and `.sat` after it, the forms Apple's OpenCL compiler emits for every `convert_<type>[_sat]
// [_<mode>]`. MSL has no spelling for them, so the kernels call the `rc_*` functions this source
// declares and never defines, and the directives below rename every call to the intrinsic in both
// companions (scripts/regen-air-fixtures). Every operand is read from a buffer, so nothing folds.
#include <metal_stdlib>
using namespace metal;

// Integer to float.
float rc_f_si_rte(int x);
float rc_f_si_rtz(int x);
float rc_f_si_rtp(int x);
float rc_f_si_rtn(int x);
float rc_f_ui_rte(uint x);
float rc_f_ui_rtz(uint x);
float rc_f_ui_rtp(uint x);
float rc_f_ui_rtn(uint x);
half rc_h_si_rte(int x);
half rc_h_si_rtz(int x);
half rc_h_si_rtp(int x);
half rc_h_si_rtn(int x);
half rc_h_us_rte(ushort x);
half rc_h_us_rtz(ushort x);
half rc_h_us_rtp(ushort x);
half rc_h_us_rtn(ushort x);
float rc_f_sl_rte(long x);
float rc_f_sl_rtz(long x);
float rc_f_sl_rtp(long x);
float rc_f_sl_rtn(long x);
float rc_f_ss_rtz(short x);
half rc_h_uc_rtp(uchar x);
half rc_h_sc_rtn(char x);
float4 rc_f4_uc4_rte(uchar4 x);
float4 rc_f4_si4_rte(int4 x);

// Float to integer.
int rc_si_f_rte(float x);
int rc_si_f_rtz(float x);
int rc_si_f_rtp(float x);
int rc_si_f_rtn(float x);
int rc_si_f_rte_sat(float x);
int rc_si_f_rtz_sat(float x);
int rc_si_f_rtp_sat(float x);
int rc_si_f_rtn_sat(float x);
uint rc_ui_f_rtz(float x);
uint rc_ui_f_rtz_sat(float x);
uint rc_ui_f_rte_sat(float x);
uchar rc_uc_f_rte_sat(float x);
char rc_sc_f_rtn_sat(float x);
short rc_ss_h_rtp_sat(half x);
ushort rc_us_h_rte(half x);
long rc_sl_f_rtz_sat(float x);
int4 rc_si4_f4_rte(float4 x);

// Integer to integer.
uchar rc_uc_si_sat(int x);
char rc_sc_si_sat(int x);
uchar rc_uc_ui_sat(uint x);
char rc_sc_ui_sat(uint x);
ushort rc_us_si_sat(int x);
short rc_ss_ui_sat(uint x);
uint rc_ui_si_sat(int x);
int rc_si_ui_sat(uint x);
short rc_ss_si_rtz(int x);
uchar rc_uc_ui_rte(uint x);
int rc_si_sl_sat(long x);
ulong rc_ul_si_sat(int x);
long rc_sl_ul_sat(ulong x);
uint rc_ui_sc_sat(char x);
int rc_si_uc_sat(uchar x);
uchar4 rc_uc4_si4_sat(int4 x);

// Float to float.
half rc_h_f_rte(float x);
half rc_h_f_rtz(float x);
half rc_h_f_rtp(float x);
half rc_h_f_rtn(float x);
float rc_f_h_rtz(half x);
half4 rc_h4_f4_rtz(float4 x);

// air-rename: _Z11rc_f_si_rtei air.convert.f.f32.s.i32.rte
// air-rename: _Z11rc_f_si_rtzi air.convert.f.f32.s.i32.rtz
// air-rename: _Z11rc_f_si_rtpi air.convert.f.f32.s.i32.rtp
// air-rename: _Z11rc_f_si_rtni air.convert.f.f32.s.i32.rtn
// air-rename: _Z11rc_f_ui_rtej air.convert.f.f32.u.i32.rte
// air-rename: _Z11rc_f_ui_rtzj air.convert.f.f32.u.i32.rtz
// air-rename: _Z11rc_f_ui_rtpj air.convert.f.f32.u.i32.rtp
// air-rename: _Z11rc_f_ui_rtnj air.convert.f.f32.u.i32.rtn
// air-rename: _Z11rc_h_si_rtei air.convert.f.f16.s.i32.rte
// air-rename: _Z11rc_h_si_rtzi air.convert.f.f16.s.i32.rtz
// air-rename: _Z11rc_h_si_rtpi air.convert.f.f16.s.i32.rtp
// air-rename: _Z11rc_h_si_rtni air.convert.f.f16.s.i32.rtn
// air-rename: _Z11rc_h_us_rtet air.convert.f.f16.u.i16.rte
// air-rename: _Z11rc_h_us_rtzt air.convert.f.f16.u.i16.rtz
// air-rename: _Z11rc_h_us_rtpt air.convert.f.f16.u.i16.rtp
// air-rename: _Z11rc_h_us_rtnt air.convert.f.f16.u.i16.rtn
// air-rename: _Z11rc_f_sl_rtel air.convert.f.f32.s.i64.rte
// air-rename: _Z11rc_f_sl_rtzl air.convert.f.f32.s.i64.rtz
// air-rename: _Z11rc_f_sl_rtpl air.convert.f.f32.s.i64.rtp
// air-rename: _Z11rc_f_sl_rtnl air.convert.f.f32.s.i64.rtn
// air-rename: _Z11rc_f_ss_rtzs air.convert.f.f32.s.i16.rtz
// air-rename: _Z11rc_h_uc_rtph air.convert.f.f16.u.i8.rtp
// air-rename: _Z11rc_h_sc_rtnc air.convert.f.f16.s.i8.rtn
// air-rename: _Z13rc_f4_uc4_rteDv4_h air.convert.f.v4f32.u.v4i8.rte
// air-rename: _Z13rc_f4_si4_rteDv4_i air.convert.f.v4f32.s.v4i32.rte
// air-rename: _Z11rc_si_f_rtef air.convert.s.i32.f.f32.rte
// air-rename: _Z11rc_si_f_rtzf air.convert.s.i32.f.f32.rtz
// air-rename: _Z11rc_si_f_rtpf air.convert.s.i32.f.f32.rtp
// air-rename: _Z11rc_si_f_rtnf air.convert.s.i32.f.f32.rtn
// air-rename: _Z15rc_si_f_rte_satf air.convert.s.i32.f.f32.rte.sat
// air-rename: _Z15rc_si_f_rtz_satf air.convert.s.i32.f.f32.rtz.sat
// air-rename: _Z15rc_si_f_rtp_satf air.convert.s.i32.f.f32.rtp.sat
// air-rename: _Z15rc_si_f_rtn_satf air.convert.s.i32.f.f32.rtn.sat
// air-rename: _Z11rc_ui_f_rtzf air.convert.u.i32.f.f32.rtz
// air-rename: _Z15rc_ui_f_rtz_satf air.convert.u.i32.f.f32.rtz.sat
// air-rename: _Z15rc_ui_f_rte_satf air.convert.u.i32.f.f32.rte.sat
// air-rename: _Z15rc_uc_f_rte_satf air.convert.u.i8.f.f32.rte.sat
// air-rename: _Z15rc_sc_f_rtn_satf air.convert.s.i8.f.f32.rtn.sat
// air-rename: _Z15rc_ss_h_rtp_satDh air.convert.s.i16.f.f16.rtp.sat
// air-rename: _Z11rc_us_h_rteDh air.convert.u.i16.f.f16.rte
// air-rename: _Z15rc_sl_f_rtz_satf air.convert.s.i64.f.f32.rtz.sat
// air-rename: _Z13rc_si4_f4_rteDv4_f air.convert.s.v4i32.f.v4f32.rte
// air-rename: _Z12rc_uc_si_sati air.convert.u.i8.s.i32.rtz.sat
// air-rename: _Z12rc_sc_si_sati air.convert.s.i8.s.i32.rtz.sat
// air-rename: _Z12rc_uc_ui_satj air.convert.u.i8.u.i32.rtz.sat
// air-rename: _Z12rc_sc_ui_satj air.convert.s.i8.u.i32.rtz.sat
// air-rename: _Z12rc_us_si_sati air.convert.u.i16.s.i32.rtz.sat
// air-rename: _Z12rc_ss_ui_satj air.convert.s.i16.u.i32.rtz.sat
// air-rename: _Z12rc_ui_si_sati air.convert.u.i32.s.i32.rtz.sat
// air-rename: _Z12rc_si_ui_satj air.convert.s.i32.u.i32.rtz.sat
// air-rename: _Z12rc_ss_si_rtzi air.convert.s.i16.s.i32.rtz
// air-rename: _Z12rc_uc_ui_rtej air.convert.u.i8.u.i32.rte
// air-rename: _Z12rc_si_sl_satl air.convert.s.i32.s.i64.rtz.sat
// air-rename: _Z12rc_ul_si_sati air.convert.u.i64.s.i32.rtz.sat
// air-rename: _Z12rc_sl_ul_satm air.convert.s.i64.u.i64.rtz.sat
// air-rename: _Z12rc_ui_sc_satc air.convert.u.i32.s.i8.rtz.sat
// air-rename: _Z12rc_si_uc_sath air.convert.s.i32.u.i8.rtz.sat
// air-rename: _Z14rc_uc4_si4_satDv4_i air.convert.u.v4i8.s.v4i32.rtz.sat
// air-rename: _Z10rc_h_f_rtef air.convert.f.f16.f.f32.rte
// air-rename: _Z10rc_h_f_rtzf air.convert.f.f16.f.f32.rtz
// air-rename: _Z10rc_h_f_rtpf air.convert.f.f16.f.f32.rtp
// air-rename: _Z10rc_h_f_rtnf air.convert.f.f16.f.f32.rtn
// air-rename: _Z10rc_f_h_rtzDh air.convert.f.f32.f.f16.rtz
// air-rename: _Z12rc_h4_f4_rtzDv4_f air.convert.f.v4f16.f.v4f32.rtz

static uint hbits(half h) { return uint(as_type<ushort>(h)); }

// Thread i of n converts in[i] (and, for the 64-bit and vector forms, its neighbours) every way,
// writing 31 words from out[31 * i]: the `float` of the signed word by rte, rtz, rtp, rtn; of the
// unsigned word; the `half` bits of the signed word; of its low 16 bits; the `float` of a 64-bit
// value (high word in[i], low word in[(i + 5) % n]) by rte, rtz, rtp, rtn; the `float` of its low
// 16 bits as a `short` (exact, by rtz); the `half` of its low byte (rtp, unsigned; rtn, signed);
// and `float4`s of its four bytes and of four words around it, by rte.
kernel void kernel_convert_rounding_int_to_float(device const uint *in [[buffer(0)]],
                                                 device uint *out [[buffer(1)]],
                                                 uint i [[thread_position_in_grid]],
                                                 uint n [[threads_per_grid]]) {
    uint u = in[i];
    int s = as_type<int>(u);
    long l = as_type<long>(uint2(in[(i + 5) % n], u));
    device uint *o = out + 31 * i;
    o[0] = as_type<uint>(rc_f_si_rte(s));
    o[1] = as_type<uint>(rc_f_si_rtz(s));
    o[2] = as_type<uint>(rc_f_si_rtp(s));
    o[3] = as_type<uint>(rc_f_si_rtn(s));
    o[4] = as_type<uint>(rc_f_ui_rte(u));
    o[5] = as_type<uint>(rc_f_ui_rtz(u));
    o[6] = as_type<uint>(rc_f_ui_rtp(u));
    o[7] = as_type<uint>(rc_f_ui_rtn(u));
    o[8] = hbits(rc_h_si_rte(s));
    o[9] = hbits(rc_h_si_rtz(s));
    o[10] = hbits(rc_h_si_rtp(s));
    o[11] = hbits(rc_h_si_rtn(s));
    ushort us = ushort(u);
    o[12] = hbits(rc_h_us_rte(us));
    o[13] = hbits(rc_h_us_rtz(us));
    o[14] = hbits(rc_h_us_rtp(us));
    o[15] = hbits(rc_h_us_rtn(us));
    o[16] = as_type<uint>(rc_f_sl_rte(l));
    o[17] = as_type<uint>(rc_f_sl_rtz(l));
    o[18] = as_type<uint>(rc_f_sl_rtp(l));
    o[19] = as_type<uint>(rc_f_sl_rtn(l));
    o[20] = as_type<uint>(rc_f_ss_rtz(short(u)));
    o[21] = hbits(rc_h_uc_rtp(uchar(u)));
    o[22] = hbits(rc_h_sc_rtn(char(u)));
    float4 b = rc_f4_uc4_rte(as_type<uchar4>(u));
    float4 w = rc_f4_si4_rte(int4(s, as_type<int>(in[(i + 1) % n]), as_type<int>(in[(i + 2) % n]),
                                  as_type<int>(in[(i + 3) % n])));
    o[23] = as_type<uint>(b.x);
    o[24] = as_type<uint>(b.y);
    o[25] = as_type<uint>(b.z);
    o[26] = as_type<uint>(b.w);
    o[27] = as_type<uint>(w.x);
    o[28] = as_type<uint>(w.y);
    o[29] = as_type<uint>(w.z);
    o[30] = as_type<uint>(w.w);
}

// Thread i converts the `float` with bits in[i] to integers, writing 24 words from out[24 * i]:
// `int` by rte, rtz, rtp, rtn; the same saturating; `uint` by rtz, saturating rtz and rte;
// `uchar` (rte) and `char` (rtn) saturating; `short` saturating (rtp) and `ushort` (rte) of its
// `half`; the two words of a saturating `long` (rtz); an `int4` (rte) of it and its neighbours.
kernel void kernel_convert_rounding_float_to_int(device const uint *in [[buffer(0)]],
                                                 device uint *out [[buffer(1)]],
                                                 uint i [[thread_position_in_grid]],
                                                 uint n [[threads_per_grid]]) {
    float x = as_type<float>(in[i]);
    half h = half(x);
    device uint *o = out + 24 * i;
    o[0] = as_type<uint>(rc_si_f_rte(x));
    o[1] = as_type<uint>(rc_si_f_rtz(x));
    o[2] = as_type<uint>(rc_si_f_rtp(x));
    o[3] = as_type<uint>(rc_si_f_rtn(x));
    o[4] = as_type<uint>(rc_si_f_rte_sat(x));
    o[5] = as_type<uint>(rc_si_f_rtz_sat(x));
    o[6] = as_type<uint>(rc_si_f_rtp_sat(x));
    o[7] = as_type<uint>(rc_si_f_rtn_sat(x));
    o[8] = rc_ui_f_rtz(x);
    o[9] = rc_ui_f_rtz_sat(x);
    o[10] = rc_ui_f_rte_sat(x);
    o[11] = uint(rc_uc_f_rte_sat(x));
    o[12] = as_type<uint>(int(rc_sc_f_rtn_sat(x)));
    o[13] = as_type<uint>(int(rc_ss_h_rtp_sat(h)));
    o[14] = uint(rc_us_h_rte(h));
    uint2 q = as_type<uint2>(rc_sl_f_rtz_sat(x));
    o[15] = q.x;
    o[16] = q.y;
    int4 v = rc_si4_f4_rte(float4(x, as_type<float>(in[(i + 1) % n]),
                                  as_type<float>(in[(i + 2) % n]), as_type<float>(in[(i + 3) % n])));
    o[17] = as_type<uint>(v.x);
    o[18] = as_type<uint>(v.y);
    o[19] = as_type<uint>(v.z);
    o[20] = as_type<uint>(v.w);
    o[21] = hbits(h);
    o[22] = in[i];
    o[23] = 0x5a5a5a5au;
}

// Thread i converts the word in[i] between integer types, writing 20 words from out[20 * i]:
// saturating to `uchar`, `char` from it as `int` and as `uint`; to `ushort` from `int`; to `short`
// from `uint`; to `uint` from `int` and back; the plain `short` of it (rtz) and `uchar` (rte);
// saturating `int` of a 64-bit value (high word in[i], low word in[(i + 7) % n]); the two words of
// the saturating `ulong` of its `int`; the two words of the saturating `long` of the 64-bit value
// as a `ulong`; saturating `uint` of its low byte as `char`; `int` of it as `uchar`; and the four
// saturating `uchar`s of it and its neighbours as `int`s, packed.
kernel void kernel_convert_rounding_int_to_int(device const uint *in [[buffer(0)]],
                                               device uint *out [[buffer(1)]],
                                               uint i [[thread_position_in_grid]],
                                               uint n [[threads_per_grid]]) {
    uint u = in[i];
    int s = as_type<int>(u);
    long l = as_type<long>(uint2(in[(i + 7) % n], u));
    device uint *o = out + 20 * i;
    o[0] = uint(rc_uc_si_sat(s));
    o[1] = as_type<uint>(int(rc_sc_si_sat(s)));
    o[2] = uint(rc_uc_ui_sat(u));
    o[3] = as_type<uint>(int(rc_sc_ui_sat(u)));
    o[4] = uint(rc_us_si_sat(s));
    o[5] = as_type<uint>(int(rc_ss_ui_sat(u)));
    o[6] = rc_ui_si_sat(s);
    o[7] = as_type<uint>(rc_si_ui_sat(u));
    o[8] = as_type<uint>(int(rc_ss_si_rtz(s)));
    o[9] = uint(rc_uc_ui_rte(u));
    o[10] = as_type<uint>(rc_si_sl_sat(l));
    uint2 a = as_type<uint2>(rc_ul_si_sat(s));
    o[11] = a.x;
    o[12] = a.y;
    uint2 b = as_type<uint2>(rc_sl_ul_sat(as_type<ulong>(l)));
    o[13] = b.x;
    o[14] = b.y;
    o[15] = rc_ui_sc_sat(char(u));
    o[16] = as_type<uint>(rc_si_uc_sat(uchar(u)));
    o[17] = as_type<uint>(rc_uc4_si4_sat(int4(s, as_type<int>(in[(i + 1) % n]),
                                              as_type<int>(in[(i + 2) % n]),
                                              as_type<int>(in[(i + 3) % n]))));
    o[18] = u;
    o[19] = 0xa5a5a5a5u;
}

// Thread i narrows the `float` with bits in[i] to `half` by rte, rtz, rtp and rtn, widens the
// first back by rtz, and narrows it and its neighbours as a `float4` by rtz: 8 words from
// out[8 * i], the `half`s as their bits.
kernel void kernel_convert_rounding_float_to_float(device const uint *in [[buffer(0)]],
                                                   device uint *out [[buffer(1)]],
                                                   uint i [[thread_position_in_grid]],
                                                   uint n [[threads_per_grid]]) {
    float x = as_type<float>(in[i]);
    device uint *o = out + 8 * i;
    half e = rc_h_f_rte(x);
    o[0] = hbits(e);
    o[1] = hbits(rc_h_f_rtz(x));
    o[2] = hbits(rc_h_f_rtp(x));
    o[3] = hbits(rc_h_f_rtn(x));
    o[4] = as_type<uint>(rc_f_h_rtz(e));
    half4 v = rc_h4_f4_rtz(float4(x, as_type<float>(in[(i + 1) % n]),
                                  as_type<float>(in[(i + 2) % n]), as_type<float>(in[(i + 3) % n])));
    uint2 p = as_type<uint2>(v);
    o[5] = p.x;
    o[6] = p.y;
    o[7] = in[i];
}
