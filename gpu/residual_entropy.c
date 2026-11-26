/* Context-mixing adaptive binary arithmetic coder for integer residual streams Q[T][N][3] (AEZ-12, version 3).
 *
 * Prediction (repeated exactly by the decoder), per atom class g, in units of the residual step s:
 *   pred = round((w0_g*fC + w1_g*Q[t][ref] + w2_g*Q[t-1][i] + w3_g*Q[t-1][ref]) / 64)
 *   fC = (C - base) / s, where C is the MDCompress style "translate and rotate" prediction: the atom's reconstructed
 *   position in frame t-1, expressed in the local frame of its three reference atoms at t-1 and placed in their
 *   local frame at t (references: already decoded atoms of this stream in this frame, or atoms of a context stream
 *   decoded before, e.g. the backbone for the side chain stream). base is the float predictor of the hybrid codec
 *   (mean structure + rigid fit, or the packer), so Q = round((x - base)/s) and x_rec = base + s Q.
 *   MDCompress level 2 ("moving_1vector") corresponds to w = (0, 64, 64, -64).
 *   Floating point is used only for fC, with FP contraction disabled, so encoder and decoder (same build) agree.
 * Binarisation of e = Q - pred: zero flag | sign | Elias-gamma length in unary | mantissa bits.
 * Each zero flag, unary and sign bit is predicted by several adaptive context models whose outputs are combined by
 * a gated logistic mixer (as in the PAQ/lpaq family), followed by an adaptive probability map (APM); mantissa bits use
 * single adaptive models. Contexts are built only from already coded data: atom class g, axis, and the magnitudes and
 * signs of the coding errors of the same atom in the two previous frames, of the reference atom in this and the
 * previous frame, and of the previous axis of the same atom.
 * Range coder: LZMA style binary range coder with carry propagation, 16 bit probabilities.
 * Everything is integer arithmetic, so encoder and decoder are bit exact on any platform. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NG 16
#define NB 20
#define NQ 8
#define NU 20
#define NBIT 32

/* ---------------- stretch / squash (12 bit), integer tables */
static int16_t STRETCH[4096];
static int squash_i(int d) { /* 4096 / (1 + exp(-d/256)), integer interpolation as in lpaq1 */
    static const int t[33] = {1,2,3,6,10,16,27,45,73,120,194,310,488,747,1101,1546,2047,2549,2994,3348,3607,3785,3901,3975,4024,4050,4068,4079,4085,4089,4092,4093,4094};
    if (d > 2047) return 4095; if (d < -2047) return 1;
    int w = d & 127; d = (d >> 7) + 16;
    return (t[d] * (128 - w) + t[d + 1] * w + 64) >> 7;
}
static int tables_ready = 0;
static void init_tables(void) {
    if (tables_ready) return;
    int pi = 0;
    for (int x = -2047; x <= 2047; x++) { int v = squash_i(x); for (int j = pi; j <= v; j++) STRETCH[j] = (int16_t)x; pi = v + 1; }
    for (int j = pi; j < 4096; j++) STRETCH[j] = 2047;
    tables_ready = 1;
}

/* ---------------- adaptive bit model: two rates, 16 bit probability of bit = 1 */
typedef struct { uint16_t a, b; } bm_t;
static inline int bm_p12(const bm_t *m) { int p = ((int)m->a + (int)m->b) >> 5; return p < 1 ? 1 : (p > 4095 ? 4095 : p); }
static inline void bm_upd(bm_t *m, int bit) {
    if (bit) { m->a += (65535 - m->a) >> 4; m->b += (65535 - m->b) >> 7; }
    else { m->a -= m->a >> 4; m->b -= m->b >> 7; }
}

/* ---------------- mixer */
#define NIN 8
typedef struct { int32_t w[NIN]; } mixw_t;
/* APM: 33 interpolation points per context */
typedef struct { uint16_t t[33]; } apm_t;
static void apm_init(apm_t *a) { for (int j = 0; j < 33; j++) a->t[j] = (uint16_t)(squash_i((j - 16) * 128) * 16); }

typedef struct {
    int nin; int st[NIN]; bm_t *m[NIN]; mixw_t *w; int pmix; apm_t *apm; int aidx, aw; int pfinal;
} pred_t;

static inline int mix_predict(pred_t *P) {
    int64_t dot = 0;
    for (int i = 0; i < P->nin; i++) { P->st[i] = STRETCH[bm_p12(P->m[i])]; dot += (int64_t)P->w->w[i] * P->st[i]; }
    P->st[P->nin] = 256; dot += (int64_t)P->w->w[P->nin] * 256;   /* bias input */
    int d = (int)(dot >> 16); if (d > 2047) d = 2047; if (d < -2047) d = -2047;
    P->pmix = squash_i(d);
    /* APM on the mixed probability */
    int s = d + 2048; int lo = s >> 7; int wgt = s & 127; P->aidx = lo; P->aw = wgt;
    int pa = ((int)P->apm->t[lo] * (128 - wgt) + (int)P->apm->t[lo + 1] * wgt) >> 11;   /* 12 bit */
    int p = (P->pmix + 3 * pa + 2) >> 2; if (p < 1) p = 1; if (p > 4095) p = 4095;
    P->pfinal = p;
    return p;
}
static inline void mix_update(pred_t *P, int bit) {
    int err = ((bit << 12) - P->pmix) * 6;
    for (int i = 0; i <= P->nin; i++) P->w->w[i] += (P->st[i] * err + 0x200) >> 10;
    for (int i = 0; i < P->nin; i++) bm_upd(P->m[i], bit);
    int target = bit ? 65535 : 0;
    int lo = P->aidx;
    P->apm->t[lo] += (target - (int)P->apm->t[lo]) * (128 - P->aw) >> 13;
    P->apm->t[lo + 1] += (target - (int)P->apm->t[lo + 1]) * P->aw >> 13;
}

/* ---------------- model tables */
typedef struct {
    /* zero flag */
    bm_t z1[NG][3][NB], z2[NG][NQ][NQ], z3[3][NQ][NQ][2], z4[NG][NQ][NQ], z5[NB][NQ], z6[NG][NQ][NQ];
    mixw_t zw[NG][NB]; apm_t za[NG][3];
    /* unary length bits */
    bm_t u1[NG][NB][NU], u2[NU][NQ][NQ], u3[NU][3][NQ], u4[NG][NU][NQ], u5[NU][NB], u6[NU][NQ][NQ];
    mixw_t uw[NU][NB]; apm_t ua[NU];
    /* sign */
    bm_t s1[NG][3][3][3], s2[3][3][3][3], s3[NG][3][3];
    mixw_t sw[NG]; apm_t sa[3];
    /* mantissa */
    bm_t mtop[NBIT][NB], mbit[NBIT][NBIT];
} model_t;

static void model_init(model_t *m) {
    memset(m, 0, sizeof(*m));
    bm_t *b = (bm_t *)m;   /* the bm_t blocks come first, then mixers/apms are re-initialised below */
    (void)b;
#define INITBM(arr) do { bm_t *p_ = (bm_t *)(arr); size_t n_ = sizeof(arr) / sizeof(bm_t); for (size_t k_ = 0; k_ < n_; k_++) { p_[k_].a = 32768; p_[k_].b = 32768; } } while (0)
#define INITW(arr, n0) do { mixw_t *p_ = (mixw_t *)(arr); size_t n_ = sizeof(arr) / sizeof(mixw_t); for (size_t k_ = 0; k_ < n_; k_++) { for (int i_ = 0; i_ < NIN; i_++) p_[k_].w[i_] = 0; for (int i_ = 0; i_ < (n0); i_++) p_[k_].w[i_] = 65536 / (n0); } } while (0)
#define INITA(arr) do { apm_t *p_ = (apm_t *)(arr); size_t n_ = sizeof(arr) / sizeof(apm_t); for (size_t k_ = 0; k_ < n_; k_++) apm_init(&p_[k_]); } while (0)
    INITBM(m->z1); INITBM(m->z2); INITBM(m->z3); INITBM(m->z4); INITBM(m->z5); INITBM(m->z6);
    INITBM(m->u1); INITBM(m->u2); INITBM(m->u3); INITBM(m->u4); INITBM(m->u5); INITBM(m->u6);
    INITBM(m->s1); INITBM(m->s2); INITBM(m->s3); INITBM(m->mtop); INITBM(m->mbit);
    INITW(m->zw, 6); INITW(m->uw, 6); INITW(m->sw, 3);
    INITA(m->za); INITA(m->ua); INITA(m->sa);
}

/* ---------------- range coder (bit = 1 has probability p12/4096) */
typedef struct { uint64_t low; uint32_t range; uint8_t cache; uint64_t csize; uint8_t *out; int64_t pos, cap; int err; } enc_t;
static inline void putb(enc_t *e, uint8_t v) { if (e->pos < e->cap) e->out[e->pos] = v; else e->err = 1; e->pos++; }
static inline void shift_low(enc_t *e) {
    if ((uint32_t)e->low < 0xFF000000u || (e->low >> 32) != 0) {
        uint8_t temp = e->cache;
        do { putb(e, (uint8_t)(temp + (uint8_t)(e->low >> 32))); temp = 0xFF; } while (--e->csize != 0);
        e->cache = (uint8_t)(e->low >> 24);
    }
    e->csize++;
    e->low = (e->low & 0x00FFFFFFu) << 8;
}
static inline void enc_raw(enc_t *e, int p12, int bit) {   /* p12 = P(bit = 1) */
    uint32_t p0 = (uint32_t)(4096 - p12) << 4;
    uint32_t bound = (e->range >> 16) * p0;
    if (!bit) e->range = bound; else { e->low += bound; e->range -= bound; }
    while (e->range < (1u << 24)) { e->range <<= 8; shift_low(e); }
}
typedef struct { uint32_t code, range; const uint8_t *in; int64_t pos, len; } dec_t;
static inline uint8_t getb(dec_t *d) { return d->pos < d->len ? d->in[d->pos++] : (d->pos++, 0); }
static inline int dec_raw(dec_t *d, int p12) {
    uint32_t p0 = (uint32_t)(4096 - p12) << 4;
    uint32_t bound = (d->range >> 16) * p0; int bit;
    if (d->code < bound) { d->range = bound; bit = 0; } else { d->code -= bound; d->range -= bound; bit = 1; }
    while (d->range < (1u << 24)) { d->range <<= 8; d->code = (d->code << 8) | getb(d); }
    return bit;
}

/* coder abstraction: one function body for encode and decode */
typedef struct { int dec; enc_t e; dec_t d; } coder_t;
static inline int code_mixed(coder_t *c, pred_t *P, int bit) {
    int p = mix_predict(P);
    if (c->dec) bit = dec_raw(&c->d, p); else enc_raw(&c->e, p, bit);
    mix_update(P, bit);
    return bit;
}
static inline int code_simple(coder_t *c, bm_t *m, int bit) {
    int p = bm_p12(m);
    if (c->dec) bit = dec_raw(&c->d, p); else enc_raw(&c->e, p, bit);
    bm_upd(m, bit);
    return bit;
}

/* ---------------- helpers */
static inline int bitlen(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
static inline int bucket(uint32_t act) {
    if (act < 4) return (int)act;
    int n = bitlen(act); int b = 4 + 2 * (n - 3) + (int)((act >> (n - 2)) & 1);
    return b < NB ? b : NB - 1;
}
static inline int qb(uint32_t v) { return v < 3 ? (int)v : (v < 5 ? 3 : (v < 9 ? 4 : (v < 17 ? 5 : (v < 65 ? 6 : 7)))); }
static inline int64_t floordiv16(int64_t s) { return (s >= 0) ? (s >> 4) : -((-s + 15) >> 4); }
static inline int sgn3(int32_t v) { return v > 0 ? 2 : (v < 0 ? 1 : 0); }
static inline uint32_t uabs(int32_t v) { return v < 0 ? (uint32_t)(-(int64_t)v) : (uint32_t)v; }

typedef struct { const double *base; double s; const double *Xc; int Nc; const int32_t *R3; } geo_t;

static inline int frame3(const double *a, const double *b, const double *c, double F[9]) {
    double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}; double n1 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
    if (!(n1 > 1e-8)) return 0; n1 = 1.0 / __builtin_sqrt(n1); e1[0] *= n1; e1[1] *= n1; e1[2] *= n1;
    double u[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]}; double d = u[0] * e1[0] + u[1] * e1[1] + u[2] * e1[2];
    u[0] -= d * e1[0]; u[1] -= d * e1[1]; u[2] -= d * e1[2]; double n2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    if (!(n2 > 1e-6)) return 0; n2 = 1.0 / __builtin_sqrt(n2); u[0] *= n2; u[1] *= n2; u[2] *= n2;
    double e3[3] = {e1[1] * u[2] - e1[2] * u[1], e1[2] * u[0] - e1[0] * u[2], e1[0] * u[1] - e1[1] * u[0]};
    F[0] = e1[0]; F[1] = e1[1]; F[2] = e1[2]; F[3] = u[0]; F[4] = u[1]; F[5] = u[2]; F[6] = e3[0]; F[7] = e3[1]; F[8] = e3[2];
    return 1;
}

static int run(coder_t *c, int32_t *Q, int T, int N, const int32_t *ref, const uint8_t *cls, const int16_t *coef, const geo_t *G) {
    double *REC = (double *)malloc(sizeof(double) * (size_t)N * 3 * 2);  /* reconstructed coordinates, frames t-1 and t */
    double fC3[3];
    init_tables();
    model_t *m = (model_t *)malloc(sizeof(model_t)); if (!m) return -2; model_init(m);
    int32_t *EB[3]; for (int k = 0; k < 3; k++) EB[k] = (int32_t *)calloc((size_t)N * 3, 4);
    for (int t = 0; t < T; t++) {
        int32_t *Qc = Q + (int64_t)t * N * 3; const int32_t *Qp = t ? Q + (int64_t)(t - 1) * N * 3 : NULL;
        int32_t *Ec = EB[t % 3]; const int32_t *Ep = t ? EB[(t + 2) % 3] : NULL; const int32_t *Epp = t > 1 ? EB[(t + 1) % 3] : NULL;
        for (int i = 0; i < N; i++) for (int ax = 0; ax < 3; ax++) {
            int g = cls[i] & (NG - 1); int r = ref[i];
            int32_t qr = (r >= 0) ? Qc[3 * r + ax] : 0; int32_t qp = Qp ? Qp[3 * i + ax] : 0;
            int32_t qpr = (Qp && r >= 0) ? Qp[3 * r + ax] : 0;
            const int16_t *cf = coef + 4 * g;
            if (ax == 0) {   /* geometric prediction for the whole atom, once per atom and frame */
                fC3[0] = fC3[1] = fC3[2] = 0.0;
                if (t > 0) {
                    double *Rc = REC + (size_t)(t & 1) * N * 3, *Rp = REC + (size_t)((t + 1) & 1) * N * 3;
                    const double *pc[3], *pp[3]; int okr = 1;
                    for (int k = 0; k < 3; k++) {
                        int u = G->R3[3 * i + k];
                        if (u < 0) { okr = 0; break; }
                        if (u < G->Nc) { pc[k] = G->Xc + ((size_t)t * G->Nc + u) * 3; pp[k] = G->Xc + ((size_t)(t - 1) * G->Nc + u) * 3; }
                        else { u -= G->Nc; if (u >= i) { okr = 0; break; } pc[k] = Rc + (size_t)u * 3; pp[k] = Rp + (size_t)u * 3; }
                    }
                    const double *bi = G->base + ((size_t)t * N + i) * 3; const double *xp = Rp + (size_t)i * 3;
                    double Fp[9], Fc[9];
                    if (okr && frame3(pp[0], pp[1], pp[2], Fp) && frame3(pc[0], pc[1], pc[2], Fc)) {
                        double v[3] = {xp[0] - pp[0][0], xp[1] - pp[0][1], xp[2] - pp[0][2]};
                        double l0 = Fp[0] * v[0] + Fp[1] * v[1] + Fp[2] * v[2], l1 = Fp[3] * v[0] + Fp[4] * v[1] + Fp[5] * v[2], l2 = Fp[6] * v[0] + Fp[7] * v[1] + Fp[8] * v[2];
                        for (int k = 0; k < 3; k++) fC3[k] = (pc[0][k] + l0 * Fc[k] + l1 * Fc[3 + k] + l2 * Fc[6 + k] - bi[k]) / G->s;
                    } else {
                        for (int k = 0; k < 3; k++) fC3[k] = (xp[k] - (G->base + ((size_t)(t - 1) * N + i) * 3)[k]) / G->s;  /* fallback: previous residual */
                    }
                }
            }
            double acc = (double)cf[0] * fC3[ax] + (double)cf[1] * (double)qr + (double)cf[2] * (double)qp + (double)cf[3] * (double)qpr;
            double pr = __builtin_floor(acc / 64.0 + 0.5);
            if (pr > 1e9) pr = 1e9; if (pr < -1e9) pr = -1e9;
            int64_t pred = (int64_t)pr;
            /* disagreement between the prediction and the previous residual of the atom (uncertainty of the predictor) */
            uint32_t dis = uabs((int32_t)(pred - (int64_t)qp)); if (t == 0) dis = 0;
            uint32_t gdis = (uint32_t)__builtin_fabs(fC3[ax] - (double)qp); if (gdis > 100000) gdis = 100000;
            int32_t ep = Ep ? Ep[3 * i + ax] : 0, epp = Epp ? Epp[3 * i + ax] : 0;
            int32_t er = (r >= 0) ? Ec[3 * r + ax] : 0, epr = (Ep && r >= 0) ? Ep[3 * r + ax] : 0;
            int32_t ea = ax ? Ec[3 * i + ax - 1] : 0, eb = ax == 2 ? Ec[3 * i] : 0;
            uint32_t aep = uabs(ep), aer = uabs(er), aepr = uabs(epr), aea = uabs(ea), aepp = uabs(epp);
            int bk = bucket(2 * aep + 2 * aer + aepr + aea);
            int bt = bucket(2 * aep + aepp + aepr); /* temporal activity */
            int32_t v = 0;
            if (!c->dec) {
                int64_t ev = (int64_t)Qc[3 * i + ax] - pred;
                if (ev > 2000000000LL || ev < -2000000000LL) { free(m); free(REC); for (int k = 0; k < 3; k++) free(EB[k]); return -3; }
                v = (int32_t)ev;
            }
            /* zero flag (bit = 1 means nonzero) */
            pred_t P; P.nin = 6; P.w = &m->zw[g][bk]; P.apm = &m->za[g][ax];
            P.m[0] = &m->z1[g][ax][bk]; P.m[1] = &m->z2[g][qb(aep)][qb(aer)]; P.m[2] = &m->z3[ax][qb(aepr)][qb(aea + uabs(eb))][aepp > 0];
            P.m[3] = &m->z4[g][qb(aepp)][qb(aep)]; P.m[4] = &m->z5[bt][qb(aer)]; P.m[5] = &m->z6[g][qb(dis)][qb(gdis)];
            int nz = code_mixed(c, &P, v != 0);
            if (nz) {
                /* sign (bit = 1 means negative) */
                pred_t S; S.nin = 3; S.w = &m->sw[g]; S.apm = &m->sa[ax];
                S.m[0] = &m->s1[g][ax][sgn3(er)][sgn3(ep)]; S.m[1] = &m->s2[ax][sgn3(epr)][sgn3(ea)][sgn3(epp)]; S.m[2] = &m->s3[g][sgn3(qr - qpr)][sgn3(qp)];
                int neg = code_mixed(c, &S, v < 0);
                uint32_t a = c->dec ? 0 : uabs(v); int n = c->dec ? 0 : bitlen(a);
                /* unary length: bits 1 while j < n-1, then 0 */
                int j = 0;
                for (;;) {
                    if (j >= 31) break;
                    int jj = j < NU ? j : NU - 1;
                    pred_t U; U.nin = 6; U.w = &m->uw[jj][bk]; U.apm = &m->ua[jj];
                    U.m[0] = &m->u1[g][bk][jj]; U.m[1] = &m->u2[jj][qb(aep)][qb(aer)]; U.m[2] = &m->u3[jj][ax][qb(aepr + aea)];
                    U.m[3] = &m->u4[g][jj][qb(aepp)]; U.m[4] = &m->u5[jj][bt]; U.m[5] = &m->u6[jj][qb(dis)][qb(gdis)];
                    int more = code_mixed(c, &U, j < n - 1);
                    if (!more) break;
                    j++;
                }
                if (c->dec) n = j + 1;
                uint32_t val = 1;
                for (int k = n - 2; k >= 0; k--) {
                    int bit = c->dec ? 0 : (int)((a >> k) & 1);
                    bit = (k == n - 2) ? code_simple(c, &m->mtop[n][bk], bit) : code_simple(c, &m->mbit[n][k], bit);
                    val = (val << 1) | (uint32_t)bit;
                }
                if (c->dec) v = neg ? -(int32_t)val : (int32_t)val;
            }
            Ec[3 * i + ax] = v;
            if (c->dec) Qc[3 * i + ax] = (int32_t)(pred + v);
            REC[(size_t)(t & 1) * N * 3 + (size_t)i * 3 + ax] = G->base[((size_t)t * N + i) * 3 + ax] + G->s * (double)Qc[3 * i + ax];
        }
    }
    free(m); free(REC); for (int k = 0; k < 3; k++) free(EB[k]);
    return 0;
}

int64_t re_encode(const int32_t *Q, int T, int N, const int32_t *ref, const uint8_t *cls, const int16_t *coef,
                  const double *base, double s, const double *Xc, int Nc, const int32_t *R3, uint8_t *out, int64_t cap) {
    geo_t G = {base, s, Xc, Nc, R3};
    coder_t c; memset(&c, 0, sizeof(c)); c.dec = 0;
    c.e.low = 0; c.e.range = 0xFFFFFFFFu; c.e.cache = 0; c.e.csize = 1; c.e.out = out; c.e.pos = 0; c.e.cap = cap; c.e.err = 0;
    int rc = run(&c, (int32_t *)Q, T, N, ref, cls, coef, &G);
    if (rc) return rc;
    for (int k = 0; k < 5; k++) shift_low(&c.e);
    return c.e.err ? -1 : c.e.pos;
}

int re_decode(const uint8_t *in, int64_t len, int T, int N, const int32_t *ref, const uint8_t *cls, const int16_t *coef,
              const double *base, double s, const double *Xc, int Nc, const int32_t *R3, int32_t *Q) {
    geo_t G = {base, s, Xc, Nc, R3};
    coder_t c; memset(&c, 0, sizeof(c)); c.dec = 1;
    c.d.code = 0; c.d.range = 0xFFFFFFFFu; c.d.in = in; c.d.pos = 0; c.d.len = len;
    for (int k = 0; k < 5; k++) c.d.code = (c.d.code << 8) | getb(&c.d);
    int rc = run(&c, Q, T, N, ref, cls, coef, &G);
    if (rc) return rc;
    return c.d.pos > len + 8 ? -1 : 0;
}

/* encoder side helper: the geometric feature fC for every atom, frame and axis, from the true integers Q */
int re_features(const int32_t *Q, int T, int N, const double *base, double s, const double *Xc, int Nc, const int32_t *R3, double *out) {
    double *REC = (double *)malloc(sizeof(double) * (size_t)N * 3 * T);
    for (size_t k = 0; k < (size_t)T * N * 3; k++) REC[k] = base[k] + s * (double)Q[k];
    for (int t = 0; t < T; t++) for (int i = 0; i < N; i++) {
        double *o = out + ((size_t)t * N + i) * 3; o[0] = o[1] = o[2] = 0.0;
        if (t == 0) continue;
        const double *pc[3], *pp[3]; int okr = 1;
        for (int k = 0; k < 3; k++) {
            int u = R3[3 * i + k];
            if (u < 0) { okr = 0; break; }
            if (u < Nc) { pc[k] = Xc + ((size_t)t * Nc + u) * 3; pp[k] = Xc + ((size_t)(t - 1) * Nc + u) * 3; }
            else { u -= Nc; if (u >= i) { okr = 0; break; } pc[k] = REC + ((size_t)t * N + u) * 3; pp[k] = REC + ((size_t)(t - 1) * N + u) * 3; }
        }
        const double *bi = base + ((size_t)t * N + i) * 3; const double *xp = REC + ((size_t)(t - 1) * N + i) * 3;
        double Fp[9], Fc[9];
        if (okr && frame3(pp[0], pp[1], pp[2], Fp) && frame3(pc[0], pc[1], pc[2], Fc)) {
            double v[3] = {xp[0] - pp[0][0], xp[1] - pp[0][1], xp[2] - pp[0][2]};
            double l0 = Fp[0] * v[0] + Fp[1] * v[1] + Fp[2] * v[2], l1 = Fp[3] * v[0] + Fp[4] * v[1] + Fp[5] * v[2], l2 = Fp[6] * v[0] + Fp[7] * v[1] + Fp[8] * v[2];
            for (int k = 0; k < 3; k++) o[k] = (pc[0][k] + l0 * Fc[k] + l1 * Fc[3 + k] + l2 * Fc[6 + k] - bi[k]) / s;
        } else {
            for (int k = 0; k < 3; k++) o[k] = (xp[k] - (base + ((size_t)(t - 1) * N + i) * 3)[k]) / s;
        }
    }
    free(REC);
    return 0;
}
