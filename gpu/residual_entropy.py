"""Context-modelled adaptive arithmetic coder for the integer residual (correction) streams (AEZ-12).

Python side of residual_entropy.c (built on first import with gcc into _residual_entropy.so).
Stream blob layout (version 4): 1 byte version | 16 x 4 int16 predictor weights per atom class (1/64 units)
| 4 byte length + lzma of the in-stream reference offsets (i - ref[i], uint8; 0 = none)
| 4 byte length + lzma of the 3 geometric reference atoms per atom (int32, unified index minus position)
| range coder payload. Everything in the blob is counted.
The decoder additionally needs the stream shape (T, N), the atom classes (topology), the float base predictor of the
stream and the step s (both already part of the codec), and for the side chain stream the decoded backbone (context
atoms). The encoder chooses the references from its own mean structure and stores them.
Predictor and contexts: see residual_entropy.c. Mechanism modelled on MDCompress (prediction from the same atom in the
previous frame and the motion of an already coded neighbour, adaptive range coding), see
benchmark/entropy/mdcompress_notes.md."""
import ctypes, lzma, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "residual_entropy.c"); LIB = os.path.join(HERE, "_residual_entropy.so")
VERSION = 4
NG = 16
CFLAGS = ["gcc", "-O2", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC"]


def _build():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        subprocess.check_call(CFLAGS + ["-o", LIB + ".tmp", SRC]); os.replace(LIB + ".tmp", LIB)
    lib = ctypes.CDLL(LIB)
    nd = lambda t: np.ctypeslib.ndpointer(t, flags="C_CONTIGUOUS")
    i32p, u8p, i16p, f64p = nd(np.int32), nd(np.uint8), nd(np.int16), nd(np.float64)
    c_int, c_i64, c_d = ctypes.c_int, ctypes.c_int64, ctypes.c_double
    lib.re_encode.restype = c_i64
    lib.re_encode.argtypes = [i32p, c_int, c_int, i32p, u8p, i16p, f64p, c_d, f64p, c_int, i32p, u8p, c_i64]
    lib.re_decode.restype = c_int
    lib.re_decode.argtypes = [u8p, c_i64, c_int, c_int, i32p, u8p, i16p, f64p, c_d, f64p, c_int, i32p, i32p]
    lib.re_features.restype = c_int
    lib.re_features.argtypes = [i32p, c_int, c_int, f64p, c_d, f64p, c_int, i32p, f64p]
    return lib


_lib = _build()


def make_refs(S, window=255):
    """in-stream reference per atom: nearest earlier atom within `window` positions in the encoder's structure S."""
    S = np.asarray(S, np.float64); N = len(S); ref = np.full(N, -1, np.int32)
    for i in range(1, N):
        lo = max(0, i - window); d = ((S[lo:i] - S[i]) ** 2).sum(1); ref[i] = lo + int(np.argmin(d))
    return ref


def prev_refs(N):
    return np.arange(-1, N - 1, dtype=np.int32)


def make_refs3(S, Sctx=None, window=255):
    """three geometric reference atoms per atom (unified index: < Nc context atom, >= Nc stream atom Nc + j, j < i):
    the nearest candidates in the encoder's structure, skipping nearly collinear choices."""
    S = np.asarray(S, np.float64); N = len(S)
    Sc = np.zeros((0, 3)) if Sctx is None else np.asarray(Sctx, np.float64); Nc = len(Sc)
    R3 = np.full((N, 3), -1, np.int32)
    for i in range(N):
        lo = max(0, i - window)
        cand = np.concatenate([Sc, S[lo:i]], 0); idx = np.concatenate([np.arange(Nc), Nc + np.arange(lo, i)])
        if len(cand) < 3: continue
        dd = ((cand - S[i]) ** 2).sum(1); o = np.argsort(dd, kind="stable")[:24]
        sel = [o[0]]
        for j in o[1:]:
            if len(sel) == 1: sel.append(j); continue
            a = cand[sel[1]] - cand[sel[0]]; b = cand[j] - cand[sel[0]]
            if np.linalg.norm(np.cross(a, b)) > 0.3 * np.linalg.norm(a) * np.linalg.norm(b): sel.append(j); break
        if len(sel) == 3: R3[i] = idx[sel]
    return R3


def _pack(arr, dtype):
    b = lzma.compress(np.ascontiguousarray(arr, dtype).tobytes(), preset=9 | lzma.PRESET_EXTREME)
    return len(b).to_bytes(4, "little") + b


def _unpack(blob, pos, dtype):
    n = int.from_bytes(blob[pos:pos + 4], "little")
    return np.frombuffer(lzma.decompress(blob[pos + 4:pos + 4 + n]), dtype), pos + 4 + n


def _geo(base, s, Xc, N, T):
    base = np.ascontiguousarray(base, np.float64); assert base.shape == (T, N, 3)
    Xc = np.zeros((T, 0, 3)) if Xc is None else Xc
    return base, float(s), np.ascontiguousarray(Xc, np.float64).reshape(-1) if Xc.size else np.zeros(1), int(Xc.shape[1])


def features(q, base, s, R3, Xc=None):
    T, N, _ = q.shape; base, s, xc, Nc = _geo(base, s, Xc, N, T)
    out = np.zeros((T, N, 3)); _lib.re_features(np.ascontiguousarray(q, np.int32), T, N, base, s, xc, Nc, np.ascontiguousarray(R3, np.int32).reshape(-1), out)
    return out


def _enc(q, ref, cls, w, base, s, xc, Nc, R3):
    T, N, _ = q.shape
    cap = int(q.size * 6 + 4096); out = np.empty(cap, np.uint8)
    n = _lib.re_encode(q, T, N, ref, cls, np.ascontiguousarray(w, np.int16).reshape(-1), base, s, xc, Nc, R3, out, cap)
    if n < 0: raise RuntimeError(f"re_encode failed ({n})")
    return out[:n].tobytes()


FIXED = {"zero": (0, 0, 0, 0), "geometric (translate and rotate)": (64, 0, 0, 0), "spatial": (0, 64, 0, 0), "temporal": (0, 0, 64, 0),
         "MDCompress level 2 analogue": (0, 64, 64, -64)}


def _cost(e):
    return np.log2(1.0 + np.abs(e)).sum()


def encode(q, cls, base, s, ref, R3, Xc=None, n_try=3, force=None):
    """q int [T,N,3] residual integers of a stream with float base predictor `base` and step s (x_rec = base + s q).
    Returns (blob, info). Candidate predictors are scored per atom class with a code length proxy; the best per class
    combination and the best global candidates are entropy coded and the smallest is kept."""
    q = np.ascontiguousarray(q, np.int32); T, N, _ = q.shape
    ref = np.ascontiguousarray(ref, np.int32); cls = np.ascontiguousarray(cls, np.uint8); R3 = np.ascontiguousarray(R3, np.int32)
    assert len(ref) == N == len(cls) == len(R3) and (ref < np.arange(N)).all() and cls.max(initial=0) < NG
    base, s, xc, Nc = _geo(base, s, Xc, N, T)
    st = max(1, T // 200); fr = np.arange(0, T, st)
    fC = features(q, base, s, R3, Xc)[fr]
    rr = np.where(ref < 0, 0, ref); v = (ref >= 0)[None, :, None]
    qprev = np.concatenate([np.zeros((1, N, 3), q.dtype), q[:-1]])[fr]
    Fs = np.stack([fC, q[fr][:, rr] * v, qprev, qprev[:, rr] * v], -1).astype(np.float64); ys = q[fr].astype(np.float64); del fC, qprev
    cands = {k: np.tile(np.array(c, np.int16), (NG, 1)) for k, c in FIXED.items()}
    for nm, cols in (("least squares, all terms", [0, 1, 2, 3]), ("least squares, linear terms", [1, 2, 3])):
        W = np.zeros((NG, 4), np.int16)
        for g in np.unique(cls):
            sel = cls == g; A = Fs[:, sel][..., cols].reshape(-1, len(cols)); y = ys[:, sel].ravel()
            if len(y) >= 30:
                w = np.linalg.lstsq(A, y, rcond=None)[0]; W[g, cols] = np.clip(np.round(64 * w), -4096, 4096)
        cands[nm] = W
    # per class proxy cost of every candidate
    groups = np.unique(cls); cost = {}
    for k, W in cands.items():
        pr = np.floor((Fs * W[cls][None, :, None, :]).sum(-1) / 64.0 + 0.5)
        e = ys - pr; cost[k] = {g: _cost(e[:, cls == g]) for g in groups}
    best_pc = np.zeros((NG, 4), np.int16); choice = {}
    for g in groups:
        k = min(cost, key=lambda kk: cost[kk][g]); best_pc[g] = cands[k][g]; choice[int(g)] = k
    cands["best per class"] = best_pc
    tot = {k: sum(cost[k].values()) for k in cost}
    order = ["best per class"] + sorted(tot, key=tot.get)[:n_try - 1]
    if force is not None: order = [force]  # analysis only: code with one named candidate (e.g. "zero": no prediction inside the coder)
    best, sizes = None, {}
    for k in order:
        b = _enc(q, ref, cls, cands[k], base, s, xc, Nc, R3.reshape(-1)); sizes[k] = len(b)
        if best is None or len(b) < len(best[2]): best = (k, cands[k], b)
    k, W, b = best
    hdr = bytes([VERSION]) + W.astype(np.int16).tobytes() + _pack(np.where(ref < 0, 0, np.arange(N) - ref), np.uint8)
    hdr += _pack((Nc + np.arange(N))[:, None] - np.where(R3 < 0, Nc + np.arange(N)[:, None], R3), np.int32)
    return hdr + b, dict(predictor=k, per_class_choice=choice, sizes=sizes, header_bytes=len(hdr))


def decode(blob, shape, cls, base, s, Xc=None):
    T, N = int(shape[0]), int(shape[1])
    assert blob[0] == VERSION
    W = np.frombuffer(blob[1:1 + 8 * NG], np.int16).copy(); pos = 1 + 8 * NG
    off, pos = _unpack(blob, pos, np.uint8); off = off.astype(np.int32)
    ref = np.where(off == 0, -1, np.arange(N, dtype=np.int32) - off).astype(np.int32)
    base, s, xc, Nc = _geo(base, s, Xc, N, T)
    d3, pos = _unpack(blob, pos, np.int32); d3 = d3.reshape(N, 3)
    u = (Nc + np.arange(N))[:, None]; R3 = np.where(d3 == 0, -1, u - d3).astype(np.int32)
    payload = np.frombuffer(blob[pos:], np.uint8).copy()
    q = np.zeros((T, N, 3), np.int32)
    rc = _lib.re_decode(payload, len(payload), T, N, ref, np.ascontiguousarray(cls, np.uint8), W, base, s, xc, Nc, np.ascontiguousarray(R3).reshape(-1), q)
    if rc != 0: raise RuntimeError(f"re_decode failed ({rc})")
    return q.astype(np.int64)
