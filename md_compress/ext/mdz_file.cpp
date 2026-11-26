// mdz_file: file front end for MDZ (Zhao et al., ICDE 2022) as shipped in SZ3 tools/mdz (include/mdz.hpp).
// The SZ3 `mdz` tool compresses and decompresses in memory only and prints the size; this program runs the same
// per axis, per batch loop as MDZ_Compress<T,2> in mdz.hpp (same predictor selection by `select`, same compressors as its VQ, MT and SZ2 functions,
// same Config: blockSize 128, quantbinCnt 1024 by default), but writes the compressed streams to a file and can
// decode them. Extra bytes over the tool's count: a 48 byte header, 13 bytes per axis (VQ level parameters, which
// the tool keeps in memory) and 9 bytes per batch (method id + length).
//   mdz_file c IN.f32 OUT.bin T N ABS_EB [BATCH=10] [METHOD=-1] [QUANTBINS=1024]   (IN: float32 [T][N][3])
//   mdz_file d IN.bin OUT.f32
// METHOD: -1 ADP (MDZ default: pick per batch among VQ/VQT or LR, and MT, every 50 batches), 0 VQ, 1 VQT, 2 MT, 3 LR, 4 TS
// Prints one line of JSON (method per batch counts) on compress.
#include <mdz.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static void wr(std::ofstream &f, const void *p, size_t n) { f.write(reinterpret_cast<const char *>(p), n); }
template <class X> static void wv(std::ofstream &f, X v) { wr(f, &v, sizeof(X)); }
template <class X> static X rv(std::ifstream &f) { X v; f.read(reinterpret_cast<char *>(&v), sizeof(X)); if (!f) throw std::runtime_error("truncated"); return v; }

// same compressor construction as VQ / MT / SZ2 in mdz.hpp (LAMMPS_compress there does not compile: set_level on the
// interface pointer), returning the stream instead of decompressing it in memory
static std::shared_ptr<concepts::CompressorInterface<float>> make(const SZ3::Config &conf, int m, float ls, float lo, int ln, float *ts0) {
    if (m == 0 || m == 1) {
        auto e = make_compressor_exaalt<float, 2>(LinearQuantizer<float>(conf.absErrorBound, conf.quantbinCnt / 2), HuffmanEncoder<int>(), Lossless_zstd(), m);
        e->set_level(ls, lo, ln); return e;
    }
    if (m == 2) return make_sz_timebased<float, 2>(conf, ts0);
    if (m == 4) return make_sz_timebased<float, 2>(conf, static_cast<float *>(nullptr));
    return make_sz<float, 2>(conf);
}

static SZ3::Config base_conf(size_t T, size_t N, double eb, int qb) {
    SZ3::Config c({T, N});
    c.errorBoundMode = SZ3::EB_ABS; c.absErrorBound = eb; c.blockSize = 128; c.quantbinCnt = qb;
    return c;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: mdz_file c IN OUT T N EB [BATCH] [METHOD] [QB] | d IN OUT\n"); return 2; }
    std::string mode = argv[1];
    if (mode == "c") {
        size_t T = atol(argv[4]), N = atol(argv[5]); double eb = atof(argv[6]);
        size_t batch = argc > 7 ? atol(argv[7]) : 10; int method = argc > 8 ? atoi(argv[8]) : -1;
        int qb = argc > 9 ? atoi(argv[9]) : 1024;
        if (batch == 0) batch = T;
        std::vector<float> xyz(T * N * 3);
        { std::ifstream fi(argv[2], std::ios::binary); fi.read(reinterpret_cast<char *>(xyz.data()), xyz.size() * 4); if (!fi) { fprintf(stderr, "short input\n"); return 1; } }
        std::ofstream fo(argv[3], std::ios::binary);
        wr(fo, "MDZF", 4); wv<uint64_t>(fo, T); wv<uint64_t>(fo, N); wv<double>(fo, eb); wv<uint64_t>(fo, batch);
        wv<int32_t>(fo, method); wv<int32_t>(fo, qb); wv<int32_t>(fo, 0); wv<int32_t>(fo, 0);
        size_t counts[5] = {0, 0, 0, 0, 0};
        for (int ax = 0; ax < 3; ax++) {
            std::vector<float> d(T * N);  // [T][N] for this axis, as MDZ_Compress<T,3> transposes
            for (size_t t = 0; t < T; t++) for (size_t a = 0; a < N; a++) d[t * N + a] = xyz[(t * N + a) * 3 + ax];
            SZ3::Config conf = base_conf(T, N, eb, qb);
            int method_batch = method == -1 ? 50 : 0;
            auto ts0 = std::vector<float>(d.data(), d.data() + N);
            float ls = 0, lo = 0; int ln = 0;
            if (method != 2 && method != 3 && method != 4) {
                size_t sn = 0.1 * N; sn = std::min(sn, (size_t)20000); sn = std::max(sn, std::min((size_t)5000, N));
                get_cluster(d.data(), N, ls, lo, ln, sn);
                if (ln > N * 0.25) ln = 0;
            }
            wv<float>(fo, ls); wv<float>(fo, lo); wv<int32_t>(fo, ln);
            // the ts0 frame is stored (zstd, lossless) when any batch uses MT; we decide after the loop, so buffer batches
            std::vector<std::pair<uint8_t, std::vector<uchar>>> out;
            int cur = method; bool need_ts0 = false;
            for (size_t ts = 0; ts < T; ts += batch) {
                conf.dims[0] = (ts + batch > T ? T - ts : batch); conf.num = conf.dims[0] * N;
                float *data = &d[ts * N];
                if (method_batch > 0 && ts / batch % method_batch == 0)
                    select<float, 2>(conf, cur, ts, d.data(), ls, lo, ln, ts0.data(), batch);
                if ((cur == 0 || cur == 1) && ln == 0) throw std::runtime_error("VQ/VQT not available on this data (no levels); use ADP, MT, LR or TS");
                auto sz = make(conf, cur, ls, lo, ln, ts0.data());
                std::vector<uchar> c(2 * conf.num * sizeof(float) + 4096);
                size_t cs = sz->compress(conf, data, c.data(), c.size());  // overwrites data with decoded values, as in mdz.hpp
                c.resize(cs); out.emplace_back((uint8_t)cur, std::move(c));
                if (cur == 2) need_ts0 = true;
                counts[cur]++;
            }
            wv<uint8_t>(fo, need_ts0 ? 1 : 0);
            if (need_ts0) {
                auto zstd = SZ3::Lossless_zstd(); size_t in = N * 4; std::vector<uchar> buf(in * 2 + 1024);
                size_t cs = zstd.compress(reinterpret_cast<uchar *>(ts0.data()), in, buf.data(), buf.size());
                wv<uint64_t>(fo, cs); wr(fo, buf.data(), cs);
            }
            for (auto &b : out) { wv<uint8_t>(fo, b.first); wv<uint64_t>(fo, b.second.size()); wr(fo, b.second.data(), b.second.size()); }
        }
        fo.close();
        printf("{\"VQ\": %zu, \"VQT\": %zu, \"MT\": %zu, \"LR\": %zu, \"TS\": %zu}\n", counts[0], counts[1], counts[2], counts[3], counts[4]);
        return 0;
    }
    if (mode == "d") {
        std::ifstream fi(argv[2], std::ios::binary); char mg[4]; fi.read(mg, 4);
        if (memcmp(mg, "MDZF", 4)) { fprintf(stderr, "not an mdz_file stream\n"); return 1; }
        size_t T = rv<uint64_t>(fi), N = rv<uint64_t>(fi); double eb = rv<double>(fi); size_t batch = rv<uint64_t>(fi);
        rv<int32_t>(fi); int qb = rv<int32_t>(fi); rv<int32_t>(fi); rv<int32_t>(fi);
        std::vector<float> xyz(T * N * 3);
        for (int ax = 0; ax < 3; ax++) {
            float ls = rv<float>(fi), lo = rv<float>(fi); int ln = rv<int32_t>(fi);
            std::vector<float> ts0(N); bool has = rv<uint8_t>(fi);
            if (has) {
                size_t cs = rv<uint64_t>(fi); std::vector<uchar> b(cs); fi.read(reinterpret_cast<char *>(b.data()), cs);
                auto zstd = SZ3::Lossless_zstd(); uchar *o = nullptr;
                size_t n = zstd.decompress(b.data(), cs, o, 0);
                if (n != N * 4) throw std::runtime_error("ts0 size");
                memcpy(ts0.data(), o, n); delete[] o;
            }
            std::vector<float> d(T * N);
            SZ3::Config conf = base_conf(T, N, eb, qb);
            for (size_t ts = 0; ts < T; ts += batch) {
                conf.dims[0] = (ts + batch > T ? T - ts : batch); conf.num = conf.dims[0] * N;
                int m = rv<uint8_t>(fi); size_t cs = rv<uint64_t>(fi);
                std::vector<uchar> b(cs); fi.read(reinterpret_cast<char *>(b.data()), cs);
                auto sz = make(conf, m, ls, lo, ln, ts0.data());
                sz->decompress(conf, b.data(), cs, &d[ts * N]);
            }
            for (size_t t = 0; t < T; t++) for (size_t a = 0; a < N; a++) xyz[(t * N + a) * 3 + ax] = d[t * N + a];
        }
        std::ofstream fo(argv[3], std::ios::binary); wr(fo, xyz.data(), xyz.size() * 4);
        return 0;
    }
    return 2;
}
