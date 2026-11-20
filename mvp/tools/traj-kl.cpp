// traj-kl: whole-trajectory KL tool.
//
// Each trajectory is evaluated in ONE forward pass at its full length, so no chunking.
// Only the listed target positions are scored: assistant-generated tokens, optionally subsampled.
//
//   base mode: run the source model and store full-vocabulary log-probs (fp16) for every scored target.
//   kl   mode: run a candidate and compute, per scored target,
//              KL(p_source || p_candidate) = sum_v p_s(v) * (log p_s(v) - log p_c(v)),
//              plus the NLL of the actual token under both models and top-1 agreement.
//
// Input traj file (little-endian):
//   char[4] "TRJ1"; uint32 n_traj;
//   per trajectory: uint32 n_tok; uint32 n_tgt; int32 tok[n_tok]; uint32 tgt[n_tgt]
//   A target t (1 <= t < n_tok) means "score the prediction of tok[t]", which is read from logits at t-1.
//
// Usage:
//   traj-kl --mode base|kl -m model.gguf --traj traj.bin --store DIR
//           [-ctk f16|q8_0|q4_0] [-ctv ...] [--ubatch 2048] [--out result.json] [--threads N]

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <string>
#include <thread>
#include <vector>

struct Traj { std::vector<int32_t> tok; std::vector<uint32_t> tgt; };

static void die(const char * msg) { fprintf(stderr, "traj-kl: %s\n", msg); exit(1); }

static std::vector<Traj> read_trajs(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) die("cannot open traj file");
    char magic[4]; uint32_t n = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TRJ1", 4) != 0) die("bad magic");
    if (fread(&n, 4, 1, f) != 1) die("bad header");
    std::vector<Traj> out(n);
    for (auto & t : out) {
        uint32_t nt = 0, ng = 0;
        if (fread(&nt, 4, 1, f) != 1 || fread(&ng, 4, 1, f) != 1) die("truncated");
        t.tok.resize(nt); t.tgt.resize(ng);
        if (fread(t.tok.data(), 4, nt, f) != nt || fread(t.tgt.data(), 4, ng, f) != ng) die("truncated");
        for (uint32_t g : t.tgt) if (g < 1 || g >= nt) die("target out of range");
        if (!std::is_sorted(t.tgt.begin(), t.tgt.end())) die("targets must be sorted");
    }
    fclose(f);
    return out;
}

static ggml_type parse_kv(const std::string & s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    die("unsupported KV type");
    return GGML_TYPE_F16;
}

// serial log-softmax (called from inside the row-parallel loop)
static void log_softmax_serial(const float * logits, float * out, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = std::max(mx, logits[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp((double) logits[i] - mx);
    const float lse = mx + (float) std::log(s);
    for (int i = 0; i < n; ++i) out[i] = logits[i] - lse;
}

int main(int argc, char ** argv) {
    std::string mode, model_path, traj_path, store, out_path, ctk = "f16", ctv = "f16";
    int ubatch = 2048, threads = 16;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) die("missing value"); return std::string(argv[++i]); };
        if      (a == "--mode")    mode = next();
        else if (a == "-m")        model_path = next();
        else if (a == "--traj")    traj_path = next();
        else if (a == "--store")   store = next();
        else if (a == "--out")     out_path = next();
        else if (a == "-ctk")      ctk = next();
        else if (a == "-ctv")      ctv = next();
        else if (a == "--ubatch")  ubatch = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else die(("unknown arg " + a).c_str());
    }
    if ((mode != "base" && mode != "kl") || model_path.empty() || traj_path.empty() || store.empty()) die("bad args");

    auto trajs = read_trajs(traj_path);
    uint32_t max_len = 0;
    for (auto & t : trajs) max_len = std::max<uint32_t>(max_len, t.tok.size());

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    auto cp = llama_context_default_params();
    cp.n_ctx = max_len + 16;
    cp.n_batch = ubatch; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = parse_kv(ctk); cp.type_v = parse_kv(ctv);
    cp.n_threads = threads; cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");

    double tot_kl = 0, tot_nll_c = 0, tot_nll_s = 0; long tot_n = 0, tot_top1 = 0;
    double t_decode = 0, t_post_total = 0;
    const auto t_start = std::chrono::steady_clock::now();
    std::string per_traj_json = "[";

    for (size_t ti = 0; ti < trajs.size(); ++ti) {
        const Traj & T = trajs[ti];
        llama_memory_clear(llama_get_memory(ctx), true);
        const std::string fpath = store + "/base_" + std::to_string(ti) + ".f16";
        FILE * fs = fopen(fpath.c_str(), mode == "base" ? "wb" : "rb");
        if (!fs) die("cannot open store file");

        // positions whose logits we need: tgt-1
        std::vector<uint32_t> need(T.tgt.size());
        for (size_t k = 0; k < T.tgt.size(); ++k) need[k] = T.tgt[k] - 1;
        size_t nk = 0;  // next target index
        double kl_sum = 0, nll_c = 0, nll_s = 0; long top1 = 0;

        llama_batch batch = llama_batch_init(ubatch, 0, 1);
        for (uint32_t p0 = 0; p0 < T.tok.size(); p0 += ubatch) {
            const uint32_t p1 = std::min<uint32_t>(T.tok.size(), p0 + ubatch);
            batch.n_tokens = 0;
            size_t k_first = nk;
            for (uint32_t p = p0; p < p1; ++p) {
                const int j = batch.n_tokens++;
                batch.token[j] = T.tok[p]; batch.pos[j] = p;
                batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
                batch.logits[j] = (nk < need.size() && need[nk] == p) ? 1 : 0;
                if (batch.logits[j]) ++nk;
            }
            auto t0 = std::chrono::steady_clock::now();
            if (llama_decode(ctx, batch) != 0) die("decode failed");
            llama_synchronize(ctx);  // decode is asynchronous; include GPU completion in t_decode
            auto t1 = std::chrono::steady_clock::now();
            t_decode += std::chrono::duration<double>(t1 - t0).count();
            // Row-parallel post-processing: one target row per thread, each with its own buffers.
            // Base rows are addressed by the target's index within this trajectory (pread, no shared file cursor).
            const long nrows = (long) (nk - k_first);
            if (mode == "base") {
                std::vector<ggml_fp16_t> blk((size_t) nrows * n_vocab);
                double s_nll = 0;
                #pragma omp parallel for reduction(+:s_nll) schedule(dynamic, 4)
                for (long r = 0; r < nrows; ++r) {
                    const size_t k = k_first + r;
                    std::vector<float> ls(n_vocab);
                    log_softmax_serial(llama_get_logits_ith(ctx, (int) (need[k] - p0)), ls.data(), n_vocab);
                    ggml_fp32_to_fp16_row(ls.data(), blk.data() + (size_t) r * n_vocab, n_vocab);
                    s_nll += -ls[T.tok[T.tgt[k]]];
                }
                if (fwrite(blk.data(), sizeof(ggml_fp16_t), blk.size(), fs) != blk.size()) die("write failed");
                nll_s += s_nll;
            } else {
                const int fd = fileno(fs);
                double s_kl = 0, s_nc = 0, s_ns = 0; long s_t1 = 0; int bad = 0;
                #pragma omp parallel for reduction(+:s_kl,s_nc,s_ns,s_t1,bad) schedule(dynamic, 4)
                for (long r = 0; r < nrows; ++r) {
                    const size_t k = k_first + r;
                    std::vector<float> lc(n_vocab), lb(n_vocab);
                    std::vector<ggml_fp16_t> h(n_vocab);
                    const off_t off = (off_t) k * n_vocab * sizeof(ggml_fp16_t);
                    if (pread(fd, h.data(), (size_t) n_vocab * sizeof(ggml_fp16_t), off) != (ssize_t) (n_vocab * sizeof(ggml_fp16_t))) { bad += 1; continue; }
                    ggml_fp16_to_fp32_row(h.data(), lb.data(), n_vocab);
                    log_softmax_serial(llama_get_logits_ith(ctx, (int) (need[k] - p0)), lc.data(), n_vocab);
                    double kl = 0; int am_s = 0, am_c = 0;
                    for (int v = 0; v < n_vocab; ++v) {
                        const float ls = lb[v];
                        if (ls > -30.0f) kl += std::exp((double) ls) * ((double) ls - lc[v]);
                        if (ls > lb[am_s]) am_s = v;
                        if (lc[v] > lc[am_c]) am_c = v;
                    }
                    const int32_t y = T.tok[T.tgt[k]];
                    s_kl += std::max(0.0, kl); s_nc += -lc[y]; s_ns += -lb[y]; s_t1 += (am_s == am_c);
                }
                if (bad) die("store truncated");
                kl_sum += s_kl; nll_c += s_nc; nll_s += s_ns; top1 += s_t1;
            }
        }
        llama_batch_free(batch);
        fclose(fs);
        t_post_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() - t_decode;

        const long n = (long) T.tgt.size();
        tot_n += n; tot_kl += kl_sum; tot_nll_c += nll_c; tot_nll_s += nll_s; tot_top1 += top1;
        char buf[320];
        snprintf(buf, sizeof buf, "%s{\"i\":%zu,\"n_tok\":%zu,\"n_tgt\":%ld,\"kl\":%.9g,\"nll_src\":%.9g,\"nll_cand\":%.9g,\"top1\":%.6g}",
                 ti ? "," : "", ti, T.tok.size(), n, n ? kl_sum / n : 0.0, n ? nll_s / n : 0.0, n ? nll_c / n : 0.0,
                 n ? (double) top1 / n : 0.0);
        per_traj_json += buf;
        fprintf(stderr, "traj %zu/%zu: n_tok=%zu n_tgt=%ld kl=%.5f  [t_decode=%.1fs t_other=%.1fs]\n", ti + 1, trajs.size(), T.tok.size(), n, n ? kl_sum / n : 0.0, t_decode, t_post_total);
    }
    per_traj_json += "]";

    char head[512];
    snprintf(head, sizeof head,
             "{\"mode\":\"%s\",\"model\":\"%s\",\"ctk\":\"%s\",\"ctv\":\"%s\",\"n_tgt\":%ld,\"kl\":%.8g,\"nll_src\":%.8g,\"nll_cand\":%.8g,\"top1\":%.6g,\"trajs\":",
             mode.c_str(), model_path.c_str(), ctk.c_str(), ctv.c_str(), tot_n,
             tot_n ? tot_kl / tot_n : 0.0, tot_n ? tot_nll_s / tot_n : 0.0, tot_n ? tot_nll_c / tot_n : 0.0,
             tot_n ? (double) tot_top1 / tot_n : 0.0);
    std::string json = std::string(head) + per_traj_json + "}";
    if (!out_path.empty()) { FILE * fo = fopen(out_path.c_str(), "w"); fputs(json.c_str(), fo); fputc('\n', fo); fclose(fo); }
    printf("%s\n", json.c_str());

    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
