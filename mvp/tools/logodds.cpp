// logodds: the detector's decision quantity for every trajectory, in one forward pass each.
//
// For each target t of each trajectory it prints logit[A] - logit[B] read from the logits at t-1, which is
// log p(A) - log p(B) for the two answer tokens (exact: the softmax normaliser cancels). Input is the
// traj-kl TRJ1 file; --shard i/n scores trajectories i, i+n, i+2n, ... so several processes can split a file.
//
// Usage:
//   logodds -m model.gguf --traj traj.bin --pair 24509,2075 --out out.tsv [--shard 0/1] [--ubatch 2048]
// Output: one line per target, "<traj index>\t<target index>\t<log-odds>".

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Traj { std::vector<int32_t> tok; std::vector<uint32_t> tgt; };

static void die(const char * msg) { fprintf(stderr, "logodds: %s\n", msg); exit(1); }

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

int main(int argc, char ** argv) {
    std::string model_path, traj_path, out_path, pair;
    int ubatch = 2048, shard_i = 0, shard_n = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) die("missing value"); return std::string(argv[++i]); };
        if      (a == "-m")        model_path = next();
        else if (a == "--traj")    traj_path = next();
        else if (a == "--out")     out_path = next();
        else if (a == "--pair")    pair = next();
        else if (a == "--ubatch")  ubatch = std::stoi(next());
        else if (a == "--shard")   { auto s = next(); if (sscanf(s.c_str(), "%d/%d", &shard_i, &shard_n) != 2) die("bad --shard"); }
        else die(("unknown arg " + a).c_str());
    }
    int tok_a = -1, tok_b = -1;
    if (model_path.empty() || traj_path.empty() || out_path.empty() || sscanf(pair.c_str(), "%d,%d", &tok_a, &tok_b) != 2) die("bad args");

    auto trajs = read_trajs(traj_path);
    uint32_t max_len = 0;
    for (size_t ti = shard_i; ti < trajs.size(); ti += shard_n) max_len = std::max<uint32_t>(max_len, trajs[ti].tok.size());

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = max_len + 16;
    cp.n_batch = ubatch; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");

    FILE * fo = fopen(out_path.c_str(), "w");
    if (!fo) die("cannot open output");
    llama_batch batch = llama_batch_init(ubatch, 0, 1);
    for (size_t ti = shard_i; ti < trajs.size(); ti += shard_n) {
        const Traj & T = trajs[ti];
        llama_memory_clear(llama_get_memory(ctx), true);
        size_t nk = 0;
        for (uint32_t p0 = 0; p0 < T.tok.size(); p0 += ubatch) {
            const uint32_t p1 = std::min<uint32_t>(T.tok.size(), p0 + ubatch);
            batch.n_tokens = 0;
            const size_t k_first = nk;
            for (uint32_t p = p0; p < p1; ++p) {
                const int j = batch.n_tokens++;
                batch.token[j] = T.tok[p]; batch.pos[j] = p;
                batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
                batch.logits[j] = (nk < T.tgt.size() && T.tgt[nk] - 1 == p) ? 1 : 0;
                if (batch.logits[j]) ++nk;
            }
            if (llama_decode(ctx, batch) != 0) die("decode failed");
            for (size_t k = k_first; k < nk; ++k) {
                const float * lg = llama_get_logits_ith(ctx, (int) (T.tgt[k] - 1 - p0));
                fprintf(fo, "%zu\t%zu\t%.7g\n", ti, k, (double) lg[tok_a] - lg[tok_b]);
            }
        }
    }
    llama_batch_free(batch);
    fclose(fo);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
