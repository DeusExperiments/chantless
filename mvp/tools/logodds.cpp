// logodds: two-token log-odds scorer for TRJ1 trajectories, with multi-sequence FlashAttention batch packing.
//
// For each target t of each trajectory it prints logit[A] - logit[B] read from the logits at t-1, which is
// log p(A) - log p(B) for the two answer tokens (exact: the softmax normaliser cancels). Input is one or more
// traj-kl TRJ1 files; --shard i/n scores trajectories i, i+n, i+2n, ... so several processes can split a file.
// When --n-seq > 1, trajectories fitting within --ubatch are length-sorted and greedily packed into shared
// microbatches with distinct seq_id slots so FlashAttention isolates causal masks in one decode call.
//
// Usage:
//   logodds -m model.gguf --traj traj.bin --pair TOK_A,TOK_B --out out.tsv
//           [--traj traj2.bin --out out2.tsv [--pair ...]]
//           [--shard 0/1] [--ubatch 4096] [--n-seq 16]
// Output: one line per target in ascending trajectory order, "<traj index>\t<target index>\t<log-odds>".

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct Traj { std::vector<int32_t> tok; std::vector<uint32_t> tgt; };

static void die(const char * msg) { fprintf(stderr, "logodds: %s\n", msg); exit(1); }

static std::vector<Traj> read_trajs(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) die(("cannot open traj file: " + path).c_str());
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

template <typename P>
static auto enable_kv_unified(P & p, int) -> decltype(p.kv_unified = true, bool()) {
    p.kv_unified = true;
    return true;
}
template <typename P>
static bool enable_kv_unified(P &, long) { return false; }

int main(int argc, char ** argv) {
    std::string model_path;
    std::vector<std::string> traj_paths, out_paths, pairs;
    int ubatch = 4096, n_seq = 16, shard_i = 0, shard_n = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) die("missing value"); return std::string(argv[++i]); };
        if      (a == "-m")        model_path = next();
        else if (a == "--traj")    traj_paths.push_back(next());
        else if (a == "--out")     out_paths.push_back(next());
        else if (a == "--pair")    pairs.push_back(next());
        else if (a == "--ubatch")  ubatch = std::stoi(next());
        else if (a == "--n-seq")   n_seq = std::max(1, std::stoi(next()));
        else if (a == "--shard")   { auto s = next(); if (sscanf(s.c_str(), "%d/%d", &shard_i, &shard_n) != 2) die("bad --shard"); }
        else die(("unknown arg " + a).c_str());
    }
    if (model_path.empty() || traj_paths.empty() || traj_paths.size() != out_paths.size() ||
        (pairs.size() != 1 && pairs.size() != traj_paths.size()) || shard_i < 0 || shard_n < 1 || shard_i >= shard_n || ubatch < 32) {
        die("bad args");
    }
    std::vector<std::pair<int, int>> parsed_pairs(traj_paths.size());
    for (size_t fi = 0; fi < traj_paths.size(); ++fi) {
        const std::string & ps = pairs[pairs.size() == 1 ? 0 : fi];
        int ta = -1, tb = -1;
        if (sscanf(ps.c_str(), "%d,%d", &ta, &tb) != 2) die("bad --pair");
        parsed_pairs[fi] = {ta, tb};
    }

    std::vector<std::vector<Traj>> all_trajs(traj_paths.size());
    uint32_t max_len = 0;
    size_t total_assigned = 0;
    for (size_t fi = 0; fi < traj_paths.size(); ++fi) {
        all_trajs[fi] = read_trajs(traj_paths[fi]);
        for (size_t ti = (size_t) shard_i; ti < all_trajs[fi].size(); ti += (size_t) shard_n) {
            max_len = std::max<uint32_t>(max_len, (uint32_t) all_trajs[fi][ti].tok.size());
            ++total_assigned;
        }
    }
    if (total_assigned == 0) {
        for (const auto & op : out_paths) {
            FILE * fo = fopen(op.c_str(), "w");
            if (!fo) die("cannot open output");
            fclose(fo);
        }
        return 0;
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");

    auto cp = llama_context_default_params();
    cp.n_batch = ubatch;
    cp.n_ubatch = ubatch;
    cp.n_seq_max = n_seq;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (n_seq <= 1) {
        cp.n_ctx = max_len + 16;
    } else if (enable_kv_unified(cp, 0)) {
        cp.n_ctx = std::max<uint32_t>(max_len + 64, (uint32_t) ubatch + 64);
    } else {
        cp.n_ctx = (max_len + 32) * (uint32_t) n_seq;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");

    llama_batch batch = llama_batch_init(ubatch, 0, 1);

    for (size_t fi = 0; fi < traj_paths.size(); ++fi) {
        const auto & trajs = all_trajs[fi];
        const int tok_a = parsed_pairs[fi].first;
        const int tok_b = parsed_pairs[fi].second;
        std::vector<std::vector<double>> res(trajs.size());

        auto score_single = [&](size_t ti) {
            const Traj & T = trajs[ti];
            res[ti].resize(T.tgt.size());
            if (T.tgt.empty()) return;
            llama_memory_clear(llama_get_memory(ctx), true);
            size_t nk = 0;
            for (uint32_t p0 = 0; p0 < T.tok.size(); p0 += (uint32_t) ubatch) {
                const uint32_t p1 = std::min<uint32_t>((uint32_t) T.tok.size(), p0 + (uint32_t) ubatch);
                batch.n_tokens = 0;
                const size_t k_first = nk;
                for (uint32_t p = p0; p < p1; ++p) {
                    const int j = batch.n_tokens++;
                    batch.token[j] = T.tok[p];
                    batch.pos[j] = p;
                    batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0;
                    bool need = false;
                    while (nk < T.tgt.size() && T.tgt[nk] - 1 == p) { need = true; ++nk; }
                    batch.logits[j] = need ? 1 : 0;
                }
                if (llama_decode(ctx, batch) != 0) die("decode failed");
                for (size_t k = k_first; k < nk; ++k) {
                    const float * lg = llama_get_logits_ith(ctx, (int) (T.tgt[k] - 1 - p0));
                    res[ti][k] = (double) lg[tok_a] - lg[tok_b];
                }
            }
        };

        if (n_seq <= 1) {
            for (size_t ti = (size_t) shard_i; ti < trajs.size(); ti += (size_t) shard_n) {
                score_single(ti);
            }
        } else {
            std::vector<size_t> short_idx, long_idx;
            for (size_t ti = (size_t) shard_i; ti < trajs.size(); ti += (size_t) shard_n) {
                if (trajs[ti].tgt.empty()) continue;
                if (trajs[ti].tok.size() <= (size_t) ubatch) short_idx.push_back(ti);
                else long_idx.push_back(ti);
            }
            std::stable_sort(short_idx.begin(), short_idx.end(), [&](size_t a, size_t b) {
                return trajs[a].tok.size() > trajs[b].tok.size();
            });

            std::vector<size_t> pack;
            std::vector<int> base_j;
            pack.reserve(n_seq);
            base_j.reserve(n_seq);

            size_t cur = 0;
            while (cur < short_idx.size()) {
                pack.clear();
                base_j.clear();
                size_t tok_sum = 0;
                while (cur < short_idx.size() && (int) pack.size() < n_seq &&
                       tok_sum + trajs[short_idx[cur]].tok.size() <= (size_t) ubatch) {
                    pack.push_back(short_idx[cur]);
                    tok_sum += trajs[short_idx[cur]].tok.size();
                    ++cur;
                }
                if (pack.size() == 1) {
                    score_single(pack[0]);
                    continue;
                }
                llama_memory_clear(llama_get_memory(ctx), true);
                batch.n_tokens = 0;
                for (size_t s = 0; s < pack.size(); ++s) {
                    const Traj & T = trajs[pack[s]];
                    base_j.push_back(batch.n_tokens);
                    size_t nk = 0;
                    for (uint32_t p = 0; p < T.tok.size(); ++p) {
                        const int j = batch.n_tokens++;
                        batch.token[j] = T.tok[p];
                        batch.pos[j] = p;
                        batch.n_seq_id[j] = 1;
                        batch.seq_id[j][0] = (llama_seq_id) s;
                        bool need = false;
                        while (nk < T.tgt.size() && T.tgt[nk] - 1 == p) { need = true; ++nk; }
                        batch.logits[j] = need ? 1 : 0;
                    }
                }
                if (llama_decode(ctx, batch) != 0) {
                    for (size_t ti : pack) score_single(ti);
                } else {
                    for (size_t s = 0; s < pack.size(); ++s) {
                        const size_t ti = pack[s];
                        const Traj & T = trajs[ti];
                        res[ti].resize(T.tgt.size());
                        for (size_t k = 0; k < T.tgt.size(); ++k) {
                            const float * lg = llama_get_logits_ith(ctx, base_j[s] + (int) (T.tgt[k] - 1));
                            res[ti][k] = (double) lg[tok_a] - lg[tok_b];
                        }
                    }
                }
            }
            for (size_t ti : long_idx) {
                score_single(ti);
            }
        }

        FILE * fo = fopen(out_paths[fi].c_str(), "w");
        if (!fo) die(("cannot open output: " + out_paths[fi]).c_str());
        for (size_t ti = (size_t) shard_i; ti < trajs.size(); ti += (size_t) shard_n) {
            for (size_t k = 0; k < res[ti].size(); ++k) {
                fprintf(fo, "%zu\t%zu\t%.17g\n", ti, k, res[ti][k]);
            }
        }
        fclose(fo);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
