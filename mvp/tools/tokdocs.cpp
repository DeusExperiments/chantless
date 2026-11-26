// tokdocs: tokenize raw documents with a GGUF's own tokenizer (vocab only, no weights on the GPU).
// Input: u32 n, then n records of u32 len + bytes. Output: u32 n, then n records of u32 len + i32 tokens.
#include "llama.h"
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

static int parse_positive_int(const char * s) {
    if (!s || !*s) return -1;
    char * end = nullptr;
    long v = strtol(s, &end, 10);
    if (*end != '\0' || v <= 0 || v > INT_MAX) return -1;
    return (int) v;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "tokdocs model.gguf in.bin out.bin [threads]\n"); return 1; }
    int nth = 8;
    if (argc > 4) {
        nth = parse_positive_int(argv[4]);
        if (nth <= 0) { fprintf(stderr, "tokdocs: invalid threads %s (must be > 0)\n", argv[4]); return 1; }
    }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.vocab_only = true; mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { llama_backend_free(); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) { llama_model_free(model); llama_backend_free(); return 2; }

    FILE * f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "tokdocs: cannot open input %s\n", argv[2]); llama_model_free(model); llama_backend_free(); return 3; }
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) { fclose(f); llama_model_free(model); llama_backend_free(); return 3; }
    std::vector<std::string> docs(n);
    for (auto & d : docs) {
        uint32_t l = 0;
        if (fread(&l, 4, 1, f) != 1) { fclose(f); llama_model_free(model); llama_backend_free(); return 3; }
        d.resize(l);
        if (l && fread(d.data(), 1, l, f) != l) { fclose(f); llama_model_free(model); llama_backend_free(); return 3; }
    }
    fclose(f);

    std::vector<std::vector<llama_token>> out(n);
    std::atomic<bool> failed{false};
    auto work = [&](int t) {
        for (uint32_t i = (uint32_t) t; i < n; i += (uint32_t) nth) {
            if (failed.load(std::memory_order_relaxed)) return;
            if (docs[i].size() > (size_t) INT32_MAX) { failed.store(true); return; }
            auto & o = out[i];
            size_t cap = docs[i].size() + 16;
            if (cap > (size_t) INT32_MAX) cap = (size_t) INT32_MAX;
            o.resize(cap);
            int32_t r = llama_tokenize(vocab, docs[i].data(), (int32_t) docs[i].size(), o.data(), (int32_t) o.size(), false, true);
            if (r < 0) {
                if (r == INT32_MIN) { failed.store(true); return; }
                int64_t need = -(int64_t) r;
                if (need <= 0 || need > (int64_t) INT32_MAX) { failed.store(true); return; }
                o.resize((size_t) need);
                r = llama_tokenize(vocab, docs[i].data(), (int32_t) docs[i].size(), o.data(), (int32_t) o.size(), false, true);
            }
            if (r < 0 || (size_t) r > o.size()) { failed.store(true); return; }
            o.resize((size_t) r);
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t) th.emplace_back(work, t);
    for (auto & t : th) t.join();
    if (failed.load()) {
        fprintf(stderr, "tokdocs: tokenization failed\n");
        llama_model_free(model); llama_backend_free();
        return 4;
    }

    FILE * fo = fopen(argv[3], "wb");
    if (!fo) { fprintf(stderr, "tokdocs: cannot open output %s\n", argv[3]); llama_model_free(model); llama_backend_free(); return 5; }
    bool w_ok = fwrite(&n, 4, 1, fo) == 1;
    for (const auto & o : out) {
        if (!w_ok) break;
        uint32_t l = (uint32_t) o.size();
        if (fwrite(&l, 4, 1, fo) != 1 || (l && fwrite(o.data(), 4, l, fo) != l)) { w_ok = false; break; }
    }
    if (fclose(fo) != 0) w_ok = false;
    llama_model_free(model); llama_backend_free();
    if (!w_ok) { fprintf(stderr, "tokdocs: write failed on %s\n", argv[3]); return 5; }
    return 0;
}
