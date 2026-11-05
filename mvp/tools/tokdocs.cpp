// tokdocs: tokenize raw documents with a GGUF's own tokenizer (vocab only, no weights on the GPU).
// Input: u32 n, then n records of u32 len + bytes. Output: u32 n, then n records of u32 len + i32 tokens.
#include "llama.h"
#include <cstdio>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "tokdocs model.gguf in.bin out.bin [threads]\n"); return 1; }
    int nth = argc > 4 ? atoi(argv[4]) : 8;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.vocab_only = true; mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    FILE * f = fopen(argv[2], "rb"); uint32_t n = 0; if (fread(&n, 4, 1, f) != 1) return 3;
    std::vector<std::string> docs(n);
    for (auto & d : docs) { uint32_t l; if (fread(&l, 4, 1, f) != 1) return 3; d.resize(l); if (l && fread(d.data(), 1, l, f) != l) return 3; }
    fclose(f);
    std::vector<std::vector<llama_token>> out(n);
    auto work = [&](int t) {
        for (uint32_t i = t; i < n; i += nth) {
            auto & o = out[i]; o.resize(docs[i].size() + 16);
            int r = llama_tokenize(vocab, docs[i].data(), docs[i].size(), o.data(), o.size(), false, true);
            if (r < 0) { o.resize(-r); r = llama_tokenize(vocab, docs[i].data(), docs[i].size(), o.data(), o.size(), false, true); }
            o.resize(r);
        }
    };
    std::vector<std::thread> th; for (int t = 0; t < nth; ++t) th.emplace_back(work, t); for (auto & t : th) t.join();
    f = fopen(argv[3], "wb"); fwrite(&n, 4, 1, f);
    for (auto & o : out) { uint32_t l = o.size(); fwrite(&l, 4, 1, f); fwrite(o.data(), 4, l, f); }
    fclose(f);
    llama_model_free(model); llama_backend_free();
    return 0;
}
