"""Render mimoagent trajectories into token sequences plus assistant-token targets for traj-kl.

Rendering uses the model's own chat_template.jinja through its Hugging Face tokenizer.
llama-server's /apply-template cannot be used here: it refuses any conversation that ends on an
assistant message containing tool calls, because it treats that as "continue this message".
Two cross-checks against the serving stack (llama-server --jinja on the same GGUF) run on every
trajectory, and a failure rejects the trajectory rather than patching it:
  (1) for each prefix ending in a user/tool message, the HF render text == llama-server /apply-template text;
  (2) HF token ids of the full render == llama-server /tokenize ids.

For assistant message i, targets run from the end of render(messages[:i], add_generation_prompt=True)
through the first <|im_end|> after it, which is the token the model emitted to stop.
Every render must be a token-prefix of the full render.

Usage: python render_trajs.py --server http://localhost:8080 --out traj.bin --meta meta.json \
           [--max-targets N --seed S] TRAJ.json [TRAJ.json ...]
"""
import argparse, json, random, struct, sys
import urllib.request


def post(server, path, body):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


TOK = None  # HF tokenizer, set in main()


def render(server, msgs, tools, gen):
    return TOK.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=gen, tokenize=False)


def server_render(server, msgs, tools):
    body = {"messages": msgs}
    if tools:
        body["tools"] = tools
    return post(server, "/apply-template", body)["prompt"]


def hf_tokenize(text):
    return TOK(text, add_special_tokens=False)["input_ids"]


def tokenize(server, text):
    return post(server, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def tokenize_pieces(server, text):
    r = post(server, "/tokenize", {"content": text, "add_special": False, "parse_special": True, "with_pieces": True})["tokens"]
    return [t["id"] for t in r], [t["piece"] if isinstance(t["piece"], str) else bytes(t["piece"]).decode("utf-8", "replace") for t in r]


def normalise(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        tc = m.get("tool_calls")
        if isinstance(tc, str):
            tc = json.loads(tc)
        if tc:
            # llama-server parses OpenAI-style string arguments into an object before templating, so the
            # template renders <parameter=...> XML: the form the model generated and saw at inference.
            # Do the same here; the template cross-check verifies it.
            tc = [dict(c) for c in tc]
            for c in tc:
                fn = dict(c.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                c["function"] = fn
            m["tool_calls"] = tc
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--max-targets", type=int, default=0, help="0 = keep all; else uniform subsample across all trajectories")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--hf-model", required=True, help="HF model dir with tokenizer + chat_template")
    ap.add_argument("trajs", nargs="+")
    a = ap.parse_args()
    global THINK_END, TOK
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(a.hf_model)
    IM_END = TOK.convert_tokens_to_ids("<|im_end|>")
    tk = tokenize(a.server, "</think>")
    if len(tk) != 1:
        sys.exit(f"</think> is not a single token: {tk}")
    THINK_END = tk[0]

    recs, meta = [], []
    for path in a.trajs:
        t = json.load(open(path))
        main_ = (t.get("trajs") or {}).get("main") or {}
        msgs, tools = normalise(main_.get("messages") or []), main_.get("tools")
        full_text = render(a.server, msgs, tools, False)
        full, pieces = tokenize_pieces(a.server, full_text)
        tgts, ok, why = [], True, ""
        n_turns = n_think = n_xcheck = 0
        if hf_tokenize(full_text) != full:
            ok, why = False, "HF vs llama-server tokenization mismatch on full render"
        for i, m in enumerate(msgs if ok else []):
            if m.get("role") in ("user", "tool") and i >= 1:
                if server_render(a.server, msgs[: i + 1], tools) != render(a.server, msgs[: i + 1], tools, True):
                    ok, why = False, f"HF vs llama-server template mismatch at message {i}"
                    break
                n_xcheck += 1
            if m.get("role") != "assistant":
                continue
            pre = hf_tokenize(render(a.server, msgs[:i], tools, True))
            if full[: len(pre)] != pre:
                ok, why = False, f"prefix mismatch at message {i}"
                break
            try:
                end = full.index(IM_END, len(pre))
            except ValueError:
                ok, why = False, f"no <|im_end|> after assistant message {i}"
                break
            span = list(range(len(pre), end + 1))
            n_turns += 1
            if not span:
                continue
            # Owner rule: exclude <think> ... </think> and the whitespace after it; score from the
            # first non-whitespace token after </think> through the end-of-turn token.
            ends = [k for k in span if full[k] == THINK_END]
            if ends:
                n_think += 1
                k = ends[-1] + 1
                while k < span[-1] + 1 and pieces[k].strip() == "":
                    k += 1
                span = list(range(k, span[-1] + 1))
            tgts.extend(span)
        meta.append({"path": path, "n_tok": len(full), "n_asst_tok": len(tgts), "n_msgs": len(msgs),
                     "n_asst_turns": n_turns, "n_turns_with_think": n_think, "n_template_xchecks": n_xcheck,
                     "ok": ok, "why": why})
        print(json.dumps(meta[-1]), file=sys.stderr)
        if ok:
            recs.append((full, tgts))

    all_t = [(ri, t) for ri, (_, ts) in enumerate(recs) for t in ts]
    if a.max_targets and len(all_t) > a.max_targets:
        rng = random.Random(a.seed)
        keep = set(rng.sample(range(len(all_t)), a.max_targets))
        per = [[] for _ in recs]
        for k, (ri, t) in enumerate(all_t):
            if k in keep:
                per[ri].append(t)
        recs = [(toks, sorted(p)) for (toks, _), p in zip(recs, per)]

    with open(a.out, "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", len(recs)))
        for toks, ts in recs:
            f.write(struct.pack("<II", len(toks), len(ts)))
            f.write(struct.pack(f"<{len(toks)}i", *toks))
            f.write(struct.pack(f"<{len(ts)}I", *ts))
    summary = {"n_traj_in": len(a.trajs), "n_traj_ok": len(recs), "n_targets": sum(len(ts) for _, ts in recs),
               "n_tokens": sum(len(t) for t, _ in recs), "max_len": max((len(t) for t, _ in recs), default=0),
               "max_targets": a.max_targets, "seed": a.seed, "trajs": meta}
    json.dump(summary, open(a.meta, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "trajs"}))


if __name__ == "__main__":
    main()
