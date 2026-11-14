"""Smoke test: per-token logprobs of an arbitrary token sequence from vLLM.

Feeds prompt+continuation as the "prompt" and asks vLLM for prompt_logprobs,
which is exactly a forward pass over the given tokens (teacher forcing).
"""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/root/autodl-tmp/models/models/teacher"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": "What is 1+1? Answer briefly."}]
    prompt_ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True
    )
    print("prompt len:", len(prompt_ids))
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.6,
        max_model_len=4096,
        dtype="bfloat16",
        max_logprobs=64,
    )
    sp = SamplingParams(max_tokens=1, prompt_logprobs=64, temperature=0.0)
    out = llm.generate(prompts=[{"prompt_token_ids": prompt_ids}], sampling_params=sp)
    pl = out[0].prompt_logprobs
    print("prompt_logprobs type:", type(pl), "len:", None if pl is None else len(pl))
    print("first entries:", pl[:2])
    for idx in (1, len(pl) - 1):
        d = pl[idx]
        tokid = prompt_ids[idx]
        print(f"pos {idx}: n_entries={len(d)} actual_in_dict={tokid in d}")
        if tokid in d:
            print("  actual logprob:", d[tokid].logprob, "rank:", d[tokid].rank)
        top = sorted(d.items(), key=lambda kv: kv[1].logprob, reverse=True)[:3]
        print("  top3:", [(k, round(v.logprob, 3)) for k, v in top])
        print("  sum of top-64 probs:", sum(__import__('math').exp(v.logprob) for v in d.values()))


if __name__ == "__main__":
    main()
