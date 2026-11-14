import os, time, json
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/root/autodl-tmp/models/models/Qwen3-1.7B"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": "Find the sum of all positive integers n such that n^2 + 1 is divisible by n + 1. Let's think step by step and output the final answer within \\boxed{}."}]
    prompts = [tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)] * 4
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=14336,
              gpu_memory_utilization=0.85, max_num_seqs=32,
              disable_log_stats=True)
    sp = SamplingParams(n=4, temperature=0.6, top_p=0.95, top_k=-1,
                        max_tokens=12288, stop_token_ids=[151645, 151643])
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    ntok = sum(len(o.token_ids) for r in outs for o in r.outputs)
    print(f"elapsed={dt:.1f}s seqs={sum(len(r.outputs) for r in outs)} tokens={ntok} tok/s={ntok/dt:.1f}")
    print(json.dumps([{"len": len(o.token_ids), "finish": o.finish_reason} for r in outs for o in r.outputs]))

if __name__ == "__main__":
    main()
