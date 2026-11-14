#!/usr/bin/env python
"""Coarsened exact matching: dh>0 vs dh<0 absorption within matched strata.

Strata are built within each prompt. Specifications (progressively stricter):
  s1: position quartile x H_T band
  s2: s1 x teacher-logp-y quartile
  s3: s2 x token category (digit/alpha/punct/space/other)
  s4: s2 x exact token id

For each stratum with enough positions on both sides, compare
rel = sum(A_kl)/sum(KL_prev) and mean A_kl. Differences are weighted by
min(n_pos, n_neg) and bootstrapped over prompts.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opd_entropy_absorption_analyze as m  # noqa: E402

HT_EDGES = np.array([0.0, 0.01, 0.1, 0.3, 0.7, 1.2, np.inf])


def step_of(tag):
    return int(tag.split("_")[-1])


def load_token_categories(tokenizer, ids):
    cache = {}

    def cat(tid):
        if tid in cache:
            return cache[tid]
        s = tokenizer.convert_ids_to_tokens(int(tid))
        if s is None:
            c = "other"
        else:
            t = s.replace("Ġ", " ").replace("Ċ", "\n")
            st = t.strip()
            if not st:
                c = "space"
            else:
                ch = st[0]
                if ch.isdigit():
                    c = "digit"
                elif ch.isalpha():
                    c = "alpha"
                elif ch in ".,;:!?()[]{}+-=*/<>|\\'\"`~@#$%^&_":
                    c = "punct"
                else:
                    c = "other"
        cache[tid] = c
        return c

    return np.array([cat(t) for t in ids], dtype=object)


def token_ids_for_pool(rollouts_path):
    import json

    ys, prompt_idx, pos = [], [], []
    with open(rollouts_path) as f:
        for line in f:
            row = json.loads(line)
            for j, tid in enumerate(row["response_ids"]):
                prompt_idx.append(row["prompt_index"])
                pos.append(j)
                ys.append(tid)
    return np.array(prompt_idx), np.array(pos), np.array(ys)


def quantile_bins(x, q=4):
    edges = np.quantile(x, np.linspace(0, 1, q + 1))
    edges[-1] += 1e-9
    return np.clip(np.digitize(x, edges[1:-1]), 0, q - 1)


def matched_diff(pidx, dh, klp, akl, strata_keys, prompts, min_side=5, min_kl=0.05):
    """Per-prompt matched difference rel(pos)-rel(neg) and absolute difference."""
    per_prompt = []
    matched_positions = 0
    total_positions = 0
    for p in prompts:
        idx = np.where(pidx == p)[0]
        total_positions += len(idx)
        groups = defaultdict(lambda: ([], []))
        for i in idx:
            key = tuple(k[i] for k in strata_keys)
            if dh[i] >= 0:
                groups[key][0].append(i)
            else:
                groups[key][1].append(i)
        terms = []
        for key, (P, N) in groups.items():
            if len(P) < min_side or len(N) < min_side:
                continue
            kp_p = np.nansum(klp[P])
            kp_n = np.nansum(klp[N])
            if kp_p < min_kl or kp_n < min_kl:
                continue
            rp = np.nansum(akl[P]) / kp_p
            rn = np.nansum(akl[N]) / kp_n
            ap = np.nanmean(akl[P])
            an = np.nanmean(akl[N])
            w = min(len(P), len(N))
            terms.append((w, rp, rn, ap, an))
            matched_positions += len(P) + len(N)
        if terms:
            W = sum(t[0] for t in terms)
            rel_pos = sum(t[0] * t[1] for t in terms) / W
            rel_neg = sum(t[0] * t[2] for t in terms) / W
            abs_pos = sum(t[0] * t[3] for t in terms) / W
            abs_neg = sum(t[0] * t[4] for t in terms) / W
            per_prompt.append((rel_pos - rel_neg, abs_pos - abs_neg, rel_pos, rel_neg))
    if not per_prompt:
        return None
    return {
        "per_prompt": np.array(per_prompt),
        "match_rate": matched_positions / max(total_positions, 1),
    }


def summarize(res, n_boot=2000):
    d = res["per_prompt"][:, 0]
    a = res["per_prompt"][:, 1]
    mean, lo, hi = m.boot_ci(d, n_boot)
    amean, alo, ahi = m.boot_ci(a, n_boot)
    return {
        "n_prompts": len(d),
        "rel_diff": mean,
        "rel_ci": [lo, hi],
        "abs_diff": amean,
        "abs_ci": [alo, ahi],
        "rel_pos": float(res["per_prompt"][:, 2].mean()),
        "rel_neg": float(res["per_prompt"][:, 3].mean()),
        "match_rate": res["match_rate"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--min-side", type=int, default=5)
    ap.add_argument("--min-kl", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    root = Path(args.dir)
    npz_dir = root / "npz"
    roll_dir = root / "rollouts"
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/models/models/Qwen3-1.7B"
    )

    pools = sorted(
        {p.stem.split("__")[0] for p in npz_dir.glob("*__*.npz")}, key=step_of
    )
    rows = []
    for pool in pools:
        files = {p.stem.split("__")[1]: p for p in npz_dir.glob(f"{pool}__*.npz")}
        if pool not in files or "teacher" not in files:
            continue
        nxts = [t for t in files if t not in (pool, "teacher")]
        if len(nxts) != 1:
            continue
        nxt = nxts[0]
        A = np.load(files[pool])
        B = np.load(files[nxt])
        T = np.load(files["teacher"])
        pidx = A["prompt_index"]
        pos = A["pos"]
        Ht = T["entropy"].astype(np.float64)
        dh = A["entropy"].astype(np.float64) - Ht
        klp, _ = m.kl_intersection(
            A["top_ids"], A["top_logp"], T["top_ids"], T["top_logp"]
        )
        kln, _ = m.kl_intersection(
            B["top_ids"], B["top_logp"], T["top_ids"], T["top_logp"]
        )
        akl = klp - kln

        # Align sampled token ids with the npz row order.
        r_pidx, r_pos, ys = token_ids_for_pool(roll_dir / f"{pool}.jsonl")
        assert np.array_equal(r_pidx, pidx) and np.array_equal(r_pos, pos), (
            f"rollout/npz misalignment for {pool}"
        )
        cats = load_token_categories(tokenizer, ys)

        prompts = np.unique(pidx)
        pos_bin = np.zeros(len(pidx), dtype=int)
        ht_bin = np.clip(np.digitize(Ht, HT_EDGES[1:-1]), 0, len(HT_EDGES) - 2)
        logp_bin = np.zeros(len(pidx), dtype=int)
        for p in prompts:
            s = pidx == p
            pos_bin[s] = quantile_bins(pos[s], 4)
            logp_bin[s] = quantile_bins(T["logp_y"][s].astype(np.float64), 4)

        specs = {
            "s1_pos_HT": [pos_bin, ht_bin],
            "s2_+logp": [pos_bin, ht_bin, logp_bin],
            "s3_+tokcat": [pos_bin, ht_bin, logp_bin, cats],
            "s4_+tokid": [pos_bin, ht_bin, logp_bin, ys],
        }
        for name, keys in specs.items():
            res = matched_diff(
                pidx, dh, klp, akl, keys, prompts, args.min_side, args.min_kl
            )
            if res is None:
                continue
            s = summarize(res, args.bootstrap)
            rows.append({"pair": f"{pool}->{nxt}", "spec": name, **s})
            print(
                f"{pool}->{nxt} {name:>11}: match={s['match_rate']:.3f} "
                f"rel pos={s['rel_pos']:.3f} neg={s['rel_neg']:.3f} "
                f"diff={s['rel_diff']:+.4f} [{s['rel_ci'][0]:+.4f},{s['rel_ci'][1]:+.4f}] "
                f"abs diff={s['abs_diff']:+.4f}",
                flush=True,
            )

    import json

    with open(root / "matched_summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[done] wrote {root / 'matched_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
