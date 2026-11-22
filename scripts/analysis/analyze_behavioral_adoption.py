#!/usr/bin/env python
"""Token Recovery -> Behavioral Adoption.

At the same reference contexts (base rollouts on held-out benchmarks), ask for
every arm:

  p_after(v*)          local probability of the teacher top-1 token
  repair_rate          fraction of contexts with p_after(v*) >= tau
  visit_rate           fraction of reference contexts reached by this arm
  usage_per_state      actual frequency of v* per reference state, over all
                       natural rollouts of this arm (0 when the state is never
                       reached)
  usage_given_visit    frequency of v* among visits to the context
  adoption_ratio       observed usage / usage expected from p_after on the
                       visited states (1 = local probability is realised)

The key comparison is FKL vs RKL on G1 (FKL-only repair): if p_after rises but
visit_rate/usage do not, the mechanism is Case C (local probability change
without behavioural adoption).  If usage rises, Phase 2 (counterfactual
utility) decides whether the adopted token is useful.

Example:
  python -I scripts/analysis/analyze_behavioral_adoption.py \
    --candidates .../candidates/candidates_base.jsonl \
    --arm-logp fixedfkl=.../arm_top1_fixedfkl.npz \
               fixedrkl=.../arm_top1_fixedrkl.npz \
               base=.../arm_top1_base.npz \
    --arm-index fixedfkl=.../index/prefix_fixedfkl.npz \
                fixedrkl=.../index/prefix_fixedrkl.npz \
                base=.../index/prefix_base.npz \
    --pair fixed --out-dir .../adoption
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exploration_common as C  # noqa: E402

PAIR_ARMS = {
    "fixed": ("fixedfkl", "fixedrkl"),
    "online": ("onfkl", "onrkl"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True)
    p.add_argument("--arm-logp", nargs="*", default=[], help="name=arm_top1_*.npz")
    p.add_argument("--arm-index", nargs="*", default=[], help="name=prefix_*.npz")
    p.add_argument("--pair", choices=sorted(PAIR_ARMS), default="fixed")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tau-student", type=float, default=0.01)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_specs(specs):
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"expected name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        out[name] = path
    return out


def load_arm_p(path, n_rows):
    d = dict(np.load(path, allow_pickle=False))
    p = np.asarray(d["p"], dtype=np.float64)
    if "candidate_id" in d:
        order = np.argsort(d["candidate_id"])
        p = p[order]
    if len(p) != n_rows:
        raise SystemExit(f"{path}: {len(p)} rows != {n_rows} candidates")
    return p


def load_arm_rank(path, n_rows):
    d = dict(np.load(path, allow_pickle=False))
    if "rank" not in d:
        return None
    rank = np.asarray(d["rank"], dtype=np.int64)
    if "candidate_id" in d:
        rank = rank[np.argsort(d["candidate_id"])]
    if len(rank) != n_rows:
        return None
    return rank


def query_index(path, prefix_hashes, suffix_hashes, v_stars, qids):
    d = dict(np.load(path, allow_pickle=False))
    hashes = d["hashes"]
    nexts = d["next"]
    qid_full = d.get("qid")
    if "suffix" in d and "suffix_next" in d:
        sh = d["suffix"]
        sn = d["suffix_next"]
        sqid = d.get("suffix_qid")
    else:
        C.log(f"WARNING: {path} has no suffix index; falling back to full prefix")
        sh, sn, sqid = hashes, nexts, qid_full
    visits = np.zeros(len(prefix_hashes), dtype=np.int64)
    cnt = np.zeros(len(prefix_hashes), dtype=np.int64)
    svisits = np.zeros(len(prefix_hashes), dtype=np.int64)
    scnt = np.zeros(len(prefix_hashes), dtype=np.int64)
    for i, (h, sh_i, v, q) in enumerate(zip(prefix_hashes, suffix_hashes, v_stars, qids)):
        lo = int(np.searchsorted(hashes, h, side="left"))
        hi = int(np.searchsorted(hashes, h, side="right"))
        if qid_full is not None and hi > lo:
            keep = qid_full[lo:hi] == q
            if keep.any():
                seg = nexts[lo:hi][keep]
                visits[i] = len(seg)
                cnt[i] = int(np.count_nonzero(seg == v))
        elif hi > lo:
            visits[i] = hi - lo
            cnt[i] = int(np.count_nonzero(nexts[lo:hi] == v))
        lo = int(np.searchsorted(sh, sh_i, side="left"))
        hi = int(np.searchsorted(sh, sh_i, side="right"))
        if sqid is not None and hi > lo:
            keep = sqid[lo:hi] == q
            if keep.any():
                seg = sn[lo:hi][keep]
                svisits[i] = len(seg)
                scnt[i] = int(np.count_nonzero(seg == v))
        elif hi > lo:
            svisits[i] = hi - lo
            scnt[i] = int(np.count_nonzero(sn[lo:hi] == v))
    return visits, cnt, svisits, scnt


def agg(mask, p, pt, pb, visits, cnt, svisits, scnt, tau):
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    out = {"n": n}
    if n == 0:
        return out
    pp = p[mask]
    tt = pt[mask]
    bb = pb[mask]
    vv = visits[mask]
    cc = cnt[mask]
    sv = svisits[mask]
    sc = scnt[mask]
    visited = vv > 0
    svisited = sv > 0
    tot_vis = int(vv.sum())
    stot_vis = int(sv.sum())
    usage = np.where(visited, cc / np.maximum(vv, 1), 0.0)
    susage = np.where(svisited, sc / np.maximum(sv, 1), 0.0)
    expected_visits = float((vv * pp).sum())
    sexpected_visits = float((sv * pp).sum())
    out.update(
        {
            "p_T_mean": float(np.nanmean(tt)),
            "p_T_median": float(np.nanmedian(tt)),
            "p_base_mean": float(np.nanmean(bb)),
            "p_after_mean": float(np.nanmean(pp)),
            "p_after_median": float(np.nanmedian(pp)),
            "repair_rate": float(np.nanmean(pp >= tau)),
            "rel_recovery_mean": float(np.nanmean(pp / np.maximum(tt, 1e-12))),
            "visit_rate": float(visited.mean()),
            "exposure_mean": float(vv.mean()),
            "usage_per_state": float(usage.mean()),
            "usage_given_visit": float(cc.sum() / tot_vis) if tot_vis else None,
            "expected_usage_per_visit": float(expected_visits / tot_vis) if tot_vis else None,
            "adoption_ratio": (float(cc.sum() / expected_visits) if expected_visits > 0 else None),
            "p_after_visited_mean": float(np.nanmean(pp[visited])) if visited.any() else None,
            "p_after_unvisited_mean": float(np.nanmean(pp[~visited])) if (~visited).any() else None,
            "suffix_visit_rate": float(svisited.mean()),
            "suffix_exposure_mean": float(sv.mean()),
            "suffix_usage_per_state": float(susage.mean()),
            "suffix_usage_given_visit": float(sc.sum() / stot_vis) if stot_vis else None,
            "suffix_adoption_ratio": (
                float(sc.sum() / sexpected_visits) if sexpected_visits > 0 else None
            ),
        }
    )
    return out


def paired_diff(a, b, mask, n_boot, seed):
    """Mean of a-b over rows in mask; bootstrap over rows."""
    mask = np.asarray(mask, dtype=bool)
    d = (a - b)[mask]
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    lo, hi = C.bootstrap_ci(d, n=n_boot, seed=seed)
    return {"n": int(len(d)), "mean": float(d.mean()), "ci": [lo, hi]}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = C.read_jsonl(args.candidates)
    n = len(rows)
    C.log(f"candidates: {n}")

    p_base = np.asarray([r["p_base"] for r in rows], dtype=np.float64)
    p_T = np.asarray([r["p_T"] for r in rows], dtype=np.float64)
    v_star = np.asarray([r["v_star"] for r in rows], dtype=np.int64)
    base_rank = np.asarray([int(r.get("base_rank", 0)) for r in rows], dtype=np.int64)
    is_control = np.asarray([bool(r["is_control"]) for r in rows])
    ref_correct = np.asarray([int(r.get("ref_correct", -1)) for r in rows], dtype=np.int64)
    rel_pos = np.asarray([float(r["rel_pos"]) for r in rows], dtype=np.float64)
    bench = np.asarray([r.get("bench", "") for r in rows], dtype=object)
    prefix_hashes = np.asarray([C.hex_to_hash(r["prefix_hash"]) for r in rows], dtype=np.uint64)
    suffix_hashes = np.asarray(
        [C.hex_to_hash(r.get("suffix_hash", r["prefix_hash"])) for r in rows], dtype=np.uint64
    )
    qids = np.asarray(
        [C.BENCH_ORDER.get(r.get("bench", ""), 99) * 1_000_000 + int(r["index"]) for r in rows],
        dtype=np.int32,
    )

    specs_p = parse_specs(args.arm_logp)
    specs_i = parse_specs(args.arm_index)
    p_by_arm = {}
    rank_by_arm = {}
    for name, path in specs_p.items():
        p_by_arm[name] = load_arm_p(path, n)
        rank_by_arm[name] = load_arm_rank(path, n)
        C.log(f"arm p [{name}] loaded")
    # The base arm's local probability is already in the candidate rows, so we
    # do not need a separate arm_top1_base.npz pass over the same model.
    if "base" not in p_by_arm:
        p_by_arm["base"] = p_base.copy()

    pair = PAIR_ARMS[args.pair]
    for name in pair:
        if name not in p_by_arm:
            raise SystemExit(
                f"pair={args.pair} needs arm-logp for {name}; got {sorted(p_by_arm)}"
            )
    groups = C.assign_groups(
        p_base, p_by_arm[pair[0]], p_by_arm[pair[1]], is_control, args.tau_student
    )
    group_names = [g for g in C.GROUPS if np.any(groups == g)]
    C.log(f"groups: { {g: int((groups == g).sum()) for g in group_names} }")

    # Query each available prefix index.
    visits_by_arm, cnt_by_arm = {}, {}
    svisits_by_arm, scnt_by_arm = {}, {}
    for name, path in specs_i.items():
        visits, cnt, svisits, scnt = query_index(
            path, prefix_hashes, suffix_hashes, v_star, qids
        )
        visits_by_arm[name] = visits
        cnt_by_arm[name] = cnt
        svisits_by_arm[name] = svisits
        scnt_by_arm[name] = scnt
        C.log(
            f"index [{name}] queried (mean visits={visits.mean():.3f}, "
            f"suffix visits={svisits.mean():.3f})"
        )

    result = {
        "settings": {
            "candidates": args.candidates,
            "pair": args.pair,
            "tau_student": args.tau_student,
            "n_rows": n,
            "arm_logp": specs_p,
            "arm_index": specs_i,
        },
        "group_counts": {g: int((groups == g).sum()) for g in group_names},
        "groups": {},
        "strata": {"rel": {}, "p": {}, "ref_correct": {}, "bench": {}},
    }

    def arm_metrics(name, mask):
        if name not in p_by_arm:
            return None
        return agg(
            mask,
            p_by_arm[name],
            p_T,
            p_base,
            visits_by_arm.get(name, np.zeros(n, dtype=np.int64)),
            cnt_by_arm.get(name, np.zeros(n, dtype=np.int64)),
            svisits_by_arm.get(name, np.zeros(n, dtype=np.int64)),
            scnt_by_arm.get(name, np.zeros(n, dtype=np.int64)),
            args.tau_student,
        )

    arm_names = [a for a in p_by_arm if a in visits_by_arm]
    if not arm_names:
        C.log("WARNING: no arm has both local p and a prefix index; "
              "the adoption table will only contain local probabilities")
    for g in group_names:
        mask = groups == g
        rec = {"n": int(mask.sum()), "arms": {}, "rank": {}}
        for name in arm_names:
            rec["arms"][name] = arm_metrics(name, mask)
        if mask.any():
            rec["rank"]["base"] = float(np.median(base_rank[mask]))
            for name in arm_names:
                if rank_by_arm.get(name) is not None:
                    rec["rank"][name] = float(np.median(rank_by_arm[name][mask]))
        # Paired FKL/RKL: all rows and both-visited rows.
        a, b = pair
        if a in visits_by_arm and b in visits_by_arm:
            ua = np.where(visits_by_arm[a] > 0, cnt_by_arm[a] / np.maximum(visits_by_arm[a], 1), 0.0)
            ub = np.where(visits_by_arm[b] > 0, cnt_by_arm[b] / np.maximum(visits_by_arm[b], 1), 0.0)
            both = mask & (visits_by_arm[a] > 0) & (visits_by_arm[b] > 0)
            rec["paired"] = {
                f"{a}_vs_{b}_all_rows": paired_diff(ua, ub, mask, args.bootstrap, args.seed),
                f"{a}_vs_{b}_both_visited": paired_diff(ua, ub, both, args.bootstrap, args.seed),
                "delta_visit_rate": float(
                    ((visits_by_arm[a][mask] > 0).astype(float)
                     - (visits_by_arm[b][mask] > 0).astype(float)).mean()
                ) if mask.any() else None,
            }
            if "base" in visits_by_arm:
                ub2 = np.where(visits_by_arm["base"] > 0,
                               cnt_by_arm["base"] / np.maximum(visits_by_arm["base"], 1), 0.0)
                both_b = mask & (visits_by_arm[a] > 0) & (visits_by_arm["base"] > 0)
                rec["paired"][f"{a}_vs_base_both_visited"] = paired_diff(
                    ua, ub2, both_b, args.bootstrap, args.seed
                )
        result["groups"][g] = rec

    # Strata: relative position, teacher probability, reference correctness.
    for key, bins in (
        ("rel", C.REL_BUCKETS),
        ("p", C.P_BUCKETS),
    ):
        for g in group_names:
            for lo, hi, label in bins:
                if key == "rel":
                    m = (groups == g) & (rel_pos >= lo) & (rel_pos < hi)
                else:
                    m = (groups == g) & (p_T >= lo) & (p_T < hi)
                if m.sum() < 5:
                    continue
                rec = {"n": int(m.sum()), "arms": {}}
                for name in arm_names:
                    rec["arms"][name] = arm_metrics(name, m)
                result["strata"][key].setdefault(g, {})[label] = rec
    for g in group_names:
        for label, m in (
            ("ref_correct", (groups == g) & (ref_correct == 1)),
            ("ref_wrong", (groups == g) & (ref_correct == 0)),
        ):
            if m.sum() < 5:
                continue
            rec = {"n": int(m.sum()), "arms": {}}
            for name in arm_names:
                rec["arms"][name] = arm_metrics(name, m)
            result["strata"]["ref_correct"].setdefault(g, {})[label] = rec
        for b in sorted(set(bench.tolist())):
            m = (groups == g) & (bench == b)
            if m.sum() < 5:
                continue
            rec = {"n": int(m.sum()), "arms": {}}
            for name in arm_names:
                rec["arms"][name] = arm_metrics(name, m)
            result["strata"]["bench"].setdefault(g, {})[b] = rec

    C.write_json_atomic(out_dir / "adoption.json", result)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    lines = ["# Token Recovery → Behavioral Adoption", ""]
    s = result["settings"]
    lines += [
        f"- candidates: `{s['candidates']}`",
        f"- rows: {s['n_rows']}  | pair: `{s['pair']}`  | tau_student: {s['tau_student']}",
        f"- group counts: {result['group_counts']}",
        "",
        "口径：参考上下文来自 base 的独立 rollout（seed 固定）；`visit_rate` 是这些",
        "上下文在该 arm 自己 rollout 中**完整前缀**出现的比例；`usage_per_state` 是每个",
        "参考状态上 v* 的实际采样频率（未访问记 0）；`adoption_ratio` = 实际采样次数 /",
        "按该 arm 局部概率 p_after 预期的采样次数（≈1 表示局部概率真正转化为行为）。",
        "",
        "## 主表",
        "",
    ]
    header = "| group | n | p_T | p_base |"
    sep = "|---|---:|---:|---:|"
    for name in arm_names:
        header += f" {name} p_after | repair | visit | usage/state | adoption |"
        sep += "---:|---:|---:|---:|---:|"
    lines += [header, sep]
    for g in group_names:
        rec = result["groups"][g]
        line = f"| {C.GROUP_LABELS.get(g, g)} | {rec['n']} |"
        # group-level p_T/p_base from any arm (same rows)
        any_arm = next(iter(rec["arms"].values()), None)
        if any_arm is None:
            line += " — | — |"
        else:
            line += f" {any_arm['p_T_mean']:.4f} | {any_arm['p_base_mean']:.4f} |"
        for name in arm_names:
            a = rec["arms"].get(name)
            if not a:
                line += " — | — | — | — | — |"
                continue
            ar = a.get("adoption_ratio")
            line += (
                f" {a['p_after_mean']:.4f} | {C.fmt_pct(a['repair_rate'], 1)} | "
                f"{C.fmt_pct(a['visit_rate'], 1)} | {a['usage_per_state']:.4f} | "
                f"{C.fmt_num(ar, 2)} |"
            )
        lines.append(line)
    lines.append("")

    lines += [
        "### v* 在模型分布中的中位排名",
        "",
        "| group | base rank |" + "".join(f" {name} rank |" for name in arm_names),
        "|---|---:|" + "---:|" * len(arm_names),
    ]
    for g in group_names:
        rec = result["groups"][g]
        line = f"| {C.GROUP_LABELS.get(g, g)} | {C.fmt_num(rec.get('rank', {}).get('base'), 1)} |"
        for name in arm_names:
            line += f" {C.fmt_num(rec.get('rank', {}).get(name), 1)} |"
        lines.append(line)
    lines.append("")

    lines += [
        "### 软匹配（最后 32 token 相同）",
        "",
        "完整前缀匹配很严格；下表把“状态”放宽为最后 32 个 token 相同，",
        "用来区分“完全没到达”与“到达了相似状态但更早的推理已分叉”。",
        "",
    ]
    header = "| group | n |"
    sep = "|---|---:|"
    for name in arm_names:
        header += f" {name} suffix visit | suffix usage/state |"
        sep += "---:|---:|"
    lines += [header, sep]
    for g in group_names:
        rec = result["groups"][g]
        line = f"| {C.GROUP_LABELS.get(g, g)} | {rec['n']} |"
        for name in arm_names:
            a = rec["arms"].get(name, {})
            line += (
                f" {C.fmt_pct(a.get('suffix_visit_rate'), 1)} | "
                f"{C.fmt_num(a.get('suffix_usage_per_state'), 4)} |"
            )
        lines.append(line)
    lines.append("")

    lines += ["## FKL vs RKL（同一批上下文）", ""]
    lines += [
        "| group | Δusage all rows [95% CI] | n | Δusage both visited [95% CI] | n | Δvisit rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in group_names:
        p = result["groups"][g].get("paired", {})
        key_all = f"{pair[0]}_vs_{pair[1]}_all_rows"
        key_both = f"{pair[0]}_vs_{pair[1]}_both_visited"
        if key_all not in p:
            continue
        d = p[key_all]
        db = p.get(key_both, {"mean": None, "ci": [None, None], "n": 0})
        lines.append(
            f"| {C.GROUP_LABELS.get(g, g)} | {C.fmt_pct(d['mean'], 2, True)} "
            f"{C.fmt_ci(*d['ci']) if d['ci'][0] is not None else ''} | {d['n']} | "
            f"{C.fmt_pct(db['mean'], 2, True) if db['mean'] is not None else '—'} "
            f"{C.fmt_ci(*db['ci']) if db['ci'][0] is not None else ''} | {db['n']} | "
            f"{C.fmt_pct(p.get('delta_visit_rate'), 2, True)} |"
        )
    lines.append("")

    lines += ["## 按相对位置分层（G1 = FKL 独有修复）", ""]
    rel = result["strata"]["rel"].get("G1", {})
    if rel:
        lines += ["| rel_pos | n | p_after FKL | repair FKL | visit FKL | usage FKL | visit RKL | usage RKL |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label, rec in rel.items():
            af = rec["arms"].get(pair[0], {})
            ar = rec["arms"].get(pair[1], {})
            lines.append(
                f"| {label} | {rec['n']} | {C.fmt_num(af.get('p_after_mean'), 4)} | "
                f"{C.fmt_pct(af.get('repair_rate'), 1)} | {C.fmt_pct(af.get('visit_rate'), 1)} | "
                f"{C.fmt_num(af.get('usage_per_state'), 4)} | {C.fmt_pct(ar.get('visit_rate'), 1)} | "
                f"{C.fmt_num(ar.get('usage_per_state'), 4)} |"
            )
    else:
        lines.append("（样本不足）")
    lines.append("")

    lines += ["## 诊断", ""]
    g1 = result["groups"].get("G1", {})
    if g1:
        af = g1["arms"].get(pair[0], {})
        ar = g1["arms"].get(pair[1], {})
        if af and ar:
            dp = (af.get("p_after_mean") or 0) - (ar.get("p_after_mean") or 0)
            du = (af.get("usage_per_state") or 0) - (ar.get("usage_per_state") or 0)
            dv = (af.get("visit_rate") or 0) - (ar.get("visit_rate") or 0)
            lines += [
                f"- G1 上 FKL−RKL：p_after {C.fmt_pct(dp, 2, True)}，"
                f"visit_rate {C.fmt_pct(dv, 2, True)}，usage/state {C.fmt_pct(du, 2, True)}。",
            ]
            if abs(dp) > 0.01 and abs(du) < 0.002:
                lines.append(
                    "- 局部概率优势明显，但实际采样几乎不变 → 支持 **Case C（只有局部概率变化）**。"
                )
            elif du > 0.002:
                lines.append(
                    "- 局部概率优势确实转成了更多实际采样 → 进入 Phase 2 判断这些 token 是否有用。"
                )
            lines.append(
                "- 注意 `adoption_ratio` 接近 1 说明在访问到的状态上，行为与局部概率一致；"
                "若 visit_rate 很低，则问题在**状态访问**而不是采样概率。"
            )
    lines.append("")
    (out_dir / "ADOPTION.md").write_text("\n".join(lines), encoding="utf-8")
    C.log(f"wrote {out_dir / 'adoption.json'} and ADOPTION.md")


if __name__ == "__main__":
    main()
