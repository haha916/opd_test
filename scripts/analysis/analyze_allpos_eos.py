#!/usr/bin/env python
"""All-position EOS-separated clustering from allpos raw files.

Each position is represented by the change in student probability on the
teacher's non-EOS top-60 tokens (rank bins), L1-normalized.  EOS mass is
reported separately; it is excluded from the clustering features.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def entropy_from_logp(lp):
    p = np.exp(lp.astype(np.float64))
    s = p.sum(-1, keepdims=True)
    p = p / np.maximum(s, 1e-12)
    return -(p * np.log(np.maximum(p, 1e-30))).sum(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/multiquery_fixed_20261109/allpos_raw")
    p.add_argument("--out-dir", default="")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eos", type=int, default=151645)
    args = p.parse_args()
    d = Path(args.dir)
    out_dir = Path(args.out_dir) if args.out_dir else d / "eos_clusters"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(d / "step_*.npz")))
    T = len(files)
    z0 = np.load(files[0])
    traj = z0["traj_indices"].astype(int)
    lens = z0["lengths"].astype(int)
    # canonical order by pool index
    order = np.argsort(traj)
    traj = traj[order]
    lens = lens[order]
    starts = np.zeros(len(traj), dtype=np.int64)
    starts[1:] = np.cumsum(lens)[:-1]
    N = int(lens.sum())
    print(f"recorded trajectories={len(traj)} positions={N} steps={T}")

    PS = np.zeros((N, T, 60), dtype=np.float32)
    EOS = np.zeros((N, T), dtype=np.float32)
    t_ids = t_lp = s_ids = s_lp = None
    for ti, f in enumerate(files):
        z = np.load(f)
        tj = z["traj_indices"].astype(int)
        pos = {int(g): i for i, g in enumerate(tj)}
        for gi, g in enumerate(traj):
            r = pos[int(g)]
            ln = int(z["lengths"][r])
            off = int(z["offsets"][r])
            st = int(starts[gi])
            pst = np.exp(z["p_s_at_t"][off:off + ln].astype(np.float32))
            PS[st:st + ln, ti, :] = pst
            # EOS from student top60 if present
            si = z["s_ids"][off:off + ln]
            sl = z["s_logp"][off:off + ln].astype(np.float32)
            e = np.zeros(ln, dtype=np.float32)
            rr, cc = np.where(si == args.eos)
            if len(rr):
                e[rr] = np.exp(sl[rr, cc])
            EOS[st:st + ln, ti] = e
            if ti == 0:
                if t_ids is None:
                    t_ids = np.zeros((N, 60), dtype=np.int32)
                    t_lp = np.zeros((N, 60), dtype=np.float32)
                    s_ids = np.zeros((N, 60), dtype=np.int32)
                    s_lp = np.zeros((N, 60), dtype=np.float32)
                t_ids[st:st + ln] = z["t_ids"][off:off + ln]
                t_lp[st:st + ln] = z["t_logp"][off:off + ln].astype(np.float32)
                s_ids[st:st + ln] = si
                s_lp[st:st + ln] = sl

    # base features
    pT = np.exp(t_lp)
    pSbase = np.exp(s_lp)
    HS0 = entropy_from_logp(s_lp)
    HT0 = entropy_from_logp(t_lp)
    qT1 = pT[:, 0]
    p_at_t1 = PS[:, 0, 0]
    p_top1 = np.exp(s_lp).max(-1)
    ratio = p_at_t1 / np.maximum(p_top1, 1e-12)
    # coverage top-p 0.5 on teacher top60
    cum = np.cumsum(pT / np.maximum(pT.sum(1, keepdims=True), 1e-12), 1)
    kk = (cum < 0.5).sum(1) + 1
    cov = np.array([PS[i, 0, :kk[i]].sum() for i in range(N)])
    # EOS column mask; non-EOS teacher tokens re-ranked by teacher logp
    eos_col = np.argmax(t_ids == args.eos, 1)
    has_eos = (t_ids == args.eos).any(1)
    PSn = np.zeros((N, T, 59), dtype=np.float32)
    for i in range(N):
        keep = np.arange(60)
        if has_eos[i]:
            keep = np.delete(keep, eos_col[i])
        else:
            keep = keep[:59]  # drop the lowest-ranked token to keep 59 columns
        order_t = keep[np.argsort(-t_lp[i, keep])]
        PSn[i] = PS[i][:, order_t]

    dp = PSn[:, -1, :] - PSn[:, 0, :]
    bins = [(0, 1), (1, 5), (5, 20), (20, 59)]
    X = np.stack([dp[:, lo:hi].sum(-1) for lo, hi in bins], -1)
    X = X / (np.abs(X).sum(-1, keepdims=True) + 1e-12)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=min(10, Xs.shape[1]), random_state=0).fit_transform(Xs)
    lab = KMeans(n_clusters=args.k, random_state=0, n_init=30).fit_predict(Z)
    sil = silhouette_score(Z, lab)
    print(f"K={args.k} silhouette={sil:.3f}")
    print(f"{'c':>2s} {'n':>6s} {'D?':>5s} {'HS0':>7s} {'HT0':>7s} {'qT1':>6s} "
          f"{'p_at1':>6s} {'ratio':>6s} {'cov':>6s} {'EOS0':>6s} {'EOS1':>6s} "
          f"{'dR1':>7s} {'dR2-5':>7s} {'dR6-20':>7s} {'dR21+':>7s}")
    rows = []
    for c in range(args.k):
        m = lab == c
        dct = {
            "cluster": c, "n": int(m.sum()),
            "HS0": float(HS0[m].mean()), "HT0": float(HT0[m].mean()),
            "qT1": float(qT1[m].mean()), "p_at_t1": float(p_at_t1[m].mean()),
            "ratio": float(ratio[m].mean()), "cov": float(cov[m].mean()),
            "EOS0": float(EOS[m, 0].mean()), "EOS1": float(EOS[m, -1].mean()),
            "dR1": float(dp[m, 0].mean()),
            "dR2_5": float(dp[m, 1:5].sum(-1).mean()),
            "dR6_20": float(dp[m, 5:20].sum(-1).mean()),
            "dR21p": float(dp[m, 20:].sum(-1).mean()),
        }
        rows.append(dct)
        print(f"{c:2d} {dct['n']:6d} {0:5d} {dct['HS0']:7.3f} {dct['HT0']:7.3f} "
              f"{dct['qT1']:6.3f} {dct['p_at_t1']:6.3f} {dct['ratio']:6.3f} "
              f"{dct['cov']:6.3f} {dct['EOS0']:6.3f} {dct['EOS1']:6.3f} "
              f"{dct['dR1']:7.4f} {dct['dR2_5']:7.4f} {dct['dR6_20']:7.4f} "
              f"{dct['dR21p']:7.4f}")
    with open(out_dir / "clusters.json", "w") as f:
        json.dump({"silhouette": sil, "clusters": rows}, f, indent=2)
    np.savez(out_dir / "allpos_eos.npz", PS=PSn, EOS=EOS, labels=lab,
             t_ids=t_ids, t_logp=t_lp, s_ids=s_ids, s_logp=s_lp,
             HS0=HS0, HT0=HT0, qT1=qT1, p_at_t1=p_at_t1, ratio=ratio, cov=cov)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
