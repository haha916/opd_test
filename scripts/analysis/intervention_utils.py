#!/usr/bin/env python
"""Probability-mass surgery for the panel interventions.

Bug being fixed
---------------
``build_state_panel.py`` stores the recovery / preserved token sets under
``state["entries"]["R"]`` and ``state["entries"]["P"]``.  The first version of
``run_panel_branches.py`` read ``state["R"]`` / ``state["P"]`` (top level),
which do not exist.  ``R`` and ``P`` were therefore empty for every state and
every withdraw/inject condition was a no-op; the differences in the old
intervention table were first-token sampling noise.

Surgery
-------
For a token set T (content tokens only):

  * withdraw / inject sets T's total mass to ``sum(p_other[T])`` and
    distributes that mass inside T according to ``p_other`` (so the target
    tokens adopt the other checkpoint's conditional distribution);
  * every stop token keeps its ``p_mu`` probability exactly;
  * every other content token (donor) is scaled proportionally to absorb the
    mass difference;
  * other special tokens are zeroed.

No global renormalisation is applied at the end, so the stop probabilities are
not perturbed.  The returned array sums to 1 up to float64 error.

For a sham, the same *relative* mass change (``sum(p_other[T]) / sum(p_mu[T])``)
is applied to a matched non-target set chosen by closest log-probability to the
real target set; donors absorb the difference in the same way.
"""

from __future__ import annotations

import numpy as np


def _clean_ids(ids, vocab):
    out = []
    for t in ids:
        t = int(t)
        if 0 <= t < vocab:
            out.append(t)
    return np.asarray(sorted(set(out)), dtype=np.int64)


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("distribution must be 1-D")
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("distribution has no probability mass")
    return p / s


def resolve_entry_tokens(state, key):
    """Return the token list stored under ``state["entries"][key]``.

    Falls back to the top-level ``state[key]`` for old files, but the panel
    states store R/P inside ``entries``.
    """
    ents = state.get("entries") or {}
    items = ents.get(key)
    if not items:
        items = state.get(key, [])
    tokens = []
    for e in items or []:
        if isinstance(e, dict):
            if "token" in e:
                tokens.append(int(e["token"]))
        else:
            tokens.append(int(e))
    return sorted(set(tokens))


def _redistribute(p, ids, new_total, *, id_weights, special, stop_ids):
    """Set the total mass of ``ids`` to ``new_total``.

    ``id_weights`` distributes the mass *inside* ``ids``.  Donor content
    tokens absorb the difference proportionally to their original mass.
    """
    vocab = len(p)
    ids = _clean_ids(ids, vocab)
    ids = ids[~special[ids]]
    ids = ids[~np.isin(ids, stop_ids)]

    # every special token (including each stop token) keeps its probability;
    # only content tokens fund the target set
    fixed = np.asarray(special, dtype=bool).copy()
    fixed[stop_ids] = True
    fixed_mass = float(p[fixed].sum())
    budget = max(0.0, 1.0 - fixed_mass)
    new_total = float(min(max(float(new_total), 0.0), budget))

    donors = ~fixed
    donors[ids] = False
    donor_ids = np.where(donors)[0]
    donor_mass = float(p[donor_ids].sum())

    p_new = p.copy()
    p_new[fixed] = p[fixed]

    if len(ids):
        w = np.asarray(id_weights, dtype=np.float64)
        if w.shape[0] != len(ids):
            raise ValueError("id_weights and ids must have the same length")
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 0.0:
            w = np.ones(len(ids), dtype=np.float64)
            s = float(len(ids))
        p_new[ids] = w * (new_total / s)

    if donor_mass > 0.0:
        p_new[donor_ids] = p[donor_ids] * ((budget - new_total) / donor_mass)
    elif len(donor_ids):
        p_new[donor_ids] = (budget - new_total) / len(donor_ids)

    return np.maximum(p_new, 0.0)


def _matched_set(p, target, special, stop_ids):
    """Tokens outside ``target`` with the closest log-probability signature."""
    pool = np.where(~special)[0]
    pool = pool[~np.isin(pool, stop_ids)]
    pool = pool[~np.isin(pool, target)]
    if len(pool) == 0:
        return pool
    lp_t = np.log(np.maximum(p[target], 1e-12))
    centre = float(np.median(lp_t))
    dist = np.abs(np.log(np.maximum(p[pool], 1e-12)) - centre)
    order = pool[np.argsort(dist)]
    return order[: len(target)]


def build_condition(p_mu, p_other, target, special, stop_ids, *, sham=False):
    """Return (modified distribution, audit info).

    ``p_mu`` is the checkpoint whose first-token distribution is changed;
    ``p_other`` supplies the target mass (withdraw: the other checkpoint is
    lower; inject: it is higher).  ``target`` is the R/P token set.
    """
    p = normalize(p_mu)
    po = normalize(p_other)
    vocab = len(p)
    special = np.asarray(special, dtype=bool)
    if special.shape[0] != vocab:
        raise ValueError(f"special mask has {special.shape[0]} entries, vocab={vocab}")
    stop_ids = _clean_ids(stop_ids, vocab)
    target = _clean_ids(target, vocab)
    target = target[~special[target]]
    target = target[~np.isin(target, stop_ids)]

    stop_mass = float(p[stop_ids].sum())
    before = float(p[target].sum()) if len(target) else 0.0
    base_info = {
        "target_before": before,
        "target_after": before,
        "moved": 0.0,
        "p_eos": stop_mass,
        "sum": float(p.sum()),
        "n_target": int(len(target)),
        "direction": 0,
        "sham": bool(sham),
    }

    if len(target) == 0:
        return p.copy(), base_info

    if not sham:
        want = float(po[target].sum())
        p_new = _redistribute(p, target, want, id_weights=po[target],
                              special=special, stop_ids=stop_ids)
        after = float(p_new[target].sum())
        info = dict(base_info)
        info.update({
            "target_after": after,
            "moved": after - before,
            "sum": float(p_new.sum()),
            "direction": int(np.sign(after - before)),
            "want": want,
        })
        return p_new, info

    # sham: matched non-target set, same relative mass change
    sset = _matched_set(p, target, special, stop_ids)
    factor = (float(po[target].sum()) / before) if before > 0.0 else 1.0
    factor = max(0.0, factor)
    sham_before = float(p[sset].sum()) if len(sset) else 0.0
    p_new = _redistribute(p, sset, sham_before * factor,
                          id_weights=p[sset], special=special, stop_ids=stop_ids)
    sham_after = float(p_new[sset].sum()) if len(sset) else 0.0
    info = dict(base_info)
    info.update({
        "sham_before": sham_before,
        "sham_after": sham_after,
        "sham_moved": sham_after - sham_before,
        "factor": factor,
        "sum": float(p_new.sum()),
    })
    return p_new, info
