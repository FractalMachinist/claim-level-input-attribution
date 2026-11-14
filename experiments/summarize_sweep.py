#!/usr/bin/env python
"""Summarize Stage 1 sweep JSONs into a table. Run from project root."""
import glob
import json

rows = []
for f in sorted(glob.glob("experiments/runs/*.json")):
    d = json.load(open(f))
    r = d["region_rows"]
    q = r.get("question", {})
    c = r.get("chaff", {})
    cm = c.get("mean_norm", 0)
    rows.append((
        d["n_chaff"], d["position"],
        q.get("mean_norm", 0), cm,
        (q.get("mean_norm", 0) / cm) if cm else float("nan"),
        q.get("share_of_sum", 0), c.get("share_of_sum", 0),
        d["reply_logp"],
    ))
rows.sort(key=lambda x: (x[0], x[1]))
print(f"{'N':>4} {'pos':<7}{'q_mean':>9}{'c_mean':>9}{'q/c':>7}"
      f"{'q_share':>9}{'c_share':>9}{'logp':>9}")
for N, pos, qm, cm, ratio, qs, cs, lp in rows:
    print(f"{N:>4} {pos:<7}{qm:>9.2f}{cm:>9.2f}{ratio:>7.2f}"
          f"{qs:>9.3f}{cs:>9.3f}{lp:>9.2f}")
