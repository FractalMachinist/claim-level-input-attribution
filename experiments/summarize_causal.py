#!/usr/bin/env python
"""Summarize Stage 1 causal JSONs: proxy-vs-measured rank agreement. Run from root."""
import glob
import json

rows = []
for f in sorted(glob.glob("experiments/runs/causal_*.json")):
    d = json.load(open(f))
    labels = d["labels"]
    proxy = d["proxy_norm"]
    meas = d["measured_dlogp"]

    def rmean(arr, lab):
        v = [a for a, l in zip(arr, labels) if l == lab]
        return sum(v) / len(v) if v else float("nan")

    rows.append((
        d["n_chaff"], d["position"],
        d["spearman_proxy_dlogp_all"],
        d["spearman_proxy_dlogp_chaff"],
        d["spearman_proxy_kl_all"],
        rmean(proxy, "question"), rmean(meas, "question"),
        rmean(proxy, "chaff"), rmean(meas, "chaff"),
    ))
rows.sort(key=lambda x: (x[0], x[1]))
print(f"{'N':>4} {'pos':<7}{'rho_all':>9}{'rho_chaff':>11}{'rho_KL':>9}"
      f"{'q_proxy':>9}{'q_meas':>9}{'c_proxy':>9}{'c_meas':>9}")
for N, pos, ra, rc, rk, qp, qm, cp, cm in rows:
    print(f"{N:>4} {pos:<7}{ra:>9.3f}{rc:>11.3f}{rk:>9.3f}"
          f"{qp:>9.2f}{qm:>9.2f}{cp:>9.3f}{cm:>9.4f}")
