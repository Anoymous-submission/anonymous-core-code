from pathlib import Path
import json, hashlib
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("--queries", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
BASE = a.base
meta = json.loads((BASE / "native_token_efficiency/test_record_snapshot.json").read_text())[
    "test_records"
]
all_uids = {r["uid"] for r in meta}
q = json.loads(a.queries.read_text())["scores_all64"]
allowed = {r["record_path"] for r in q if not r["known_overlap"]}
meta = [r for r in meta if any(r["path"].endswith("/" + x) or r["path"] == x for x in allowed)]
uids = [r["uid"] for r in meta]
assert len(uids) == len(set(uids)) == 559
names, groups = np.unique([r["path"] for r in meta], return_inverse=True)
n_episodes = len(names)
metrics = ["ade_mm", "fde_mm"]
out = {}
scores = {}
sources = {}
for name, loc in [
    ("Language", "language/correct"),
    ("Language masked", "language/no_text"),
    ("None", "none/no_text"),
]:
    a = []
    for seed in range(3):
        p = BASE / f"evaluation_final/{loc}/seed{seed}/scores.json"
        records = json.loads(p.read_text())
        d = {r["uid"]: r for r in records}
        assert len(records) == 701 and set(d) == all_uids
        for r in meta:
            q = d[r["uid"]]
            assert q["seed"] == seed
            for k in ["path", "caption_uid", "query_start", "caption_last_frame"]:
                assert q[k] == r[k]
            assert q["caption_last_frame"] <= q["query_start"]
        a.append([[d[u][k] for k in metrics] for u in uids])
        sources[str(p.relative_to(BASE))] = hashlib.sha256(p.read_bytes()).hexdigest()
    scores[name] = np.array(a)
    out[name] = {
        k: {
            "mean": float(scores[name][:, :, j].mean()),
            "generation_sd": float(scores[name][:, :, j].mean(1).std(ddof=1)),
        }
        for j, k in enumerate(metrics)
    }
draws = np.random.default_rng(20260923).integers(0, n_episodes, (10000, n_episodes))
size = np.bincount(groups)
den = size[draws].sum(1)
contrasts = {}
for reference in ["None", "Language masked"]:
    contrasts[reference] = {}
    for j, k in enumerate(metrics):
        gain = (scores[reference][:, :, j] - scores["Language"][:, :, j]).mean(0)
        tot = np.bincount(groups, weights=gain)
        boot = tot[draws].sum(1) / den
        contrasts[reference][k] = {
            "gain": float(gain.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
result = {
    "models": out,
    "contrasts_baseline_minus_language": contrasts,
    "windows": len(uids),
    "episodes": n_episodes,
    "generation_seeds": [0, 1, 2],
    "sources_sha256": sources,
    "eligible_record_paths": sorted({r["path"] for r in meta}),
    "window_uids": uids,
}
a.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"models": out, "contrasts": contrasts}, indent=2))
