from pathlib import Path
import json, hashlib
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
ROOT = a.inputs
OUT = a.out
OUT.mkdir(parents=True, exist_ok=False)
load = lambda p: json.loads((ROOT / p).read_text())
q = load("main/queries.json")
meta = {r["sample_index"]: r for r in q["scores_all64"]}
ids = sorted(i for i, r in meta.items() if not r["known_overlap"] and not r["padding"])
assert len(ids) == 50
excluded = {r["record_path"] for r in meta.values() if r["known_overlap"]}
cfg = load("main/config.json")
contract = cfg["evaluation_contract"]
train = set(contract["train_record_ids"])
hold = set(contract["holdout_record_ids"])
assert len(train) == 1252 and len(hold) == 64 and not train & hold
labels = np.array([meta[i]["instruction"] for i in ids])
groups = sorted(set(labels))
assert len(groups) == 14
rng = np.random.default_rng(20260926)
draw = rng.integers(0, len(groups), (10000, len(groups)))
counts = np.array([(labels == g).sum() for g in groups])


def ci(values):
    v = np.array(values)
    sums = np.array([v[labels == g].sum() for g in groups])
    means = sums[draw].sum(1) / counts[draw].sum(1)
    return dict(mean=float(v.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist())


def read_rows(path):
    p = ROOT / path
    r = [json.loads(x) for x in p.read_text().splitlines()]
    return r


base = read_rows("main/results/ID/samples.jsonl")
lighting = read_rows("lighting/results/ID/samples.jsonl")


def group(rows, condition):
    rr = {r["sample_index"]: r for r in rows if r["condition"] == condition}
    assert sorted(rr) == ids
    return rr


baseg = {c: group(base, c) for c in ["correct", "motion_only", "video_only", "none"]}
lightg = {c: group(lighting, c) for c in ["correct", "none", "wrong"]}
for c in ["correct", "none"]:
    for i in ids:
        for f in [
            "initial_noise_seed",
            "within_batch_index",
            "state_sha256",
            "latent_sha256",
            "target_state_raw",
            "query_rgb_sha256",
            "robot_state_sha256",
        ]:
            assert baseg[c][i][f] == lightg[c][i][f], (c, i, f)
conditions = {**baseg, "wrong": lightg["wrong"]}


def metrics(rr):
    vectors = {k: [] for k in ["ade", "fde", "state_mse", "pixel_mse", "lpips"]}
    for i in ids:
        r = rr[i]
        assert r.get("record_path", meta[i]["record_path"]) == meta[i]["record_path"]
        assert r["robot_start"] == meta[i]["robot_start_frame"]
        assert meta[i]["record_path"] in hold - excluded
        for f in [
            "target_state_raw",
            "initial_noise_seed",
            "within_batch_index",
            "query_rgb_sha256",
            "robot_state_sha256",
        ]:
            assert r[f] == baseg["correct"][i][f], (i, f)
        p = np.array(r["predicted_state_raw"])
        t = np.array(r["target_state_raw"])
        d = np.linalg.norm(p[:, :3] - t[:, :3], axis=1)
        np.testing.assert_allclose(d, r["translation_by_step"], rtol=3e-5, atol=5e-5)
        vectors["ade"].append(float(d.mean()))
        vectors["fde"].append(float(d[-1]))
        vectors["state_mse"].append(r["state_mse"])
        vectors["pixel_mse"].append(r["rgb_mse"])
        vectors["lpips"].append(np.mean(r["lpips_by_frame"]))
    return dict(summary={k: ci(v) for k, v in vectors.items()}, vectors=vectors)


static = {c: metrics(rr) for c, rr in conditions.items()}
prefix = {}
for c in ["video_none", "z0", "z01", "z012"]:
    rr = group(read_rows(f"prefix/results/Full/{c}/samples.jsonl"), "correct")
    prefix[c] = metrics(rr)
for c, other in [("video_none", "motion_only"), ("z012", "correct")]:
    np.testing.assert_array_equal(prefix[c]["vectors"]["ade"], static[other]["vectors"]["ade"])
# Recompute factorial metrics on exactly the same original indices/anchors.
raw = json.loads((ROOT / "factorial/raw.json").read_text())
archive = load("factorial/results_archive.json")
std = np.array(cfg["state_stats"]["robot_std"])
fact = {}
unfiltered = {}
for f in raw:
    assert f["sha256"] == archive["raw_hashes"]["eval/" + f["name"] + "/metrics.samples.jsonl"]
    model, seed = f["name"].rsplit("_s", 1)
    for variant in ["none", "real", "other_task"]:
        allrows = {r["sample_index"]: r for r in f["rows"] if r["variant"] == variant}
        assert sorted(allrows) == list(range(64))
        rows = [allrows[i] for i in ids]
        v = []
        p20 = []
        for i, r in zip(ids, rows):
            assert (
                r["target_path"] == meta[i]["record_path"]
                and r["robot_start_frame"] == meta[i]["robot_start_frame"]
                and r["overflow_padded_frames"] == 0
            )
            delta = (
                np.array(r["predicted_state_normalized"]) - np.array(r["target_state_normalized"])
            ) * std
            d = np.linalg.norm(delta[:, :3], axis=1)
            np.testing.assert_allclose(d, r["translation_by_step"], rtol=5e-5, atol=1e-3)
            v.append(float(d.mean()))
            p20.append(float(np.all(d <= 20)))
        fact.setdefault(model, {}).setdefault(variant, []).append(
            dict(seed=int(seed), ade=v, p20=p20)
        )
fm = {}
fv = {}
for model, conds in fact.items():
    fm[model] = {}
    fv[model] = {}
    for c, r in conds.items():
        a = np.mean([x["ade"] for x in r], axis=0)
        p = np.mean([x["p20"] for x in r], axis=0)
        fv[model, c] = a
        fm[model][c] = dict(ade=float(a.mean()), p20=float(100 * p.mean()), ade_ci=ci(a)["ci95"])
A = "decoupled_nodemo"
B = "decoupled_demo50"
C = "coupled_nodemo"
D = "coupled_demo50"
val = lambda m, c: fv[m, c]
contrasts = {
    "C_absent-A_absent": val(C, "none") - val(A, "none"),
    "D_absent-B_absent": val(D, "none") - val(B, "none"),
    "B_absent-A_absent": val(B, "none") - val(A, "none"),
    "D_absent-C_absent": val(D, "none") - val(C, "none"),
    "B_correct-B_absent": val(B, "real") - val(B, "none"),
    "D_correct-D_absent": val(D, "real") - val(D, "none"),
    "B_correct-B_incorrect": val(B, "real") - val(B, "other_task"),
    "absent_interaction": (val(D, "none") - val(C, "none")) - (val(B, "none") - val(A, "none")),
    "coupling_context_gain_difference": (val(D, "none") - val(D, "real"))
    - (val(B, "none") - val(B, "real")),
}
# Existing delayed-context window IDs already use the identical recording eligibility list.
sel = load("arrival/selection.json")
accepted = {r["sample_index"]: r for r in sel["accepted_rows"]}
assert len(accepted) == 83
for r in accepted.values():
    assert (
        r["record_path"] in hold - excluded
        and not r["padding"]
        and r["robot_start_frame"] + 128 < r["available_frames"]
    )
for schedule in ["8plus24", "32plus96"]:
    d = load(f"arrival/{schedule}_samples.json")
    for arm, rows in d.items():
        assert {r["sample_index"] for r in rows} == set(accepted)
        for r in rows:
            a = accepted[r["sample_index"]]
            assert (
                r["record_path"] == a["record_path"]
                and r["robot_start_frame"] == a["robot_start_frame"]
                and not r["padding"]
                and not r["known_overlap"]
            )
output = dict(
    window_ids=ids,
    recording_count=len({meta[i]["record_path"] for i in ids}),
    instruction_count=14,
    train_records=len(train),
    heldout_record_ids=len(hold),
    file_identity_disjoint=True,
    delay83_same_recording_filter_verified=True,
    delay_episodes=42,
    static=static,
    prefix=prefix,
    factorial=fm,
    contrasts={k: ci(v) for k, v in contrasts.items()},
    static_wrong_minus_none={
        k: ci(np.array(static["wrong"]["vectors"][k]) - static["none"]["vectors"][k])
        for k in ["ade", "state_mse", "pixel_mse", "lpips"]
    },
    uncertainty="10000 paired instruction-cluster bootstrap draws; generation seeds averaged per window first; fixed trained models",
)
(OUT / "results.json").write_text(json.dumps(output, indent=2))
print(
    json.dumps(
        {
            k: output[k]
            for k in [
                "recording_count",
                "instruction_count",
                "delay83_same_recording_filter_verified",
                "factorial",
                "contrasts",
                "static_wrong_minus_none",
            ]
        },
        indent=2,
    )
)
