"""Versioned equal-context view over unchanged source/learner simulations."""

import sys, json, hashlib, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
ORIG = BASE / "spatial_fixed"
ADAPT = BASE / "adaptive_spatial"
sys.path.insert(0, str(ROOT))
from task_adapter import source, target, parameters, QUERY_DIMS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, j):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(j, indent=2))
    t.replace(p)


def main(task):
    oldfreeze = json.loads((ORIG / "FORMAL_FREEZE.json").read_text())
    data = {}
    manifests = {}
    raw = {}
    fields = json.loads((ROOT / "SOURCE_CONTEXT_FIELDS.json").read_text())["fields"][task]
    for policy in ["fixed", "adaptive"]:
        paths = []
        for i, original in enumerate(oldfreeze["data"][task]):
            original = Path(original)
            src = original if policy == "fixed" else ADAPT / task / "train" / f"{i:02d}"
            dst = ROOT / task / policy / "train" / f"{i:02d}"
            dst.mkdir(parents=True, exist_ok=False)
            meta = json.loads((src / "COMPLETE.json").read_text())
            audit = np.load(src / "source_audit.npz")
            original_audit = np.load(original / "source_audit.npz")
            q = audit["query"]
            assert np.array_equal(q, original_audit["query"])
            assert np.array_equal(q[::2], q[1::2])
            assert q.shape[1] == len(fields) == QUERY_DIMS[task]
            context = np.pad(q, ((0, 0), (0, 16 - q.shape[1]))).astype(np.float32)
            np.save(dst / "source_context.npy", context)
            # The existing data contract rejects symlinks outside the manifest directory.
            # Hard links preserve immutable bytes without another copy of the RGB arrays.
            for name in meta["files"]:
                os.link((src / name).resolve(), dst / name)
            meta = {
                **meta,
                "source_context_fields": "Full source initial state, goal/waypoint, geometry,known mass; no hidden physics or observed response",
                "source_context_common_to_all_modes": True,
                "source_policy": policy,
                "learner_records_unchanged": True,
            }
            meta["files"] = {n: sha(dst / n) for n in [*meta["files"], "source_context.npy"]}
            write(dst / "COMPLETE.json", meta)
            assert sha(dst / "records.npz") == sha(original / "records.npz")
            paths.append(str(dst))
            manifests[str(dst / "COMPLETE.json")] = sha(dst / "COMPLETE.json")
            raw.update({str((dst / n).resolve()): h for n, h in meta["files"].items()})
        data[policy] = paths
        write(
            ROOT / task / policy / "FORMAL_FREEZE.json",
            dict(
                ready=True,
                data={task: paths},
                data_manifests={
                    str(Path(p) / "COMPLETE.json"): sha(Path(p) / "COMPLETE.json") for p in paths
                },
                raw_data_files=raw.copy(),
                code={p.name: sha(p) for p in (ROOT).glob("*.py")},
                version="equal_source_geometry_v1",
                source_policy=policy,
                seed=0,
                updates_per_run=4096,
            ),
        )
    # Reconstruct source query from the existing, fixed evaluation seed, and check
    # it against all unchanged ID learner queries and physical siblings.
    original = BASE / "spatial_paired/formal" / task
    r = dict(np.load(original / "records.npz"))
    rng = np.random.default_rng(dict(gate=2309301000, bank=2309302000, ramp=2309303000)[task])
    contexts = []
    cache = ROOT / task / "recovered_source_context.npy"
    receipt = ROOT / task / "RNG_CONTEXT_RECOVERY.json"
    if task == "ramp" and receipt.exists():
        proof = json.loads(receipt.read_text())
        assert (
            proof["passed"]
            and proof["target_rng_records_verified"] == 512
            and proof["full_native_target_and_rng_parity_checks"] == 8
        )
        assert (
            sha(cache) == proof["contexts_sha256"]
            and sha(original / "records.npz") == proof["records_sha256"]
            and sha(original / "source_motion.npy") == proof["source_motion_sha256"]
        )
        contexts = np.load(cache)
        assert contexts.shape == (256, 16)
    else:
        for f in range(128):
            sq, sa, sm = source(task, rng)
            qs = [target(task, rng, ood=False) for _ in range(4)]
            ps, _ = parameters(task, rng)
            for si in range(2):
                ids = np.flatnonzero((r["source_id"] == 2 * f + si) & (r["condition"] == 0))
                assert all(
                    np.array_equal(
                        r["query"][i, : QUERY_DIMS[task]], qs[int(r["query_index"][i])][0]
                    )
                    for i in ids
                )
                assert all(np.array_equal(r["physics"][i, : len(ps[si])], ps[si]) for i in ids)
                contexts.append(np.pad(sq, (0, 16 - len(sq))))
        contexts = np.asarray(contexts, dtype=np.float32)
    assert np.array_equal(contexts[::2], contexts[1::2])
    for policy in ["fixed", "adaptive"]:
        src = original if policy == "fixed" else ADAPT / task / "evaluation"
        dst = ROOT / task / policy / "evaluation"
        dst.mkdir(parents=True, exist_ok=False)
        for n in ["records.npz", "source_motion.npy", "source_rgb.npy"]:
            os.link((src / n).resolve(), dst / n)
        np.save(dst / "source_context.npy", contexts)
        assert sha(dst / "records.npz") == sha(original / "records.npz")
        write(
            dst / "DATA_COMPLETE.json",
            dict(
                learner_records_sha=sha(dst / "records.npz"),
                source_context_sha=sha(dst / "source_context.npy"),
                common_context_identical_between_policies=True,
                hidden_physics_not_in_context=True,
                records=4096,
                families=128,
            ),
        )
    for a, b in zip(data["fixed"], data["adaptive"]):
        assert sha(Path(a) / "source_context.npy") == sha(Path(b) / "source_context.npy")
    write(
        ROOT / task / "DATA_READY.json",
        dict(
            source_data_retained=True,
            common_source_context_verified=True,
            training_records=16384,
            training_families=2048,
            seed=0,
            learner_labels_unchanged=True,
        ),
    )


if __name__ == "__main__":
    main(sys.argv[1])
