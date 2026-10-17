"""Summarise Horde results over robots: python -m src.tools.horde_report results/hd2_*.tar.gz > docs/horde_results.md"""
import json
import sys
import tarfile
from pathlib import Path


def load(paths):
	out = {}
	for p in map(Path, paths):
		with tarfile.open(p) as tf:
			m = next(x for x in tf.getmembers() if x.name.endswith("horde.json"))
			out[p.name.replace(".tar.gz", "").replace("hd2_", "")] = json.loads(tf.extractfile(m).read())
	return out


def main(paths):
	R = load(paths)
	cols = [("reward_free_auroc", "Horde, reward-free"), ("ensemble_D_auroc", "Ensemble D (reward-free)"),
	        ("reward_audit_auroc", "Reward audit (ours)"), ("combined_auroc", "Reward audit + Horde"),
	        ("per_signal_auroc_mean", "Per-signal AUROC"), ("localisation_top1", "Localise: Horde"),
	        ("localisation_ensemble", "Localise: ensemble"), ("localisation_majority", "Localise: majority"),
	        ("localisation_chance", "Localise: chance")]
	L = ["# Horde per-signal audits: results", "", "| Robot | " + " | ".join(c for _, c in cols) + " |", "| :-- |" + " :-: |" * len(cols)]
	for n, r in sorted(R.items()):
		L.append(f"| {n} | " + " | ".join(f"{r[k]:.3f}" if isinstance(r.get(k), float) else "–" for k, _ in cols) + " |")
	mean = lambda k: sum(r[k] for r in R.values() if isinstance(r.get(k), float)) / max(1, sum(isinstance(r.get(k), float) for r in R.values()))
	L.append("| **Mean** | " + " | ".join(f"**{mean(k):.3f}**" for k, _ in cols) + " |")
	L += ["", "## Signals most often the wrong one (hallucinated windows)", ""]
	for n, r in sorted(R.items()):
		names = r.get("signal_names") or []
		top = [f"{names[i] if names else i} ({c})" for i, c in r.get("most_wrong_signals", [])]
		L.append(f"- {n}: " + ", ".join(top))
	print("\n".join(L))


if __name__ == "__main__":
	main(sys.argv[1:])
