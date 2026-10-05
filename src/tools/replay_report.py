"""Claims T1-T3 of docs/preregistration_audit_replay.md from the training runs' final_eval.json files.

	python -m src.tools.replay_report results/rep_*.tar.gz > docs/replay_results.md
"""
import json
import re
import sys
import tarfile
from pathlib import Path

import torch

from src.tools.pool_lambda import pooled_ci


def load(paths) -> dict:
	"""{(task, seed): {mode: final_eval dict}} from run tarballs or directories named rep_<mode>_<task>_s<seed>."""
	out = {}
	for p in map(Path, paths):
		m = re.match(r"rep_(uniform|lba|D)_(\w+?)_s(\d+)", p.name)
		if not m:
			continue
		if p.is_dir():
			text = next(p.rglob("final_eval.json")).read_text()
		else:
			with tarfile.open(p) as tf:
				member = next((x for x in tf.getmembers() if x.name.endswith("final_eval.json")), None)
				if member is None:
					print(f"warning: {p.name} has no final_eval.json (unfinished?)", file=sys.stderr)
					continue
				text = tf.extractfile(member).read().decode()
		out.setdefault((m.group(2), int(m.group(3))), {})[m.group(1)] = json.loads(text)
	return out


def diff(runs: dict, a: str, b: str) -> dict:
	per = [torch.tensor(r[a]["successes"]) - torch.tensor(r[b]["successes"]) for r in runs.values() if a in r and b in r]
	return pooled_ci(per)


def run(runs: dict) -> str:
	ci = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] ({d['models']} task-seed pairs)"
	L = ["# Detector-guided training: results", "", "| Task | Seed | " + " | ".join(f"{m} success" for m in ("uniform", "lba", "D")) +
	     " | " + " | ".join(f"{m} imag. error" for m in ("uniform", "lba", "D")) + " |", "| :-- | :-: |" + " :-: |" * 6]
	for (task, seed), r in sorted(runs.items()):
		s = [f"{r[m]['episode_success']:.2f}" if m in r else "–" for m in ("uniform", "lba", "D")]
		e = [f"{r[m].get('return_error_last', float('nan')):.3f}" if m in r else "–" for m in ("uniform", "lba", "D")]
		L.append(f"| {task} | {seed} | " + " | ".join(s + e) + " |")
	t1, t2 = diff(runs, "lba", "uniform"), diff(runs, "lba", "D")
	err = {m: [r[m]["return_error_last"] for r in runs.values() if m in r and "return_error_last" in r[m]] for m in ("uniform", "lba", "D")}
	mean = {m: sum(v) / len(v) if v else float("nan") for m, v in err.items()}
	L += ["", "## Pre-registered claims", "",
	      f"- **T1** success, lba − uniform: {ci(t1)} → {'passed' if t1['models'] and t1['ci95'][0] > 0 else 'failed'}",
	      f"- **T2** success, lba − D: {ci(t2)} → {'passed' if t2['models'] and t2['ci95'][0] > 0 else 'failed'}",
	      f"- **T3** imagination error, lba {mean['lba']:.3f} vs uniform {mean['uniform']:.3f} (D {mean['D']:.3f}) → "
	      f"{'passed' if mean['lba'] < mean['uniform'] else 'failed'}",
	      "", f"Exploratory: D − uniform success {ci(diff(runs, 'D', 'uniform'))}"]
	return "\n".join(L)


if __name__ == "__main__":
	print(run(load(sys.argv[1:])))
