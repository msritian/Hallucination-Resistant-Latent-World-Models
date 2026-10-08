"""Summarise the DeepMind Control audit: python -m src.tools.dmc_report results/dmc48_g*.tar.gz [results/dmc317_g*.tar.gz]"""
import json
import math
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

DET = ("LBA", "BellmanOnly", "NoBellman", "B", "E", "A", "Aa")


def main(paths):
	models = defaultdict(dict)
	for p in map(Path, paths):
		with tarfile.open(p) as tf:
			m = next(x for x in tf.getmembers() if x.name.endswith("results.json"))
			models[p.name.split("_")[0]].update(json.loads(tf.extractfile(m).read()))
	L = []
	for model, res in sorted(models.items()):
		ok = {t: r for t, r in res.items() if r["n_pos_cal"] >= 20}
		L += [f"## {model}: {len(res)} tasks ({len(ok)} with >= 20 calibration positives)", "",
		      "| Task | Return | " + " | ".join(DET) + " |", "| :-- | :-: |" + " :-: |" * len(DET)]
		for t, r in sorted(res.items()):
			a = {d: r["all"].get(d, float("nan")) for d in DET}
			best = max((v for v in a.values() if not math.isnan(v)), default=None)
			L.append(f"| {t}{'' if t in ok else ' *'} | {r['return']:.0f} | " +
			         " | ".join(("**%.3f**" % a[d]) if a[d] == best else ("%.3f" % a[d]) for d in DET) + " |")
		for case in ("all", "gradual", "sudden"):
			means = {d: [ok[t][case][d] for t in ok if d in ok[t][case] and not math.isnan(ok[t][case][d])] for d in DET}
			L.append(f"| **mean {case}** | | " + " | ".join(f"**{sum(v) / len(v):.3f}**" if v else "–" for v in means.values()) + " |")
		rivals = ("B", "E", "A", "Aa")
		wins = sum(1 for t in ok if ok[t]["all"]["LBA"] >= max(ok[t]["all"][d] for d in rivals if not math.isnan(ok[t]["all"][d])))
		L += ["", f"Audit best on {wins} of {len(ok)} tasks (all hallucinations). * = fewer than 20 calibration positives.", ""]
	print("\n".join(L))


if __name__ == "__main__":
	main(sys.argv[1:])
