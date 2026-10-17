"""OPE trust study: does a consistency score predict how wrong a policy's off-policy (imagined) value estimate is?

	python -m src.tools.ope_trust_report results/ope_*.tar.gz
Policy level: estimate = mean imagined return, truth = mean real return over the same starts; Spearman between each
score (averaged over the policy's episodes) and |estimate - truth|. Episode level: Spearman(score, |R_hat - R|).
"""
import math
import sys

from src.tools.imag_eval_report import load, spearman

SCORES = ("sarsa_mean", "A_mean", "Aa_mean", "D_mean", "B_mean", "LBA", "D", "B")


def main(paths):
	data = load(paths)
	L = ["# OPE trust: does the score predict the estimation error?", "",
	     "| Robot | level | " + " | ".join(SCORES) + " |", "| :-- | :-- |" + " :-: |" * len(SCORES)]
	acc = {s: [] for s in SCORES}
	for name, rows in data.items():
		pols = sorted({r["policy"] for r in rows})
		agg = {}
		for p in pols:
			m = [r for r in rows if r["policy"] == p]
			agg[p] = dict(err=abs(sum(r["imagined_return"] for r in m) / len(m) - sum(r["real_return"] for r in m) / len(m)),
			              **{s: sum(r.get(s, float("nan")) for r in m) / len(m) for s in SCORES})
		err = [agg[p]["err"] for p in pols]
		cells = []
		for s in SCORES:
			v = [agg[p][s] for p in pols]
			rho = spearman(v, err) if not any(math.isnan(x) for x in v) else float("nan")
			cells.append(rho); acc[s].append(rho)
		L.append(f"| {name} | policy ({len(pols)}) | " + " | ".join(f"{c:+.2f}" for c in cells) + " |")
		gap = [abs(r["imagined_return"] - r["real_return"]) for r in rows]
		L.append(f"| {name} | episode ({len(rows)}) | " + " | ".join(
			f"{spearman([r.get(s, float('nan')) for r in rows], gap):+.2f}" if not any(math.isnan(r.get(s, float("nan"))) for r in rows) else "–"
			for s in SCORES) + " |")
		L.append(f"| {name} | per-policy error | " + ", ".join(f"{p} {agg[p]['err']:.2f}" for p in pols) + " |" + " |" * (len(SCORES) - 1))
	L.append("| **mean (policy level)** | | " + " | ".join(
		f"**{sum(x for x in v if not math.isnan(x)) / max(1, sum(not math.isnan(x) for x in v)):+.2f}**" for v in acc.values()) + " |")
	print("\n".join(L))


if __name__ == "__main__":
	main(sys.argv[1:])
