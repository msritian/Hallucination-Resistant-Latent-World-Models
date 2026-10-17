"""Compounding error on DeepMind Control (released models, no dynamics ensembles): does critic-head spread (B, a
Q-ensemble) keep tracking growing imagination error, versus anchored / cumulative Bellman checks?

	python -m src.tools.dmc_compounding results/dmcH24_*.tar.gz
"""
import sys
import tarfile
import tempfile
from pathlib import Path

import torch

from src.preflight.metrics import auroc

SIG = ("LBA", "Aa", "Ac", "A", "B", "E")


def main(paths):
	per = []
	for p in map(Path, paths):
		d = Path(tempfile.mkdtemp())
		with tarfile.open(p) as tf:
			tf.extractall(d)
		per += [torch.load(f, weights_only=False) for f in d.rglob("perstep_*.pt")]
	err, grow, auc = [], {s: [] for s in SIG}, {s: [] for s in SIG}
	for t in per:
		E, pos, keep = t["E"].float(), t["pos"], t["keep"]
		L = E.shape[0]
		err.append(E.mean(1))
		for s in SIG:
			sc = t["scores"][s]
			on = torch.tensor([float(sc[k][pos[k]].mean()) if pos[k].any() else float("nan") for k in range(L)])
			grow[s].append(on)
			auc[s].append(torch.tensor([auroc(sc[k][keep[k]], pos[k][keep[k]]) if pos[k][keep[k]].any() and (~pos[k][keep[k]]).any()
			                            else float("nan") for k in range(L)]))
	m = lambda v: torch.nanmean(torch.stack(v), 0)
	e = m(err)
	print(f"{len(per)} tasks. True error growth first-3 -> last-3: x{float(e[-3:].mean() / e[:3].mean()):.2f}")
	print("signal  growth(on hallucinated)  AUROC first-3  last-3")
	for s in SIG:
		g, a = m(grow[s]), m(auc[s])
		print(f"{s:6s}  x{float(g[-3:].mean() / g[:3].mean()):.2f}                 {float(a[:3].mean()):.2f}         {float(a[-3:].mean()):.2f}")
	# per-task statistics: growth ratio per task, bootstrap 95% CI over tasks, and consistency counts
	ratio = {s: torch.tensor([float(v[-3:].nanmean() / v[:3].nanmean()) for v in grow[s]]) for s in SIG}
	ok = ~torch.stack([r.isnan() for r in ratio.values()]).any(0)          # tasks with hallucinated windows early and late
	g = torch.Generator().manual_seed(0)
	print(f"\nPer-task growth ratio (first 3 -> last 3 steps, hallucinated windows), {int(ok.sum())} tasks: mean [95% CI]")
	for s in SIG:
		r = ratio[s][ok]
		boots = torch.stack([r[torch.randint(len(r), (len(r),), generator=g)].mean() for _ in range(2000)])
		print(f"  {s:6s} x{float(r.mean()):.2f} [{float(boots.quantile(0.025)):.2f}, {float(boots.quantile(0.975)):.2f}]   "
		      f"grows (>1) on {int((r > 1).sum())}/{len(r)} tasks")
	for anc in ("Aa", "Ac", "LBA"):
		print(f"  {anc} grows more than B on {int((ratio[anc][ok] > ratio['B'][ok]).sum())}/{int(ok.sum())} tasks")


if __name__ == "__main__":
	main(sys.argv[1:])
