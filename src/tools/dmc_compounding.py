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


if __name__ == "__main__":
	main(sys.argv[1:])
