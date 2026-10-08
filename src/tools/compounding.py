"""Do detection signals keep tracking COMPOUNDING error along latent rollouts? ("Biased Dreams" question)

	python -m src.tools.compounding /tmp/feat/feat_*.tar.gz
For robots with dynamics ensembles: per imagined step k, AUROC of each signal for that step's hallucination label (same
labels as the main results), the mean true error, and each signal's mean on hallucinated windows (does it keep rising
with the error, or flatten like ensembles in "Biased Dreams"?).
"""
import sys

import torch

from src.preflight.metrics import auroc
from src.tools import audit_ablations as aa

SIG = ("LBA", "Aa", "Ac", "A", "D", "M", "B")


def main(paths):
	dumps = {r: d for r, d in aa.load_dumps(paths).items() if r.endswith("/grounded") and "D" in d["ev"]["signals"]}
	per_k = {s: [] for s in SIG}; err_k = []; rise = {s: [] for s in SIG}
	for r, d in dumps.items():
		ev = d["ev"]; L = ev["E"].shape[0]
		pos, keep, _, _ = aa.labels(ev, d["cal"]["E"])
		m = aa.fit(d, aa.CRITIC, True)
		sc = {"LBA": aa.score(m, d, aa.CRITIC, True)}
		for s in SIG[1:]:
			sc[s] = ev["signals"][s][:L].float()
		err_k.append(ev["E"].float().mean(1))
		for s in SIG:
			row = []
			for k in range(L):
				kk = keep[k]
				row.append(auroc(sc[s][k][kk], pos[k][kk]) if pos[k][kk].any() and (~pos[k][kk]).any() else float("nan"))
			per_k[s].append(torch.tensor(row))
			# signal on hallucinated windows, late vs early, relative to the error's own growth
			on_pos = [float(sc[s][k][pos[k]].mean()) if pos[k].any() else float("nan") for k in range(L)]
			rise[s].append(torch.tensor(on_pos))
	n = len(dumps)
	nanmean = lambda t: torch.nanmean(torch.stack(t), 0)
	E = nanmean(err_k)
	L = E.shape[0]
	print(f"{n} robots with ensembles. Mean true imagination error by step: " + " ".join(f"{x:.2f}" for x in E))
	print("\nAUROC per imagined step (1 = first imagined transition):")
	print("signal " + " ".join(f"k{k + 1:>5d}" for k in range(L)) + "   early(1-3) late(9-11) change")
	for s in SIG:
		a = nanmean(per_k[s])
		e, l = float(a[:3].mean()), float(a[-3:].mean())
		print(f"{s:6s} " + " ".join(f"{x:6.2f}" for x in a) + f"   {e:.2f}       {l:.2f}     {l - e:+.2f}")
	print("\nGrowth from early (1-3) to late (9-11) steps, on hallucinated windows: signal ratio vs error ratio")
	er = float(E[-3:].mean() / E[:3].mean())
	for s in SIG:
		v = nanmean(rise[s])
		print(f"  {s:6s} signal x{float(v[-3:].mean() / v[:3].mean()):.2f}   (true error x{er:.2f})")


if __name__ == "__main__":
	main(sys.argv[1:])
