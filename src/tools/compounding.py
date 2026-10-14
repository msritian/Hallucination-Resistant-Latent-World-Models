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
TRAJ = ("Dt", "Mt")   # trajectory-level ensembles (PETS / STEVE-style), in dumps made after 2026-10-13


def main(paths):
	global SIG
	dumps = {r: d for r, d in aa.load_dumps(paths).items() if r.endswith("/grounded") and "D" in d["ev"]["signals"]}
	if all(t in d["ev"]["signals"] for d in dumps.values() for t in TRAJ):
		SIG = SIG + TRAJ
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
	print("signal " + " ".join(f"k{k + 1:>5d}" for k in range(L)) + "   first-3  last-3  change")
	for s in SIG:
		a = nanmean(per_k[s])
		e, l = float(a[:3].mean()), float(a[-3:].mean())
		print(f"{s:6s} " + " ".join(f"{x:6.2f}" for x in a) + f"   {e:.2f}       {l:.2f}     {l - e:+.2f}")
	print("\nGrowth from the first 3 to the last 3 imagined steps, on hallucinated windows: signal ratio vs error ratio")
	er = float(E[-3:].mean() / E[:3].mean())
	for s in SIG:
		v = nanmean(rise[s])
		print(f"  {s:6s} signal x{float(v[-3:].mean() / v[:3].mean()):.2f}   (true error x{er:.2f})")
	# per-robot statistics: growth ratio per robot, bootstrap 95% CI over robots, and consistency counts
	ratio = {s: torch.tensor([float(v[-3:].nanmean() / v[:3].nanmean()) for v in rise[s]]) for s in SIG}
	g = torch.Generator().manual_seed(0)
	print("\nPer-robot growth ratio (first 3 -> last 3 steps, hallucinated windows): mean [95% CI over robots]")
	for s in SIG:
		r = ratio[s]
		boots = torch.stack([r[torch.randint(len(r), (len(r),), generator=g)].mean() for _ in range(2000)])
		print(f"  {s:6s} x{float(r.mean()):.2f} [{float(boots.quantile(0.025)):.2f}, {float(boots.quantile(0.975)):.2f}]   "
		      f"grows (>1) on {int((r > 1).sum())}/{len(r)} robots")
	for ens in ("D", "M", "B") + tuple(t for t in TRAJ if t in SIG):
		for anc in ("Aa", "Ac", "LBA"):
			print(f"  {anc} grows more than {ens} on {int((ratio[anc] > ratio[ens]).sum())}/{len(ratio[ens])} robots")

	if all("shared_bias" in d["ev"] for d in dumps.values()):
		shared_bias_report(dumps)


def shared_bias_report(dumps):
	"""Do the ensemble members drift TOGETHER? Error of the ensemble mean vs member spread, steps first-3 vs last-3."""
	print("\nShared bias (heads rolled out independently from the real start), mean over robots:")
	print("  quantity                         first-3   last-3   growth")
	keys = (("lat_mean_err", "latent error of the ensemble mean"), ("lat_spread", "latent spread across heads"),
	        ("ret_mean_err", "return error of the ensemble mean"), ("ret_spread", "return spread across heads"))
	per = {k: torch.stack([d["ev"]["shared_bias"][k].float().mean(1) for d in dumps.values()]) for k, _ in keys}
	for k, name in keys:
		v = per[k].mean(0)
		print(f"  {name:32s} {float(v[:3].mean()):8.3f} {float(v[-3:].mean()):8.3f}   x{float(v[-3:].mean() / v[:3].mean()):.2f}")
	for err, spr, what in (("lat_mean_err", "lat_spread", "latent"), ("ret_mean_err", "ret_spread", "return")):
		ratio = per[err] / per[spr].clamp_min(1e-8)                                  # [robots, steps]: error per unit of spread
		early, late = ratio[:, :3].mean(1), ratio[:, -3:].mean(1)
		print(f"  {what}: error / spread  first-3 {float(early.mean()):.2f} -> last-3 {float(late.mean()):.2f};  "
		      f"rises on {int((late > early).sum())}/{len(late)} robots;  error > spread at the end on {int((late > 1).sum())}/{len(late)}")


if __name__ == "__main__":
	main(sys.argv[1:])
