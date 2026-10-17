"""Training-maturity study (docs/preregistration_maturity.md): does detection get better as the same robot trains longer?

	python -m src.tools.maturity mfeat_*.tar.gz
Inputs are dumps of one model snapshot each, named mfeat_matmodels_<robot>_<step>.tar.gz (cluster/maturity.sub). Per
robot and snapshot: agent success, the learned audit's AUROC (fitted on that snapshot's own calibration half, as in the
main results), the anchored residual and the best ensemble rival, and the critic floor C of Proposition 1 (median |C|,
also divided by the spread of the critic's values), and the signal-to-floor ratio median |I| on hallucinated steps /
median |C|. Then the pre-registered criteria M1-M4.
"""
import re
import sys
from pathlib import Path

import torch

from src.tools import audit_ablations as aa

TOTAL = {"push": 200_000, "stack": 500_000, "ycb": 1_000_000}
EARLY, LATE = 0.25, 1.0
MIN_LABELS = 20


def parse(path: str):
	m = re.match(r"mfeat_matmodels_(.+)_(\d+)\.tar\.gz$", Path(path).name)
	if not m:
		raise ValueError(f"unexpected dump name {path}")
	robot, step = m.group(1), int(m.group(2))
	task = next(t for t in TOTAL if f"_{t}_" in robot)
	return robot, round(step / TOTAL[task], 2)


def measure(d) -> dict:
	ev = d["ev"]
	pos, keep, _, _ = aa.labels(ev, d["cal"]["E"])
	neg = keep & ~pos
	out = dict(success=float(d["success"][:, -1].float().mean()), n_pos=int(pos.sum()), n_neg=int(neg.sum()))
	if out["n_pos"] < MIN_LABELS or out["n_neg"] < MIN_LABELS:
		return out
	m = aa.fit(d, aa.CRITIC, True)
	if m is not None:
		out["LBA"] = aa.aurocs(aa.score(m, d, aa.CRITIC, True), d)["all"]
	for s in ("Aa", "D", "M"):
		if s in ev["signals"]:
			out[s] = aa.aurocs(ev["signals"][s][:pos.shape[0]].float(), d)["all"]
	if "floor" in ev:
		L = pos.shape[0]
		C, I = ev["floor"]["C"][:L].abs(), ev["floor"]["I"][:L].abs()
		v = ev["v_hat"].float()
		spread = float(torch.quantile(v.flatten(), 0.95) - torch.quantile(v.flatten(), 0.05)) + 1e-8
		out["floor"] = float(C[keep].median())
		out["floor_rel"] = out["floor"] / spread
		out["snr"] = float(I[pos].median()) / (out["floor"] + 1e-8)
	return out


def boot_ci(x: torch.Tensor, n: int = 2000, seed: int = 0):
	g = torch.Generator().manual_seed(seed)
	b = torch.stack([x[torch.randint(len(x), (len(x),), generator=g)].mean() for _ in range(n)])
	return float(x.mean()), float(b.quantile(0.025)), float(b.quantile(0.975))


def main(paths):
	res = {}
	for p in paths:
		robot, frac = parse(p)
		dumps = aa.load_dumps([p])
		d = next(v for k, v in dumps.items() if k.endswith("/grounded"))
		res.setdefault(robot, {})[frac] = measure(d)
	fracs = sorted({f for r in res.values() for f in r})
	f3 = lambda v: "  –  " if v is None else f"{v:.3f}"
	for key, name in (("success", "agent success"), ("LBA", "learned audit AUROC"), ("Aa", "anchored residual AUROC"),
	                  ("D", "ensemble D AUROC"), ("floor_rel", "critic floor |C| / value spread"), ("snr", "median |I| on hallucinations / median |C|")):
		print(f"\n{name}\n{'robot':26s} " + " ".join(f"{f:>6.2f}" for f in fracs))
		for r in sorted(res):
			print(f"{r:26s} " + " ".join(f3(res[r].get(f, {}).get(key)).rjust(6) for f in fracs))
		col = [aa.mean(res[r].get(f, {}).get(key) for r in res) for f in fracs]
		print(f"{'mean':26s} " + " ".join(f3(c).rjust(6) for c in col))

	ok = [r for r in res if all(res[r].get(f, {}).get("LBA") is not None for f in (EARLY, LATE))]
	print(f"\nPre-registered criteria ({len(ok)}/{len(res)} robots with both the {EARLY:.0%} and {LATE:.0%} snapshots):")
	if not ok:
		return
	delta = torch.tensor([res[r][LATE]["LBA"] - res[r][EARLY]["LBA"] for r in ok])
	m, lo, hi = boot_ci(delta)
	print(f"  M1 detection improves:   mean AUROC(100%) - AUROC(25%) = {m:+.3f} [{lo:+.3f}, {hi:+.3f}], higher on "
	      f"{int((delta > 0).sum())}/{len(ok)} robots -> {'PASS' if lo > 0 else 'FAIL'}")
	print(f"  M2 not inflated early:   lower bound {lo:+.3f} > -0.03 -> {'PASS' if lo > -0.03 else 'FAIL'}")
	fl = [r for r in ok if "floor_rel" in res[r][EARLY] and "floor_rel" in res[r][LATE]]
	falls = sum(res[r][LATE]["floor_rel"] < res[r][EARLY]["floor_rel"] for r in fl)
	need = -(-7 * len(fl) // 9)   # 7 of 9, scaled if robots are missing
	print(f"  M3 critic floor shrinks: relative floor lower at 100% on {falls}/{len(fl)} robots (need {need}) -> "
	      f"{'PASS' if fl and falls >= need else 'FAIL'}")
	ds = torch.tensor([res[r][LATE]["success"] - res[r][EARLY]["success"] for r in ok])
	m, lo, hi = boot_ci(ds)
	print(f"  M4 sanity, agents improve: mean success(100%) - success(25%) = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
	pts = [(res[r][f]["success"], res[r][f]["LBA"]) for r in res for f in res[r] if res[r][f].get("LBA") is not None]
	rank = lambda v: torch.tensor(v).argsort().argsort().float()
	rho = float(torch.corrcoef(torch.stack([rank([p[0] for p in pts]), rank([p[1] for p in pts])]))[0, 1])
	print(f"  reported: Spearman(agent success, LBA AUROC) over {len(pts)} (robot, snapshot) points = {rho:+.2f}")


if __name__ == "__main__":
	main(sys.argv[1:])
