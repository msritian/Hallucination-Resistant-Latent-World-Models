"""Is detection easy only because weak models make huge, obvious errors?

	python -m src.tools.difficulty --success success.json feat_*.tar.gz
Three checks on the main (H=12) dumps:
1. per robot: how sloppy the model is (eps_hall in reward units), how far past the bar its hallucinations are, and the
   agent's success rate next to the detector's AUROC;
2. AUROC by error size: hallucinations only just past the bar (1-2x eps_hall) vs moderate (2-4x) vs huge (>4x), each
   against all correct steps, stratified by imagined step;
3. strong agents (success >= 0.7) vs weak agents, overall and on the near-threshold hallucinations only.
"""
import argparse
import json

import torch

from src.preflight.metrics import stratified_auroc
from src.tools import audit_ablations as aa

SIG = ("LBA", "Aa", "A", "B", "E", "D", "M")
BINS = ((1, 2, "1-2x"), (2, 4, "2-4x"), (4, float("inf"), ">4x"))


def per_robot(d):
	ev, cal = d["ev"], d["cal"]
	pos, keep, _, _ = aa.labels(ev, cal["E"])
	hall = torch.quantile(cal["E"][0].flatten().float(), 0.99).item()
	neg = keep & ~pos
	steps = torch.arange(pos.shape[0]).view(-1, 1).expand_as(pos)
	m = aa.fit(d, aa.CRITIC, True)
	sc = {"LBA": aa.score(m, d, aa.CRITIC, True)} if m is not None else {}
	for s in SIG[1:]:
		if s in ev["signals"]:
			sc[s] = ev["signals"][s][:pos.shape[0]].float()
	ratio = ev["E"].float() / hall
	out = {"hall": hall, "n_pos": int(pos.sum()), "ratio_med": float(ratio[pos].median()) if pos.any() else float("nan"),
	       "frac_huge": float((ratio[pos] > 4).float().mean()) if pos.any() else float("nan"), "auc": {}, "bin": {}}
	for s, v in sc.items():
		out["auc"][s] = stratified_auroc(v[keep], pos[keep], steps[keep])
		for lo, hi, name in BINS:
			p = pos & (ratio >= lo) & (ratio < hi)
			k = p | neg
			out["bin"][(s, name)] = stratified_auroc(v[k], p[k], steps[k]) if p.sum() >= 10 else float("nan")
	return out


def main(argv=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("--success", required=True, help="JSON {robot/critic: success rate}")
	ap.add_argument("paths", nargs="+")
	a = ap.parse_args(argv)
	succ = json.load(open(a.success))
	dumps = aa.load_dumps(a.paths)
	res = {r: per_robot(d) for r, d in sorted(dumps.items())}
	mean = lambda xs: aa.mean(list(xs))

	print("1. Per robot (eps_hall in reward units; ratio = hallucination error / eps_hall)")
	print(f"{'robot':42s} {'success':>7s} {'eps_hall':>8s} {'#hall':>6s} {'med ratio':>9s} {'>4x':>5s} {'LBA':>6s} {'best rival':>14s}")
	for r, o in res.items():
		riv = {s: v for s, v in o["auc"].items() if s in ("B", "E", "D", "M")}
		best = max(riv, key=riv.get) if riv else None
		print(f"{r:42s} {succ.get(r, float('nan')):7.2f} {o['hall']:8.3f} {o['n_pos']:6d} {o['ratio_med']:9.2f} "
		      f"{o['frac_huge']:5.2f} {aa.fmt(o['auc'].get('LBA'))} {(best or '-') + ' ' + aa.fmt(riv.get(best)):>14s}")

	print("\n2. AUROC by hallucination size (positives in the bin vs all correct steps), mean over robots")
	print(f"{'signal':7s} " + " ".join(f"{n:>7s}" for _, _, n in BINS) + "    all")
	for s in SIG:
		row = [mean(o["bin"].get((s, n), float("nan")) for o in res.values()) for _, _, n in BINS]
		print(f"{s:7s} " + " ".join(aa.fmt(x).rjust(7) for x in row) + "  " + aa.fmt(mean(o["auc"].get(s, float("nan")) for o in res.values())))

	print("\n3. Strong (success >= 0.7) vs weak agents")
	groups = {"strong": [r for r in res if succ.get(r, 0) >= 0.7], "weak": [r for r in res if succ.get(r, 0) < 0.7]}
	print(f"{'group':7s} {'n':>3s} {'eps_hall':>8s} {'med ratio':>9s} {'>4x':>5s} {'LBA all':>8s} {'LBA 1-2x':>8s} {'B all':>6s} {'E all':>6s} {'D all':>6s}")
	for g, rs in groups.items():
		o = [res[r] for r in rs]
		print(f"{g:7s} {len(rs):3d} {mean(x['hall'] for x in o):8.3f} {mean(x['ratio_med'] for x in o):9.2f} "
		      f"{mean(x['frac_huge'] for x in o):5.2f} {aa.fmt(mean(x['auc'].get('LBA', float('nan')) for x in o)):>8s} "
		      f"{aa.fmt(mean(x['bin'].get(('LBA', '1-2x'), float('nan')) for x in o)):>8s} "
		      + " ".join(aa.fmt(mean(x['auc'].get(s, float('nan')) for x in o)).rjust(6) for s in ("B", "E", "D")))
	# rank correlation across robots: agent success vs detector AUROC, and model sloppiness vs AUROC
	rs = [r for r in res if r in succ and "LBA" in res[r]["auc"]]
	rank = lambda v: torch.tensor(v).argsort().argsort().float()
	spear = lambda x, y: float(torch.corrcoef(torch.stack([rank(x), rank(y)]))[0, 1])
	lba = [res[r]["auc"]["LBA"] for r in rs]
	print(f"\nSpearman over {len(rs)} robots: success vs LBA AUROC {spear([succ[r] for r in rs], lba):+.2f};  "
	      f"eps_hall vs LBA AUROC {spear([res[r]['hall'] for r in rs], lba):+.2f};  "
	      f"median ratio vs LBA AUROC {spear([res[r]['ratio_med'] for r in rs], lba):+.2f}")


if __name__ == "__main__":
	main()
