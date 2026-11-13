"""Grounding 2x2 report (docs/preregistration_grounding.md): claims H1-H3 from ga_<variant>_<task>_s<seed> archives.

	python -m src.tools.grounding_report results/ga_*.tar.gz
"""
import sys

import numpy as np

from src.tools.general_report import ARMS, MIN_SUCCESS, load

VARIANTS = ("grounded", "noaux", "stockaux", "stock")
REAL, IMAG = ("grounded", "noaux"), ("stockaux", "stock")
B = 5000


def main(paths: list):
	rng = np.random.default_rng(0)
	S, excluded = {}, []
	for p in paths:
		name = p.split("/")[-1].replace(".tar.gz", "")
		arms, coll = load(p)
		if not all(a in arms for a in ARMS):
			print(f"incomplete {name}: {sorted(arms)}")
			continue
		if coll is None or coll < MIN_SUCCESS:
			excluded.append((name, round(coll or 0, 3)))
			continue
		seeds = sorted(r["seed"] for r in arms["k1"])
		S[name] = {a: np.array([{r["seed"]: r["success"] for r in arms[a]}[s] for s in seeds]) for a in ARMS}
	group = {v: sorted(n for n in S if n.split("_")[1] == v) for v in VARIANTS}
	print(f"included per variant: { {v: len(g) for v, g in group.items()} }; excluded (collection < {MIN_SUCCESS}): {excluded}\n")
	print(f"{'variant':10s} " + " ".join(f"{a:>8s}" for a in ARMS))
	for v in VARIANTS:
		if group[v]:
			print(f"{v:10s} " + " ".join(f"{np.mean([S[n][a].mean() for n in group[v]]):8.3f}" for a in ARMS))

	def gain(a, b):            # per robot gain function over an index dict
		return lambda n, i: (S[n][a][i] - S[n][b][i]).mean()

	def contrast(g, pos, neg):
		"""mean gain over robots of the pos variants (each variant weighted equally) minus the same for neg."""
		def stat(idx):
			m = lambda vs: np.mean([np.mean([g(n, idx[n]) for n in group[v]]) for v in vs]) if vs else 0.0
			return m(pos) - m(neg)
		full = stat({n: np.arange(len(S[n]["k1"])) for n in S})
		bs = [stat({n: rng.integers(0, len(S[n]["k1"]), len(S[n]["k1"])) for n in S}) for _ in range(B)]
		return full, *np.percentile(bs, [2.5, 97.5])

	def show(tag, g, pos, neg=(), claim=True):
		if not all(group[v] for v in pos + neg):
			print(f"{tag:40s} missing variant")
			return
		m, lo, hi = contrast(g, pos, neg)
		print(f"{tag:40s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  " + (("PASS" if lo > 0 else "FAIL") if claim else ""))

	G4, G6, R4 = gain("IR@4", "k4"), gain("IR@6", "k6"), gain("IR@4", "random@4")
	print("\nclaims (G = IR@4 - k4):")
	show("H1 G(real) - G(imagined)", G4, REAL, IMAG)
	show("H2 G(grounded_noaux)", G4, ("noaux",))
	show("H3 G(grounded) - G(stock)", G4, ("grounded",), ("stock",))
	print("\nreported:")
	show("b=6: G6(real) - G6(imagined)", G6, REAL, IMAG, False)
	show("IR@4-random@4: real - imagined", R4, REAL, IMAG, False)
	show("ensemble: G(aux) - G(no aux)", G4, ("grounded", "stockaux"), ("noaux", "stock"), False)
	for v in VARIANTS:
		show(f"{v}: IR@4 - k4", G4, (v,), (), False)
		show(f"{v}: IR@6 - k6", G6, (v,), (), False)
		show(f"{v}: IR@4 - random@4", R4, (v,), (), False)
		show(f"{v}: PE@4 - k4", gain("PE@4", "k4"), (v,), (), False)
		show(f"{v}: k1 - k4", gain("k1", "k4"), (v,), (), False)
	print("\nper robot IR@4 - k4:")
	for n in sorted(S):
		print(f"  {n:28s} {S[n]['IR@4'].mean() - S[n]['k4'].mean():+.3f}  (k1 {S[n]['k1'].mean():.2f})")


if __name__ == "__main__":
	main(sys.argv[1:])
