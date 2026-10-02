"""Full per-robot detection tables (simulator labels) as Markdown.

	python -m src.tools.detection_tables results_local/real*_*.tar.gz > docs/meeting/detection_tables.md

Rows: robots (task, seed, critic type, role). Columns: detectors. One table per case (all / gradual / sudden),
the best detector per robot in bold, plus averages per task and overall. Ensemble detectors exist only on robots
trained with our critic, so averages are given both over all robots (no-extra-model detectors) and over the robots
with our critic (all detectors, like-for-like).
"""
import math
import re
import sys

from src.tools import detect_select as ds

METHODS = [("LBA", "Ours (tree)"), ("LBA_lin", "Ours (linear)"), ("Aa", "Ours: anchored only"),
           ("M", "MOBILE-style*"), ("D", "Dynamics ensemble*"),
           ("B", "Critic disagreement"), ("E", "ELVIS-style")]
CASES = [("real_all", "All hallucinations"), ("real_gradual", "Gradual hallucinations"), ("real_sudden", "Sudden hallucinations")]
ROLE = {"10": "practice", "40": "new", "20": "extra", "30": "extra"}


def describe(name: str) -> tuple:
	task = "Push" if "push" in name else "StackCube" if "stack" in name else "YCB" if "ycb" in name else "Peg" if "peg" in name else "?"
	seed = re.search(r"_s(\d+)", name).group(1)
	critic = "ours" if name.endswith("/grounded") else "normal"
	role = "extra" if task == "Peg" else ROLE.get(seed, "")  # Peg was never used to design the method
	return task, seed, critic, role


def fmt(v):
	return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def mean(xs):
	xs = [x for x in xs if x is not None and not math.isnan(x)]
	return sum(xs) / len(xs) if xs else float("nan")


def main(argv=None):
	paths = argv or sys.argv[1:]
	models = {}
	for p in paths:
		models.update(ds.load(p))
	order = {"Push": 0, "StackCube": 1, "YCB": 2, "Peg": 3}
	rows = sorted(models, key=lambda n: (order.get(describe(n)[0], 9), int(describe(n)[1]), describe(n)[2] != "ours"))
	out = ["# Detection results per robot (judge: the simulator)", "",
	       "AUROC: how well a detector ranks real hallucinations above correct imagined steps (0.5 = guessing, 1.0 = perfect).",
	       "Best detector per robot in **bold**. \\* = needs 4 extra world models; these exist only on robots trained with our critic.",
	       "Roles: practice = used to design the method (seed 10); new = trained afterwards, only tested (seed 40); "
	       "extra = trained earlier, not used to design the tree reader (seeds 20, 30).", ""]
	for key, title in CASES:
		out += [f"## {title}", "", "| Task | Seed | Critic | Role | " + " | ".join(l for _, l in METHODS) + " |",
		        "| :-- | :-: | :-- | :-- |" + " :-: |" * len(METHODS)]
		for n in rows:
			task, seed, critic, role = describe(n)
			vals = [models[n].get(k, {}).get(key) for k, _ in METHODS]
			best = max((v for v in vals if v is not None and not math.isnan(v)), default=None)
			cells = [f"**{fmt(v)}**" if v is not None and v == best else fmt(v) for v in vals]
			out.append(f"| {task} | {seed} | {critic} | {role} | " + " | ".join(cells) + " |")
		# averages
		out += ["", f"**Averages ({title.lower()})**", "",
		        "| Group | Robots | " + " | ".join(l for _, l in METHODS) + " |", "| :-- | :-: |" + " :-: |" * len(METHODS)]
		groups = [("All robots", rows), ("Robots with our critic (like-for-like, all methods)", [n for n in rows if n.endswith("/grounded")])]
		for task in ["Push", "StackCube", "YCB", "Peg"]:
			g = [n for n in rows if describe(n)[0] == task and n.endswith("/grounded")]
			if g:
				groups.append((f"{task}, robots with our critic", g))
		for label, members in groups:
			vals = []
			for k, _ in METHODS:
				vs = [models[n].get(k, {}).get(key) for n in members if k in models[n]]
				vals.append(mean(vs) if vs else float("nan"))
			best = max(v for v in vals if not math.isnan(v))
			cells = [f"**{fmt(v)}**" if v == best else fmt(v) for v in vals]
			out.append(f"| {label} | {len(members)} | " + " | ".join(cells) + " |")
		out.append("")
	# win counts vs each rival on robots with our critic
	g = [n for n in rows if n.endswith("/grounded")]
	out += ["## How often ours (tree) beats each method, robot by robot (robots with our critic)", "",
	        "| Compared with | " + " | ".join(t for _, t in CASES) + " |", "| :-- |" + " :-: |" * len(CASES)]
	for k, label in METHODS[1:]:
		cells = []
		for key, _ in CASES:
			pairs = [(models[n]["LBA"][key], models[n][k][key]) for n in g if k in models[n] and key in models[n].get(k, {})]
			cells.append(f"{sum(a > b for a, b in pairs)} of {len(pairs)}" if pairs else "–")
		out.append(f"| {label} | " + " | ".join(cells) + " |")
	print("\n".join(out))


if __name__ == "__main__":
	main()
