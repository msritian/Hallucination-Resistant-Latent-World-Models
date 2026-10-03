"""Bar charts of detection results (simulator-judged) for the meeting.

	python -m src.tools.plot_detection_summary --dev results_local/real2_*.tar.gz --exam results_local/real_seed_grounded_*_s40.tar.gz --out docs/meeting/figs
"""
import argparse
from pathlib import Path

from src.tools import detect_select as ds

METHODS = [  # (key, label, uses extra models?)
	("LBA", "Ours: learned Bellman audit", False),
	("Aa", "Ours: anchored check alone", False),
	("M", "MOBILE-style (4 extra models)", True),
	("D", "Dynamics ensemble (4 extra models)", True),
	("B", "Critic disagreement", False),
	("E", "ELVIS-style", False),
]
TESTS = [("real_all", "All hallucinations"), ("real_gradual", "Gradual"), ("real_sudden", "Sudden")]
COLORS = {"LBA": "#1f77b4", "Aa": "#9ecae1", "M": "#ff7f0e", "D": "#fdae6b", "B": "#bdbdbd", "E": "#e0e0e0"}


def load_all(paths) -> dict:
	models = {}
	for p in paths:
		models.update(ds.load(p))
	return models


ANY_ROBOT = ("LBA", "Aa", "B", "E")  # methods available on every robot (no extra models)


def grounded_only(models: dict) -> dict:
	"""Robots trained with our critic: the only ones that also have the 4 extra dynamics models."""
	return {n: m for n, m in models.items() if n.endswith("/grounded")}


def bars(ax, models, title, keys=None):
	import numpy as np
	methods = [m for m in METHODS if keys is None or m[0] in keys]
	x = np.arange(len(TESTS))
	w = 0.8 / len(methods)
	print(f"\n{title}")
	for i, (k, label, extra) in enumerate(methods):
		vals = [ds.score(models, k, t) for t, _ in TESTS]
		print(f"  {label:38s} " + "  ".join(f"{v:.3f}" for v in vals))
		b = ax.bar(x + (i - len(methods) / 2 + 0.5) * w, vals, w, label=label, color=COLORS[k],
		           edgecolor="black" if k == "LBA" else "none", hatch="//" if extra else None)
		if k == "LBA":
			for rect, v in zip(b, vals):
				ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
	ax.axhline(0.5, color="grey", ls=":", lw=1)
	ax.text(len(TESTS) - 0.45, 0.51, "chance", color="grey", fontsize=8)
	ax.set_xticks(x)
	ax.set_xticklabels([n for _, n in TESTS])
	ax.set_ylim(0.2, 1.0)
	ax.set_ylabel("AUROC (higher = better; 0.5 = guessing)")
	ax.set_title(title)


def main(argv=None):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	p = argparse.ArgumentParser()
	p.add_argument("--dev", nargs="+", required=True)
	p.add_argument("--exam", nargs="+", required=True)
	p.add_argument("--out", required=True)
	p.add_argument("--prefix", default="", help="filename prefix, e.g. 'all_' for the all-robots version")
	p.add_argument("--label", default="brand-new robots", help="how to describe the --exam robots in titles")
	a = p.parse_args(argv)
	out = Path(a.out)
	out.mkdir(parents=True, exist_ok=True)
	dev, exam = load_all(a.dev), load_all(a.exam)

	# 1. Main result on new robots, like-for-like: (a) all robots, methods available everywhere;
	#    (b) robots with our critic, all methods (the ensemble methods exist only there)
	fig, axes = plt.subplots(1, 2, figsize=(18, 5.2), sharey=True, gridspec_kw=dict(width_ratios=[4, 7]))
	bars(axes[0], exam, f"(a) All {len(exam)} robots: methods needing no extra models", ANY_ROBOT)
	g = grounded_only(exam)
	bars(axes[1], g, f"(b) The {len(g)} robots with our critic: ALL methods, same robots")
	axes[0].legend(fontsize=8, loc="lower right")
	axes[1].legend(fontsize=8, loc="lower right", ncol=2)
	fig.suptitle(f"Detecting real hallucinations on {a.label} (judge: the simulator)")
	fig.tight_layout()
	fig.savefig(out / (a.prefix + "fig_detection_new_robots.png"), dpi=150)

	# 2. Practice vs new robots, all methods on the same (our-critic) robots
	fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
	gd = grounded_only(dev)
	bars(axes[0], gd, f"Practice robots with our critic ({len(gd)}): all methods")
	bars(axes[1], g, f"New robots with our critic ({len(g)}): all methods")
	axes[1].legend(fontsize=8, loc="lower right", ncol=2)
	fig.tight_layout()
	fig.savefig(out / (a.prefix + "fig_detection_practice_vs_new.png"), dpi=150)

	# 3. Per task on new robots: the robot with our critic, all methods (like-for-like)
	tasks = {}
	for name, m in g.items():
		task = "Push" if "push" in name else "StackCube" if "stack" in name else "YCB" if "ycb" in name else "Peg" if "peg" in name else name
		tasks.setdefault(task, {})[name] = m
	fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), sharey=True)
	for ax, (task, ms) in zip(axes if len(tasks) > 1 else [axes], sorted(tasks.items())):
		bars(ax, ms, f"{task}: {len(ms)} robot(s) with our critic, all methods")
	axes[-1].legend(fontsize=7, loc="lower right", ncol=2)
	fig.tight_layout()
	fig.savefig(out / (a.prefix + "fig_detection_per_task.png"), dpi=150)

	# 4. Robot-by-robot (robots with our critic): ours vs the two strongest rivals
	import numpy as np
	order = {"push": 0, "stack": 1, "ycb": 2, "peg": 3}
	def label_of(n):
		t = "push" if "push" in n else "stack" if "stack" in n else "ycb" if "ycb" in n else "peg"
		sd = n.split("_s")[-1].split("/")[0].split(".")[0]
		return t, int(sd), f"{dict(push='Push', stack='Stack', ycb='YCB', peg='Peg')[t]} s{sd}"
	names = sorted(g, key=lambda n: (order[label_of(n)[0]], label_of(n)[1]))
	fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
	for ax, rival, rlabel in [(axes[0], "M", "MOBILE-style (4 extra models)"), (axes[1], "D", "dynamics ensemble (4 extra models)")]:
		x = np.arange(len(names))
		ours = [g[n]["LBA"]["real_all"] for n in names]
		theirs = [g[n][rival]["real_all"] for n in names]
		ax.bar(x - 0.2, ours, 0.4, label="ours", color=COLORS["LBA"])
		ax.bar(x + 0.2, theirs, 0.4, label=rlabel, color=COLORS[rival])
		for i, (o, t) in enumerate(zip(ours, theirs)):
			ax.text(i, max(o, t) + 0.01, "✓" if o > t else "✗", ha="center", fontsize=11, color="green" if o > t else "red")
		ax.set_xticks(x)
		ax.set_xticklabels([label_of(n)[2] for n in names], rotation=35, ha="right", fontsize=9)
		ax.set_ylim(0.4, 1.02)
		won = sum(o > t for o, t in zip(ours, theirs))
		ax.set_title(f"Each robot with our critic: ours vs {rlabel}\nours higher on {won} of {len(names)} robots (all hallucinations)")
		ax.legend(fontsize=8, loc="lower right")
	fig.tight_layout()
	fig.savefig(out / (a.prefix + "fig_detection_per_robot.png"), dpi=150)
	print(f"wrote 4 figures to {out}")


if __name__ == "__main__":
	main()
