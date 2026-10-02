import tarfile

import torch

from src.tools import pool_lambda as pl


def _write(tmp, name, rates, n=40, seed=0):
	g = torch.Generator().manual_seed(seed)
	d = tmp / name / "evalout"
	d.mkdir(parents=True)
	rows = ["arm,episode,seed,success,reward"]
	for arm, p in rates.items():
		succ = (torch.rand(n, generator=g) < p).int().tolist()
		rows += [f"{arm},{i},{7000 + i},{s},0.0" for i, s in enumerate(succ)]
	(d / "episodes.csv").write_text("\n".join(rows) + "\n")
	tar = tmp / f"{name}.tar.gz"
	with tarfile.open(tar, "w:gz") as tf:
		tf.add(d, arcname="evalout")
	return tar


def test_claims_detect_clear_effects(tmp_path):
	rates = {"standard_H3": 0.3, "stock_tdmpc2_H3": 0.3}
	for H in (12, 24):
		rates |= {pl.ELVIS.format(H=H): 0.6, pl.CONST.format(H=H): 0.6,
		          pl.PEN.format(H=H, b=1.0): 1.0, pl.PEN.format(H=H, b=0.5): 0.8}
	paths = [_write(tmp_path, f"eval_m{i}_lambda", rates, seed=i) for i in range(3)]
	models = {p.name: pl.read_episodes(str(p)) for p in paths}
	c = pl.claims(models)
	assert c["C2"]["passed"] and c["C3"]["passed"] and c["C2"]["models"] == 3
	assert "stock_tdmpc2_H3" in pl.success_table(models)
	pl.main([str(p) for p in paths])


def test_no_effect_is_not_passed():
	same = {"a": {s: float(s % 2) for s in range(50)}, "b": {s: float(s % 2) for s in range(50)}}
	r = pl.pooled_ci(pl.paired({"m": same}, [("a", "b")]))
	assert r["mean"] == 0 and r["ci95"][0] <= 0 <= r["ci95"][1]


def test_grounding_pairs_and_pessimism(tmp_path):
	base = {"stock_tdmpc2_H3": 0.0, "standard_H3": 0.3, "standard_H12": 0.2, "standard_H12_vmin": 0.6,
	        "standard_H3_vmin": 0.3, "standard_H24": 0.1, "standard_H24_vmin": 0.5,
	        "elvis_H12_const0.8": 0.4, "elvis_H12_const0.8_vmin": 0.7, "elvis_H24_const0.8_vmin": 0.7}
	g = _write(tmp_path, "eval_seed_grounded_stack_s20_lambda", dict(base, stock_tdmpc2_H3=0.9), seed=1)
	s = _write(tmp_path, "eval_seed_stock_stack_s20_lambda", dict(base, stock_tdmpc2_H3=0.1), seed=2)
	models = pl.load([str(g), str(s)], "_lambda")
	gr = pl.grounding(models)
	assert set(gr["per_pair"]) == {"stack_s20"} and gr["pooled_grounded_minus_stock"]["mean"] > 0.5
	pz = pl.pessimism(pl.load([str(_write(tmp_path, "eval_x_pessimism", base, seed=3))], "_pessimism"))
	assert pz["H12_min_vs_mean"]["mean"] > 0.2
	assert pl.model_key("bk_grounded_stackcube_s10") == ("grounded", "stack_s10")
