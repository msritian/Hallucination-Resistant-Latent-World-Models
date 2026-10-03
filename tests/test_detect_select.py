import json
import tarfile

from src.tools import detect_select as ds


def _report(tmp, name, ours, base):
	res = {}
	for s in ds.CANDIDATES:
		res[s] = {t: {"auroc": ours.get(s, 0.5)} for t in ds.TESTS}
	for b in ds.BASELINES:
		res[b] = {t: {"auroc": base} for t in ds.TESTS}
	d = tmp / name / "reanalysis"
	d.mkdir(parents=True)
	(d / "report.json").write_text(json.dumps({"grounded": {"results": res}, "stock": {"results": res}}))
	tar = tmp / f"detect_{name}.tar.gz"
	with tarfile.open(tar, "w:gz") as tf:
		tf.add(d, arcname="reanalysis")
	return tar


def test_select_and_confirm(tmp_path):
	t = _report(tmp_path, "m1", {"A+Aa+P": 0.95, "A+Aa": 0.9}, base=0.8)
	models = ds.load(str(t))
	assert set(models) == {"m1/grounded", "m1/stock"}
	assert ds.select(models)["chosen"] == "A+Aa+P"
	c = ds.confirm(models, "A+Aa+P")
	assert c["passed_all"] and c["gradual"]["baselines"]["B"] == 0.8
	assert not ds.confirm(models, "Ab+Acb")["passed_all"]
	ds.main(["select", str(t)])


def test_confirm_real_tests(tmp_path):
	import json, tarfile
	res = {s: {t: {"auroc": 0.9} for t in ds.REAL_TESTS} for s in ds.CANDIDATES}
	res.update({b: {t: {"auroc": 0.7} for t in ds.REAL_TESTS} for b in ds.BASELINES})
	d = tmp_path / "r" / "reanalysis"
	d.mkdir(parents=True)
	(d / "report.json").write_text(json.dumps({"grounded": {"results": res}}))
	c = ds.confirm(ds.load(str(tmp_path / "r")), "A+Aa+P", ds.REAL_TESTS)
	assert c["passed_all"] and set(c) >= set(ds.REAL_TESTS)


def test_margins():
	models = {f"m{i}": {"X": {"t": 0.8 + 0.01 * i}, "R": {"t": 0.7}} for i in range(5)}
	r = ds.margins(models, "X", ("R",), ("t",))["t"]["R"]
	assert r["models_won"] == 5 and r["ci95"][0] > 0 and abs(r["mean_diff"] - 0.12) < 1e-9
