import json

from src.utils import mirror


def _step(path):
	return json.loads(path.read_text())["step"]


def _ckpt(d, step):
	d.mkdir(parents=True, exist_ok=True)
	(d / "checkpoint.pt").write_text(json.dumps({"step": step}))


def test_push_then_pull_after_eviction(tmp_path):
	out, mir = tmp_path / "run", tmp_path / "staging" / "runs" / "x"
	_ckpt(out, 50_000)
	(out / "train.csv").write_text("step\n50000\n")
	(out / "videos").mkdir()
	(out / "videos" / "a.mp4").write_text("v")
	mirror.push(out, mir)
	assert _step(mir / "checkpoint.pt") == 50_000 and (mir / "videos" / "a.mp4").exists()
	# eviction: fresh sandbox with an older (HTCondor-restored) checkpoint
	fresh = tmp_path / "fresh"
	_ckpt(fresh, 10_000)
	assert mirror.pull(fresh, mir, _step)
	assert _step(fresh / "checkpoint.pt") == 50_000 and (fresh / "train.csv").read_text() == "step\n50000\n"


def test_pull_keeps_newer_local(tmp_path):
	out, mir = tmp_path / "run", tmp_path / "mir"
	_ckpt(out, 80_000)
	_ckpt(mir, 60_000)
	assert not mirror.pull(out, mir, _step)
	assert _step(out / "checkpoint.pt") == 80_000
	empty = tmp_path / "empty"
	empty.mkdir()
	assert not mirror.pull(empty, tmp_path / "nope", _step)
