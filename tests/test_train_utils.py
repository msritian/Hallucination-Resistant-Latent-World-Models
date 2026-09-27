import torch
from tensordict import TensorDict

from src.utils.csvlog import CSVLog


def test_csvlog_adds_new_columns_and_keeps_rows(tmp_path):
	log = CSVLog(tmp_path / "train.csv")
	log.write(dict(step=50, reward=1.0))
	log.write(dict(step=100, reward=2.0, consistency_loss=0.5))
	log = CSVLog(tmp_path / "train.csv")  # reopen (resume)
	log.write(dict(step=150, reward=3.0, consistency_loss=0.4))
	lines = (tmp_path / "train.csv").read_text().strip().splitlines()
	assert lines[0] == "step,reward,consistency_loss"
	assert lines[1:] == ["50,1.0,", "100,2.0,0.5", "150,3.0,0.4"]


def test_episode_checkpoint_roundtrip_is_plain_tensors():
	eps = [TensorDict(obs=torch.randn(51, 35), action=torch.randn(51, 7), reward=torch.randn(51),
	                  terminated=torch.zeros(51), batch_size=(51,)) for _ in range(3)]
	saved = torch.stack(eps).to_dict()
	assert all(isinstance(v, torch.Tensor) for v in saved.values())
	restored = TensorDict(saved, batch_size=saved["reward"].shape[:2])
	assert restored.shape == (3, 51)
	assert torch.equal(restored["obs"][1], eps[1]["obs"])
