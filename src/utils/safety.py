"""Safeguards for resuming and checkpointing training runs."""
import torch


def episodes_that_fit(n_episodes: int, entries_per_episode: int, capacity: int) -> int:
	"""Number of most recent episodes whose entries fit in a replay buffer of the given capacity."""
	return min(n_episodes, capacity // entries_per_episode)


def model_is_finite(model) -> bool:
	return all(torch.isfinite(p).all() for p in model.parameters())
