"""The learned Bellman audit on the GPU, for scoring every MPPI candidate at every planning step.

- ``GPUTrees``: exact torch evaluation of a fitted sklearn HistGradientBoostingClassifier (same splits, same leaf
  values, float64), batched over all candidates and steps. ``check`` compares it with sklearn on real features.
- ``gpu_features``: the audit's per-step features (as ``run_preflight._features``) without leaving the GPU.
- ``candidate_audit``: signals + features + probabilities p_t [L, N] for N imagined plans from start latents z0.
"""
import torch

from src.preflight.run_preflight import CRITIC_FEATURES, plan_signals


class GPUTrees:
	def __init__(self, clf, device):
		trees = [t[0] for t in clf._predictors]            # binary classification: one tree per boosting iteration
		n_max = max(len(t.nodes) for t in trees)
		T = len(trees)
		f = lambda name, dtype, fill: torch.full((T, n_max), fill, dtype=dtype)
		self.feature, self.threshold = f("feature", torch.long, 0), f("thr", torch.float64, 0.0)
		self.left, self.right = f("left", torch.long, 0), f("right", torch.long, 0)
		self.leaf, self.value = f("leaf", torch.bool, True), f("value", torch.float64, 0.0)
		self.missing_left = f("ml", torch.bool, True)
		for i, t in enumerate(trees):
			n = t.nodes
			assert not n["is_categorical"].any(), "categorical splits are not supported"
			k = len(n)
			self.feature[i, :k] = torch.as_tensor(n["feature_idx"].astype("int64"))
			self.threshold[i, :k] = torch.as_tensor(n["num_threshold"].astype("float64"))
			self.left[i, :k] = torch.as_tensor(n["left"].astype("int64"))
			self.right[i, :k] = torch.as_tensor(n["right"].astype("int64"))
			self.leaf[i, :k] = torch.as_tensor(n["is_leaf"].astype(bool))
			self.value[i, :k] = torch.as_tensor(n["value"].astype("float64"))
			self.missing_left[i, :k] = torch.as_tensor(n["missing_go_to_left"].astype(bool))
		self.depth = int(max(t.nodes["depth"].max() for t in trees)) + 1
		self.baseline = float(clf._baseline_prediction.reshape(-1)[0])
		for name in ("feature", "threshold", "left", "right", "leaf", "value", "missing_left"):
			setattr(self, name, getattr(self, name).to(device))
		self.T = T

	@torch.no_grad()
	def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
		"""X [M, F] -> P(hallucination) [M] (float32)."""
		X = X.to(self.feature.device, torch.float64)
		M = X.shape[0]
		node = torch.zeros(self.T, M, dtype=torch.long, device=X.device)
		t_idx = torch.arange(self.T, device=X.device).view(-1, 1)
		for _ in range(self.depth):
			leaf = self.leaf[t_idx, node]
			feat = self.feature[t_idx, node]
			x = X[torch.arange(M, device=X.device).view(1, -1).expand(self.T, M), feat]
			go_left = torch.where(torch.isnan(x), self.missing_left[t_idx, node], x <= self.threshold[t_idx, node])
			nxt = torch.where(go_left, self.left[t_idx, node], self.right[t_idx, node])
			node = torch.where(leaf, node, nxt)
		raw = self.baseline + self.value[t_idx, node].sum(0)
		return torch.sigmoid(raw).float()

	def check(self, clf, X: torch.Tensor, tol: float = 1e-5) -> float:
		"""Max |GPU - sklearn| probability on X [M, F]; raises if above tol (guards against sklearn version changes)."""
		ref = torch.as_tensor(clf.predict_proba(X.cpu().numpy())[:, 1], dtype=torch.float32)
		err = float((self.predict_proba(X).cpu() - ref).abs().max())
		if err > tol:
			raise RuntimeError(f"GPU trees disagree with sklearn: max error {err:.2e}")
		return err


def gpu_features(sig: dict, roll, L: int) -> torch.Tensor:
	"""Same columns and order as run_preflight._features(sig, roll, CRITIC_FEATURES, L, True): [L, N, F]."""
	cols = []
	for k in CRITIC_FEATURES:
		if k in sig:
			x = sig[k][:L].float()
			cols += [x, torch.cummax(x, 0).values]
	cols += [roll.v_mean[1:L + 1].squeeze(-1).float(), roll.r_hat[:L].squeeze(-1).float()]
	N = cols[0].shape[1]
	cols.append(torch.arange(L, dtype=torch.float32, device=cols[0].device).view(-1, 1).expand(L, N))
	return torch.stack(cols, -1)


@torch.no_grad()
def candidate_audit(agent, trees: GPUTrees, bias: torch.Tensor, z0: torch.Tensor, actions: torch.Tensor,
                    beta: float = 1.0):
	"""p_t [L, N] for every candidate and step, and the rollout (no dynamics-ensemble signals: not audit inputs)."""
	sig, roll = plan_signals(agent, z0, actions, beta, bias=bias, ensemble=False)
	L = actions.shape[0] - 1
	X = gpu_features(sig, roll, L)
	p = trees.predict_proba(X.reshape(-1, X.shape[-1])).view(L, -1)
	return p, roll
