from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def frozen(*modules: nn.Module):
	"""Temporarily set ``requires_grad=False`` on all parameters, restoring the previous flags on exit.

	Unlike ``torch.no_grad()``, gradients still flow *through* the modules into their inputs.
	"""
	saved = [(p, p.requires_grad) for m in modules for p in m.parameters()]
	try:
		for p, _ in saved:
			p.requires_grad_(False)
		yield
	finally:
		for p, flag in saved:
			p.requires_grad_(flag)
