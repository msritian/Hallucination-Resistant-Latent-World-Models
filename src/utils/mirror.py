"""Keep a persistent copy of a training run (e.g. on CHTC staging) so evictions lose little progress."""
import os
import pwd
import shutil
from pathlib import Path
from typing import Callable, Optional


def staging_base() -> Optional[Path]:
	"""/staging/<first letter>/<user> if it exists and is writable (CHTC layout), else None."""
	try:
		user = pwd.getpwuid(os.getuid()).pw_name
	except KeyError:
		user = os.environ.get("USER", "")
	if not user:
		return None
	base = Path(f"/staging/{user[:1]}/{user}")
	return base if base.is_dir() and os.access(base, os.W_OK) else None


def pull(out: Path, mirror: Path, step_of: Callable[[Path], int]) -> bool:
	"""Copy the mirror into ``out`` if its checkpoint is newer. Returns True if restored."""
	remote, local = mirror / "checkpoint.pt", out / "checkpoint.pt"
	if remote.exists() and step_of(remote) > (step_of(local) if local.exists() else -1):
		shutil.copytree(mirror, out, dirs_exist_ok=True)
		return True
	return False


def push(out: Path, mirror: Path) -> None:
	"""Copy every file of ``out`` into the mirror; the checkpoint is replaced atomically."""
	for f in out.rglob("*"):
		if f.is_file() and f.suffix != ".tmp":
			dst = mirror / f.relative_to(out)
			dst.parent.mkdir(parents=True, exist_ok=True)
			if f.name == "checkpoint.pt":
				tmp = dst.with_suffix(".tmp")
				shutil.copy2(f, tmp)
				os.replace(tmp, dst)
			else:
				shutil.copy2(f, dst)
