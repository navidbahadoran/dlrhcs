"""Repository-root and path helpers.

The housing data scripts must run from any current working directory and from
repository folders with different names.  These helpers resolve paths against a
validated project root without requiring Git at runtime.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]


def _validate_repo_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    required = [root / "dlrhcs", root / "scripts"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise ValueError(f"invalid DLRHCS repository root {root}: missing {', '.join(missing)}")
    if not (root / "dlrhcs").is_dir() or not (root / "scripts").is_dir():
        raise ValueError(f"invalid DLRHCS repository root {root}: project markers are not directories")
    return root


def _discover_from(start: Path) -> Path:
    cur = start.expanduser().resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / "dlrhcs").is_dir() and (candidate / "scripts").is_dir():
            return _validate_repo_root(candidate)
    raise ValueError(f"could not locate DLRHCS repository root from {start}")


def find_repo_root(start: Optional[PathLike] = None,
                   explicit: Optional[PathLike] = None,
                   env_var: str = "DLRHCS_ROOT") -> Path:
    """Resolve the repository root by explicit path, environment, then discovery."""
    if explicit not in (None, ""):
        return _validate_repo_root(Path(explicit))
    env = os.environ.get(env_var)
    if env:
        return _validate_repo_root(Path(env))
    if start not in (None, ""):
        return _discover_from(Path(start))
    return _discover_from(Path(__file__))


def resolve_repo_path(path: PathLike, root: Path) -> Path:
    """Resolve user paths: absolute as-is, relative against the repo root."""
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def repo_relative(path: PathLike, root: Path) -> str:
    """Return a stable repository-relative path where possible."""
    p = Path(path).expanduser().resolve()
    root = root.expanduser().resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()
