from __future__ import annotations

from pathlib import Path
from typing import Literal


def list_entries(
  path: str | Path,
  kind: Literal["files", "dirs", "both"] = "files",
  recursive: bool = False,
  absolute: bool = False,
  sort: bool = True,
) -> list[Path]:
  """
  List entries under `path`.

  Args:
    path: Directory to scan.
    kind: "files" -> only files, "dirs" -> only subdirectories, "both" -> both.
    recursive: If True, traverse subdirectories.
    absolute: If True, return absolute Paths.
    sort: If True, sort results by path string.

  Returns:
    A list of pathlib.Path objects.

  Raises:
    FileNotFoundError: if path does not exist.
    NotADirectoryError: if path is not a directory.
    ValueError: if kind is invalid.
  """
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(f"Path does not exist: {p}")
  if not p.is_dir():
    raise NotADirectoryError(f"Not a directory: {p}")

  if recursive:
    it = p.rglob("*")
  else:
    it = p.iterdir()

  out: list[Path] = []
  for x in it:
    if kind == "files" and x.is_file():
      out.append(x)
    elif kind == "dirs" and x.is_dir():
      out.append(x)
    elif kind == "both":
      out.append(x)
    elif kind not in ("files", "dirs", "both"):
      raise ValueError('kind must be "files", "dirs", or "both"')

  if absolute:
    out = [x.resolve() for x in out]
  if sort:
    out.sort(key=lambda z: str(z))

  return out
