"""
cleanup.py
----------
Delete leftover `temp_dubbing_*` working directories.

Without `--keep_temp` the pipeline works inside a `tempfile.TemporaryDirectory`
that cleans itself up. With `--keep_temp` the working directory is deliberately
left behind, and those pile up fast: a few minutes of video becomes hundreds of
megabytes of WAVs, Demucs stems, and per-chunk artifacts.

Deleting is not undoable, so this module is deliberately narrow:

  * only directories whose name starts with `temp_dubbing` are ever considered;
  * only directly inside the output directory — no recursive walk;
  * symlinks are never followed and never deleted through;
  * anything holding a resumable checkpoint is kept unless explicitly included,
    because deleting it silently throws away hours of ASR and translation work.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import config

log = config.get_logger(__name__)


# The one prefix this module will ever delete. Anything else in the output
# directory — finished videos, `artifacts/` — is off limits by construction.
TEMP_DIR_PREFIX = "temp_dubbing"

# A directory holding these is mid-run or resumable; `--resume` reads them to
# skip ASR and translation. Removing one is throwing away real compute.
_CHECKPOINT_GLOB = "checkpoint_*.json"


@dataclass
class TempDirInfo:
    """A candidate directory and what deleting it would cost."""

    path: Path
    size_bytes: int
    file_count: int
    checkpoints: list[str]
    age_days: float

    @property
    def resumable(self) -> bool:
        return bool(self.checkpoints)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def _dir_stats(path: Path) -> tuple[int, int]:
    """(total bytes, file count), skipping anything unreadable."""
    total = 0
    count = 0
    for item in path.rglob("*"):
        try:
            if item.is_symlink() or not item.is_file():
                continue
            total += item.stat().st_size
            count += 1
        except OSError:
            continue
    return total, count


def _newest_mtime(path: Path) -> float:
    """
    Most recent mtime anywhere in the tree.

    The directory's own mtime is not enough: on Windows it often reflects
    creation rather than the last write, which would make an active run look
    stale and eligible for deletion.
    """
    newest = 0.0
    try:
        newest = path.stat().st_mtime
    except OSError:
        pass
    for item in path.rglob("*"):
        try:
            newest = max(newest, item.stat().st_mtime)
        except OSError:
            continue
    return newest


def find_temp_dirs(output_dir: str | Path | None = None) -> list[TempDirInfo]:
    """
    Describe every `temp_dubbing*` directory in *output_dir*, largest first.

    Read-only: this never deletes anything. Use it to show the user what a
    cleanup would remove before it happens.
    """
    import time

    base = Path(output_dir) if output_dir is not None else Path(config.OUTPUT_DIR)
    if not base.is_dir():
        return []

    now = time.time()
    found: list[TempDirInfo] = []
    for entry in sorted(base.iterdir()):
        # Name check first, then is_dir(): a symlink to elsewhere passes is_dir()
        # and we must not delete through it.
        if not entry.name.startswith(TEMP_DIR_PREFIX) or entry.is_symlink():
            continue
        if not entry.is_dir():
            continue

        size, count = _dir_stats(entry)
        checkpoints = sorted(p.name for p in entry.glob(_CHECKPOINT_GLOB))
        age_days = max(0.0, (now - _newest_mtime(entry)) / 86_400)
        found.append(
            TempDirInfo(
                path=entry,
                size_bytes=size,
                file_count=count,
                checkpoints=checkpoints,
                age_days=age_days,
            )
        )

    found.sort(key=lambda d: d.size_bytes, reverse=True)
    return found


def clean_temp_dirs(
    output_dir: str | Path | None = None,
    *,
    dry_run: bool = True,
    keep_resumable: bool = True,
    min_age_days: float = 0.0,
    only: Optional[Iterable[str | Path]] = None,
    exclude: Optional[Iterable[str | Path]] = None,
) -> tuple[list[TempDirInfo], int]:
    """
    Delete leftover temp working directories.

    Parameters
    ----------
    output_dir     : Directory to scan (default config.OUTPUT_DIR).
    dry_run        : Report what would be deleted and delete nothing.
                     Defaults to True so a mistaken call is harmless.
    keep_resumable : Skip directories holding checkpoints. On by default —
                     those represent finished ASR and translation work that
                     `--resume` can still use.
    min_age_days   : Only touch directories untouched for this long. Protects
                     a run happening right now in another terminal.
    only           : Restrict to these specific directories (names or paths).
                     Anything not matching a discovered temp dir is ignored,
                     so a typo deletes nothing.
    exclude        : Never delete these, whatever the other rules say. Callers
                     pass the directory of the run in progress.

    Returns
    -------
    (deleted, bytes_freed) — on a dry run, (would_delete, bytes_that_would_free).
    """
    candidates = find_temp_dirs(output_dir)
    if not candidates:
        log.info("No %s* directories found.", TEMP_DIR_PREFIX)
        return [], 0

    if only is not None:
        wanted = {Path(o).name for o in only}
        unknown = wanted - {c.path.name for c in candidates}
        for name in sorted(unknown):
            log.warning("Not a known temp directory, skipping: %s", name)
        candidates = [c for c in candidates if c.path.name in wanted]

    targets: list[TempDirInfo] = []
    protected = {Path(e).resolve() for e in exclude} if exclude else set()
    for info in candidates:
        if protected and info.path.resolve() in protected:
            log.debug("Keeping %s — in use by the current run.", info.path.name)
            continue
        if keep_resumable and info.resumable:
            log.info(
                "Keeping %s — has %d checkpoint(s), still resumable (%.0f MB).",
                info.path.name,
                len(info.checkpoints),
                info.size_mb,
            )
            continue
        if min_age_days > 0 and info.age_days < min_age_days:
            log.info(
                "Keeping %s — modified %.1f day(s) ago, under the %.1f day threshold.",
                info.path.name,
                info.age_days,
                min_age_days,
            )
            continue
        targets.append(info)

    if not targets:
        log.info("Nothing to delete.")
        return [], 0

    total = sum(t.size_bytes for t in targets)
    if dry_run:
        log.info(
            "Dry run — would delete %d directory(ies), freeing %.0f MB:",
            len(targets),
            total / (1024 * 1024),
        )
        for info in targets:
            log.info("  %s  (%.0f MB, %d files)", info.path.name, info.size_mb, info.file_count)
        log.info("Re-run without --dry_run to delete.")
        return targets, total

    deleted: list[TempDirInfo] = []
    freed = 0
    for info in targets:
        try:
            shutil.rmtree(info.path)
        except OSError as exc:
            # A file held open by another process is the usual cause on Windows.
            log.warning("Could not delete %s: %s", info.path.name, exc)
            continue
        deleted.append(info)
        freed += info.size_bytes
        log.info("Deleted %s (%.0f MB)", info.path.name, info.size_mb)

    log.info("Cleanup complete: %d directory(ies), %.0f MB freed.", len(deleted), freed / (1024 * 1024))
    return deleted, freed


def auto_clean_after_run(work_dir: Path, *, keep_temp: bool) -> None:
    """
    Post-run hook: drop this run's working directory unless it was asked for.

    Only ever touches the directory just used, and only when it matches the
    temp prefix — a caller passing something unexpected gets a no-op, not a
    deleted folder.
    """
    if keep_temp:
        return
    work_dir = Path(work_dir)
    if not work_dir.is_dir() or work_dir.is_symlink():
        return
    if not work_dir.name.startswith(TEMP_DIR_PREFIX):
        return
    try:
        shutil.rmtree(work_dir)
        log.info("Removed working directory %s", work_dir.name)
    except OSError as exc:
        log.warning("Could not remove %s: %s", work_dir.name, exc)


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python cleanup.py",
        description="Delete leftover temp_dubbing_* working directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cleanup.py                      # list what exists, delete nothing\n"
            "  python cleanup.py --delete             # delete, keeping resumable dirs\n"
            "  python cleanup.py --delete --all       # delete everything, checkpoints too\n"
            "  python cleanup.py --delete --older_than 7\n"
            "  python cleanup.py --delete --all --only temp_dubbing_videoplayback\n"
        ),
    )
    p.add_argument("--output_dir", default=None, help="Directory to scan (default: config.OUTPUT_DIR)")
    p.add_argument("--delete", action="store_true", help="Actually delete. Without this, only reports.")
    p.add_argument(
        "--all",
        action="store_true",
        help="Include directories with checkpoints. This discards resumable ASR/translation work.",
    )
    p.add_argument("--older_than", type=float, default=0.0, metavar="DAYS",
                   help="Only delete directories untouched for this many days")
    p.add_argument("--only", nargs="+", default=None, metavar="NAME",
                   help="Restrict to these directory names")
    return p


def main() -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()

    if not args.delete:
        found = find_temp_dirs(args.output_dir)
        if not found:
            print(f"No {TEMP_DIR_PREFIX}* directories found.")
            return 0
        total = sum(f.size_bytes for f in found)
        print(f"{len(found)} temp directory(ies), {total / (1024 * 1024):.0f} MB total:\n")
        for info in found:
            flag = f"resumable, {len(info.checkpoints)} checkpoint(s)" if info.resumable else "no checkpoints"
            print(f"  {info.size_mb:>7.0f} MB  {info.age_days:>5.1f}d  [{flag}]  {info.path.name}")
        print("\nNothing deleted. Add --delete to remove them.")
        return 0

    clean_temp_dirs(
        args.output_dir,
        dry_run=False,
        keep_resumable=not args.all,
        min_age_days=args.older_than,
        only=args.only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
