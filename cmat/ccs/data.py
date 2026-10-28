"""Strict, non-destructive transit-timing input contract for CCS.

Canonical CSV columns: epoch,t_mid_bjd_tdb,sigma_sec. Additional optional
columns: instrument,reference,timing_group. Duplicate epochs are permitted.
Legacy timestamps must be explicitly audited and converted before import.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TimingData:
    epoch: np.ndarray
    time_bjd_tdb: np.ndarray
    sigma_sec: np.ndarray
    instrument: tuple[str, ...] = ()
    reference: tuple[str, ...] = ()
    timing_group: tuple[str, ...] = ()
    input_sha256: str = "synthetic"

    def __post_init__(self) -> None:
        e = np.asarray(self.epoch)
        t = np.asarray(self.time_bjd_tdb, dtype=float)
        s = np.asarray(self.sigma_sec, dtype=float)
        if e.ndim != 1 or t.ndim != 1 or s.ndim != 1:
            raise ValueError("Epoch, time and uncertainty must be 1D arrays")
        if not (len(e) == len(t) == len(s)) or len(e) < 3:
            raise ValueError("At least three matched transit times are required")
        if not np.isfinite(e).all() or not np.equal(e, np.rint(e)).all():
            raise ValueError("Epochs must be finite integers")
        if not np.isfinite(t).all() or not np.isfinite(s).all() or (s <= 0).any():
            raise ValueError("Invalid timing value or nonpositive sigma_sec")
        if np.unique(e).size < 3:
            raise ValueError("At least three distinct epochs are required")
        for name in ("instrument", "reference", "timing_group"):
            items = getattr(self, name)
            if items and len(items) != len(e):
                raise ValueError(f"{name} must match the number of observations")
        object.__setattr__(self, "epoch", e.astype(np.int64))
        object.__setattr__(self, "time_bjd_tdb", t)
        object.__setattr__(self, "sigma_sec", s)

    @property
    def n(self) -> int:
        return len(self.epoch)

    @property
    def span_days(self) -> float:
        return float(self.time_bjd_tdb.max() - self.time_bjd_tdb.min())


def load_timings_csv(path: str | Path) -> TimingData:
    """Read canonical BJD_TDB timings, refusing ambiguous time standards."""
    path = Path(path)
    raw = path.read_bytes()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"epoch", "t_mid_bjd_tdb", "sigma_sec"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Canonical timing CSV requires {sorted(required)}; "
                "do not silently interpret tmid/terr or unknown time scales"
            )
        rows = list(reader)
    try:
        epochs = np.array([int(row["epoch"]) for row in rows], dtype=np.int64)
        times = np.array([float(row["t_mid_bjd_tdb"]) for row in rows])
        errors = np.array([float(row["sigma_sec"]) for row in rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timing record in {path}") from exc
    return TimingData(
        epoch=epochs,
        time_bjd_tdb=times,
        sigma_sec=errors,
        instrument=tuple(row.get("instrument", "") for row in rows),
        reference=tuple(row.get("reference", "") for row in rows),
        timing_group=tuple(row.get("timing_group", "") for row in rows),
        input_sha256=hashlib.sha256(raw).hexdigest(),
    )


def convert_wd_style_csv(
    src: str | Path, dst: str | Path, *, assume_bjd_tdb: bool = False
) -> Path:
    """Explicit, opt-in converter for epoch,tmid,terr (terr in *days*).

    Scientific provenance is not verified by this conversion. Caller must
    explicitly certify that the input uses BJD_TDB.
    """
    if not assume_bjd_tdb:
        raise ValueError("Conversion requires explicit assume_bjd_tdb=True")
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite {dst}")
    with src.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"epoch", "tmid", "terr"}.issubset(rows[0]):
        raise ValueError("Expected nonempty epoch,tmid,terr legacy CSV")
    payload = []
    for row in rows:
        payload.append({
            "epoch": str(int(row["epoch"])),
            "t_mid_bjd_tdb": str(float(row["tmid"])),
            "sigma_sec": format(float(row["terr"]) * 86400.0, ".16g"),
            "instrument": row.get("telescope", ""),
            "reference": row.get("source", ""),
            "timing_group": row.get("source", ""),
        })
    with dst.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(payload[0]))
        writer.writeheader()
        writer.writerows(payload)
    load_timings_csv(dst)
    return dst
