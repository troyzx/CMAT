"""Safe CCS CLI: no automatic DVC pull or large computations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import convert_wd_style_csv, load_timings_csv
from .likelihood import fit_reference_ephemeris, score_model
from .validation import synthetic_profile_check


def _write_json(payload: dict, output: Path | None) -> None:
    result = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if output is None:
        print(result)
    else:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(result + "\n")
        print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m cmat.ccs")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="fast synthetic test without REBOUND")
    demo.add_argument("--out", type=Path)
    audit = sub.add_parser("audit", help="inspect a canonical timing CSV")
    audit.add_argument("--timings", type=Path, required=True)
    audit.add_argument("--out", type=Path)
    conv = sub.add_parser("convert-legacy", help="explicit WD style timing conversion")
    conv.add_argument("--source", type=Path, required=True)
    conv.add_argument("--out", type=Path, required=True)
    conv.add_argument("--assume-bjd-tdb", action="store_true")
    pilot = sub.add_parser("pilot", help="one REBOUND M0-M3 comparison, bounded")
    pilot.add_argument("--timings", type=Path, required=True)
    pilot.add_argument("--star-mass-msun", type=float, required=True)
    pilot.add_argument("--inner-mass-mjup", type=float, required=True)
    pilot.add_argument("--outer-mass-mjup", type=float, required=True)
    pilot.add_argument("--outer-period-day", type=float, required=True)
    pilot.add_argument("--outer-phase-deg", type=float, default=0)
    pilot.add_argument("--mode", choices=["fixed", "offset", "linear", "median"],
                       default="linear")
    pilot.add_argument("--max-epoch-span", type=int, default=3000)
    pilot.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        _write_json(synthetic_profile_check(), args.out)
    elif args.command == "convert-legacy":
        print(convert_wd_style_csv(args.source, args.out,
              assume_bjd_tdb=args.assume_bjd_tdb))
    elif args.command == "audit":
        data = load_timings_csv(args.timings)
        t0, p, e_ref = fit_reference_ephemeris(data)
        fit = score_model(
            data, 0*data.sigma_sec, reference_t0_bjd=t0,
            reference_period_day=p, reference_epoch=e_ref, mode="linear",
        )
        _write_json(dict(status="INPUT_QC_ONLY", n=data.n,
                         n_unique_epochs=len(set(data.epoch.tolist())),
                         span_days=data.span_days, sigma_min_sec=float(data.sigma_sec.min()),
                         sigma_max_sec=float(data.sigma_sec.max()),
                         input_sha256=data.input_sha256,
                         reference_t0_bjd_tdb=t0, reference_period_day=p,
                         null_linear_chi2=fit.chi2), args.out)
    elif args.command == "pilot":
        from .search import compare_candidate
        data = load_timings_csv(args.timings)
        result = compare_candidate(
            data, star_mass_msun=args.star_mass_msun,
            inner_mass_mjup=args.inner_mass_mjup,
            outer_mass_mjup=args.outer_mass_mjup,
            outer_period_day=args.outer_period_day,
            outer_phase_deg=args.outer_phase_deg,
            ephemeris_mode=args.mode,
            max_epoch_span=args.max_epoch_span,
        )
        _write_json(result, args.out)


if __name__ == "__main__":
    main()
