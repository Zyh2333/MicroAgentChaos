"""Command line entry point for offline evidence-graph audits.

    .venv/bin/python -m experiments.evidence_assoc.cli \
        output/MDOC/MicroCEARCL/samples/2026-11-16-01:40:43 --out /tmp/evidence-audit.json
"""
import argparse
import json
from pathlib import Path

from .audit import aggregate, audit_artifact
from .relation import Policy


def _artifacts(paths):
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
    out = []
    for file in files:
        if file.name == "mapping.json" or file.name.startswith(
                ("results-", "metrics-", "selection-")):
            continue
        try:
            head = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(head, dict) and "rules" in head and "layers" in head:
            out.append(file)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="artifact JSON files or sample directories")
    parser.add_argument("--out", help="write the full report here")
    parser.add_argument("--degree-budget", type=int, default=6)
    parser.add_argument("--limit", type=int, help="audit at most N artifacts")
    parser.add_argument("--ucg-out", help="write one Unexplained Case Graph snapshot per sample here")
    parser.add_argument("--ucg-status", choices=("unexplained", "all"), default="unexplained",
                        help="which samples to snapshot (default: only unexplained)")
    parser.add_argument("--ucg-reasons", nargs="+", default=None,
                        help="unexplained reasons to include (default: OutOfTaxonomy InsufficientSupport Mixed)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    policy = Policy()
    files = _artifacts(args.paths)
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error("no MicroCEARCL artifacts found in the given paths")

    reports = []
    for file in files:
        try:
            report = audit_artifact(file, policy=policy)
        except Exception as error:  # a broken artifact must not abort the batch
            print(f"[skip] {file.name}: {type(error).__name__}: {error}")
            continue
        report["graph"]["degree_budget"] = args.degree_budget
        reports.append(report)
        if not args.quiet:
            rq1, rq2 = report["rq1"], report["rq2"]
            print(f"[{report['sample']}] obs={report['observations']:4d} "
                  f"same={report['graph']['edges']['SAME']:4d} "
                  f"dep={report['graph']['edges']['DEPENDENT']:4d} "
                  f"unk={report['graph']['unknown_total']:5d} | "
                  f"RQ1 dec={rq1['decided']:4d} prec={rq1['precision']:.3f} "
                  f"cover={rq1['coverage']:.3f} | "
                  f"RQ2 bundles={rq2['propositions']['bundles']:3d} "
                  f"cites={rq2['propositions']['citations']:4d} "
                  f"single={rq2['propositions']['single_source_support']:2d}")

    total = aggregate(reports)
    payload = {"aggregate": total, "reports": reports,
               "configuration": {"policy": policy.__dict__,
                                 "degree_budget": args.degree_budget}}
    if args.ucg_out:
        from .ucg import write_pool
        manifest = write_pool(files, args.ucg_out,
                              responses=args.ucg_reasons,
                              include_all=args.ucg_status == "all")
        payload["ucg_pool"] = {"out": args.ucg_out, "count": manifest["count"],
                               "skipped": len(manifest["skipped"])}
        print(f"ucg pool -> {args.ucg_out} ({manifest['count']} snapshot(s), "
              f"{len(manifest['skipped'])} skipped)")
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.out}")
    print(json.dumps(total, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
