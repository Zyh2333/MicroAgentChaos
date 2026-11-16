"""Inspect one evidence tool without an LLM: python -m ...tool SAMPLE TOOL."""
import argparse
import json
from .io import load_sample
from .registry import Registry, MODULES
from .utils.observation_summary import READERS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sample")
    ap.add_argument("tool", choices=MODULES)
    ap.add_argument("--service")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--baseline")
    ap.add_argument('--topology-file')
    ap.add_argument('--arguments', type=json.loads, default={}, help='该工具的额外参数 JSON 对象')
    ap.add_argument("--evidence-mode", choices=("runtime",), default="runtime")
    ap.add_argument("--mask-csv-gen-auth", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if not isinstance(args.arguments, dict):
        ap.error('--arguments 必须为 JSON 对象')
    sample = load_sample(args.sample, args.baseline, args.evidence_mode,
                         mask_csv_gen_auth=args.mask_csv_gen_auth, topology_file=args.topology_file)
    arguments = ({'limit': args.limit} if args.tool in READERS else {}) | args.arguments
    if args.service:
        arguments["service"] = args.service
    output = Registry(sample).invoke(args.tool, arguments, args.stage)
    print(json.dumps({"result": output, "warnings": sample.warnings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
