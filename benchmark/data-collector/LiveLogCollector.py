#!/usr/bin/env python3
"""Capture Kubernetes service logs throughout a benchmark window."""

import argparse
import LogCollector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='agent-network')
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--interval', type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    LogCollector.watch_logs(args.namespace, args.start, args.output, args.interval)


if __name__ == '__main__':
    main()
