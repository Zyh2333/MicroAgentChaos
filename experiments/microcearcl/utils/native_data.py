"""Small, provenance-preserving helpers for native observation files."""
from datetime import datetime, timezone
import re

from ..core import number, reference
from .background_noise import policy


def epoch(value):
    if number(value):
        return float(value)
    if not isinstance(value, str):
        return None
    # Python 3.10 does not accept the collector's nanosecond ISO timestamps.
    text = re.sub(r'(\.\d{6})\d+', r'\1', value.strip()).replace('Z', '+00:00')
    try:
        stamp = datetime.fromisoformat(text)
        # Native CSV timestamps use the collector's UTC wall clock.
        return (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        return None


def log_identity(file):
    pod, _, container = file.stem.partition('_')
    service = container or re.sub(r'-[a-z0-9]{6,10}-[a-z0-9]{5}$', '', pod)
    if pod.startswith(service + '-edge-'):
        service += '-edge'
    return service, pod


def services(sample):
    return sorted(set(sample.cache.get('services', [])) |
                  {log_identity(f)[0] for f in sample.ns('log').glob('*.log')})


def logs(sample, service=None, start=None, end=None):
    """Stream original logs; no semantic filters, labels, or chaos directory access."""
    lower, upper = epoch(start), epoch(end)
    for value, parsed in ((start, lower), (end, upper)):
        if value is not None and parsed is None:
            raise ValueError('时间必须为 ISO 时间或 epoch 秒')
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError('start 必须小于 end')
    for file in sorted(sample.ns('log').glob('*.log')):
        name, pod = log_identity(file)
        if service is not None and service != name:
            continue
        relative = str(file.relative_to(sample.path))
        with file.open(encoding='utf-8', errors='replace') as handle:
            for line_no, line in enumerate(handle, 1):
                stamp = epoch(line.split(maxsplit=1)[0]) if line.strip() else None
                if not sample.in_window(stamp):
                    continue
                if lower is not None and (stamp is None or stamp < lower):
                    continue
                if upper is not None and (stamp is None or stamp >= upper):
                    continue
                text = policy(sample).text(line, service=name, channel='runtime_logs',
                    ref={'file': relative, 'line': line_no})
                if line.strip() and not text.strip():
                    continue
                yield {'service': name, 'pod': pod, 'timestamp': stamp,
                       'text': text.rstrip('\r\n'), 'ref': reference(relative, line_no)}
