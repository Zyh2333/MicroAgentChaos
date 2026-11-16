"""Shared ranking formulas and batch style, grouped by collaboration/fault mode."""
from contextlib import redirect_stdout
from pathlib import Path
import results_log


def summarize(tag, rows, categories=()):
    print('\n\n' + '=' * 96 + f'\nSUMMARY [{tag}]\n' + '=' * 96)
    print(f"{'collaboration':<20}{'fault_mode':<15}{'fault':<32}{'rank':<6}{'gt':<30}{'top1'}")
    for row in rows:
        print(f"{row['topology_mode']:<20}{row['fault_mode']:<15}{row.get('fault',''):<32}"
              f"{row.get('rank',0):<6}{row.get('gt',''):<30}{row.get('top1') or ''}")
    print(f"Localization accuracy  top1={sum(r.get('rank') == 1 for r in rows)}/{len(rows)}")
    print('\nRanking metrics (rank 0/missing = miss)')
    print(f"{'stage':<26}" + ''.join(f'{"ACC@"+str(k):>9}' for k in results_log.TOP_K_CUTOFFS)
          + ''.join(f'{"AVG@"+str(k):>9}' for k in results_log.TOP_K_CUTOFFS) + f"{'MRR':>9}{'n':>5}")
    results_log._pm(tag, rows)
    for topology in sorted({r['topology_mode'] for r in rows}):
        subset = [r for r in rows if r['topology_mode'] == topology]
        results_log._pm(topology, subset)
        for fault_mode in sorted({r['fault_mode'] for r in subset}):
            print(f'  [{topology}/{fault_mode}]')
            results_log._pm('    ' + fault_mode, [r for r in subset if r['fault_mode'] == fault_mode])
    for category in categories:
        results_log._pm(f'  [{category}]', [r for r in rows if r.get('category') == category])
    successful = sum(r['status'] == 'success' for r in rows)
    print(f'\nBatch  completed={len(rows)} successful={successful} failed={len(rows)-successful}')
    for stage in ('read', 'localize', 'total'):
        values = [r['timing'][stage] for r in rows if stage in r.get('timing', {})]
        if values:
            print(f'  mean {stage}: {sum(values)/len(values):.3f}s ({len(values)})')


def write_sublogs(root, ts, tag, rows, categories=()):
    subsets = {}
    for row in rows:
        topology, fault_mode = row['topology_mode'], row['fault_mode']
        subsets.setdefault((topology,), []).append(row)
        subsets.setdefault((topology, fault_mode), []).append(row)
    for key, subset in subsets.items():
        folder = Path(root).joinpath(*key)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f'batch-{tag}_{ts}.log').open('w', encoding='utf-8') as output:
            with redirect_stdout(output):
                print(f'Sub-level log [{"/".join(key)}] for [{tag}] ({ts})')
                summarize(tag, subset, categories)
