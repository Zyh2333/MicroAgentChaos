"""Unique task completion provenance and five early local symptom features."""
from collections import defaultdict
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

EARLY_SECONDS=30
LOCAL=['node_error_rate','link_error_rate','mean_span_seconds','p95_span_seconds','retry_rate']
FEATURES=['early_'+k for k in LOCAL]
END=re.compile(r'^(\S+) PIPELINE ([0-9a-f]{32}) TIME COST TOTAL: ([0-9.eE+-]+)\s*$')


def completion_times(sample,tasks,start,end):
    positive,zero=defaultdict(dict),defaultdict(set)
    task_ids={t['task_id'] for t in tasks}
    for source in sorted((Path(sample)/'agent-network/log').glob('*planner*.log')):
        with source.open(errors='replace') as handle:
            for line_no,line in enumerate(handle,1):
                match=END.match(line.strip())
                if not match or match[2] not in task_ids:continue
                timestamp,seconds=pd.Timestamp(match[1]).timestamp(),float(match[3])
                if not np.isfinite(timestamp) or not np.isfinite(seconds):continue
                if seconds>0:positive[match[2]].setdefault((timestamp,seconds),(str(source),line_no))
                elif seconds==0:zero[match[2]].add(timestamp)
    records={}
    for task in tasks:
        tid=task['task_id'];markers=positive.get(tid,{})
        unique=len(markers)==1
        timestamp,provenance=(next(iter(markers))[0],next(iter(markers.values()))) if unique else (np.nan,('', ''))
        records[tid]={'completion_timestamp':timestamp,'completion_log_file':provenance[0],
                     'completion_log_line':provenance[1],'n_positive_end_markers':len(markers),
                     'n_zero_end_markers':len(zero.get(tid,())),
                     'in_fault_completion_window':bool(start<=timestamp<end) if unique else None,
                     'completion_time_basis':'unique_positive_pipeline_end' if unique else 'missing_or_ambiguous'}
    matched=sum(np.isfinite(r['completion_timestamp']) for r in records.values())
    return {'n_timed_tasks':matched,'completion_times_complete':bool(tasks) and matched==len(tasks),
            'n_ambiguous_completion_tasks':sum(r['n_positive_end_markers']>1 for r in records.values())},records


def early_features(tasks,completion,start,end,all_times_complete):
    cutoff=min(start+EARLY_SECONDS,end)
    observed=[t for t in tasks if start<=completion[t['task_id']]['completion_timestamp']<cutoff]
    nodes=sum(t['node_n'] for t in observed);links=sum(t['link_n'] for t in observed)
    durations=[v for t in observed for v in t['span_seconds']]
    metrics={'node_error_rate':sum(t['node_bad'] for t in observed)/nodes if nodes else np.nan,
             'link_error_rate':sum(t['link_bad'] for t in observed)/links if links else np.nan,
             'mean_span_seconds':float(np.mean(durations)) if durations else np.nan,
             'p95_span_seconds':float(np.percentile(durations,95)) if durations else np.nan,
             'retry_rate':sum(t['retry_spans'] for t in observed)/nodes if nodes else np.nan}
    reasons=[]
    if not all_times_complete:reasons.append('missing_or_ambiguous_task_completion_time')
    if not observed:reasons.append('no_completed_tasks_in_early_window')
    if not all(np.isfinite(v) for v in metrics.values()):reasons.append('unobserved_early_local_feature')
    return {**{'early_'+k:v for k,v in metrics.items()},'early_start':start,'early_cutoff':cutoff,
            'early_seconds':cutoff-start,'early_n_observed_tasks':len(observed),
            'early_feature_task_ids_json':json.dumps([t['task_id'] for t in observed]),
            'early_features_reliable':not reasons,'early_exclusion_reason':';'.join(reasons),
            'early_feature_basis':'retrospective_completed_task_prefix_not_live_span_snapshot'}
