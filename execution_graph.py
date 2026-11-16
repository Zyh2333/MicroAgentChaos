"""execution-graph parsing / indexing / slicing / aggregation."""
import os
import re
import glob
import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union

# System-independent event discovery.  These are generic log/exception markers,
# not a catalogue of root causes.  The actual signature is extracted from each
# dataset's own log text below.
_ERROR_EVENT_RE = re.compile(
    r'(?i)(?:\b(?:error|exception|fatal|critical|panic|failed|failure|'
    r'timeout|timed\s+out|denied|unavailable|quota|oom|killed)\b|'
    r'\b\w*Error\b|Traceback\s*\(most recent call last\))'
)
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_ISO_TS_RE = re.compile(
    r'\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+(?:Z|[+-]\d{2}:?\d{2})?\b')
_UUID_RE = re.compile(
    r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.I)
_HEX_ID_RE = re.compile(r'\b[0-9a-f]{16,}\b', re.I)
_IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b')
_URL_RE = re.compile(r'https?://\S+', re.I)
_DYNAMIC_FIELD_RE = re.compile(
    r'(?i)(\b(?:request[_ -]?id|trace[_ -]?id|span[_ -]?id|task[_ -]?id)\b'
    r'\s*[:=]\s*)["\']?[^\s,}\)"\']+["\']?')
_NUMBER_RE = re.compile(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])')
_STACK_BOILERPLATE_RE = re.compile(
    r'(?i)^(?:traceback\s*\(most recent call last\):|'
    r'during handling of the above exception.*|raise\s+\w+)')


def _normalize_log_signature(text: str, max_chars: int = 500) -> str:
    """Turn a concrete error event into a stable, readable log template."""
    text = _ANSI_RE.sub('', text).strip()
    text = _ISO_TS_RE.sub('<TIMESTAMP>', text)
    text = _UUID_RE.sub('<ID>', text)
    text = _HEX_ID_RE.sub('<ID>', text)
    text = _IP_RE.sub('<IP>', text)
    text = _URL_RE.sub('<URL>', text)
    text = _DYNAMIC_FIELD_RE.sub(r'\1<ID>', text)
    text = _NUMBER_RE.sub('<NUM>', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    # Drop transport/runtime prefixes so the same event can match across pods
    # and logging frameworks.
    text = re.sub(r'^(?:<timestamp>\s*)+', '', text)
    text = re.sub(r'^\[(?:<timestamp>|<num>|[,.: -])+\]\s*', '', text)
    # Different layers often wrap the same exception differently, e.g.
    # ``openai.RateLimitError: ...`` versus
    # ``Exception: (RateLimitError("..."), "Task Failed")``. Canonicalize
    # both to the concrete exception class + message so they match.
    for match in re.finditer(r'\b(?:[a-z_]\w*\.)*([a-z_]\w*error)\b', text, re.I):
        error_class = match.group(1).lower()
        if error_class == 'error':
            continue
        tail = text[match.end():]
        tail = re.sub(r'^[\s:("\']+', '', tail)
        tail = re.sub(
            r'["\']?\)?\s*,\s*["\']task failed["\']\)?\s*$', '', tail,
            flags=re.I)
        tail = tail.strip(' \t\r\n:\"\'()')
        text = f'{error_class}: {tail}'.rstrip(' :')
        break
    return text[:max_chars]


def _is_actionable_log_match(content: str, signatures) -> bool:
    """Reject known configuration text that merely names an error concept."""
    if (re.search(r'\b(?:response|connect|read)?timeout\s*=', content, re.I)
            and not re.search(r'\b(?:error|exception|failed|fatal)\b', content, re.I)):
        return False
    return True


class VertexSlice:
    """Failure evidence extracted for a single execution-graph vertex."""

    def __init__(self, key, level, status, time, token, cost, task):
        self.key = key
        self.level = level
        self.status = status
        self.time = time
        self.token = token
        self.cost = cost
        self.task = task
        self.empty_results: List[str] = []   # visible results left empty
        self.error_texts: List[str] = []     # raw error snippets
        self.signatures: List[str] = []       # normalized error signatures
        self.status_failed = False            # e.g. status=8 summarizer
        self.failed = False

    def to_dict(self, budget=1500):
        return {
            'vertex': self.key,
            'level': self.level,
            'status': self.status,
            'time': self.time,
            'token': self.token,
            'failed': self.failed,
            'failure_sources': (["status"] if self.status_failed else []) +
                               (["signature"] if self.signatures else []),
            'signatures': sorted(set(self.signatures)),
            'empty_results': self.empty_results,
            'error_texts': [t[:budget] for t in self.error_texts[:3]],
            'task': (self.task or '')[:budget],
        }


class ExecGraph:
    def __init__(self, trace_id, task, summary, total_level, participated,
                 planning_result, vertexes: Dict[str, VertexSlice],
                 ordered_vertexes: Optional[List[str]] = None,
                 task_status=None):
        self.trace_id = trace_id
        self.task = task
        self.summary = summary
        self.total_level = total_level
        self.participated = participated or []
        self.planning_result = planning_result
        self.vertexes = vertexes
        self.ordered_vertexes = ordered_vertexes or list(vertexes)
        self.task_status = task_status

    @property
    def failed(self):
        return (self.task_status in _FAILED_TASK_STATUSES or
                any(v.signatures for v in self.vertexes.values()))


def _match_signatures(text: str) -> List[str]:
    """Dynamically extract normalized error-event templates from text."""
    signatures = []
    seen = set()
    for line in text.splitlines() or [text]:
        if not _ERROR_EVENT_RE.search(line):
            continue
        structural = _ANSI_RE.sub('', line)
        structural = _ISO_TS_RE.sub('', structural).strip()
        structural = re.sub(r'^\[[^\]]+\]\s*', '', structural)
        if _STACK_BOILERPLATE_RE.search(structural):
            continue
        signature = _normalize_log_signature(line)
        if _STACK_BOILERPLATE_RE.search(signature):
            continue
        if signature and signature not in seen:
            seen.add(signature)
            signatures.append(signature)
    return signatures


def _stringify(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _parse_vertex(key, span, level) -> VertexSlice:
    vs = VertexSlice(
        key=key,
        level=level,
        status=span.get('status'),
        time=span.get('time'),
        token=span.get('token'),
        cost=span.get('cost'),
        task=span.get('task'),
    )
    # collect a text blob from messages/params/results to scan for errors
    blob_parts = []
    for m in span.get('messages', []) or []:
        c = str(m.get('content', ''))
        blob_parts.append(c)
        if _match_signatures(c):
            vs.error_texts.append(c)
    for r in span.get('results', []) or []:
        val = r.get('value', None)
        if val in (None, '', '[]', '{}'):
            vs.empty_results.append(r.get('name'))
        text = _stringify(val)
        blob_parts.append(text)
        if val and _match_signatures(text):
            vs.error_texts.append(str(val))
    for p in span.get('params', []) or []:
        val = p.get('value')
        text = _stringify(val)
        blob_parts.append(text)
        if val and _match_signatures(text):
            vs.error_texts.append(str(val))

    blob = '\n'.join(x for x in blob_parts if x)
    vs.signatures = _match_signatures(blob)
    # `visible` and empty outputs are descriptive only. They are not reliable
    # failure criteria; a vertex fails directly only on an error signature.
    vs.failed = bool(vs.signatures)
    return vs


_PLANNER_VERTEX = 'AgentNetworkPlannerGroup/AgentNetworkPlanner'
_SUMMARIZER_VERTEX = 'AgentNetworkSummarizerGroup'
_SUMMARIZER_TASK_STATUSES = {7, 8}
_FAILED_TASK_STATUSES = {3, 8}


def _synthetic_vertex(key, level):
    """Create a topology-only vertex absent from level_spans.

    Planner can be omitted on an empty graph and Summarizer is normally not
    persisted in level_routes at all.  Synthetic vertices restore the actual
    end-to-end route without inventing direct failure evidence.
    """
    return VertexSlice(key=key, level=level, status=None, time=None,
                       token=None, cost=None, task=None)


def _service_route_vertex(key: str) -> str:
    """Canonical service-level route vertex for a concrete vertex key."""
    if key == _PLANNER_VERTEX:
        return key
    return key.split('/', 1)[0]


def _merge_vertex_evidence(target: VertexSlice, source: VertexSlice):
    """Roll concrete-agent evidence up to its service Group vertex."""
    target.signatures = sorted(set(target.signatures) | set(source.signatures))
    target.error_texts = list(dict.fromkeys(target.error_texts + source.error_texts))
    target.empty_results = list(dict.fromkeys(
        target.empty_results + source.empty_results))
    target.failed = bool(target.signatures)


def parse_execution_graph(path: str, trace_metadata: Optional[Dict] = None
                          ) -> Optional[ExecGraph]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except Exception as e:
        print(f'[execution_graph] skip {os.path.basename(path)}: {e}')
        return None
    trace_metadata = trace_metadata or {}
    vertexes: Dict[str, VertexSlice] = {}
    span_order = []
    route_order = []
    route_output_evidence = {}
    for ld in d.get('level_details', []) or []:
        level = ld.get('level')
        # level_routes is the authoritative execution order. Keep sources and
        # destinations even when one of them has no level_spans payload.
        for source, destinations in (ld.get('level_routes', {}) or {}).items():
            if source not in route_order:
                route_order.append(source)
            for destination, route_detail in (destinations or {}).items():
                if destination not in route_order:
                    route_order.append(destination)
                # Params on an outgoing route are the source vertex's output
                # as delivered to the downstream vertex.
                for param in (route_detail or {}).get('params', []) or []:
                    val = param.get('value')
                    text = _stringify(val)
                    signatures = _match_signatures(text)
                    if signatures:
                        evidence = route_output_evidence.setdefault(
                            _service_route_vertex(source),
                            {'signatures': set(), 'error_texts': []})
                        evidence['signatures'].update(signatures)
                        evidence['error_texts'].append(str(val))
        for key, span in (ld.get('level_spans', {}) or {}).items():
            vertexes[key] = _parse_vertex(key, span, level)
            span_order.append((level if level is not None else float('inf'), key))
    span_order.sort(key=lambda item: item[0])

    # call_chains.path is the already-linearized form of level_routes and is
    # preferred when available. Fall back to route order, then span order.
    raw_route = list(trace_metadata.get('path') or route_order or
                     [key for _, key in span_order])
    # level_routes contains both service groups and their internal concrete
    # agents. RCA topology is service-level: retain xxxGroup only, except for
    # the planner's canonical level-1 vertex.
    ordered_vertexes = [
        key for key in raw_route
        if key == _PLANNER_VERTEX or '/' not in key
    ]
    if _PLANNER_VERTEX in ordered_vertexes:
        ordered_vertexes.remove(_PLANNER_VERTEX)
    ordered_vertexes.insert(0, _PLANNER_VERTEX)

    # Retain any span that was not present in the flattened route metadata.
    for _, key in span_order:
        if '/' not in key and key not in ordered_vertexes:
            ordered_vertexes.append(key)
    task_status = trace_metadata.get('task_status')
    if task_status in _SUMMARIZER_TASK_STATUSES:
        if _SUMMARIZER_VERTEX in ordered_vertexes:
            ordered_vertexes.remove(_SUMMARIZER_VERTEX)
        ordered_vertexes.append(_SUMMARIZER_VERTEX)
    for level, key in enumerate(ordered_vertexes, start=1):
        if key not in vertexes:
            vertexes[key] = _synthetic_vertex(key, level)

    # Concrete xxxGroup/xxAgent spans are diagnostic inputs for the service,
    # not separate topology nodes. Merge their signatures into xxxGroup.
    for key, source in list(vertexes.items()):
        group_key = _service_route_vertex(key)
        if group_key == key:
            continue
        if group_key not in vertexes:
            vertexes[group_key] = _synthetic_vertex(group_key, source.level)
        _merge_vertex_evidence(vertexes[group_key], source)
    for key, evidence in route_output_evidence.items():
        if key not in vertexes:
            vertexes[key] = _synthetic_vertex(key, None)
        target = vertexes[key]
        target.signatures = sorted(
            set(target.signatures) | evidence['signatures'])
        target.error_texts = list(dict.fromkeys(
            target.error_texts + evidence['error_texts']))
        target.failed = bool(target.signatures)
    if task_status == 8:
        # Status 8 specifically means the final summarization failed. This is
        # status evidence rather than an invented text/error signature.
        vertexes[_SUMMARIZER_VERTEX].status_failed = True
        vertexes[_SUMMARIZER_VERTEX].failed = True
    return ExecGraph(
        trace_id=d.get('trace_id'),
        task=d.get('task'),
        summary=d.get('summary'),
        total_level=d.get('total_level'),
        participated=d.get('participated_vertexes'),
        planning_result=d.get('planning_result'),
        vertexes=vertexes,
        ordered_vertexes=ordered_vertexes,
        task_status=task_status,
    )


def _load_trace_metadata(sample_ns_dir: str) -> Dict[str, Dict]:
    """Index task_status and the level_routes-derived path by trace id."""
    path = os.path.join(sample_ns_dir, 'graph', 'call_chains.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            rows = json.load(f)
    except Exception as e:
        print(f'[execution_graph] cannot read call_chains.json: {e}')
        return {}
    return {
        str(row.get('trace_id') or row.get('task_id')): row
        for row in (rows or [])
        if row.get('trace_id') or row.get('task_id')
    }


def load_execution_graphs(sample_ns_dir: str, max_traces: Optional[int] = 40,
                          candidate_services: Optional[List[str]] = None
                          ) -> List[ExecGraph]:
    """Load execution graphs from <sample>/<ns>/graph/graph_json/*.json.

    Prioritizes non-trivial / failed traces (larger files first) so the cap
    keeps the informative ones.  When ``candidate_services`` is supplied,
    only traces containing at least one vertex mapped to a candidate service
    are returned.  The cap is applied *after* this filtering, so large traces
    belonging only to unrelated services cannot crowd relevant traces out.
    """
    gdir = os.path.join(sample_ns_dir, 'graph', 'graph_json')
    files = glob.glob(os.path.join(gdir, '*.json'))
    files.sort(key=lambda p: os.path.getsize(p), reverse=True)
    trace_metadata = _load_trace_metadata(sample_ns_dir)
    graphs = []
    candidate_set = set(candidate_services) if candidate_services is not None else None
    for p in files:
        trace_id = os.path.splitext(os.path.basename(p))[0]
        g = parse_execution_graph(p, trace_metadata.get(trace_id))
        if g is None or not g.vertexes:
            continue
        if candidate_set is not None and not graph_matches_services(g, candidate_set):
            continue
        graphs.append(g)
        if max_traces is not None and len(graphs) >= max_traces:
            break
    return graphs


# Explicit trace-vertex-group -> metric service stem mapping (authoritative,
# from data-collector/analysis/joint_analysis.py). Full service name is
# f"{SVC_NS_PREFIX}-{stem}".
#
# Two datasets share the same 'agent-network' namespace but use different agent
# groups. Switch with the AGENT_DATASET env var (default 'MDOC'):
#     export AGENT_DATASET=MARBLEBench
SVC_NS_PREFIX = 'agent-network'

_GROUP2SVC_MDOC = {
    "AgentNetworkPlannerGroup": "planner",
    "AgentNetworkSummarizerGroup": "summarizer",
    "WordGenerationAgentGroup": "word-gen",
    "PdfGenAgentGroup":         "pdf-gen",
    "PdfAgentGroup":            "pdf-parsing",
    "DirectionAgentGroup":      "direction",
    "OCRParserGroup":           "ocr",
    "CSVGeneratorAgentGroup":   "csv-gen",
    "ImageGenAgentGroup":       "image-gen",
    "ImageAgentGroup":          "image",           # OCR-front image tool service
    "ExcelGenGroup":            "excel-gen",
    "ExcelGroup":               "excel-parsing",   # Excel read/parse
    "WordAgentGroup":           "word-parsing",    # Word read/parse
}

# MARBLEBench: full service names are agent-network-marble-* (+ planner/summarizer);
# stems are the names with the SVC_NS_PREFIX stripped so group_to_service's
# f"{SVC_NS_PREFIX}-{stem}" reconstruction stays identical to MDOC.
_GROUP2SVC_MARBLE = {
    "AgentNetworkPlannerGroup":    "planner",
    "AgentNetworkSummarizerGroup": "summarizer",
    "MarbleWorldAgentGroup":       "marble-world",
    "MarbleCodingAgentGroup":      "marble-coding",
    "MarbleDatabaseAgentGroup":    "marble-database",
    "MarbleWerewolfAgentGroup":    "marble-werewolf",
    "MarbleResearchAgentGroup":    "marble-research",
    "MarbleWebAgentGroup":         "marble-web",
    "MarbleMinecraftAgentGroup":   "marble-minecraft",
}

GROUP2SVC_BY_DATASET = {
    'MDOC': _GROUP2SVC_MDOC,
    'MARBLEBench': _GROUP2SVC_MARBLE,
}

AGENT_DATASET = os.environ.get('AGENT_DATASET', 'MDOC').strip()
GROUP2SVC = GROUP2SVC_BY_DATASET.get(AGENT_DATASET, _GROUP2SVC_MDOC)


def group_to_service(vertex_key: str, services=None) -> Optional[str]:
    """Map an execution-graph vertex key (e.g. 'OCRParserGroup/ocr_tool' or
    'WordGenerationAgentGroup') to its k8s service name via GROUP2SVC.
    If `services` is given, returns the actual matching column name."""
    # Identity pass-through: datasets whose call_chains.json already stores
    # service-level paths (e.g. the RE2 adapter) list real service names as
    # vertices. An agent-network vertex key such as 'OCRParserGroup/ocr_tool'
    # is never equal to a metric service name, so this is backward-compatible.
    if services is not None and vertex_key in services:
        return vertex_key
    stem = GROUP2SVC.get(vertex_key.split('/')[0])
    if stem is None:
        return None
    full = f'{SVC_NS_PREFIX}-{stem}'
    if services is None or full in services:
        return full
    for s in services:                       # fallback: match by stem suffix
        if s == full or s.endswith('-' + stem):
            return s
    return full


def match_service_to_vertexes(service: str, graph: ExecGraph) -> List[str]:
    """Service-level route vertexes mapped to ``service``.

    Concrete xxxGroup/xxAgent spans have already been rolled up and are never
    counted as separate execution-chain vertexes.
    """
    return [key for key in graph.ordered_vertexes
            if group_to_service(key) == service]


def graph_matches_services(graph: ExecGraph, services) -> bool:
    """Whether a graph contains a vertex belonging to one of ``services``."""
    service_set = set(services)
    return any(group_to_service(key, service_set) in service_set
               for key in graph.vertexes)


def graph_failed_for_services(graph: ExecGraph, services) -> bool:
    """Whether status failure or a vertex signature implicates this scope.

    Status and vertex-signature evidence are a union: 3 is task failure, 7 is
    status success (but may still carry a vertex error), and 8 is summarizer
    failure. Empty/visible outputs never participate.
    """
    service_set = set(services)
    status_failed = bool(_status_failed_services(graph) & service_set)
    vertex_failed = any(
        graph.vertexes[key].signatures and
        group_to_service(key, service_set) in service_set
        for key in graph.ordered_vertexes
    )
    return status_failed or vertex_failed


def _status_failed_services(graph: ExecGraph) -> set:
    """Services implicated by task status alone."""
    service_chain = _trace_service_chain(graph)
    if graph.task_status == 3:
        # The task stopped in the middle of the route. Attribute the status
        # failure to the last service actually reached, never to Summarizer.
        # TODO: for non-linear execution, return all services on the service_chain
        return {service_chain[-1]} if service_chain else set()
    if graph.task_status == 8:
        return {'agent-network-summarizer'} & set(service_chain)
    return set()


def _signature_failed_services(graph: ExecGraph) -> set:
    """Services whose canonical route vertex carries an error signature."""
    return {
        service
        for key in graph.ordered_vertexes
        for service in [group_to_service(key)]
        if service is not None and graph.vertexes[key].signatures
    }


def _trace_service_chain(graph: ExecGraph) -> List[str]:
    """Map the level-ordered vertex path to a collapsed service call chain."""
    chain = []
    for key in graph.ordered_vertexes:
        service = group_to_service(key)
        if service is not None and (not chain or chain[-1] != service):
            chain.append(service)
    return chain


def aggregate_problem_trace_chains(
        graphs: List[ExecGraph], problem_services: Optional[List[str]] = None
        ) -> Dict:
    """Aggregate complete paths for every failed trace in the loaded scope.

    A problem trace is the union of task_status failure (3/8) and a service
    vertex error signature, restricted to the supplied candidate scope. Once
    a trace qualifies, its complete service-level chain and all direct failed
    vertexes are retained.
    Service coverage counts a service at most once per trace; chain occurrences
    retain repeated non-consecutive visits.
    """
    problem_graphs = []
    seen_problem_trace_ids = set()
    for graph in graphs:
        qualifies = (graph_failed_for_services(graph, problem_services)
                     if problem_services is not None else graph.failed)
        if not qualifies or graph.trace_id in seen_problem_trace_ids:
            continue
        seen_problem_trace_ids.add(graph.trace_id)
        problem_graphs.append(graph)
    chain_trace_ids = {}
    service_trace_count = Counter()
    service_chain_occurrences = Counter()
    all_service_trace_count = Counter()
    all_service_chain_occurrences = Counter()
    status_total_trace_ids = defaultdict(set)
    status_error_trace_ids = defaultdict(set)
    vertex_total_trace_ids = defaultdict(set)
    vertex_error_trace_ids = defaultdict(set)
    traces = []

    # Status and vertex evidence are collected separately, then combined by
    # trace-id set union so overlap between the two channels is counted once.
    for graph in graphs:
        all_service_chain = _trace_service_chain(graph)
        chain_services = set(all_service_chain)
        status_failed_services = _status_failed_services(graph)
        signature_failed_services = _signature_failed_services(graph)
        trace_id = graph.trace_id
        all_service_trace_count.update(chain_services)
        all_service_chain_occurrences.update(all_service_chain)
        for service in chain_services:
            vertex_total_trace_ids[service].add(trace_id)
        for service in signature_failed_services:
            vertex_error_trace_ids[service].add(trace_id)
        if graph.task_status in {3, 7, 8}:
            for service in chain_services:
                status_total_trace_ids[service].add(trace_id)
            for service in status_failed_services:
                status_error_trace_ids[service].add(trace_id)

    for graph in problem_graphs:
        vertex_chain = list(graph.ordered_vertexes)
        service_chain = _trace_service_chain(graph)
        chain_trace_ids.setdefault(tuple(service_chain), []).append(graph.trace_id)
        service_trace_count.update(set(service_chain))
        service_chain_occurrences.update(service_chain)

        failed_vertexes = []
        failed_services = set()
        for key in vertex_chain:
            vertex = graph.vertexes[key]
            if not vertex.failed:
                continue
            service = group_to_service(key)
            if service is not None:
                failed_services.add(service)
            failed_vertexes.append({
                'vertex': key,
                'service': service,
                'level': vertex.level,
                'signatures': sorted(set(vertex.signatures)),
                'empty_results': vertex.empty_results,
            })
        # Every graph in this loop is already a failed/problem trace. Count a
        # service when that failed trace traverses it; separately retain direct
        # vertex failures so a resource-only root cause is not reported as
        # absent merely because its vertex contains no error text.
        traces.append({
            'trace_id': graph.trace_id,
            'task_status': graph.task_status,
            'vertex_chain': vertex_chain,
            'service_chain': service_chain,
            'failed_vertexes': failed_vertexes,
        })

    total = len(problem_graphs)
    chain_frequencies = [
        {
            'service_chain': list(chain),
            'trace_count': len(trace_ids),
            'trace_ratio': round(len(trace_ids) / total, 4) if total else 0.0,
            'trace_ids': trace_ids,
        }
        for chain, trace_ids in chain_trace_ids.items()
    ]
    chain_frequencies.sort(key=lambda item: item['trace_count'], reverse=True)

    all_total = len({graph.trace_id for graph in graphs})
    services = sorted(all_service_trace_count,
                      key=lambda svc: (all_service_trace_count[svc],
                                       all_service_chain_occurrences[svc]),
                      reverse=True)
    service_frequencies = [
        {
            'service': service,
            'traces_containing_service': len(vertex_total_trace_ids[service]),
            'trace_coverage': (
                round(len(vertex_total_trace_ids[service]) / all_total, 4)
                if all_total else 0.0),
            'chain_occurrences': all_service_chain_occurrences[service],
            'problem_trace_count': service_trace_count[service],
            'problem_trace_ratio': (
                round(service_trace_count[service] / total, 4)
                if total else 0.0),
            'failed_trace_count': len(
                status_error_trace_ids[service] |
                vertex_error_trace_ids[service]),
            'failed_trace_ratio': (
                round(len(status_error_trace_ids[service] |
                          vertex_error_trace_ids[service]) /
                      len(vertex_total_trace_ids[service]), 4)
                if vertex_total_trace_ids[service] else 0.0),
            'direct_failed_trace_count': len(vertex_error_trace_ids[service]),
            'direct_failed_trace_ratio': (
                round(len(vertex_error_trace_ids[service]) /
                      len(vertex_total_trace_ids[service]), 4)
                if vertex_total_trace_ids[service] else 0.0),
            'status_error_count': len(status_error_trace_ids[service]),
            'status_total_count': len(status_total_trace_ids[service]),
            'vertex_error_count': len(vertex_error_trace_ids[service]),
            'vertex_total_count': len(vertex_total_trace_ids[service]),
            'combined_error_count': len(
                status_error_trace_ids[service] |
                vertex_error_trace_ids[service]),
            'combined_total_count': len(
                status_total_trace_ids[service] |
                vertex_total_trace_ids[service]),
            'combined_error_ratio': (
                round(len(status_error_trace_ids[service] |
                          vertex_error_trace_ids[service]) /
                      len(status_total_trace_ids[service] |
                          vertex_total_trace_ids[service]), 4)
                if (status_total_trace_ids[service] |
                    vertex_total_trace_ids[service])
                else 0.0),
        }
        for service in services
    ]
    return {
        'scope': ('union of task_status failures and service vertex-signature '
                  'failures within loaded candidate-related graphs'),
        'problem_trace_definition': (
            'task_status in {3, 8}, or a service vertex has an error signature; '
            'the implicated scope intersects a top-k candidate'),
        'num_problem_traces': total,
        'traces': traces,
        'call_chain_frequencies': chain_frequencies,
        'service_frequencies': service_frequencies,
    }


def _parse_log_ts(token: str):
    """Parse a leading ISO8601 log timestamp (e.g. 2026-09-11T05:33:21.3Z).
    Returns a tz-naive pandas Timestamp, or None."""
    try:
        import pandas as pd
        return pd.to_datetime(token, utc=True).tz_localize(None)
    except Exception:
        return None


def scan_service_logs(ns_dir: str, service: str, start=None, end=None,
                      cfg=None, max_snippets: int = 5,
                      anomaly_type_union: Optional[Union[str, List[str]]] = None,
                      line_filter=None
                      ) -> Dict:
    """Scan a service's container log(s) for normalized error signatures
    within [start, end]. The root-cause error text (e.g. 429 quota) lives in
    logs, not the execution graph. Log files end with `_<service>.log`."""
    budget = getattr(cfg, 'exec_slice_max_chars', 1500)
    min_count = max(1, int(getattr(cfg, 'log_signature_min_count', 2)))
    logdir = os.path.join(ns_dir, 'log')
    files = glob.glob(os.path.join(logdir, '*_' + service + '.log'))
    signature_counts = Counter()
    snippets_by_signature = {}
    matched_lines = 0
    for f in files:
        try:
            fh = open(f, 'r', encoding='utf-8', errors='ignore')
        except Exception:
            continue
        with fh:
            for line_no, line in enumerate(fh, 1):
                line = line.rstrip('\n')
                if line_filter is not None:
                    line = line_filter(line, f, line_no)
                if not line:
                    continue
                content = line
                # Most log lines contain no candidate error event. Match first so
                # we only pay the comparatively high timestamp-parsing cost
                # for lines that could actually become evidence.
                # Phase 1: dynamically extract a normalized template from the
                # concrete event. Phase 2 below uses that template as the key
                # for matching/counting related events in this service's logs.
                sigs = _match_signatures(line)
                if not sigs or not _is_actionable_log_match(line, sigs):
                    continue
                if ' ' in line:
                    tok, rest = line.split(' ', 1)
                    ts = _parse_log_ts(tok)
                    if ts is not None:
                        content = rest
                        if start is not None and end is not None and not (start <= ts <= end):
                            continue
                matched_lines += 1
                signature_counts.update(set(sigs))
                for sig in set(sigs):
                    bucket = snippets_by_signature.setdefault(sig, [])
                    if len(bucket) < max_snippets:
                        bucket.append(content[:budget])
    # Frequency gate: a one-off signature is retained only in the diagnostic
    # rare count, never promoted into root-cause signatures/snippets.
    frequent_counts = {
        sig: n for sig, n in signature_counts.items() if n >= min_count
    }
    rare_counts = {
        sig: n for sig, n in signature_counts.items() if n < min_count
    }
    signatures = sorted(
        frequent_counts,
        key=lambda sig: (frequent_counts[sig], sig),
        reverse=True,
    )
    snippets = []
    seen_snippets = set()
    for sig in signatures:
        for snippet in snippets_by_signature.get(sig, []):
            if snippet not in seen_snippets:
                seen_snippets.add(snippet)
                snippets.append(snippet)
            if len(snippets) >= max_snippets:
                break
        if len(snippets) >= max_snippets:
            break
    if isinstance(anomaly_type_union, str):
        anomaly_type_union = [anomaly_type_union]
    try:
        duration_minutes = max((end - start).total_seconds() / 60.0, 1.0 / 60.0)
    except (AttributeError, TypeError):
        duration_minutes = None
    return {'service': service,
            'anomaly_type_union': list(anomaly_type_union or ['unknown']),
            'signatures': signatures,
            'signature_counts': frequent_counts,
            'signature_frequency_per_minute': (
                {sig: round(n / duration_minutes, 4)
                 for sig, n in frequent_counts.items()}
                if duration_minutes is not None else None
            ),
            'min_signature_count': min_count,
            'count': sum(frequent_counts.values()),
            'matched_lines': matched_lines,
            'discarded_rare_count': sum(rare_counts.values()),
            'snippets': snippets}


def build_log_evidence(ns_dir: str, services: List[str], start=None, end=None,
                       cfg=None,
                       anomaly_type_union: Optional[Union[
                           str, List[str], Dict[str, Union[str, List[str]]]
                       ]] = None
                       ) -> Dict[str, Dict]:
    anomaly_type_union = anomaly_type_union or ['unknown']
    ev = {}
    for svc in services:
        service_types = (anomaly_type_union.get(svc, ['unknown'])
                         if isinstance(anomaly_type_union, dict)
                         else anomaly_type_union)
        ev[svc] = scan_service_logs(
            ns_dir, svc, start, end, cfg,
            anomaly_type_union=service_types,
        )
    return ev


def aggregate_topk_slices(topk_services: List[str], graphs: List[ExecGraph],
                          cfg, log_evidence: Optional[Dict[str, Dict]] = None,
                          shared_over_services: Optional[List[str]] = None,
                          coarse_ranking: Optional[List] = None,
                          candidate_metadata: Optional[Dict[str, Dict]] = None,
                          gnn_service_ranking: Optional[List] = None,
                          trace_analysis_graphs: Optional[List[ExecGraph]] = None
                          ) -> Dict:
    """Build the aggregated context for the LLM localizer.

    Also computes cross-service dynamic-template support. This is auxiliary
    evidence for a systemic root cause; GNN/severity remain primary.
    """
    budget = getattr(cfg, 'exec_slice_max_chars', 1500)
    log_evidence = log_evidence or {}
    candidate_metadata = candidate_metadata or {}
    trace_analysis_graphs = (trace_analysis_graphs
                             if trace_analysis_graphs is not None else graphs)
    candidates = []
    sig_seen_per_service = {}            # signature -> set(service)
    sig_log_counts = {}                  # signature -> {service: occurrences}

    for rank, svc in enumerate(topk_services, start=1):
        matched = []
        svc_sigs = set()
        for g in graphs:
            if not g.failed:
                continue
            for key in match_service_to_vertexes(svc, g):
                vs = g.vertexes[key]
                if vs.failed:
                    matched.append({'trace_id': g.trace_id, **vs.to_dict(budget)})
                    svc_sigs.update(vs.signatures)
        # de-dup by (vertex, signatures) and cap
        seen = set()
        uniq = []
        for m in matched:
            skey = (m['vertex'], tuple(m['signatures']))
            if skey in seen:
                continue
            seen.add(skey)
            uniq.append(m)
        log_ev = log_evidence.get(svc, {})
        svc_sigs.update(log_ev.get('signatures', []))
        candidates.append({
            'rank': rank,
            'service': svc,
            'coarse_evidence': candidate_metadata.get(svc, {}),
            'anomaly_type_union': log_ev.get('anomaly_type_union', ['unknown']),
            'matched_failed_vertexes': uniq[:5],
            'log_signatures': log_ev.get('signatures', []),
            'log_signature_counts': log_ev.get('signature_counts', {}),
            'log_signature_frequency_per_minute': log_ev.get(
                'signature_frequency_per_minute'),
            'discarded_rare_log_signatures': log_ev.get('discarded_rare_count', 0),
            'log_errors': log_ev.get('snippets', []),
            'signatures': sorted(svc_sigs),
        })

    # Shared signatures are computed over the explicitly selected service set:
    # union of log + exec-graph signatures per candidate service.
    services_for_shared = shared_over_services or topk_services
    exec_sig_by_service = {}
    for g in graphs:
        for key, vs in g.vertexes.items():
            if vs.signatures:
                for svc in services_for_shared:
                    if key in match_service_to_vertexes(svc, g):
                        exec_sig_by_service.setdefault(svc, set()).update(vs.signatures)
    for svc in services_for_shared:
        svc_log_counts = log_evidence.get(svc, {}).get('signature_counts', {})
        sigs = set(svc_log_counts)
        sigs.update(exec_sig_by_service.get(svc, set()))
        for s in sigs:
            sig_seen_per_service.setdefault(s, set()).add(svc)
        for s, count in svc_log_counts.items():
            sig_log_counts.setdefault(s, {})[svc] = count

    # Keep every frequent dynamic template for evidence ranking, but reserve
    # `shared_signatures` for templates observed in at least two services.
    # This prevents one noisy service from being mistaken for a systemic fault.
    frequent_signatures = [
        {'signature': s,
         'num_services': len(svcs),
         'services': sorted(svcs),
         'log_occurrences': sum(sig_log_counts.get(s, {}).values()),
         'per_service_log_counts': sig_log_counts.get(s, {})}
        for s, svcs in sig_seen_per_service.items()
    ]
    frequent_signatures.sort(
        key=lambda d: (d['num_services'], d['log_occurrences'], d['signature']),
        reverse=True)
    shared_signatures = [s for s in frequent_signatures
                         if s['num_services'] >= 2]

    # a compact global view: any failed trace summaries
    candidate_failed_graphs = [
        g for g in graphs if graph_failed_for_services(g, topk_services)
    ]
    problem_trace_analysis = aggregate_problem_trace_chains(
        trace_analysis_graphs, problem_services=topk_services)
    failed_traces = [{'trace_id': g.trace_id, 'task': (g.task or '')[:budget],
                     'summary': (g.summary or '')[:budget]}
                    for g in candidate_failed_graphs][:5]

    # coarse severity ranking (success_rate + latency + resource-deviation);
    # crucial for resource faults that leave no shared error signature.
    coarse = None
    if coarse_ranking:
        coarse = [{'service': s, 'severity': round(float(v), 4), **d}
                  for s, v, d in coarse_ranking[:8]]
    gnn_coarse = None
    if gnn_service_ranking:
        gnn_coarse = [
            {'service': svc, 'gnn_score': round(float(score), 4)}
            for svc, score in gnn_service_ranking[:8]
        ]

    return {
        'global': {
            'num_traces': len(graphs),
            'num_trace_analysis_graphs': len(trace_analysis_graphs),
            'num_failed_traces': len(candidate_failed_graphs),
            'num_problem_traces': problem_trace_analysis['num_problem_traces'],
            'failed_trace_examples': failed_traces,
        },
        'coarse_ranking': coarse,
        'gnn_service_ranking': gnn_coarse,
        'problem_trace_analysis': problem_trace_analysis,
        'candidates': candidates,
        'frequent_signatures': frequent_signatures,
        'shared_signatures': shared_signatures,
    }
