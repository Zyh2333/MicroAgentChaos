"""Per-sample hypothesis nodes, provenance edges and rooted subgraph overlap."""
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
import json

from .schema import Relation
from .propositions import Claim, CONTRADICTION, SUPPORT


def proposition_claims(proposition_evidence):
    """Project contracted edges into the roles used by proposition assessment."""
    claims = []
    for rule, roles in proposition_evidence.items():
        support = list(dict.fromkeys([*roles.get("basis", []), *roles.get("confirms", [])]))
        if support:
            claims.append(Claim(rule, SUPPORT, support))
        if roles.get("refutes"):
            claims.append(Claim(rule, CONTRADICTION, list(roles["refutes"])))
    return claims


def _ids(values, field):
    if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values):
        raise ValueError(f"{field} 需要非空 ID 数组")
    return list(dict.fromkeys(values))


class HypothesisMemory:
    def __init__(self, nodes=None):
        self.nodes = {}
        self.proposition_evidence = {}
        self.last_id = 0
        if nodes:
            self.commit(deepcopy(nodes))

    def snapshot(self):
        return {"hypotheses": deepcopy(self.nodes),
                "proposition_evidence": deepcopy(self.proposition_evidence)}

    def commit(self, nodes):
        """Contract confirmed nodes after the enclosing action succeeds."""
        changed = {key: node for key, node in nodes.items() if node != self.nodes.get(key)}
        for key, node in changed.items():
            self.last_id = max(self.last_id, int(key[1:]))
            if node["status"] != "confirmed":
                continue
            for rule in node["supports"]:
                roles = self.proposition_evidence.setdefault(rule, {})
                for edge in node["evidence"]:
                    ids = roles.setdefault(edge["role"], [])
                    if edge["id"] not in ids:
                        ids.append(edge["id"])
        self.nodes = {key: node for key, node in nodes.items() if node["status"] != "confirmed"}
        return changed

    def revised(self, updates, graph, allowed_rules, round_no):
        """Apply each valid update to a copy and report rejected entries separately."""
        if not isinstance(updates, list):
            return None, [{"message": "hypotheses 必须为数组"}]
        nodes = deepcopy(self.nodes)
        errors = []
        for index, update in enumerate(updates):
            try:
                self._apply_update(nodes, update, graph, allowed_rules, round_no)
            except ValueError as exc:
                error = {"index": index, "message": str(exc)}
                if isinstance(update, dict) and isinstance(update.get("id"), str):
                    error["id"] = update["id"]
                errors.append(error)
        return nodes, errors

    def _apply_update(self, nodes, update, graph, allowed_rules, round_no):
        if not isinstance(update, dict):
            raise ValueError("hypotheses 的每项必须是对象")
        ids = _ids(update.get("evidence_ids"), "evidence_ids")
        missing = [eid for eid in ids if eid not in graph.observations]
        if missing:
            raise ValueError(f"假设只能引用已读的原始记录 ID，未找到 {', '.join(missing)}")
        role = update.get("role", "basis")
        if role not in ("basis", "confirms", "refutes"):
            raise ValueError("role 必须是 basis、confirms 或 refutes")
        key = update.get("id")
        statement = update.get("statement", "")
        if not isinstance(statement, str):
            raise ValueError("statement 必须是字符串")
        statement = statement.strip()
        if key is None:
            if not statement or role != "basis":
                raise ValueError("新假设需要 statement 和 basis 依据，更新请提供已有 id")
            key = next((k for k, n in nodes.items() if n["statement"] == statement), None)
            if key is None:
                next_id = max((int(k[1:]) for k in nodes), default=0)
                key = f"H{max(self.last_id, next_id) + 1:04d}"
        elif not isinstance(key, str) or key not in nodes:
            raise ValueError("假设 id 未登记，新建时省略 id")
        node = nodes.get(key)
        if node and statement and statement != node["statement"]:
            raise ValueError("更新假设时须保持原陈述，不同陈述请新建")
        rules = []
        if node is None or "propositions" in update:
            rules = _ids(update.get("propositions"), "propositions")
            if set(rules) - allowed_rules:
                raise ValueError("propositions 必须引用有效的规则 ID 或 层:类型")
        if node is None:
            node = {"id": key, "statement": statement, "created_round": round_no,
                    "status": "pending", "supports": {}, "evidence": []}
        previous = {(e["id"], e["role"]) for e in node["evidence"]}
        additions = [i for i in ids if (i, role) not in previous]
        if role != "basis" and additions:
            opposite = "refutes" if role == "confirms" else "confirms"
            opposing_classes = {graph.class_of(e["id"]) for e in node["evidence"]
                                if e["role"] == opposite}
            conflicts = [eid for eid in additions if graph.class_of(eid) in opposing_classes]
            if conflicts:
                raise ValueError(f"同一观测不能既证实又反驳原陈述，冲突记录 {', '.join(conflicts)}")
            if node["status"] == "confirmed" and role == "refutes":
                raise ValueError("本次提交已确认该假设，其证据将转入命题")
            node["status"] = "confirmed" if role == "confirms" else "refuted"
        for rule in rules:
            node["supports"].setdefault(rule, round_no)
        node["evidence"].extend({"id": i, "role": role, "round": round_no} for i in additions)
        nodes[key] = node

    @staticmethod
    def _subgraph(node, graph, by_rule, dependent):
        # Tagged node and edge tuples form a single set of equally weighted elements.
        elements, records = set(), set()

        def cite(source, role, observation_id):
            target = graph.class_of(observation_id)
            records.add(target)
            elements.add(("observation", target))
            elements.add((source, role, target))

        for edge in node["evidence"]:
            cite("@root", edge["role"], edge["id"])
        for rule in node["supports"]:
            elements.add(("proposition", rule))
            elements.add(("@root", "supports", rule))
            for role, eid in by_rule[rule]:
                cite(rule, role, eid)
        for source in records:
            for target in dependent.get(source, ()):
                if target in records:
                    elements.add((source, "DEPENDENT", target))
        return elements, records

    def view(self, graph, claims, budget):
        """Present pairs in exact Jaccard order, including their inherited evidence."""
        by_rule = defaultdict(set)
        for rule, roles in self.proposition_evidence.items():
            for role, ids in roles.items():
                by_rule[rule].update((role, eid) for eid in ids)
        for claim in claims:
            for eid in claim.evidence_ids:
                by_rule[claim.rule_id].add((claim.kind, eid))
            for eid in claim.control_ids:
                by_rule[claim.rule_id].add(("control", eid))
        dependent = defaultdict(set)
        for edge in graph.edges.values():
            if edge.relation == Relation.DEPENDENT:
                a, b = graph.class_of(edge.a), graph.class_of(edge.b)
                if a != b:
                    dependent[a].add(b)
        pending = {key: self._subgraph(n, graph, by_rule, dependent)
                   for key, n in self.nodes.items() if n["status"] == "pending"}
        pairs = []
        for a, b in combinations(sorted(pending), 2):
            left, right = pending[a][0], pending[b][0]
            shared = len(left & right)
            if shared:
                union = len(left) + len(right) - shared
                pairs.append((shared / union, a, b, shared, union))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        rows, visible = [], {}
        view = {"nodes": visible, "pairs": rows, "total_nodes": len(self.nodes),
                "positive_pairs": len(pairs)}

        def compact(key):
            n = self.nodes[key]
            return {"statement": n["statement"][:240], "status": n["status"],
                    "propositions": list(n["supports"]), "evidence_ids": [
                        e["id"] for e in n["evidence"][-3:]]}

        def fits():
            return len(json.dumps(view, ensure_ascii=False)) <= budget

        for score, a, b, shared, union in pairs:
            added = [k for k in (a, b) if k not in visible]
            visible.update((k, compact(k)) for k in added)
            left, right = pending[a][1], pending[b][1]
            rows.append({"ids": [a, b], "overlap": score, "intersection": shared, "union": union,
                         "shared_evidence": sorted(left & right)[:4],
                         "unique_evidence": {a: sorted(left - right)[:4], b: sorted(right - left)[:4]}})
            if not fits():
                rows.pop()
                for k in added:
                    del visible[k]
                break
        for key in reversed(self.nodes):
            if key not in visible:
                visible[key] = compact(key)
                if not fits():
                    del visible[key]
        return view if fits() else {}
