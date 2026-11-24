"""Open-world evidence association for heterogeneous agent-system observations.

The package is deliberately independent of the benchmark vocabulary: it takes
evidence records (text + references + tool + stage) and returns a dual-relation
evidence graph over them, plus proposition support strength and next-step needs.
"""
from .associate import Associator
from .compat import Corpus, build_corpus, compatibility
from .graph import EvidenceGraph
from .relation import Policy, Verdict, classify
from .schema import (Edge, Features, Observation, PropositionState, Ref, Relation,
                     Dimension)

__all__ = ["Associator", "Corpus", "build_corpus", "compatibility", "EvidenceGraph",
           "Policy", "Verdict", "classify", "Edge", "Features", "Observation",
           "PropositionState", "Ref", "Relation", "Dimension"]
