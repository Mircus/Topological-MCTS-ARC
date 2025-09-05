
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List
from .game import ARCGame
from .complex import SimplicialComplex

@dataclass
class MCTSNode:
    state: ARCGame
    parent: Optional["MCTSNode"] = None
    action: Optional[Any] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    def ucb1(self, c=1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.value_sum / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits + 1) / self.visits) if self.parent else 0.0
        return exploit + explore

class TopologicalMCTSEngine:
    def __init__(self, topology_weight: float = 0.5, ucb_c: float = 1.414, rollout_depth: int = 10):
        self.topology_weight = float(topology_weight)
        self.ucb_c = float(ucb_c)
        self.rollout_depth = int(rollout_depth)

    def topological_bonus(self, state: ARCGame) -> float:
        comp = SimplicialComplex.from_grid(state.initial)
        return comp.centrality_score()

    def select(self, node: MCTSNode) -> MCTSNode:
        cur = node
        while cur.children:
            def score(child: MCTSNode) -> float:
                base = child.ucb1(c=self.ucb_c)
                topo = self.topology_weight * self.topological_bonus(child.state)
                return base + topo
            cur = max(cur.children, key=score)
        return cur

    def expand(self, node: MCTSNode) -> MCTSNode:
        if node.state.is_terminal():
            return node
        actions = node.state.legal_actions()
        random.shuffle(actions)
        for a in actions:
            child = MCTSNode(state=node.state.step(a), parent=node, action=a)
            node.children.append(child)
        return random.choice(node.children) if node.children else node

    def rollout(self, state: ARCGame) -> float:
        cur = state.clone()
        depth = 0
        while (not cur.is_terminal()) and depth < self.rollout_depth:
            actions = cur.legal_actions()
            if not actions:
                break
            cur = cur.step(random.choice(actions))
            depth += 1
        return cur.reward()

    def backprop(self, node: MCTSNode, value: float):
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value
            cur = cur.parent

    def run(self, root_state: ARCGame, iterations: int = 200) -> Tuple[Any, float]:
        root = MCTSNode(state=root_state)
        for _ in range(iterations):
            leaf = self.select(root)
            child = self.expand(leaf)
            value = self.rollout(child.state)
            self.backprop(child, value)
        if not root.children:
            return None, 0.0
        best = max(root.children, key=lambda c: c.visits)
        return best.action, (best.value_sum / max(1, best.visits))
