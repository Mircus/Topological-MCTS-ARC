
import networkx as nx
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GameInvariants:
    beta0: int  # number of connected components
    beta1: int  # cyclomatic number approximation
    symmetry: float  # simple symmetry score proxy
    spectral_radius: float

def compute_invariants(G: nx.Graph) -> GameInvariants:
    # β0 via connected components
    beta0 = nx.number_connected_components(G)
    # β1 ~ cycles = m - n + c (cyclomatic number)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    beta1 = m - n + beta0
    # symmetry proxy: ratio of distinct degrees
    degs = [d for _, d in G.degree()]
    symmetry = 1.0 / (1 + len(set(degs)))
    # spectral radius of adjacency
    import numpy as np
    A = nx.to_numpy_array(G)
    w = np.linalg.eigvals(A) if A.size else [0.0]
    spectral_radius = float(max(abs(w)))
    return GameInvariants(beta0=beta0, beta1=int(beta1), symmetry=float(symmetry), spectral_radius=spectral_radius)

def signature_dict(inv: GameInvariants) -> Dict[str, Any]:
    return dict(beta0=inv.beta0, beta1=inv.beta1, symmetry=inv.symmetry, spectral_radius=inv.spectral_radius)
