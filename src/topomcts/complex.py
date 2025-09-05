
import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from dataclasses import dataclass

@dataclass
class SimplicialComplex:
    """
    Basic graph-based 'complex' built from grid adjacency of missing/filled cells.
    Provides Laplacian and Fiedler vector proxy.
    """
    G: nx.Graph

    @staticmethod
    def from_grid(grid: np.ndarray) -> "SimplicialComplex":
        # Build a simple adjacency graph over grid cells; edges for 4-neighbors
        n, m = grid.shape
        G = nx.Graph()
        for i in range(n):
            for j in range(m):
                G.add_node((i, j), value=int(grid[i, j]))
        # 4-neighborhood (only right/down to avoid duplicates)
        for i in range(n):
            for j in range(m):
                for di, dj in [(1,0), (0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < n and 0 <= nj < m:
                        G.add_edge((i, j), (ni, nj))
        return SimplicialComplex(G)

    def laplacian(self):
        A = nx.to_numpy_array(self.G, nodelist=list(self.G.nodes()))
        L = csgraph.laplacian(A, normed=False)
        return L

    def fiedler_vector(self):
        # Compute the second-smallest eigenvector of L (Fiedler). For small grids only.
        L = self.laplacian()
        w, v = np.linalg.eigh(L)
        if len(w) > 1:
            return v[:, 1]
        return v[:, 0]

    def centrality_score(self) -> float:
        # A toy 'centrality' proxy = variance of the Fiedler vector
        f = self.fiedler_vector()
        return float(np.var(f))
