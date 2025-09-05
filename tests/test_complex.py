
import numpy as np
from topomcts.complex import SimplicialComplex

def test_fiedler_vector_runs():
    grid = np.array([[0,0,0],[0,-1,1],[1,1,1]])
    sc = SimplicialComplex.from_grid(grid)
    f = sc.fiedler_vector()
    assert len(f) == grid.size
