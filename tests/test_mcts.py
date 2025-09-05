
import numpy as np
from topomcts.game import ARCGame
from topomcts.mcts import TopologicalMCTSEngine

def test_mcts_runs():
    init = np.array([[0,-1,1],[1,1,1],[1,1,1]])
    tgt = np.array([[0,0,1],[1,1,1],[1,1,1]])
    g = ARCGame(init, tgt, alphabet=(0,1))
    eng = TopologicalMCTSEngine()
    act, val = eng.run(g, iterations=10)
    assert True  # ran without exceptions
