
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
import random

def generate_task(grid_size=(3,3), alphabet=(0,1,2,3,4), missing=2, seed=None):
    rng = np.random.default_rng(seed)
    n, m = grid_size
    target = rng.integers(low=alphabet[0], high=alphabet[-1]+1, size=(n,m))
    initial = target.copy()
    all_pos = [(i,j) for i in range(n) for j in range(m)]
    rng.shuffle(all_pos)
    for (i,j) in all_pos[:missing]:
        initial[i,j] = -1
    return initial, target

def generate_dataset(num_tasks=50, seed=42) -> List[Dict]:
    rng = random.Random(seed)
    out = []
    for k in range(num_tasks):
        r = rng.random()
        if r < 0.60:
            gs = (3,3)
        elif r < 0.85:
            gs = (4,4)
        else:
            gs = (5,5)
        missing = rng.choice([1,2,3])
        init, tgt = generate_task(gs, missing=missing, seed=seed+k)
        out.append(dict(id=k, initial=init, target=tgt))
    return out
