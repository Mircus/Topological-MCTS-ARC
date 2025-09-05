
import argparse, csv, time
from pathlib import Path
from topomcts.datasets import generate_dataset
from topomcts.game import ARCGame
from topomcts.mcts import TopologicalMCTSEngine

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--out", type=str, default="experiments/results/baselines.csv")
    args = ap.parse_args()

    ds = generate_dataset(num_tasks=10, seed=42)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id","method","success","quality","time_s"])
        for trial in range(args.trials):
            for task in ds:
                game = ARCGame(task["initial"], task["target"])
                eng = TopologicalMCTSEngine(topology_weight=0.5, ucb_c=1.414, rollout_depth=10)
                t0 = time.time()
                action, val = eng.run(game, iterations=args.iterations)
                dt = time.time() - t0
                w.writerow([task["id"], "tmcts", float(game.reward()), float(game.quality(game.initial, game.target)), f"{dt:.4f}"])

if __name__ == "__main__":
    main()
