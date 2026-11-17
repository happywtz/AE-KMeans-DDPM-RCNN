"""Train an RCNN on a user-supplied event array, not bundled study observations."""

import argparse
from dataclasses import asdict
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from flood_core import RCNN
from flood_core.preprocessing import QPScaler
from flood_core.training import train_rcnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing output")
    if args.updates < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Positive updates and learning rate required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    events = np.load(args.training_events, allow_pickle=False)
    scaler = QPScaler.fit(events)
    model = RCNN(256, 24).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses = train_rcnn(model, optimizer, scaler.transform(events), args.updates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "scaler": asdict(scaler),
                "seed": args.seed, "updates": args.updates, "loss": losses[-1]}, args.output)
    print(f"Saved user-trained RCNN: {args.output}")


if __name__ == "__main__":
    main()
