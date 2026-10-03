"""Print the recorded Flickr8k test scores and selected checkpoint."""

import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "results" / "metrics_comparison.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    selection = json.loads((root / "best_checkpoint_selection.json").read_text())
    for row in rows:
        print(
            f"{row['model']:12s}  images={row['n_test_images']}  "
            f"BLEU-4={float(row['BLEU-4']):.4f}  "
            f"CIDEr={float(row['CIDEr']):.4f}  "
            f"seconds/image={float(row['seconds_per_image']):.4f}"
        )
    print(
        f"Selected epoch: {selection['epoch']}; "
        f"validation loss: {selection['validation_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
