from pathlib import Path
from statistics import NormalDist

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision.transforms.functional import to_tensor
from PIL import Image
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("samples_dir", "./logs/samples", "Directory to save samples")
flags.DEFINE_string("save_dir", "./logs/metrics", "Directory to save metrics")


def main(_):
    samples_dir = Path(FLAGS.samples_dir)

    targets = []
    reconstructs = []

    for dataset_item_dir in samples_dir.iterdir():
        if not dataset_item_dir.is_dir():
            continue

        for sample_dir in dataset_item_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            target = Image.open(sample_dir / "target.png")
            reconstructed = Image.open(sample_dir / "reconstructed.png")

            targets.append(to_tensor(target))
            reconstructs.append(to_tensor(reconstructed))

    targets = torch.stack(targets)
    reconstructs = torch.stack(reconstructs)

    metrics = {
        "ssim": StructuralSimilarityIndexMeasure(reduction="none", data_range=(0, 1)),
        "psnr": PeakSignalNoiseRatio(
            reduction="none", data_range=(0, 1), dim=(1, 2, 3)
        ),
    }

    for name, metric in metrics.items():
        value = metric(reconstructs, targets).cpu().numpy()
        mean, lower, upper = confidence_interval(value)

        print(f"{name}: {mean:.4f} ({lower:.4f} - {upper:.4f})")


def confidence_interval(data, confidence=0.95):
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.0)
    h = dist.stdev * z / ((len(data) - 1) ** 0.5)
    return dist.mean, dist.mean - h, dist.mean + h


if __name__ == "__main__":
    app.run(main)
