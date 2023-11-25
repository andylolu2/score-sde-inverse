from pathlib import Path

import numpy as np
from absl import app, flags
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms.functional import to_tensor
import lpips

FLAGS = flags.FLAGS
flags.DEFINE_string("samples_dir", "./logs/samples", "Directory to save samples")


def main(_):
    samples_dir = Path(FLAGS.samples_dir)

    metrics = {
        "ssim": StructuralSimilarityIndexMeasure(data_range=(0, 1)),
        "psnr": PeakSignalNoiseRatio(data_range=(0, 1)),
        "lpips": lpips.LPIPS(net="alex"),
    }
    values = {name: [] for name in metrics.keys()}

    for dataset_item_dir in samples_dir.iterdir():
        if not dataset_item_dir.is_dir():
            continue

        for sample_dir in dataset_item_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            target = Image.open(sample_dir / "target.png")
            reconstructed = Image.open(sample_dir / "reconstructed.png")
            target = to_tensor(target).unsqueeze(0)
            reconstructed = to_tensor(reconstructed).unsqueeze(0)

            for name, metric in metrics.items():
                value = metric(reconstructed, target)
                values[name].append(value.cpu().numpy())

    for name, value in values.items():
        mean = np.mean(value)
        low, median, high = bootstrap(value)

        print(f"{name}: {mean:.4f} ({low:.4f}, {median:.4f}, {high:.4f})")


def bootstrap(data, n=10000, func=np.mean):
    """Bootstrap estimate of CI for statistic.

    Returns the 5%, 50%, 95% percentiles of the bootstrap distribution.
    """
    data = np.array(data)

    samples = np.random.choice(data, size=(n, len(data)))
    stats = [func(s) for s in samples]

    return np.percentile(stats, [5, 50, 95])


if __name__ == "__main__":
    app.run(main)
