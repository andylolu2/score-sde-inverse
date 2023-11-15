from pathlib import Path

import torch
import numpy as np
from PIL import Image
from absl import app, flags, logging

from configs.ve.cifar10_ncsnpp_deep_continuous import get_config as get_cifar10_config
from configs.ve.celebahq_256_ncsnpp_continuous import get_config as get_celeba_config
from score_inverse.models.utils import create_model
from score_inverse.models.ema import ExponentialMovingAverage
from score_inverse.tasks.deblur import DeblurTask
from score_inverse.datasets import CelebA, CIFAR10
from score_inverse.sde import get_sde
from score_inverse.datasets.scalers import get_data_inverse_scaler, get_data_scaler
from score_inverse.sampling import get_corrector, get_predictor
from score_inverse.sampling.inverse import get_pc_inverse_solver

FLAGS = flags.FLAGS
flags.DEFINE_enum("dataset", "cifar10", ["cifar10", "celeba"], "Dataset to use.")
flags.DEFINE_integer("num_scales", 100, "Number of discretisation steps")
flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_integer("num_batches", 1, "Number of samples to generate")
flags.DEFINE_integer("samples_per_image", 1, "No. of reconstructed samples per image")
flags.DEFINE_string("save_dir", "./logs/samples", "Directory to save samples")
flags.DEFINE_float("lambda_", 0.1, "Lambda parameter for inverse task")
flags.DEFINE_enum("task", "deblur_gaussian", ["deblur_gaussian"], "Inverse task to use")


def main(_):
    if FLAGS.dataset == "cifar10":
        config = get_cifar10_config()
        ckpt_path = "checkpoints/ve/cifar10_ncsnpp_deep_continuous/checkpoint_12.pth"
        dataset = CIFAR10()
    elif FLAGS.dataset == "celeba":
        config = get_celeba_config()
        ckpt_path = "checkpoints/ve/celebahq_256_ncsnpp_continuous/checkpoint_48.pth"
        dataset = CelebA(img_size=config.data.img_size)
    else:
        raise ValueError(f"Unknown dataset {FLAGS.dataset}")

    config.model.num_scales = FLAGS.num_scales
    config.eval.batch_size = FLAGS.batch_size

    score_model = load_checkpoint(config, ckpt_path)
    inverse_task = get_inverse_task(config, dataset, FLAGS.task)
    data_loader = get_dataloader(dataset, FLAGS.samples_per_image, FLAGS.batch_size)
    sampling_fn = get_sampling_fn(config, dataset, inverse_task, lambda_=FLAGS.lambda_)

    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, (ds_indices, s_indices, x) in enumerate(data_loader):
        if i == FLAGS.num_batches:
            break

        logging.info("Sampling batch %d...", i)

        x = x.to(device=config.device)
        y = inverse_task.A(x)
        x_hat, _ = sampling_fn(score_model, y)

        for i in range(FLAGS.batch_size):
            source = tensor_to_image(y[i])
            target = tensor_to_image(x[i])
            reconstructed = tensor_to_image(x_hat[i])

            save_dir_sample = save_dir / str(ds_indices[i]) / str(s_indices[i])
            save_dir_sample.mkdir(parents=True, exist_ok=True)
            source.save(save_dir_sample / f"source.png", "PNG")
            target.save(save_dir_sample / f"target.png", "PNG")
            reconstructed.save(save_dir_sample / f"reconstructed.png", "PNG")


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    x = np.clip(x.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    if x.shape[-1] == 1:
        x = x[:, :, 0]
    return Image.fromarray(x)


def get_dataloader(dataset, samples_per_image: int, batch_size: int):
    dataset_idx, sample_idx, batch = [], [], []
    for i in range(len(dataset)):
        for j in range(samples_per_image):
            dataset_idx.append(i)
            sample_idx.append(j)
            batch.append(dataset[i])

            if len(batch) == batch_size:
                yield dataset_idx, sample_idx, torch.stack(batch)
                dataset_idx, sample_idx, batch = [], [], []


def get_inverse_task(config, dataset, task_name: str):
    if task_name == "deblur_gaussian":
        return DeblurTask(dataset.img_size, kernel_type="gaussian", kernel_size=5).to(
            device=config.device
        )
    else:
        raise ValueError(f"Unknown inverse task {task_name}")


def load_checkpoint(config, ckpt_path):
    logging.info("Loading checkpoint from %s", ckpt_path)
    loaded_state = torch.load(ckpt_path, map_location=config.device)
    score_model = create_model(config)
    # Still need to load the base model state since non-trainable params aren't covered by EMA
    score_model.load_state_dict(loaded_state["model"], strict=False)

    # Replace trainable model params with EMA params
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    ema.load_state_dict(loaded_state["ema"])
    ema.copy_to(score_model.parameters())

    return score_model


def get_sampling_fn(config, dataset, inverse_task, lambda_: float = 1):
    inverse_scaler = get_data_inverse_scaler(config)
    sde, sampling_eps = get_sde(config)

    sampling_shape = (config.eval.batch_size, *dataset.img_size)
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_inverse_solver(
        sde=sde,
        shape=sampling_shape,
        predictor=predictor,
        corrector=corrector,
        inverse_scaler=inverse_scaler,
        snr=config.sampling.snr,
        n_steps=config.sampling.n_steps_each,
        probability_flow=config.sampling.probability_flow,
        continuous=config.training.continuous,
        denoise=config.sampling.noise_removal,
        eps=sampling_eps,
        device=config.device,
        inverse_task=inverse_task,
        lambda_=lambda_,
    )
    return sampling_fn


if __name__ == "__main__":
    app.run(main)
