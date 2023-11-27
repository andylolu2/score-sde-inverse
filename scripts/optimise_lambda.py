from absl import app, flags
from bayes_opt import BayesianOptimization
import pandas as pd

import torch

import sys
import os
sys.path.append(os.getcwd())

from scripts.shared_utils import SharedUtils, compute_psnr

FLAGS = flags.FLAGS
flags.DEFINE_enum("dataset", "cifar10", ["cifar10", "celeba"], "Dataset to use.")
flags.DEFINE_integer("num_scales", 100, "Number of discretisation steps")
flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_integer("num_batches", 1, "Number of samples to generate")
flags.DEFINE_integer("samples_per_image", 1, "Number of reconstructed samples per image")
flags.DEFINE_integer("iterations", 10, "Number of Bayesian optimization iterations to perform")
flags.DEFINE_integer("init_points", 2, "Number of random Bayesian optimization initialization points")
flags.DEFINE_enum("task", "denoise", ["deblur_gaussian", "denoise"], "Inverse task to use")

iteration_counter = 0
results = []

def main(_):
    global iteration_counter

    utils = SharedUtils(FLAGS, train=True)

    pbounds = {'lambda_value': (0.0, 1.0)}

    optimizer = BayesianOptimization(
        f=lambda lambda_value: objective_function(utils, lambda_value, FLAGS.num_batches),
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=FLAGS.init_points,
        n_iter=FLAGS.init_points,
    )

    best_lambda = optimizer.max['params']['lambda_value']
    print("Best lambda value:", best_lambda)
    print(f"Total iterations: {iteration_counter}")

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    df.to_csv('optimization_results_2.csv', index=False)


def objective_function(utils, lambda_value, num_batches):
    global iteration_counter
    iteration_counter += 1
    print(f"Running iteration {iteration_counter} with lambda value {lambda_value}")
    sampling_fn = utils.get_sampling_fn(lambda_value)
    dataset = torch.stack([utils.dataset[ind] for ind in range(utils.config.eval.batch_size*num_batches)])

    samples = []

    for i in range(num_batches):
        print('Sampling batch', i)
        y = utils.inverse_task.forward(dataset[i*utils.config.eval.batch_size:(i+1)*utils.config.eval.batch_size].to(utils.config.device))
        sample, _ = sampling_fn(utils.score_model, y)
        samples.append(sample.detach().cpu())

    samples = torch.cat(samples)

    psnr = compute_psnr(dataset, samples)
    torch.cuda.empty_cache()

    results.append({'iteration': iteration_counter, 'lambda': lambda_value, 'psnr': psnr})
    return psnr


if __name__ == "__main__":
    app.run(main)
