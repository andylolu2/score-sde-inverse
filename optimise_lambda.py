from pathlib import Path
from absl import app, flags, logging
from bayes_opt import BayesianOptimization
import pandas as pd



from shared_utils import SharedUtils, compute_metrics

FLAGS = flags.FLAGS
flags.DEFINE_enum("dataset", "cifar10", ["cifar10", "celeba"], "Dataset to use.")
flags.DEFINE_integer("num_scales", 100, "Number of discretisation steps")
flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_integer("num_batches", 1, "Number of samples to generate")
flags.DEFINE_integer("samples_per_image", 1, "No. of reconstructed samples per image")
flags.DEFINE_enum("task", "denoise", ["deblur_gaussian", "denoise"], "Inverse task to use")

iteration_counter = 0
results = []

def main(_):
    global iteration_counter

    print("let's go!!!!!")
    utils = SharedUtils(FLAGS)

    pbounds = {'lambda_value': (0.0, 1.0)}
    n_iter = 10

    optimizer = BayesianOptimization(
        f=lambda lambda_value: objective_function(utils, lambda_value),
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=n_iter,
    )

    best_lambda = optimizer.max['params']['lambda_value']
    print("Best lambda value:", best_lambda)
    print(f"Total iterations: {iteration_counter}")

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    df.to_csv('optimization_results_2.csv', index=False)


def objective_function(utils, lambda_value):
    global iteration_counter
    validation_size = 10
    iteration_counter += 1
    print(f"Running iteration {iteration_counter} with lambda value {lambda_value}")
    sampling_fn = utils.get_sampling_fn(lambda_value)
    psnr_total, count = 0, 0
    validation_set = utils.get_validation_set(validation_size)

    for i in range(len(validation_set)):
        x = validation_set[i]
        print(f"Sampling image {i}...")

        x = x[None, :].to(utils.config.device)
        y = utils.inverse_task.A(x)
        x_hat, _ = sampling_fn(utils.score_model, y)

        print(f"Computing metrics for image {i}...")
        psnr = compute_metrics(x, x_hat)

        psnr_total += psnr
        count += 1

    mean_psnr = psnr_total / count
    results.append({'iteration': iteration_counter, 'lambda': lambda_value, 'psnr': mean_psnr})
    return mean_psnr


if __name__ == "__main__":
    app.run(main)
