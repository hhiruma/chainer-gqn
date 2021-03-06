import argparse
import math
import multiprocessing
import os
import random
import sys
import time

import chainer
import chainer.functions as cf
import chainermn
import cupy
import numpy as np
from chainer.backends import cuda

import gqn
from hyperparams import HyperParameters
from model import Model
from optimizer import AdamOptimizer, MomentumSGDOptimizer, RMSpropOptimizer
from scheduler import Scheduler


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")


def to_gpu(array):
    if isinstance(array, np.ndarray):
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if isinstance(array, cupy.ndarray):
        return cuda.to_cpu(array)
    return array


def main():
    ##############################################
    # To avoid OpenMPI bug
    multiprocessing.set_start_method("forkserver")
    p = multiprocessing.Process(target=print, args=("", ))
    p.start()
    p.join()
    ##############################################

    try:
        os.mkdir(args.snapshot_directory)
    except:
        pass

    comm = chainermn.create_communicator()
    device = comm.intra_rank
    print("device", device, "/", comm.size)
    cuda.get_device(device).use()
    xp = cupy

    dataset = gqn.data.Dataset(args.dataset_directory)

    hyperparams = HyperParameters()
    hyperparams.generator_share_core = args.generator_share_core
    hyperparams.generator_share_prior = args.generator_share_prior
    hyperparams.generator_generation_steps = args.generation_steps
    hyperparams.generator_u_channels = args.u_channels
    hyperparams.generator_share_upsampler = args.generator_share_upsampler
    hyperparams.generator_subpixel_convolution_enabled = args.generator_subpixel_convolution_enabled
    hyperparams.inference_share_core = args.inference_share_core
    hyperparams.inference_share_posterior = args.inference_share_posterior
    hyperparams.inference_downsampler_channels = args.inference_downsampler_channels
    hyperparams.h_channels = args.h_channels
    hyperparams.z_channels = args.z_channels
    hyperparams.representation_channels = args.representation_channels
    hyperparams.pixel_n = args.pixel_n
    hyperparams.pixel_sigma_i = args.initial_pixel_variance
    hyperparams.pixel_sigma_f = args.final_pixel_variance
    if comm.rank == 0:
        hyperparams.save(args.snapshot_directory)
        print(hyperparams)

    model = Model(hyperparams, snapshot_directory=args.snapshot_directory)
    model.to_gpu()

    optimizer = AdamOptimizer(
        model.parameters,
        communicator=comm,
        mu_i=args.initial_lr,
        mu_f=args.final_lr)
    if comm.rank == 0:
        print(optimizer)

    scheduler = Scheduler(
        sigma_start=args.initial_pixel_variance,
        sigma_end=args.final_pixel_variance,
        final_num_updates=args.pixel_n)
    if comm.rank == 0:
        print(scheduler)

    pixel_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        scheduler.pixel_variance**2,
        dtype="float32")
    pixel_ln_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(scheduler.pixel_variance**2),
        dtype="float32")
    num_pixels = hyperparams.image_size[0] * hyperparams.image_size[1] * 3

    random.seed(0)
    subset_indices = list(range(len(dataset.subset_filenames)))

    representation_shape = (
        args.batch_size,
        hyperparams.representation_channels) + hyperparams.chrz_size

    current_training_step = 0
    for iteration in range(args.training_iterations):
        mean_kld = 0
        mean_nll = 0
        mean_mse = 0
        mean_elbo = 0
        total_num_batch = 0
        subset_size_per_gpu = len(subset_indices) // comm.size
        if len(subset_indices) % comm.size != 0:
            subset_size_per_gpu += 1
        start_time = time.time()

        for subset_loop in range(subset_size_per_gpu):
            random.shuffle(subset_indices)
            subset_index = subset_indices[comm.rank]
            subset = dataset.read(subset_index)
            iterator = gqn.data.Iterator(subset, batch_size=args.batch_size)

            for batch_index, data_indices in enumerate(iterator):
                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints = subset[data_indices]

                # (batch, views, height, width, channels) ->  (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images = images / 255.0
                images += np.random.uniform(
                    0, 1.0 / 256.0, size=images.shape).astype(np.float32)

                total_views = images.shape[1]

                # Sample observations
                num_views = random.choice(range(total_views + 1))
                if current_training_step == 0 and num_views == 0:
                    num_views = 1  # avoid OpenMPI error

                observation_view_indices = list(range(total_views))
                random.shuffle(observation_view_indices)
                observation_view_indices = observation_view_indices[:num_views]

                if num_views > 0:
                    representation = model.compute_observation_representation(
                        images[:, observation_view_indices],
                        viewpoints[:, observation_view_indices])
                else:
                    representation = xp.zeros(
                        representation_shape, dtype="float32")
                    representation = chainer.Variable(representation)

                # Sample query
                query_index = random.choice(range(total_views))
                query_images = images[:, query_index]
                query_viewpoints = viewpoints[:, query_index]

                # Transfer to gpu
                query_images = to_gpu(query_images)
                query_viewpoints = to_gpu(query_viewpoints)

                z_t_param_array, mean_x = model.sample_z_and_x_params_from_posterior(
                    query_images, query_viewpoints, representation)

                # Compute loss
                ## KL Divergence
                loss_kld = chainer.Variable(xp.zeros((), dtype=xp.float32))
                for params in z_t_param_array:
                    mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p = params
                    kld = gqn.functions.gaussian_kl_divergence(
                        mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)
                    loss_kld += cf.sum(kld)

                ##Negative log-likelihood of generated image
                loss_nll = cf.sum(
                    gqn.functions.gaussian_negative_log_likelihood(
                        query_images, mean_x, pixel_var, pixel_ln_var))

                # Calculate the average loss value
                loss_nll = loss_nll / args.batch_size
                loss_kld = loss_kld / args.batch_size

                loss = (loss_nll / scheduler.pixel_variance) + loss_kld

                model.cleargrads()
                loss.backward()
                optimizer.update(current_training_step)

                loss_nll = float(loss_nll.data) + math.log(256.0)
                loss_kld = float(loss_kld.data)

                elbo = -(loss_nll + loss_kld)

                loss_mse = float(
                    cf.mean_squared_error(query_images, mean_x).data)

                if comm.rank == 0:
                    printr(
                        "Iteration {}: Subset {} / {}: Batch {} / {} - elbo: {:.2f} - loss: nll: {:.2f} mse: {:.5f} kld: {:.5f} - lr: {:.4e} - pixel_variance: {:.5f} - step: {}  ".
                        format(iteration + 1, subset_loop + 1,
                               subset_size_per_gpu, batch_index + 1,
                               len(iterator), elbo, loss_nll, loss_mse,
                               loss_kld, optimizer.learning_rate,
                               scheduler.pixel_variance,
                               current_training_step))

                total_num_batch += 1
                current_training_step += comm.size
                mean_kld += loss_kld
                mean_nll += loss_nll
                mean_mse += loss_mse
                mean_elbo += elbo

                scheduler.step(current_training_step)
                pixel_var[...] = scheduler.pixel_variance**2
                pixel_ln_var[...] = math.log(scheduler.pixel_variance**2)

            if comm.rank == 0:
                model.serialize(args.snapshot_directory)

        if comm.rank == 0:
            elapsed_time = time.time() - start_time
            mean_elbo /= total_num_batch
            mean_nll /= total_num_batch
            mean_mse /= total_num_batch
            mean_kld /= total_num_batch
            print(
                "\033[2KIteration {} - elbo: {:.2f} - loss: nll: {:.2f} mse: {:.5f} kld: {:.5f} - lr: {:.4e} - pixel_variance: {:.5f} - step: {} - time: {:.3f} min".
                format(iteration + 1, mean_elbo, mean_nll, mean_mse, mean_kld,
                       optimizer.learning_rate, scheduler.pixel_variance,
                       current_training_step, elapsed_time / 60))
            model.serialize(args.snapshot_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory", "-dataset", type=str, default="dataset_train")
    parser.add_argument(
        "--snapshot-directory", "-snapshot", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument(
        "--training-iterations", "-iter", type=int, default=2 * 10**6)
    parser.add_argument("--generation-steps", "-gsteps", type=int, default=12)
    parser.add_argument("--initial-lr", "-mu-i", type=float, default=0.0005)
    parser.add_argument("--final-lr", "-mu-f", type=float, default=0.0005)
    parser.add_argument(
        "--initial-pixel-variance", "-ps-i", type=float, default=2.0)
    parser.add_argument(
        "--final-pixel-variance", "-ps-f", type=float, default=0.7)
    parser.add_argument("--pixel-n", "-pn", type=int, default=2 * 10**5)
    parser.add_argument("--pretrain-pixel-n", "-ppn", type=int, default=10000)
    parser.add_argument("--h-channels", "-ch", type=int, default=64)
    parser.add_argument("--z-channels", "-cz", type=int, default=3)
    parser.add_argument("--u-channels", "-cu", type=int, default=16)
    parser.add_argument(
        "--representation-channels", "-cr", type=int, default=256)
    parser.add_argument(
        "--inference-downsampler-channels", "-cix", type=int, default=128)
    parser.add_argument(
        "--generator-share-core", "-g-share-core", action="store_true")
    parser.add_argument(
        "--generator-share-prior", "-g-share-prior", action="store_true")
    parser.add_argument(
        "--generator-share-upsampler",
        "-g-share-upsampler",
        action="store_true")
    parser.add_argument(
        "--generator-subpixel-convolution-enabled",
        "-g-subpixel-convolution",
        action="store_true")
    parser.add_argument(
        "--inference-share-core", "-i-share-core", action="store_true")
    parser.add_argument(
        "--inference-share-posterior",
        "-i-share-posterior",
        action="store_true")
    parser.add_argument("--num-bits-x", "-bits", type=int, default=8)
    args = parser.parse_args()
    main()
