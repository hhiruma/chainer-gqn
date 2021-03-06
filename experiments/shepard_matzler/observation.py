import argparse
import math
import time
import sys
import os
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda

sys.path.append("../../")
import gqn
from hyperparams import HyperParameters
from model import Model


def make_uint8(image):
    if (image.shape[0] == 3):
        image = image.transpose(1, 2, 0)
    image = to_cpu(image)
    return np.uint8(np.clip(image * 255, 0, 255))


def to_gpu(array):
    if isinstance(array, np.ndarray):
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if isinstance(array, cupy.ndarray):
        return cuda.to_cpu(array)
    return array


def generate_random_query_viewpoint(num_generation, xp):
    view_radius = 3
    eye = np.random.normal(size=3)
    eye = tuple(view_radius * (eye / np.linalg.norm(eye)))
    center = (0, 0, 0)
    yaw = gqn.math.yaw(eye, center)
    pitch = gqn.math.pitch(eye, center)
    query_viewpoints = xp.array(
        (eye[0], eye[1], eye[2], math.cos(yaw), math.sin(yaw), math.cos(pitch),
         math.sin(pitch)),
        dtype=np.float32)
    query_viewpoints = xp.broadcast_to(
        query_viewpoints, (num_generation, ) + query_viewpoints.shape)
    return query_viewpoints


def rotate_query_viewpoint(angle_rad, num_generation, xp):
    view_radius = 3
    eye = (view_radius * math.sin(angle_rad),
           view_radius * math.sin(angle_rad),
           view_radius * math.cos(angle_rad))
    center = (0, 0, 0)
    yaw = gqn.math.yaw(eye, center)
    pitch = gqn.math.pitch(eye, center)
    query_viewpoints = xp.array(
        (eye[0], eye[1], eye[2], math.cos(yaw), math.sin(yaw), math.cos(pitch),
         math.sin(pitch)),
        dtype=np.float32)
    query_viewpoints = xp.broadcast_to(
        query_viewpoints, (num_generation, ) + query_viewpoints.shape)
    return query_viewpoints


def add_annotation(axis, array):
    text = axis.text(-155, -60, "observations", fontsize=18)
    array.append(text)
    text = axis.text(-30, -60, "neural rendering", fontsize=18)
    array.append(text)


def main():
    try:
        os.mkdir(args.output_directory)
    except:
        pass

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    dataset = gqn.data.Dataset(args.dataset_path)

    hyperparams = HyperParameters(snapshot_directory=args.snapshot_path)
    model = Model(hyperparams, snapshot_directory=args.snapshot_path)
    if using_gpu:
        model.to_gpu()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 5))

    axis_observation_array = []
    axis_observation_array.append(fig.add_subplot(2, 4, 1))
    axis_observation_array.append(fig.add_subplot(2, 4, 2))
    axis_observation_array.append(fig.add_subplot(2, 4, 5))
    axis_observation_array.append(fig.add_subplot(2, 4, 6))

    for axis in axis_observation_array:
        axis.axis("off")

    axis_generation_array = []
    axis_generation_array.append(fig.add_subplot(2, 4, 3))
    axis_generation_array.append(fig.add_subplot(2, 4, 4))
    axis_generation_array.append(fig.add_subplot(2, 4, 7))
    axis_generation_array.append(fig.add_subplot(2, 4, 8))

    for axis in axis_generation_array:
        axis.axis("off")

    num_views_per_scene = 4
    num_generation = 4
    total_frames_per_rotation = 24

    image_shape = (3, ) + hyperparams.image_size
    blank_image = make_uint8(np.full(image_shape, 0))
    file_number = 1

    with chainer.no_backprop_mode():
        for subset in dataset:
            iterator = gqn.data.Iterator(subset, batch_size=1)

            for data_indices in iterator:
                artist_frame_array = []

                observed_image_array = xp.zeros(
                    (num_views_per_scene, ) + image_shape, dtype=np.float32)
                observed_viewpoint_array = xp.zeros(
                    (num_views_per_scene, 7), dtype=np.float32)

                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints, original_images = subset[data_indices]

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images = images / 255.0
                images += np.random.uniform(
                    0, 1.0 / 256.0, size=images.shape).astype(np.float32)

                batch_index = 0

                # Generate images without observations
                r = xp.zeros(
                    (
                        num_generation,
                        hyperparams.representation_channels,
                    ) + hyperparams.chrz_size,
                    dtype=np.float32)

                angle_rad = 0
                for t in range(total_frames_per_rotation):
                    artist_array = []

                    for axis in axis_observation_array:
                        axis_image = axis.imshow(
                            make_uint8(blank_image),
                            interpolation="none",
                            animated=True)
                        artist_array.append(axis_image)

                    query_viewpoints = rotate_query_viewpoint(
                        angle_rad, num_generation, xp)
                    generated_images = model.generate_image(
                        query_viewpoints, r, xp)

                    for j, axis in enumerate(axis_generation_array):
                        image = make_uint8(generated_images[j])
                        axis_image = axis.imshow(
                            image, interpolation="none", animated=True)
                        artist_array.append(axis_image)

                    angle_rad += 2 * math.pi / total_frames_per_rotation

                    # plt.pause(1e-8)
                    axis = axis_generation_array[-1]
                    add_annotation(axis, artist_array)
                    artist_frame_array.append(artist_array)

                # Generate images with observations
                for m in range(num_views_per_scene):
                    observed_image = images[batch_index, m]
                    observed_viewpoint = viewpoints[batch_index, m]

                    observed_image_array[m] = to_gpu(observed_image)
                    observed_viewpoint_array[m] = to_gpu(observed_viewpoint)

                    r = model.compute_observation_representation(
                        observed_image_array[None, :m + 1],
                        observed_viewpoint_array[None, :m + 1])

                    r = cf.broadcast_to(r, (num_generation, ) + r.shape[1:])

                    angle_rad = 0
                    for t in range(total_frames_per_rotation):
                        artist_array = []

                        for axis, observed_image in zip(
                                axis_observation_array, observed_image_array):
                            axis_image = axis.imshow(
                                make_uint8(observed_image),
                                interpolation="none",
                                animated=True)
                            artist_array.append(axis_image)

                        query_viewpoints = rotate_query_viewpoint(
                            angle_rad, num_generation, xp)
                        generated_images = model.generate_image(
                            query_viewpoints, r, xp)

                        for j in range(num_generation):
                            axis = axis_generation_array[j]
                            axis_image = axis.imshow(
                                make_uint8(generated_images[j]),
                                interpolation="none",
                                animated=True)
                            artist_array.append(axis_image)

                        angle_rad += 2 * math.pi / total_frames_per_rotation
                        # plt.pause(1e-8)

                        axis = axis_generation_array[-1]
                        add_annotation(axis, artist_array)
                        artist_frame_array.append(artist_array)

                plt.tight_layout()
                plt.subplots_adjust(
                    left=None,
                    bottom=None,
                    right=None,
                    top=None,
                    wspace=0,
                    hspace=0)
                anim = animation.ArtistAnimation(
                    fig,
                    artist_frame_array,
                    interval=1 / 24,
                    blit=True,
                    repeat_delay=0)

                anim.save(
                    "{}/shepard_matzler_{}.gif".format(
                        args.output_directory, file_number),
                    writer="imagemagick")
                anim.save(
                    "{}/shepard_matzler_{}.mp4".format(
                        args.output_directory, file_number),
                    writer="ffmpeg",
                    fps=12)
                file_number += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--output-directory", "-out", type=str, default="gif")
    args = parser.parse_args()
    main()
