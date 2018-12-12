import argparse
import math
import time
import sys
import os
import random
import cv2
import copy

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda

sys.path.append("./../../")
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

def func_anim_upate(i, fig, snapshot_array):
    snapshot = snapshot_array[0][i]
    snapshot.print_to_fig(fig, frame_num=i)


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

    num_views_per_scene = 4
    num_generation = 2 # lessened from 4 to 2 (remaining 2 used for original outpu)
    num_original = 2
    total_frames_per_rotation = 24

    image_shape = (3, ) + hyperparams.image_size
    blank_image = make_uint8(np.full(image_shape, 0))
    file_number = 1

    with chainer.no_backprop_mode():
        for subset in dataset:
            iterator = gqn.data.Iterator(subset, batch_size=1)

            for data_indices in iterator:
                snapshot_array = []

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

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                original_images = original_images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                original_images = original_images / 255.0
                original_images += np.random.uniform(
                    0, 1.0 / 256.0, size=original_images.shape).astype(np.float32)

                batch_index = 0

                # Generate images without observations
                r = xp.zeros(
                    (
                        num_generation,
                        hyperparams.representation_channels,
                    ) + hyperparams.chrz_size,
                    dtype=np.float32)

                angle_rad = 0
                current_scene_original_images_cpu = original_images[batch_index]
                current_scene_original_images = to_gpu(current_scene_original_images_cpu)

                kl_div_sum = 0
                kl_div_list = np.zeros((1 + num_views_per_scene, total_frames_per_rotation))

                gqn.animator.Snapshot.make_graph(
                    id='kl_div_graph',
                    pos=7,
                    graph_type='plot',
                    frame_in_rotation=total_frames_per_rotation,
                    num_of_data_per_graph=num_views_per_scene + 1,
                    trivial_settings={
                        'colors': ['red', 'blue', 'green', 'orange', 'white'],
                        'markers': ['o', 'o', 'o', 'o', 'o']
                    }
                )

                for t in range(total_frames_per_rotation):
                    snapshot = gqn.animator.Snapshot((2, 4))
                    # grid_master = GridSpec(nrows=2, ncols=4, height_ratios=[1, 1])
                    # snapshot = gqn.animator.Snapshot(layout_settings={
                    #     'subplot_count': 8,
                    #     'grid_master': grid_master,
                    #     'subplots': [
                    #         {
                    #             'subplot_id': i + 1,
                    #             'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[i//4, i%4])
                    #         }
                    #         for i in range(8)
                    #     ]
                    # })

                    for i in [1, 2, 5, 6]:
                        snapshot.add_media(
                            media_type='image',
                            media_data=make_uint8(blank_image),
                            media_position=i
                        )
                        if i == 1:
                            snapshot.add_title(
                                text='Observed',
                                target_media_pos=i
                            )

                    query_viewpoints = rotate_query_viewpoint(
                        angle_rad, num_generation, xp)
                    generated_images = model.generate_image(
                        query_viewpoints, r, xp)

                    kl_div = gqn.math.get_KL_div(
                        to_cpu(current_scene_original_images[t]),
                        to_cpu(generated_images[0]))

                    for i in [3]:
                        snapshot.add_media(
                            media_type='image',
                            media_data=make_uint8(generated_images[0]),
                            media_position=i
                        )
                        snapshot.add_title(
                            text='Generated',
                            target_media_pos=i
                        )

                    for i in [4]:
                        snapshot.add_media(
                            media_type='image',
                            media_data=make_uint8(current_scene_original_images[t]),
                            media_position=i
                        )
                        snapshot.add_title(
                            text='Original',
                            target_media_pos=i
                        )

                    gqn.animator.Snapshot.add_graph_data(
                        graph_id='kl_div_graph',
                        data_id='kl_div_data_0',
                        new_data=kl_div,
                        frame_num=t,
                    )

                    snapshot.add_title(
                        text='KL Divergence',
                        target_media_pos=7
                    )

                    snapshot_array.append(snapshot)

                    angle_rad += 2 * math.pi / total_frames_per_rotation


                # Generate images with observations
                for m in range(num_views_per_scene):
                    kl_div_sum = 0
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
                        snapshot = gqn.animator.Snapshot((2, 4))
                        # grid_master = GridSpec(nrows=2, ncols=4, height_ratios=[1, 1])
                        # snapshot = gqn.animator.Snapshot(layout_settings={
                        #     'subplot_count': 8,
                        #     'grid_master': grid_master,
                        #     'subplots': [
                        #         {
                        #             'subplot_id': i + 1,
                        #             'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[i//4, i%4])
                        #         }
                        #         for i in range(8)
                        #     ]
                        # })

                        for i, observed_image in zip([1, 2, 5, 6], observed_image_array):
                            snapshot.add_media(
                                media_type='image',
                                media_data=make_uint8(observed_image),
                                media_position=i
                            )
                            if i == 1:
                                snapshot.add_title(
                                    text='Observed',
                                    target_media_pos=i
                                )

                        query_viewpoints = rotate_query_viewpoint(
                            angle_rad, num_generation, xp)
                        generated_images = model.generate_image(
                            query_viewpoints, r, xp)

                        kl_div = gqn.math.get_KL_div(
                            to_cpu(current_scene_original_images[t]),
                            to_cpu(generated_images[0]))

                        for i in [3]:
                            snapshot.add_media(
                                media_type='image',
                                media_data=make_uint8(generated_images[0]),
                                media_position=i
                            )
                            snapshot.add_title(
                                text='Generated',
                                target_media_pos=i
                            )

                        for i in [4]:
                            snapshot.add_media(
                                media_type='image',
                                media_data=make_uint8(current_scene_original_images[t]),
                                media_position=i
                            )
                            snapshot.add_title(
                                text='Original',
                                target_media_pos=i
                            )

                        gqn.animator.Snapshot.add_graph_data(
                            graph_id='kl_div_graph',
                            data_id='kl_div_data_' + str(m+1),
                            new_data=kl_div,
                            frame_num=t,
                        )

                        snapshot.add_title(
                            text='KL Divergence',
                            target_media_pos=7
                        )

                        angle_rad += 2 * math.pi / total_frames_per_rotation
                        # plt.pause(1e-8)

                        snapshot_array.append(snapshot)

                plt.subplots_adjust(
                    left=None,
                    bottom=None,
                    right=None,
                    top=None,
                    wspace=0,
                    hspace=0)

                anim = animation.FuncAnimation(
                    fig,
                    func_anim_upate,
                    fargs = (fig, [snapshot_array]),
                    interval=1/24,
                    frames= (num_views_per_scene + 1) * total_frames_per_rotation
                )

                anim.save(
                    "{}/shepard_matzler_{}.mp4".format(
                        args.output_directory, file_number),
                    writer="ffmpeg",
                    fps=12)




                if not os.path.exists("{}/shepard_matzler_{}".format(args.output_directory, file_number)):
                    os.mkdir("{}/shepard_matzler_{}".format(
                        args.output_directory, file_number))

                picData=[]
                for i in range((num_views_per_scene) * total_frames_per_rotation):
                    snapshot = snapshot_array[i+total_frames_per_rotation]
                    _media = snapshot.get_subplot(3)
                    media= _media['body']
                    picData.append(media['media_data'])
                    figu = plt.figure()
                    plt.axis('off')
                    plt.imshow(media['media_data'])
                    plt.savefig("{}/shepard_matzler_{}/{}.png".format(
                        args.output_directory, file_number, i))
                    plt.close(figu)

                bigfig = plt.figure(figsize=(20, 10))
                for i in range(num_views_per_scene):
                    for j in range(total_frames_per_rotation):
                        plt.subplot(num_views_per_scene,total_frames_per_rotation,(i*total_frames_per_rotation+j+1))
                        plt.axis('off')
                        plt.imshow(picData[i*total_frames_per_rotation+j])
                plt.savefig("{}/shepard_matzler_{}_ALL.png".format(
                    args.output_directory, file_number))
                plt.close(bigfig)



                file_number += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-cubes", "-cubes", type=int, default=5)
    parser.add_argument("--num-colors", "-colors", type=int, default=12)
    parser.add_argument("--file-name", "-file", type=str, default="001.npy")
    parser.add_argument("--output-directory", "-out", type=str, default="output")
    args = parser.parse_args()
    main()
