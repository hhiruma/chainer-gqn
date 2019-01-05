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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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

    dataset_train = gqn.data.Dataset(args.dataset_path)

    if not args.snapshot_path == None:
        assert args.snapshot_path_1 == None, 'snapshot_path already specified'
        assert args.snapshot_path_2 == None, 'snapshot_path already specified'
        hyperparams_1 = HyperParameters(snapshot_directory=args.snapshot_path)
        model_1 = Model(hyperparams_1, snapshot_directory=args.snapshot_path)
        hyperparams_2 = HyperParameters(snapshot_directory=args.snapshot_path)
        model_2 = Model(hyperparams_2, snapshot_directory=args.snapshot_path)
    elif not (args.snapshot_path_1 == None or args.snapshot_path_2 == None):
        hyperparams_1 = HyperParameters(snapshot_directory=args.snapshot_path_1)
        model_1 = Model(hyperparams_1, snapshot_directory=args.snapshot_path_1)
        hyperparams_2 = HyperParameters(snapshot_directory=args.snapshot_path_2)
        model_2 = Model(hyperparams_2, snapshot_directory=args.snapshot_path_2)
    else:
        raise TypeError('snapshot path incorrectly specified')

    if using_gpu:
        model_1.to_gpu()
        model_2.to_gpu()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 8))

    num_views_per_scene = 6
    num_generation = 2
    total_frames_per_rotation = 24

    image_shape = (3, ) + hyperparams_1.image_size
    blank_image = make_uint8(np.full(image_shape, 0))
    file_number = 1

    with chainer.no_backprop_mode():
        #for subset_1, subset_2 in zip(dataset_1, dataset_train):
        for i, subset in enumerate(dataset_train):
            iterator_train = gqn.data.Iterator(subset, batch_size=1)
            #iterator_1 = gqn.data.Iterator(subset_1, batch_size=1)
            #iterator_2 = gqn.data.Iterator(subset_2, batch_size=1)

            #for data_indices_1, data_indices_2 in zip(iterator_1, iterator_2):
            for j, data_indices in enumerate(iterator_train):
                if j == 0:
                    continue
                snapshot_array = []

                observed_image_array_1 = xp.zeros(
                    (num_views_per_scene, ) + image_shape, dtype=np.float32)
                observed_viewpoint_array_train = xp.zeros(
                    (num_views_per_scene, 7), dtype=np.float32)
                observed_image_array_2 = xp.zeros(
                    (num_views_per_scene, ) + image_shape, dtype=np.float32)
                observed_viewpoint_array_2 = xp.zeros(
                    (num_views_per_scene, 7), dtype=np.float32)


                test_data_path = os.path.join(args.dataset_path, 'test_data', str(i)+'_'+str(j)+'.npy')
                if not os.path.exists(test_data_path):
                    raise TypeError('test data not found')
                test_data = np.load(test_data_path)

                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images_1, viewpoints, original_images_1 = subset[data_indices]
                print('images_1.shape')
                print(images_1.shape)
                #images_2, viewpoints_2, original_images_2 = subset_2[data_indices_2]
                images_2, original_images_2  = test_data
                print('test_data:')
                print(len(test_data))
                print(type(test_data))
                print('images_2')
                print(len(images_2))
                print(type(images_2))
                images_2 = np.array([np.array(images_2)])      
                print(images_2.shape)
                print(type(images_2))
                print('original_images_2')
                print(len(original_images_2))
                print(type(original_images_2))
                original_images_2 = np.array([np.array(original_images_2)])      

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                images_1 = images_1.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images_1 = images_1 / 255.0
                images_1 += np.random.uniform(0, 1.0 / 256.0, size=images_1.shape).astype(np.float32)
                images_2 = images_2.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images_2 = images_2 / 255.0
                images_2 += np.random.uniform(0, 1.0 / 256.0, size=images_2.shape).astype(np.float32)

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                original_images_1 = original_images_1.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                original_images_1 = original_images_1 / 255.0
                original_images_1 += np.random.uniform( 0, 1.0 / 256.0, size=original_images_1.shape).astype(np.float32)
                original_images_2 = original_images_2.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                original_images_2 = original_images_2 / 255.0
                original_images_2 += np.random.uniform( 0, 1.0 / 256.0, size=original_images_2.shape).astype(np.float32)

                batch_index = 0

                # Generate images without observations
                r_1 = xp.zeros(
                    (
                        num_generation,
                        hyperparams_1.representation_channels,
                    ) + hyperparams_1.chrz_size,
                    dtype=np.float32)

                r_2 = xp.zeros(
                    (
                        num_generation,
                        hyperparams_2.representation_channels,
                    ) + hyperparams_2.chrz_size,
                    dtype=np.float32)

                angle_rad = 0
                current_scene_original_images_cpu_1 = original_images_1[batch_index]
                current_scene_original_images_1 = to_gpu(current_scene_original_images_cpu_1)
                current_scene_original_images_cpu_2 = original_images_2[batch_index]
                current_scene_original_images_2 = to_gpu(current_scene_original_images_cpu_2)

                gqn.animator.Snapshot.make_graph(
                    id='sq_d_graph_1',
                    pos=8,
                    graph_type='plot',
                    mode='sequential',
                    frame_in_rotation=total_frames_per_rotation,
                    num_of_data_per_graph=num_views_per_scene + 1,
                    trivial_settings={
                        'colors': ['red', 'blue', 'green', 'orange', 'white', 'cyan', 'magenta'],
                        'markers': ['', '', '', '', '', '', '']
                    }
                )

                gqn.animator.Snapshot.make_graph(
                    id='sq_d_graph_2',
                    pos=16,
                    graph_type='plot',
                    mode='sequential',
                    frame_in_rotation=total_frames_per_rotation,
                    num_of_data_per_graph=num_views_per_scene + 1,
                    trivial_settings={
                        'colors': ['red', 'blue', 'green', 'orange', 'white', 'cyan', 'magenta'],
                        'markers': ['', '', '', '', '', '', '']
                    }
                )

                gqn.animator.Snapshot.make_graph(
                    id='sq_d_avg_graph',
                    pos=17,
                    graph_type='bar',
                    mode='simultaneous',
                    frame_in_rotation=9,
                    frame_per_cycle=total_frames_per_rotation,
                    num_of_data_per_graph=2,
                    trivial_settings={
                        'colors': ['red', 'blue'],
                        'markers': ['', ''],
                        'legends': ['Train', 'Test']
                    }
                )

                sq_d_sums_1 = [0 for i in range(num_views_per_scene+1)]
                sq_d_sums_2 = [0 for i in range(num_views_per_scene+1)]
                for t in range(total_frames_per_rotation):
                    grid_master = GridSpec(nrows=4, ncols=9, height_ratios=[1,1,1,1])
                    grid_master.update(wspace=0.5, hspace=0.8)
                    snapshot = gqn.animator.Snapshot(unify_ylim=True, layout_settings={
                        'subplot_count': 17,
                        'grid_master': grid_master,
                        'subplots': [
                            { 'subplot_id': 1,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[0, 0]) },
                            { 'subplot_id': 2,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[0, 1]) },
                            { 'subplot_id': 3,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[0, 2]) },
                            { 'subplot_id': 4,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[1, 0]) },
                            { 'subplot_id': 5,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[1, 1]) },
                            { 'subplot_id': 6,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[1, 2]) },
                            { 'subplot_id': 7,   'subplot': GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=grid_master[0:2, 3:5]) },
                            { 'subplot_id': 8,   'subplot': GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=grid_master[0:2, 5:7]) },

                            { 'subplot_id': 9,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[2, 0]) },
                            { 'subplot_id': 10,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[2, 1]) },
                            { 'subplot_id': 11,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[2, 2]) },
                            { 'subplot_id': 12,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[3, 0]) },
                            { 'subplot_id': 13,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[3, 1]) },
                            { 'subplot_id': 14,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[3, 2]) },
                            { 'subplot_id': 15,  'subplot': GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=grid_master[2:4, 3:5]) },
                            { 'subplot_id': 16,  'subplot': GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=grid_master[2:4, 5:7]) },

                            { 'subplot_id': 17,  'subplot': GridSpecFromSubplotSpec(nrows=4, ncols=2, subplot_spec=grid_master[0:4, 7:9]) }
                        ]
                    })

                    for i in [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]:
                        snapshot.add_media(
                            media_type='image',
                            media_data=make_uint8(blank_image),
                            media_position=i
                        )
                        if i == 1:
                            snapshot.add_title(text='Train',target_media_pos=i)
                        if i == 9:
                            snapshot.add_title(text='Test', target_media_pos=i)

                    query_viewpoints = rotate_query_viewpoint(
                        angle_rad, num_generation, xp)
                    generated_images_1 = model_1.generate_image(
                        query_viewpoints, r_1, xp)
                    generated_images_2 = model_2.generate_image(
                        query_viewpoints, r_2, xp)

                    total_sq_d_1, _ = gqn.math.get_squared_distance(
                        to_cpu(current_scene_original_images_1[t]),
                        to_cpu(generated_images_1[0]))
                    sq_d_sums_1[0] += total_sq_d_1

                    total_sq_d_2, _ = gqn.math.get_squared_distance(
                        to_cpu(current_scene_original_images_2[t]),
                        to_cpu(generated_images_2[0]))
                    sq_d_sums_2[0] += total_sq_d_2

                    for i in [7]:
                        snapshot.add_media(
                            media_type='image',
                            media_data=make_uint8(generated_images_1[0]),
                            media_position=i
                        )
                        snapshot.add_title(
                            text='GQN Output',
                            target_media_pos=i
                        )

                    for i in [15]:
                        snapshot.add_media(
                            media_type='image',
                            media_data=make_uint8(generated_images_2[0]),
                            media_position=i
                        )
                        snapshot.add_title(
                            text='GQN Output',
                            target_media_pos=i
                        )

                    for i in [8, 16]:
                        snapshot.add_title(
                            text='Squared Distance',
                            target_media_pos=i
                        )

                    gqn.animator.Snapshot.add_graph_data(
                        graph_id='sq_d_graph_1',
                        data_id='sq_d_data_0',
                        new_data=total_sq_d_1,
                        frame_num=t,
                    )

                    gqn.animator.Snapshot.add_graph_data(
                        graph_id='sq_d_graph_2',
                        data_id='sq_d_data_0',
                        new_data=total_sq_d_2,
                        frame_num=t,
                    )

                    if t == total_frames_per_rotation - 1:
                      sq_d_sums_1[0] /= total_frames_per_rotation
                      sq_d_sums_2[0] /= total_frames_per_rotation
                      gqn.animator.Snapshot.add_graph_data(
                          graph_id='sq_d_avg_graph',
                          data_id='sq_d_data_0',
                          new_data=sq_d_sums_1[0],
                          frame_num=0
                      )
                      gqn.animator.Snapshot.add_graph_data(
                          graph_id='sq_d_avg_graph',
                          data_id='sq_d_data_1',
                          new_data=sq_d_sums_2[0],
                          frame_num=0
                      )

                    snapshot_array.append(snapshot)

                    angle_rad += 2 * math.pi / total_frames_per_rotation

                # Generate images with observations
                for m in range(num_views_per_scene):
                    kl_div_sum = 0
                    observed_image_1 = images_1[batch_index, m]
                    observed_viewpoint_1 = viewpoints[batch_index, m]
                    observed_image_2 = images_2[batch_index, m]
                    observed_viewpoint_2 = viewpoints[batch_index, m]

                    observed_image_array_1[m] = to_gpu(observed_image_1)
                    observed_viewpoint_array_train[m] = to_gpu(observed_viewpoint_1)
                    observed_image_array_2[m] = to_gpu(observed_image_2)
                    observed_viewpoint_array_2[m] = to_gpu(observed_viewpoint_2)

                    r_1 = model_1.compute_observation_representation(
                        observed_image_array_1[None, :m + 1],
                        observed_viewpoint_array_train[None, :m + 1])
                    r_2 = model_2.compute_observation_representation(
                        observed_image_array_2[None, :m + 1],
                        observed_viewpoint_array_2[None, :m + 1])

                    r_1 = cf.broadcast_to(r_1, (num_generation, ) + r_1.shape[1:])
                    r_2 = cf.broadcast_to(r_2, (num_generation, ) + r_2.shape[1:])

                    angle_rad = 0
                    for t in range(total_frames_per_rotation):
                        grid_master = GridSpec(nrows=4, ncols=9, height_ratios=[1,1,1,1])
                        grid_master.update(wspace=0.5, hspace=0.8)
                        snapshot = gqn.animator.Snapshot(unify_ylim=True, layout_settings={
                            'subplot_count': 17,
                            'grid_master': grid_master,
                            'subplots': [
                              { 'subplot_id': 1,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[0, 0]) },
                              { 'subplot_id': 2,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[0, 1]) },
                              { 'subplot_id': 3,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[0, 2]) },
                              { 'subplot_id': 4,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[1, 0]) },
                              { 'subplot_id': 5,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[1, 1]) },
                              { 'subplot_id': 6,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[1, 2]) },
                              { 'subplot_id': 7,   'subplot': GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=grid_master[0:2, 3:5]) },
                              { 'subplot_id': 8,   'subplot': GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=grid_master[0:2, 5:7]) },

                              { 'subplot_id': 9,   'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[2, 0]) },
                              { 'subplot_id': 10,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[2, 1]) },
                              { 'subplot_id': 11,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[2, 2]) },
                              { 'subplot_id': 12,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[3, 0]) },
                              { 'subplot_id': 13,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[3, 1]) },
                              { 'subplot_id': 14,  'subplot': GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=grid_master[3, 2]) },
                              { 'subplot_id': 15,  'subplot': GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=grid_master[2:4, 3:5]) },
                              { 'subplot_id': 16,  'subplot': GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=grid_master[2:4, 5:7]) },

                              { 'subplot_id': 17,  'subplot': GridSpecFromSubplotSpec(nrows=4, ncols=2, subplot_spec=grid_master[0:4, 7:9]) }
                            ]
                        })

                        for i, observed_image in zip([1, 2, 3, 4, 5, 6], observed_image_array_1):
                            snapshot.add_media(
                                media_type='image',
                                media_data=make_uint8(observed_image),
                                media_position=i
                            )
                            if i == 1:
                                snapshot.add_title(text='Train', target_media_pos=i)

                        for i, observed_image in zip([9, 10, 11, 12, 13, 14], observed_image_array_2):
                            snapshot.add_media(
                                media_type='image',
                                media_data=make_uint8(observed_image),
                                media_position=i
                            )
                            if i == 9:
                                snapshot.add_title(text='Test', target_media_pos=i)

                        query_viewpoints = rotate_query_viewpoint(
                            angle_rad, num_generation, xp)
                        generated_images_1 = model_1.generate_image(
                            query_viewpoints, r_1, xp)
                        generated_images_2 = model_2.generate_image(
                            query_viewpoints, r_2, xp)

                        total_sq_d_1, _ = gqn.math.get_squared_distance(
                            to_cpu(current_scene_original_images_1[t]),
                            to_cpu(generated_images_1[0]))
                        sq_d_sums_1[m+1] += total_sq_d_1

                        total_sq_d_2, _ = gqn.math.get_squared_distance(
                            to_cpu(current_scene_original_images_2[t]),
                            to_cpu(generated_images_2[0]))
                        sq_d_sums_2[m+1] += total_sq_d_2

                        for i in [7]:
                            snapshot.add_media(
                                media_type='image',
                                media_data=make_uint8(generated_images_1[0]),
                                media_position=i
                            )
                            snapshot.add_title(
                                text='GQN Output',
                                target_media_pos=i
                            )

                        for i in [15]:
                            snapshot.add_media(
                                media_type='image',
                                media_data=make_uint8(generated_images_2[0]),
                                media_position=i
                            )
                            snapshot.add_title(
                                text='GQN Output',
                                target_media_pos=i
                            )

                        for i in [8, 16]:
                            snapshot.add_title(
                                text='Squared Distance',
                                target_media_pos=i
                            )

                        gqn.animator.Snapshot.add_graph_data(
                            graph_id='sq_d_graph_1',
                            data_id='sq_d_data_' + str(m+1),
                            new_data=total_sq_d_1,
                            frame_num=t,
                        )

                        gqn.animator.Snapshot.add_graph_data(
                            graph_id='sq_d_graph_2',
                            data_id='sq_d_data_' + str(m+1),
                            new_data=total_sq_d_2,
                            frame_num=t,
                        )

                        if t == total_frames_per_rotation - 1:
                          sq_d_sums_1[m+1] /= total_frames_per_rotation
                          sq_d_sums_2[m+1] /= total_frames_per_rotation
                          gqn.animator.Snapshot.add_graph_data(
                              graph_id='sq_d_avg_graph',
                              data_id='sq_d_data_0',
                              new_data=sq_d_sums_1[m+1],
                              frame_num=m+1
                          )

                          gqn.animator.Snapshot.add_graph_data(
                              graph_id='sq_d_avg_graph',
                              data_id='sq_d_data_1',
                              new_data=sq_d_sums_2[m+1],
                              frame_num=m+1
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
                file_number += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str)
    parser.add_argument("--snapshot-path", "-snapshot", type=str)
    parser.add_argument("--snapshot-path-1", "-snapshot1", type=str)
    parser.add_argument("--snapshot-path-2", "-snapshot2", type=str)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-cubes", "-cubes", type=int, default=5)
    parser.add_argument("--num-colors", "-colors", type=int, default=12)
    parser.add_argument("--file-name", "-file", type=str, default="001.npy")
    parser.add_argument("--output-directory", "-out", type=str, default="output")
    args = parser.parse_args()
    main()
