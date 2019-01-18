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
from model_for_EE import EEModel

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

    dataset = gqn.data.Dataset(args.dataset_directory)

    hyperparams = HyperParameters(snapshot_directory=args.snapshot_path)
    model = Model(hyperparams, snapshot_directory=args.snapshot_path)
    if using_gpu:
        model.to_gpu()


    num_views_per_scene = 4
    num_generation = 2 # lessened from 4 to 2 (remaining 2 used for original outpu)


    image_shape = (3, ) + hyperparams.image_size
    file_number = 1

    original_picData = [] # 元データのpicの集合体
    genrerated_picData = [] #生成されたデータの集合体



    # これ以下のwit〜のやつはモデル学習に使うデータセットを作る用

    with chainer.no_backprop_mode():

        for subset in dataset:
            if file_number == 2:break
            count_scenes = 0

            iterator = gqn.data.Iterator(subset, batch_size=1)

            for data_indices in iterator:
                count_scenes+=1
                if count_scenes == 500 :break



                observed_image_array = xp.zeros(
                    (num_views_per_scene, ) + image_shape, dtype=np.float32)
                observed_viewpoint_array = xp.zeros(
                    (num_views_per_scene, 7), dtype=np.float32)

                # shape: (batch, views, height, width, channels)
                # range: [-1, 1]
                images, viewpoints, original_images = subset[data_indices]

                # (batch, views, height, width, channels) -> (batch, views, channels, height, width)
                # 学習に使った画像
                images = images.transpose((0, 1, 4, 2, 3)).astype(np.float32)
                images = images / 255.0
                images += np.random.uniform(
                    0, 1.0 / 256.0, size=images.shape).astype(np.float32)


                batch_index = 0

                # rの初期化
                r = xp.zeros(
                    (
                        num_generation,
                        hyperparams.representation_channels,
                    ) + hyperparams.chrz_size,
                    dtype=np.float32)


                current_scene_images_cpu = images[batch_index]
                current_scene_images = to_gpu(current_scene_images_cpu)

                current_scene_viewpoints_cpu = viewpoints[batch_index]
                current_scene_viewpoints = to_gpu(current_scene_viewpoints_cpu)

                # Generate images with observations
                for m in range(num_views_per_scene):
                    if m < 3:
                        continue

                    # 現在入力されている画像の番号がm, num_views_per_sceneの数だけ枚数がある
                    # for文を回すたびにobserved_imageを更新して，rに新しい画像を入れている
                    observed_image = images[batch_index, m]
                    observed_viewpoint = viewpoints[batch_index, m]

                    observed_image_array[m] = to_gpu(observed_image)
                    observed_viewpoint_array[m] = to_gpu(observed_viewpoint)

                    # 現在のシーンに対してrを更新
                    r = model.compute_observation_representation(
                        observed_image_array[None, :m + 1],
                        observed_viewpoint_array[None, :m + 1])


                    r = cf.broadcast_to(r, (num_generation, ) + r.shape[1:])

                    # 入力の画像とviewpoint１５セットに対して出力をだす
                    for t in range(len(images[batch_index])):

                        data_viewpoint = xp.broadcast_to(
                            current_scene_viewpoints[t], (num_generation,7))

                        # traindataのviewpointを使ってimageを出力
                        generated_images = model.generate_image(
                            data_viewpoint, r, xp)

                        # それぞれのデータを形を（64, 64, 3）から（3, 64, 64）に変更してしまって取っておく
                        m1 = cupy.asnumpy(generated_images[0])
                        m2 = cupy.asnumpy(current_scene_images[t])

                        original_picData.append(m1.astype(np.float32))
                        genrerated_picData.append(m2.astype(np.float32))

                print("\rfile-number: {} / {}  scene-number: {} / {}   numbers of Photos: {}".format(file_number, dataset.__len__(), count_scenes, iterator.__len__(), len(original_picData)), end='')


            print("")
            file_number += 1





    print("____start training____")
    # これ以降がモデルを学習させるところ
    original_picData = np.asarray(original_picData)
    genrerated_picData = np.asarray(genrerated_picData)
    N = len(original_picData)
    print("number of TrainData :", N)
    batch_size = args.batch_size

    Emodel = EEModel()
    if using_gpu:
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(Emodel)

    num_of_epoch = args.generation_steps

    lossboxes=[]

    for epoch in range(1,num_of_epoch+1):
        lossbox = 0

        perm = np.random.permutation(N) # Nはデータ全体の数，０からN-1までの数をランダムに並べた配列を作成
        count = 0
        for i in range(0, N, batch_size):
            count += 1

            x = original_picData[perm[i : (i+batch_size)]]
            y = genrerated_picData[perm[i : (i+batch_size)]]

            cal_loss = Emodel(x)
            sq_error = cf.squared_error(x,y)
            for j in range(len(sq_error)):
                sum_error = np.zeros(cal_loss.shape, dtype=np.float32)
                sum_error[j][0] = cf.sum(sq_error[j]).data
            sum_error = chainer.Variable(sum_error)
            loss = cf.mean_squared_error(cal_loss, sum_error)

            Emodel.cleargrads()
            loss.backward()
            optimizer.update()
            b_n =math.ceil(count*100*batch_size/N)
            if b_n >=100: b_n = 100
            print("\rEpoch {}   :   Batch {} / {}".format(epoch, count, math.ceil(N/batch_size)), end='')

            lossbox += loss.data

        print("")
        print("LossMean  :  {}".format(lossbox/math.ceil(N/batch_size)))
        lossboxes.append(lossbox/math.ceil(N/batch_size))


    Emodel.serialize("testEEmodel.model", args.output_directory)
    print("epoch_loss: ")
    print(lossboxes)
    loss_path = os.path.join(args.output_directory,"Loss_Memo.txt")
    f = open(loss_path,'w')
    for x in lossboxes:
        f.write(str(x) + "\n")
    f.close()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory", "-dataset", type=str, default="dataset_train") #メインモデル学習に使ったdatasetを渡す
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True) #学習済みのメインモデルを渡す
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--output-directory", "-out", type=str, default="output") #ここに学習したモデルが入る
    parser.add_argument("--generation-steps", "-gsteps", type=int, default=300)
    parser.add_argument("--batch-size", "-b", type=int, default=12)
    args = parser.parse_args()
    main()
