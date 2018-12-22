###########################################3
#   内容：GQNのモデルの定義
###########################################3

import os
import sys
import uuid

import chainer
import chainer.functions as cf
import chainer.links as nn
import cupy
from chainer.backends import cuda
from chainer.initializers import HeNormal
from chainer.serializers import load_hdf5, save_hdf5

import gqn
from hyperparams import HyperParameters


class Model():
    def __init__(self, hyperparams: HyperParameters, snapshot_directory=None):
        assert isinstance(hyperparams, HyperParameters)
        # ジェネレーターのLSTMの数
        self.generation_steps = hyperparams.generator_generation_steps
        # ハイパーパラメータを格納
        self.hyperparams = hyperparams
        # モデルを保存しやすいようにネットワークのパーツをまとめて入れておく入れ物
        self.parameters = chainer.ChainList()


        # Generator Networkの生成（デコーダー）
        #  返り値：
        #     generation_cores：
        #           デコーダーのLSTMの部分が入ってる
        #     generation_priors：
        #           潜在変数zを作る元になる平均と分散を出力する（つまり分布）
        #           Priorの名が示すとおり潜在変数の <<事前分布>>
        #     generation_upsamplers：
        #           LSTMの出力をアップサンプルするLink.
        #           LSTMのSkip connectionを担うuに代入する際アップサンプルシたものを使う
        #     generation_map_u_x：
        #           upsamplerから帰ってきたものを画像に変換するときもある
        #           （基本的に16チャンネル担っててそれを3チャンネルに治す
        self.generation_cores, self.generation_priors, self.generation_upsamplers, self.generation_map_u_x = self.build_generation_network(
            generation_steps=self.generation_steps,
            h_channels=hyperparams.h_channels,
            z_channels=hyperparams.z_channels,
            u_channels=hyperparams.generator_u_channels)


        # Inference Networkの生成（エンコーダー）
        #  返り値:
        #      inference_cores:
        #            エンコーダーのLSTMの部分が入ってる
        #      inference_posteriors:
        #           潜在変数zを作る元になる平均と分散を出力する（つまり分布）
        #           Posteriorの名前が示すとおり潜在変数の <<事後分布>>
        #      inference_downsampler_x:
        #           入力の画像xをダウンサンプルするLink
        #           LSTMに入れられるようにダウンサンプルする必要がある
        self.inference_cores, self.inference_posteriors, self.inference_downsampler_x = self.build_inference_network(
            generation_steps=self.generation_steps,
            h_channels=hyperparams.h_channels,
            z_channels=hyperparams.z_channels,
            downsampler_channels=hyperparams.inference_downsampler_channels)


        # Representation Networkの生成（Representationの生成ネットワーク)
        # 返り値：
        #      representation_network:
        #           ネットワーク本体（CNN)
        #           指定したアーキテクチャのものが帰ってくる
        self.representation_network = self.build_representation_network(
            architecture=hyperparams.representation_architecture,
            r_channels=hyperparams.representation_channels)


        # スナップショットディレクトリが指定されていればソロを読み込む
        if snapshot_directory:
            try:
                filepath = os.path.join(snapshot_directory, self.filename)
                if os.path.exists(filepath) and os.path.isfile(filepath):
                    print("loading {}".format(filepath))
                    load_hdf5(filepath, self.parameters)
            except Exception as error:
                print(error)


    # generation のネットワークを生成する
    def build_generation_network(self, generation_steps, h_channels,
                                 z_channels, u_channels):
        # 生成したものを入れておくリスト
        #    以下生成時にself.parametersにも入れる。
        #    parametersの中にあるやつが最終的にファイルに保存される。
        #    このリストに入れておくとがあとからメソッドで特定のLinkを呼び出しやすい。
        #    self.parametersにappendするLinkオブジェクトとcore_arrayなどにappendする
        #    オブジェクトはpythonの「共有渡し」という性質により共有される。
        #    故に片方の重みが更新されればもう片方も自動的に更新される。

        core_array = []
        prior_array = []
        upsampler_h_u_array = []

        # Chain List登録するのためにはLinkをinit_scope内で登録する必要がある
        #    こうやって登録しないとbackwardsでLinkの重みが更新されない
        with self.parameters.init_scope():
            # LSTM core
            num_cores = 1 if self.hyperparams.generator_share_core else generation_steps
            for _ in range(num_cores):
                core = gqn.nn.generator.Core(h_channels=h_channels)
                core_array.append(core)
                self.parameters.append(core)

            # z prior sampler
            num_priors = 1 if self.hyperparams.generator_share_prior else generation_steps
            for _ in range(num_priors):
                prior = gqn.nn.generator.Prior(z_channels=z_channels)
                prior_array.append(prior)
                self.parameters.append(prior)

            # upsampler (h -> u)
            num_upsamplers = 1 if self.hyperparams.generator_share_upsampler else generation_steps
            scale = 4
            for _ in range(num_upsamplers):
                if self.hyperparams.generator_subpixel_convolution_enabled:
                    upsampler = gqn.nn.upsampler.SubPixelConvolutionUpsampler(
                        channels=u_channels * scale**2, scale=scale)
                else:
                    upsampler = gqn.nn.upsampler.DeconvolutionUpsampler(
                        channels=u_channels)
                upsampler_h_u_array.append(upsampler)
                self.parameters.append(upsampler)

            # 1x1 conv (u -> x)
            if u_channels == 3:
                map_u_x = None
            else:
                map_u_x = nn.Convolution2D(
                    u_channels,
                    3,
                    ksize=1,
                    stride=1,
                    pad=0,
                    initialW=HeNormal(0.1))
                self.parameters.append(map_u_x)

        return core_array, prior_array, upsampler_h_u_array, map_u_x

    # inference のネットワークを生成する
    def build_inference_network(self, generation_steps, h_channels, z_channels,
                                downsampler_channels):
        # 生成したものを入れておくリスト
        #    以下生成時にself.parametersにも入れる。
        #    parametersの中にあるやつが最終的にファイルに保存される
        #    このリストに入れておくとがあとからメソッドで特定のLinkを呼び出しやすい
        #    self.parametersにappendするLinkオブジェクトとcore_arrayなどにappendする
        #    オブジェクトはpythonの「共有渡し」という性質により共有される。
        #    故に片方の重みが更新されればもう片方も自動的に更新される。

        core_array = []
        posterior_array = []

        # Chain List登録するのためにはLinkをinit_scope内で登録する必要がある
        #    こうやって登録しないとbackwardsでLinkの重みが更新されない
        with self.parameters.init_scope():
            num_cores = 1 if self.hyperparams.inference_share_core else generation_steps
            for t in range(num_cores):
                # LSTM core
                core = gqn.nn.inference.Core(h_channels=h_channels)
                core_array.append(core)
                self.parameters.append(core)

            # z posterior sampler
            num_posteriors = 1 if self.hyperparams.inference_share_posterior else generation_steps
            for t in range(num_posteriors):
                posterior = gqn.nn.inference.Posterior(z_channels=z_channels)
                posterior_array.append(posterior)
                self.parameters.append(posterior)

            # x downsampler
            downsampler_x_h = gqn.nn.downsampler.Downsampler(
                channels=downsampler_channels)
            self.parameters.append(downsampler_x_h)

        return core_array, posterior_array, downsampler_x_h

    # representation のネットワークを生成する
    def build_representation_network(self, architecture, r_channels):
        if architecture == "tower":
            layer = gqn.nn.representation.TowerNetwork(r_channels=r_channels)
            with self.parameters.init_scope():
                self.parameters.append(layer)
            return layer

        raise NotImplementedError

    def to_gpu(self):
        self.parameters.to_gpu()

    def cleargrads(self):
        self.parameters.cleargrads()

    @property
    def filename(self):
        return "model.hdf5"

    def serialize(self, path):
        self.serialize_parameter(path, self.filename, self.parameters)

    def serialize_parameter(self, path, filename, params):
        tmp_filename = str(uuid.uuid4())
        save_hdf5(os.path.join(path, tmp_filename), params)
        os.rename(
            os.path.join(path, tmp_filename), os.path.join(path, filename))

    def generate_initial_state(self, batch_size, xp):
        initial_h_gen = xp.zeros(
            (
                batch_size,
                self.hyperparams.h_channels,
            ) + self.hyperparams.chrz_size,
            dtype="float32")
        initial_c_gen = xp.zeros(
            (
                batch_size,
                self.hyperparams.h_channels,
            ) + self.hyperparams.chrz_size,
            dtype="float32")
        initial_u = xp.zeros(
            (
                batch_size,
                self.hyperparams.generator_u_channels,
            ) + self.hyperparams.image_size,
            dtype="float32")
        initial_h_enc = xp.zeros(
            (
                batch_size,
                self.hyperparams.h_channels,
            ) + self.hyperparams.chrz_size,
            dtype="float32")
        initial_c_enc = xp.zeros(
            (
                batch_size,
                self.hyperparams.h_channels,
            ) + self.hyperparams.chrz_size,
            dtype="float32")
        return initial_h_gen, initial_c_gen, initial_u, initial_h_enc, initial_c_enc

    def get_generation_core(self, l):
        if self.hyperparams.generator_share_core:
            return self.generation_cores[0]
        return self.generation_cores[l]

    def get_generation_prior(self, l):
        if self.hyperparams.generator_share_prior:
            return self.generation_priors[0]
        return self.generation_priors[l]

    def get_generation_upsampler(self, t):
        if self.hyperparams.generator_share_upsampler:
            return self.generation_upsamplers[0]
        return self.generation_upsamplers[t]

    def get_inference_core(self, l):
        if self.hyperparams.inference_share_core:
            return self.inference_cores[0]
        return self.inference_cores[l]

    def get_inference_posterior(self, l):
        if self.hyperparams.inference_share_posterior:
            return self.inference_posteriors[0]
        return self.inference_posteriors[l]

    # def compute_information_gain(self, x, r):
    #     xp = cuda
    #     h0_gen, c0_gen, u_0, h0_enc, c0_enc = self.generate_initial_state(
    #         1, xp)
    #     loss_kld = 0

    #     hl_enc = h0_enc
    #     cl_enc = c0_enc
    #     hl_gen = h0_gen
    #     cl_gen = c0_gen
    #     ul_enc = u_0

    #     xq = self.inference_downsampler(x)

    #     for l in range(self.generation_steps):
    #         inference_core = self.get_inference_core(l)
    #         inference_posterior = self.get_inference_posterior(l)
    #         generation_core = self.get_generation_core(l)
    #         generation_piror = self.get_generation_prior(l)

    #         h_next_enc, c_next_enc = inference_core.forward_onestep(
    #             hl_gen, hl_enc, cl_enc, xq, v, r)

    #         mean_z_q = inference_posterior.compute_mean_z(hl_enc)
    #         ln_var_z_q = inference_posterior.compute_ln_var_z(hl_enc)
    #         ze_l = cf.gaussian(mean_z_q, ln_var_z_q)

    #         mean_z_p = generation_piror.compute_mean_z(hl_gen)
    #         ln_var_z_p = generation_piror.compute_ln_var_z(hl_gen)

    #         h_next_gen, c_next_gen, u_next_enc = generation_core.forward_onestep(
    #             hl_gen, cl_gen, ul_enc, ze_l, v, r)

    #         kld = gqn.nn.functions.gaussian_kl_divergence(
    #             mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)

    #         loss_kld += cf.sum(kld)

    #         hl_gen = h_next_gen
    #         cl_gen = c_next_gen
    #         ul_enc = u_next_enc
    #         hl_enc = h_next_enc
    #         cl_enc = c_next_enc

    def compute_observation_representation(self, images, viewpoints):
        batch_size = images.shape[0]
        num_views = images.shape[1]

        # (batch, views, channels, height, width) -> (batch * views, channels, height, width)
        images = images.reshape((batch_size * num_views, ) + images.shape[2:])
        viewpoints = viewpoints.reshape((batch_size * num_views, ) +
                                        viewpoints.shape[2:])

        # Transfer to gpu
        xp = self.parameters.xp
        if xp is cupy:
            images = cuda.to_gpu(images)
            viewpoints = cuda.to_gpu(viewpoints)

        # Add noise
        images += xp.random.uniform(
            0, 1.0 / 256.0, size=images.shape).astype(xp.float32)

        r = self.representation_network(images, viewpoints)

        # (batch * views, channels, height, width) -> (batch, views, channels, height, width)
        r = r.reshape((batch_size, num_views) + r.shape[1:])

        # Sum element-wise across views
        r = cf.sum(r, axis=1)

        return r

    def map_u_x(self, x):
        if self.generation_map_u_x is None:
            return x
        return self.generation_map_u_x(x)

    def sample_z_and_x_params_from_posterior(self, x, v, r):
        batch_size = x.shape[0]
        xp = cuda.get_array_module(x)

        h_t_gen, c_t_gen, u_t, h_t_enc, c_t_enc = self.generate_initial_state(
            batch_size, xp)

        downsampled_x = self.inference_downsampler_x(x)

        v_broadcast_shape = (
            h_t_enc.shape[0],
            v.shape[1],
        ) + h_t_enc.shape[2:]
        v = xp.reshape(v, v.shape + (1, 1))
        v = xp.broadcast_to(v, shape=v_broadcast_shape)

        z_t_params_array = []

        for t in range(self.generation_steps):
            inference_core = self.get_inference_core(t)
            inference_posterior = self.get_inference_posterior(t)
            generation_core = self.get_generation_core(t)
            generation_piror = self.get_generation_prior(t)
            generation_upsampler = self.get_generation_upsampler(t)

            h_next_enc, c_next_enc = inference_core(h_t_gen, h_t_enc, c_t_enc,
                                                    downsampled_x, v, r)

            mean_z_q, ln_var_z_q = inference_posterior.compute_parameter(
                h_t_enc)
            z_t = cf.gaussian(mean_z_q, ln_var_z_q)

            mean_z_p, ln_var_z_p = generation_piror.compute_parameter(h_t_gen)

            h_next_gen, c_next_gen = generation_core(h_t_gen, c_t_gen, z_t, v,
                                                     r)

            z_t_params_array.append((mean_z_q, ln_var_z_q, mean_z_p,
                                     ln_var_z_p))

            u_t = u_t + generation_upsampler(h_next_gen)
            h_t_gen = h_next_gen
            c_t_gen = c_next_gen
            h_t_enc = h_next_enc
            c_t_enc = c_next_enc

        mean_x = self.map_u_x(u_t)
        return z_t_params_array, mean_x

    def generate_image(self, v, r, xp):
        batch_size = v.shape[0]
        h_t_gen, c_t_gen, u_t, _, _ = self.generate_initial_state(
            batch_size, xp)

        v_broadcast_shape = (
            h_t_gen.shape[0],
            v.shape[1],
        ) + h_t_gen.shape[2:]
        v = xp.reshape(v, v.shape + (1, 1))
        v = xp.broadcast_to(v, shape=v_broadcast_shape)

        for t in range(self.generation_steps):
            generation_core = self.get_generation_core(t)
            generation_piror = self.get_generation_prior(t)
            generation_upsampler = self.get_generation_upsampler(t)

            mean_z_p, ln_var_z_p = generation_piror.compute_parameter(h_t_gen)
            z_t = cf.gaussian(mean_z_p, ln_var_z_p)

            h_next_gen, c_next_gen = generation_core(h_t_gen, c_t_gen, z_t, v,
                                                     r)

            u_t = u_t + generation_upsampler(h_next_gen)
            h_t_gen = h_next_gen
            c_t_gen = c_next_gen

        mean_x = self.map_u_x(u_t)
        return mean_x.data

    def generate_canvas_states(self, v, r, xp):
        batch_size = v.shape[0]
        h_t_gen, c_t_gen, u_t, _, _ = self.generate_initial_state(
            batch_size, xp)

        v_broadcast_shape = (
            h_t_gen.shape[0],
            v.shape[1],
        ) + h_t_gen.shape[2:]
        v = xp.reshape(v, v.shape + (1, 1))
        v = xp.broadcast_to(v, shape=v_broadcast_shape)

        u_t_array = []

        for t in range(self.generation_steps):
            generation_core = self.get_generation_core(t)
            generation_piror = self.get_generation_prior(t)
            generation_upsampler = self.get_generation_upsampler(t)

            mean_z_p, ln_var_z_p = generation_piror.compute_parameter(h_t_gen)
            z_t = cf.gaussian(mean_z_p, ln_var_z_p)

            h_next_gen, c_next_gen = generation_core(h_t_gen, c_t_gen, z_t, v,
                                                     r)

            u_t = u_t + generation_upsampler(h_next_gen)
            h_t_gen = h_next_gen
            c_t_gen = c_next_gen

            u_t_array.append(u_t)

        return u_t_array

    def reconstruct_image(self, query_images, v, r, xp):
        batch_size = v.shape[0]
        h0_g, c0_g, u0, h0_e, c0_e = self.generate_initial_state(
            batch_size, xp)

        hl_e = h0_e
        cl_e = c0_e
        hl_g = h0_g
        cl_g = c0_g
        ul_e = u0

        xq = self.inference_downsampler(query_images)

        for l in range(self.generation_steps):
            inference_core = self.get_inference_core(l)
            inference_posterior = self.get_inference_posterior(l)
            generation_core = self.get_generation_core(l)

            he_next, ce_next = inference_core(hl_g, hl_e, cl_e, xq, v, r)

            ze_l = inference_posterior.sample_z(hl_e)

            hg_next, cg_next, ue_next = generation_core(
                hl_g, cl_g, ul_e, ze_l, v, r)

            hl_g = hg_next
            cl_g = cg_next
            ul_e = ue_next
            hl_e = he_next
            cl_e = ce_next

        x = self.generation_observation.compute_mean_x(ul_e)
        return x.data
