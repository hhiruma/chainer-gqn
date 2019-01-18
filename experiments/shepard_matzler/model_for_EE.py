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


class EEModel(chainer.Chain):
    def __init__(self):
        super(EEModel, self).__init__()
        with self.init_scope():
            self.cn1 = nn.Convolution2D(3, 16, ksize=3)
            self.cn2 = nn.Convolution2D(16, 32, ksize=3)
            self.cn3 = nn.Convolution2D(32, 64, ksize=3)
            self.cn4 = nn.Convolution2D(64, 64, ksize=3)
            self.cn5 = nn.Convolution2D(64, 64, ksize=3)
            self.l1 = nn.Linear(None,500)
            self.l2 = nn.Linear(500,1)
            self.parameters = chainer.ChainList()


    def __call__(self,x):
        h1 = cf.relu(self.cn1(x))
        h2 = cf.max_pooling_2d(cf.relu(self.cn2(h1)),2)
        h3 = cf.relu(self.cn3(h2))
        h4 = cf.max_pooling_2d(cf.relu(self.cn4(h3)),2)
        h5 = cf.max_pooling_2d(cf.relu(self.cn5(h4)),2)
        h6 = cf.relu(self.l1(h5))
        return self.l2(h6)

    def serialize(self,Name, path):
        self.serialize_parameter(path, Name, self.parameters)

    def serialize_parameter(self, path, filename, params):
        tmp_filename = str(uuid.uuid4())
        save_hdf5(os.path.join(path, tmp_filename), params)
        os.rename(
            os.path.join(path, tmp_filename), os.path.join(path, filename))
