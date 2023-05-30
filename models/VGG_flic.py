#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

# alexnet 정의 방식과는 살짝 다르게 links 배열에 (레이어이름, 레이어코드)의 튜플 형식으로 레이어 구성
class VGG_BN(chainer.Chain):

    def __init__(self, n_joints):
        self.train = True

        super(VGG_BN, self).__init__()
        links = [('conv1_1', L.Convolution2D(3, 64, 3, stride=1, pad=1))]
        # BatchNormalization(size): 배치 정규화 진행
        links += [('bn1_1', L.BatchNormalization(64))]
        links += [('conv1_2', L.Convolution2D(64, 64, 3, stride=1, pad=1))]
        links += [('bn1_2', L.BatchNormalization(64))]

        links += [('conv2_1', L.Convolution2D(64, 128, 3, stride=1, pad=1))]
        links += [('bn2_1', L.BatchNormalization(128))]
        links += [('conv2_2', L.Convolution2D(128, 128, 3, stride=1, pad=1))]
        links += [('bn2_2', L.BatchNormalization(128))]

        links += [('conv3_1', L.Convolution2D(128, 256, 3, stride=1, pad=1))]
        links += [('bn3_1', L.BatchNormalization(256))]
        links += [('conv3_2', L.Convolution2D(256, 256, 3, stride=1, pad=1))]
        links += [('bn3_2', L.BatchNormalization(256))]
        links += [('conv3_3', L.Convolution2D(256, 256, 3, stride=1, pad=1))]
        links += [('bn3_3', L.BatchNormalization(256))]

        links += [('conv4_1', L.Convolution2D(256, 512, 3, stride=1, pad=1))]
        links += [('bn4_1', L.BatchNormalization(512))]
        links += [('conv4_2', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('bn4_2', L.BatchNormalization(512))]
        links += [('conv4_3', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('bn4_3', L.BatchNormalization(512))]

        links += [('conv5_1', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('bn5_1', L.BatchNormalization(512))]
        links += [('conv5_2', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('bn5_2', L.BatchNormalization(512))]
        links += [('conv5_3', L.Convolution2D(512, 512, 3, stride=1, pad=1))]
        links += [('bn5_3', L.BatchNormalization(512))]
        # Linear에서 입력을 None으로 결정할 경우, 최초의 순전파 시 크기가 결정됨
        links += [('fc6', L.Linear(None, 4096))]
        links += [('fc7', L.Linear(4096, 4096))]

        # 오류 수정: out_size 없는 것 대신 n_joints 변수
        # links += [('fc8', L.Linear(4096, out_size ** 2))]
        links += [('fc8', L.Linear(4096, n_joints * 2))]
        for link in links:
            self.add_link(*link)

    # init에서 정의한 links에 있는 레이어들을 이용해 relu, max pooling 등으로 신경망 구성
    def __call__(self, x):
        h = F.relu(self.bn1_1(self.conv1_1(x), test=not self.train))
        h = F.relu(self.bn1_2(self.conv1_2(h), test=not self.train))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bn2_1(self.conv2_1(h), test=not self.train))
        h = F.relu(self.bn2_2(self.conv2_2(h), test=not self.train))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bn3_1(self.conv3_1(h), test=not self.train))
        h = F.relu(self.bn3_2(self.conv3_2(h), test=not self.train))
        h = F.relu(self.bn3_3(self.conv3_3(h), test=not self.train))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bn4_1(self.conv4_1(h), test=not self.train))
        h = F.relu(self.bn4_2(self.conv4_2(h), test=not self.train))
        h = F.relu(self.bn4_3(self.conv4_3(h), test=not self.train))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.bn5_1(self.conv5_1(h), test=not self.train))
        h = F.relu(self.bn5_2(self.conv5_2(h), test=not self.train))
        h = F.relu(self.bn5_3(self.conv5_3(h), test=not self.train))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        return self.fc8(h)