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

# 일반적으로 잘 알려져 있는 alexnet 구조와 동일
# 일부 변형 alexnet 구조 사진 말고, 기본 alexnet 사진을 참조하면 좋음
# 크게 __init__에서는 필요한 레이어를 정의, __call__ 에서는 해당 레이어들을 쌓아서 신경망 구성
# chainer에 있는 Convolution2D, linear(dense 레이어), relu, max_pooling_2d를 사용

class AlexNet(chainer.Chain):

    def __init__(self, n_joints):
        super(AlexNet, self).__init__(
            # Convolution2D(in_channels, out_channels, ksize, stride=1, pad=0)
            # 아래 코드는 입력 채널:3, 출력 채널: 96, 필터 크기: 11인 예시
            conv1=L.Convolution2D(3, 96, 11, stride=4, pad=1),
            conv2=L.Convolution2D(96, 256, 5, stride=1, pad=2),
            conv3=L.Convolution2D(256, 384, 3, stride=1, pad=1),
            conv4=L.Convolution2D(384, 384, 3, stride=1, pad=1),
            conv5=L.Convolution2D(384, 256, 3, stride=1, pad=1),

            # Linear(in_size, out_size)
            # 아래 코드는 입력크기 9216, 출력크기 4096 예시
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),

            # 마지막 레이어만 전통 alexnet과 다름. 
            # 전통 alexnet은 1000개, 그러나 우리 모델의 마지막은 n_joints*2로 분류
            fc8=L.Linear(4096, n_joints * 2)
        )
        self.train = True

    # __init__에서 정의한 레이어들을 이용하여 신경망 구성
    # 활성화함수는 relu를 쓰고, max pooling도 하고, 특히 lrn (배치 정규화 기법) 작업도 있음
    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        # dropout도 마찬가지로 전통 alexnet과 동일
        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.6)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.6)

        return self.fc8(h)