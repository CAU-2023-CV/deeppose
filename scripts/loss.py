from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABC

import numpy
from chainer import reporter

import chainer
from chainer.utils import type_check


class MeanSquaredError(chainer.Function):
    """Mean squared error (a.k.a. Euclidean loss) function.

    In forward method, it calculates mean squared error between two variables
    with ignoring all elements that the value of ignore_joints at the same
    position is 0.

    """

    def __init__(self):
        self.count = 0
        self.diff = 0.0

    def check_type_forward(self, in_types):
        #type_check.expect(in_types.size() == 2)
        # type_check.expect(
        #     in_types[0].dtype == numpy.float32,
        #     in_types[1].dtype == numpy.float32,
        #     in_types[2].dtype == numpy.int32,
        #     in_types[0].shape == in_types[1].shape,
        #     in_types[1].shape == in_types[2].shape,
        # )
        type_check.expect(in_types.size()==3)

    def forward(self, inputs):
        x, t, ignore = inputs
        xp = chainer.backend.get_array_module(x)  # chainer.cuda -> chainer.backend
        self.count = int(ignore.sum())
        self.diff = (numpy.dot(x,ignore) - numpy.dot(t,ignore)).astype(xp.float32)
        diff = self.diff.ravel()
        return xp.asarray(diff.dot(diff) / self.count, dtype=xp.float32),

    def backward(self, inputs, gy):
        coeff = gy[0] * gy[0].dtype.type(2. / self.count)
        gx0 = coeff * self.diff
        return gx0, -gx0, None


def mean_squared_error(x0, x1, ignore):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return MeanSquaredError()(x0, x1, ignore)


class PoseEstimationError(chainer.Chain):

    @classmethod
    def from_params(cls, *args, **kwargs): #?? 공식문서 봐도 딱히 나와있는 내용은 없는데 추가하라고 에러뜸
        pass

    def __init__(self, predictor):
        super(PoseEstimationError, self).__init__(predictor=predictor)
        self.lossfun = mean_squared_error
        self.y = None
        self.loss = None

    def __call__(self, *args):
        x, t, ignore = args[:3]
        self.y = None
        self.loss = None
        self.pre_rec = None
        self.y = self.predictor(x)
        self.loss = self.lossfun(self.y, t, ignore)
        reporter.report({'loss': self.loss}, self)
        return self.loss
    
#     Exception in main training loop: shapes (128,28) and (128,32) not aligned: 28 (dim 1) != 128 (dim 0)
# Traceback (most recent call last):
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\training\trainer.py", line 343, in run
#     update()
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\training\updaters\standard_updater.py", line 240, in update
#     self.update_core()
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\training\updaters\parallel_updater.py", line 131, in update_core
#     loss = loss_func(*in_arrays)
#   File "C:\Users\pc03\deeppose\scripts\loss.py", line 81, in __call__
#     self.loss = self.lossfun(self.y, t, ignore)
#   File "C:\Users\pc03\deeppose\scripts\loss.py", line 60, in mean_squared_error
#     return MeanSquaredError()(x0, x1, ignore)
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\function.py", line 307, in __call__
#     ret = node.apply(inputs)
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\function_node.py", line 334, in apply
#     outputs = self.forward(in_data)
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\function.py", line 179, in forward
#     return self._function.forward(inputs)
#   File "C:\Users\pc03\deeppose\scripts\loss.py", line 43, in forward
#     self.diff = (numpy.dot(x,ignore) - numpy.dot(t,ignore)).astype(xp.float32)
#   File "<__array_function__ internals>", line 180, in dot
# Will finalize trainer extensions and updater before reraising the exception.
# Traceback (most recent call last):
#   File "C:\Users\pc03\deeppose\scripts\train.py", line 243, in <module>
#     trainer.run()
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\training\trainer.py", line 376, in run
#     six.reraise(*exc_info)
#   File "C:\Users\pc03\anaconda3\lib\site-packages\six.py", line 719, in reraise
#     raise value
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\training\trainer.py", line 343, in run
#     update()
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\training\updaters\standard_updater.py", line 240, in update
#     self.update_core()
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\training\updaters\parallel_updater.py", line 131, in update_core
#     loss = loss_func(*in_arrays)
#   File "C:\Users\pc03\deeppose\scripts\loss.py", line 81, in __call__
#     self.loss = self.lossfun(self.y, t, ignore)
#   File "C:\Users\pc03\deeppose\scripts\loss.py", line 60, in mean_squared_error
#     return MeanSquaredError()(x0, x1, ignore)
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\function.py", line 307, in __call__
#     ret = node.apply(inputs)
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\function_node.py", line 334, in apply
#     outputs = self.forward(in_data)
#   File "C:\Users\pc03\anaconda3\lib\site-packages\chainer\function.py", line 179, in forward
#     return self._function.forward(inputs)
#   File "C:\Users\pc03\deeppose\scripts\loss.py", line 43, in forward
#     self.diff = (numpy.dot(x,ignore) - numpy.dot(t,ignore)).astype(xp.float32)
#   File "<__array_function__ internals>", line 180, in dot
# ValueError: shapes (128,28) and (128,32) not aligned: 28 (dim 1) != 128 (dim 0)
