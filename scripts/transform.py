#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2 as cv
import numpy as np
import chainer


class Transform(object):

    def __init__(self, args):
        print(args.items())
        [setattr(self, key, value) for key, value in args.items()]

    def transform(self, datum, datadir, train=True,
                  fname_index=0, joint_index=1):
        img_fn = '%s/images/%s' % (datadir, datum[self.fname_index])
        if not os.path.exists(img_fn):
            raise Exception('%s is not exist' % img_fn)
        self._img = cv.imread(img_fn)
        self.orig = self._img.copy()
        self._joints = np.asarray([int(float(p)) for p in datum[joint_index:]])
        #self._joints = list(zip(coords[0::2], coords[1::2]))

        if hasattr(self, 'padding') and hasattr(self, 'shift'):
            self.crop()
        if hasattr(self, 'flip'):
            self.fliplr()
        if hasattr(self, 'resize'):
            self.resizeFunc()
        if hasattr(self, 'lcn'):
            self.contrast()

        # joint pos centerization
        h, w, c = self._img.shape
        center_pt = np.array([w / 2, h / 2], dtype=np.float32)  # x,y order
        joints = list(zip(self._joints[0::2], self._joints[1::2]))
        joints = np.array(joints, dtype=np.float32) - center_pt
        #joints -= np.array([center_pt[0], center_pt[1]])
        joints[:, 0] /= w
        joints[:, 1] /= h
        self._joints = joints.flatten() #coord_normalize

        return self._img, self._joints

    def crop(self):
        # image cropping
        joints = self._joints.reshape((len(self._joints) / 2, 2))
        x, y, w, h = cv.boundingRect(np.asarray([joints.tolist()]))

        # bounding rect extending
        inf, sup = self.padding
        r = sup - inf
        pad_w_r = np.random.rand() * r + inf  # inf~sup
        pad_h_r = np.random.rand() * r + inf  # inf~sup
        x -= (w * pad_w_r - w) / 2
        y -= (h * pad_h_r - h) / 2
        w *= pad_w_r
        h *= pad_h_r

        # shifting
        x += np.random.rand() * self.shift * 2 - self.shift
        y += np.random.rand() * self.shift * 2 - self.shift

        # clipping
        x, y, w, h = [int(z) for z in [x, y, w, h]]
        x = np.clip(x, 0, self._img.shape[1] - 1)
        y = np.clip(y, 0, self._img.shape[0] - 1)
        w = np.clip(w, 1, self._img.shape[1] - (x + 1))
        h = np.clip(h, 1, self._img.shape[0] - (y + 1))
        self._img = self._img[y:y + h, x:x + w]

        # joint shifting
        joints = np.asarray([(j[0] - x, j[1] - y) for j in joints])
        self._joints = joints.flatten()

    def apply_zoom(self, image, joints, center_x, center_y, fx=None, fy=None):
        joint_vecs = joints - np.array([center_x, center_y])
        #print(fx,fy)
        if fx is None and fy is None:
            zoom = 1.0 + np.random.uniform(-self.zoom_range, self.zoom_range)
            fx, fy = zoom, zoom
        image = cv.resize(image, None, fx=fx, fy=fy)
        joint_vecs *= np.array([fx, fy])
        center_x, center_y = center_x * fx, center_y * fy
        joints = joint_vecs + np.array([center_x, center_y])
        return image, joints, center_x, center_y
    
    def resizeFunc(self):
        if not isinstance(self.resize, int):
            raise Exception('self.size should be int')

        orig_h, orig_w = self._img.shape[:2]
        self._joints[0::2] = self._joints[0::2] / float(orig_w) * self.resize
        self._joints[1::2] = self._joints[1::2] / float(orig_h) * self.resize
        self._img = cv.resize(self._img, (self.resize, self.resize), interpolation=cv.INTER_NEAREST)


    def contrast(self):
        if self.lcn:
            if not self._img.dtype == np.float32:
                self._img = self._img.astype(np.float32)
            # local contrast normalization
            for ch in range(self._img.shape[2]):
                im = self._img[:, :, ch]
                im = (im - np.mean(im)) / \
                     (np.std(im) + np.finfo(np.float32).eps)
                self._img[:, :, ch] = im

    def fliplr(self):
        if np.random.randint(2) == 1 and self.flip:
            self._img = np.fliplr(self._img)
            self._joints[0::2] = self._img.shape[1] - self._joints[0:: 2]
            joints = list(zip(self._joints[0::2], self._joints[1::2]))

            # shoulder
            joints[2], joints[4] = joints[4], joints[2]
            # elbow
            joints[1], joints[5] = joints[5], joints[1]
            # wrist
            joints[0], joints[6] = joints[6], joints[0]

            self._joints = np.array(joints).flatten()

    def revert(self, img, pred):
        h, w, c = img.shape
        center_pt = np.array([w / 2, h / 2])
        joints = np.array(list(zip(pred[0::2], pred[1::2])))  # x,y order
        joints[:, 0] *= w
        joints[:, 1] *= h
        joints += center_pt
        #joints = chainer.as_array(joints)
        for i in range(joints.shape[0]):
            for j in range(joints.shape[1]):
                joints[i][j] = float(joints[i][j].item())
        #joints = chainer.as_array(joints)
        joints = joints.astype(np.int32)
        if hasattr(self, 'lcn') and self.lcn:
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)

        return img, joints
