from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile

import grep as grep
from transform import Transform
import chainer
from chainer import backends
from chainer import serializers
from chainer import Variable

import argparse
import cv2 as cv
import glob
import importlib
import importlib.machinery
import numpy as np
import os
import re
import sys
import loss
import dataset


def cropping(img, joints, min_dim):
    # image cropping
    _joints = joints.reshape((len(joints) // 2, 2))
    posi_joints = [(j[0], j[1]) for j in _joints if j[0] > 0 and j[1] > 0]
    x, y, w, h = cv.boundingRect(np.asarray([posi_joints]))
    if w < min_dim:
        w = min_dim
    if h < min_dim:
        h = min_dim

    # bounding rect extending
    x -= (w * 1.5 - w) / 2
    y -= (h * 1.5 - h) / 2
    w *= 1.5
    h *= 1.5

    # clipping
    x, y, w, h = [int(z) for z in [x, y, w, h]]  # int로
    x = np.clip(x, 0, img.shape[1] - 1)  # min보다 작은건 min으로, max보다 큰건 max로
    y = np.clip(y, 0, img.shape[0] - 1)
    w = np.clip(w, 1, img.shape[1] - (x + 1))
    h = np.clip(h, 1, img.shape[0] - (y + 1))
    img = img[y:y + h, x:x + w]

    # joint shifting
    _joints = np.asarray([(j[0] - x, j[1] - y) for j in _joints])
    joints = _joints.flatten()

    return img, joints


def resize(img, joints, size):
    orig_h, orig_w = img.shape[:2]
    joints[0::2] = joints[0::2] / float(orig_w) * size
    joints[1::2] = joints[1::2] / float(orig_h) * size
    img = cv.resize(img, (size, size), interpolation=cv.INTER_NEAREST)

    return img, joints


def contrast(img):
    if not img.dtype == np.float32:
        img = img.astype(np.float32)
    # global contrast normalization
    img -= img.reshape(-1, 3).mean(axis=0)
    img -= img.reshape(-1, 3).std(axis=0) + 1e-5

    return img


def input_transform(datum, datadir, fname_index, joint_index, min_dim, gcn):
    img_fn = '%s/images/%s' % (datadir, datum[fname_index])
    if not os.path.exists(img_fn):
        raise IOError('%s is not exist' % img_fn)

    img = cv.imread(img_fn)
    joints = np.asarray([int(float(p)) for p in datum[joint_index:]])
    img, joints = cropping(img, joints, min_dim)
    img, joints = resize(img, joints, img.shape[0])  # 정사각형으로 가정
    if gcn:
        img = contrast(img)
    else:
        img /= 255.0

    return img, joints


def load_model(args):
    model_fn = os.path.basename(args.model)
    model_name = model_fn.split('.')[0]
    loader = importlib.machinery.SourceFileLoader(model_name, args.model)
    model = loader.load_module()
    model = getattr(model, model_name)
    model = model(args.joint_num)

    #conv1, conv2, conv3, conv4, conv5, fc6, fc7, fc8
    path_base="predictor/"
    tmp = dict(np.load(args.param))
    for i in tmp:
        tmp[i] = np.nan_to_num(tmp[i], nan=0)
    np.savez(args.param, **tmp)
    
    serializers.load_npz(args.param, model.conv1, path_base+"conv1/")
    serializers.load_npz(args.param, model.conv2, path_base+"conv2/")
    serializers.load_npz(args.param, model.conv3, path_base+"conv3/")
    serializers.load_npz(args.param, model.conv4, path_base+"conv4/")
    serializers.load_npz(args.param, model.conv5, path_base+"conv5/")
    serializers.load_npz(args.param, model.fc6, path_base+"fc6/")
    serializers.load_npz(args.param, model.fc7, path_base+"fc7/")
    serializers.load_npz(args.param, model.fc8, path_base+"fc8/")
    #serializers.load_npz(args.param, model)
    model.train = False #??

    return model


def load_data(trans, args, x):
    c= 3
    #s = args.size
    d = args.joint_num * 2
    
    #input_data = np.zeros((len(x),c,s,s))
    input_data=[]
    label = np.zeros((len(x),d))

    for i, line in enumerate(x):
        d,t = trans.transform(line.split(','), args.datadir, args.fname_index, args.joint_index)
        input_data.append(d.transpose((2,0,1)))
        label[i] = t

    return np.array(input_data), label


def create_tiled_image(perm, out_dir, result_dir, epoch, suffix, N=25):
    fnames = np.array(sorted(glob.glob('%s/*%s.jpg' % (out_dir, suffix))))
    tile_fnames = fnames[perm[:N]]

    h, w, pad = 220, 220, 2
    side = int(np.ceil(np.sqrt(len(tile_fnames))))
    canvas = np.zeros((side * h + pad * (side + 1),
                       side * w + pad * (side + 1), 3))

    for i, fname in enumerate(tile_fnames):
        img = cv.imread(fname)
        x = w * (i % side) + pad * (i % side + 1)
        y = h * (i // side) + pad * (i // side + 1)
        canvas[y:y + h, x:x + w, :] = img

    if args.resize > 0:
        canvas = cv.resize(canvas, (args.resize, args.resize))
    cv.imwrite('%s/test_%d_tiled_%s.jpg' % (result_dir, epoch, suffix), canvas)


def draw_joints(image, joints, prefix, ignore_joints):
    skeleton_lines=[
        [0,1],
        [1,2],
        [2,6],
        [6,3],
        [3,4],
        [4,5],
        [6,7],
        [7,8],
        [8,9],
        [10,11],
        [11,12],
        [12,8],
        [8,13],
        [13,14],
        [14,15]
    ]
    N=16
    if image.shape[2] != 3:
        _image = image.transpose(1, 2, 0).copy()
    else:
        _image = image.copy()

    if joints.ndim == 1:
        joints = np.array(list(zip(joints[0::2], joints[1::2])))

    if ignore_joints.ndim == 1:
        ignore_joints = np.array(
            list(zip(ignore_joints[0::2], ignore_joints[1::2])))

    available=[False]*N
    point=[False]*N
    for i, (x, y) in enumerate(joints):
        if ignore_joints is not None \
                and (ignore_joints[i][0] == 0 or ignore_joints[i][1] == 0):
            continue
        _image = cv.circle(_image, (int(x), int(y)), 2, (0, 0, 255), -1)
        _image = cv.putText(
            _image, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX,
            0.3, (255, 255, 255), 3)
        _image = cv.putText(
            _image, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX,
            0.3, (0, 0, 0), 1)
        available[i]=True
        point[i]=(x,y)

    for lines in skeleton_lines:
        start = lines[0]
        end = lines[1]
        if available[start] and available[end] :
            _image = cv.line(_image, point[start], point[end], (0,0,255),2)
    _, fn_img = tempfile.mkstemp()
    basename = os.path.basename(fn_img)
    fn_img = fn_img.replace(basename, basename)
    fn_img = fn_img + '.png'
    return _image
    #cv.imwrite(fn_img, _image)


def test(args):
    # test data
    test_fn = '%s/test_joints.csv' % args.datadir
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    # load model
    if args.gpu >= 0:
        backends.cuda.get_device_from_array(args.gpu).use()
    model = load_model(args)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # create output dir
    epoch = int(10)  # args.aram에서 epoch-(숫자) 형식 찾아서 0번째 -> int로
    result_dir = os.path.dirname(args.param)
    out_dir = '%s/test_%d' % (result_dir, epoch)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_log = '%s.log' % out_dir
    fp = open(out_log, 'w')

    mean_error = 0.0
    N = len(test_dl)
    
    ## pcp code
    # to reference variable (not normal variable)
    arm_up_total = [0]
    arm_lo_total = [0]
    leg_up_total = [0]
    leg_lo_total = [0]

    arm_up_correct = [0]
    arm_lo_correct = [0]
    leg_up_correct = [0]
    leg_lo_correct = [0]

    for i in range(0, N, args.batchsize):
        lines = test_dl[args.batchsize*i:args.batchsize*(i+1)]
        trans = Transform(vars(args))
        input_data, labels = load_data(trans, args, lines)

        if args.gpu >= 0:
            input_data = backends.cuda.to_gpu(input_data.astype(np.float32))
            labels = backends.cuda.to_gpu(labels.astype(np.float32))
        else:
            input_data = input_data.astype(np.float32)
            labels = labels.astype(np.float32)
        with chainer.using_config('volatile', True):
            x = chainer.Variable(input_data)
            t = chainer.Variable(labels)
            model = loss.PoseEstimationError(model)
            model.predictor.train=False
            ig = labels.astype(np.float32)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i][j]==-1:
                        ig[i][j] = 0
                    else:
                        ig[i][j]=1
            model(x,t,ig)

        if args.gpu >= 0:
            preds = backends.cuda.to_cpu(model.pred.data)
            input_data = backends.cuda.to_cpu(input_data)
            labels = backends.cuda.to_cpu(labels)
        else:
            preds = model.predictor(x)#?? -> model 코드가 바꼈음
            
        for n, line in enumerate(lines):
            img_fn = line.split(',')[args.fname_index]
            img = input_data[n].transpose((1, 2, 0))
            pred = preds[n]
            img_pred, pred = trans.revert(img, pred)

            # turn label data into image coordinates
            label = labels[n]
            img_label, label = trans.revert(img, label)

            # calc mean_error
            #print("before",joints)
            error = np.linalg.norm(pred - label) / len(pred)
            mean_error += error
            
            ## pcp
            ## calc pcp metric distance
            dis_arm_up = (np.linalg.norm(label[11] - label[12]) + np.linalg.norm(label[13] - label[14])) / 2
            dis_arm_lo = (np.linalg.norm(label[10] - label[11]) + np.linalg.norm(label[14] - label[15])) / 2
            dis_leg_up = (np.linalg.norm(label[1] - label[2]) + np.linalg.norm(label[3] - label[4])) / 2
            dis_leg_lo = (np.linalg.norm(label[0] - label[1]) + np.linalg.norm(label[4] - label[5])) / 2

            ## pcp count
            for metric, correct_count, total_count in [ (dis_arm_up, arm_up_correct, arm_up_total) , \
                (dis_arm_lo, arm_lo_correct, arm_lo_total), (dis_leg_up, leg_up_correct, leg_up_total) , (dis_leg_lo, leg_lo_correct, leg_lo_total)]:
                
                for i in range(len(label)):
                    total_count[0] += 1
                    dis = np.linalg.norm(label[i] - pred[i])
                    if dis < metric:
                        correct_count[0] += 1

            # create pred, label tuples
            img_pred = np.array(img_pred.copy())
            img_label = np.array(img_label.copy())
            pred = np.array([tuple(p) for p in pred]) #prediction..?
            label = np.array([tuple(p) for p in label]) #label
            ig_pred = [0 if v == -1 else 1 for v in pred.flatten()]
            ig_pred  = np.array(list(zip(ig_pred[0::2], ig_pred[1::2])))
            ig_label = [0 if v == -1 else 1 for v in label.flatten()]
            ig_label  = np.array(list(zip(ig_label[0::2], ig_label[1::2])))
            

            # all limbs
            img_label = draw_joints(
                img_label, label, args.draw_limb, ig_label)
            img_pred = draw_joints(
                img_pred, pred, args.draw_limb, ig_pred)

            msg = '{:5}/{:5} {}\terror:{}\tmean_error:{}'.format(
                i + n, N, img_fn, error, mean_error / (i + n + 1))
            print(msg, file=fp)
            print(msg)
            
            fn, ext = os.path.splitext(img_fn)
            tr_fn = '%s/%d-%d_%s_pred%s' % (out_dir, i, n, fn, ext)
            la_fn = '%s/%d-%d_%s_label%s' % (out_dir, i, n, fn, ext)
            cv.imwrite(tr_fn, img_pred)
            cv.imwrite(la_fn, img_label)
        
        ## pcp
        ## pcp count printing
        print("---- pcp value ----")
        print("%7s %7s %7s %7s %7s" % ("arm_up", "arm_lo", "leg_up", "leg_lo", "avg"))
        print("%7s %7s %7s %7s %7s" % (round(arm_up_correct[0]/arm_up_total[0] ,3), \
            round(arm_lo_correct[0]/arm_lo_total[0], 3),\
            round(leg_up_correct[0]/leg_up_total[0], 3),\
            round(leg_lo_correct[0]/leg_lo_total[0], 3),\
            round((arm_up_correct[0]/arm_up_total[0] + arm_lo_correct[0]/arm_lo_total[0] + leg_up_correct[0]/leg_up_total[0] + leg_lo_correct[0]/leg_lo_total[0])/4, 3))
            )
        print("-------------------")
        # print("arm_up: " , )
        # print("arm_lo: " , )
        # print("leg_up: " , )
        # print("leg_lo: " , round(leg_lo_correct[0]/leg_lo_total[0], 2))


def tile(args):
    # create output dir
    #epoch = int(re.search('epoch-([0-9]+)', args.param).groups()[0])
    epoch=10
    result_dir = os.path.dirname(args.param)
    out_dir = '%s/test_%d' % (result_dir, epoch)
    if not os.path.exists(out_dir):
        raise Exception('%s is not exist' % out_dir)

    # save tiled image of randomly chosen results and labels
    n_img = len(glob.glob('%s/*pred*' % (out_dir)))
    perm = np.random.permutation(n_img)
    create_tiled_image(perm, out_dir, result_dir, epoch, 'pred', args.n_imgs)
    create_tiled_image(perm, out_dir, result_dir, epoch, 'label', args.n_imgs)


if __name__ == '__main__': #또다른 메인..? 이거는 테스트용인듯.
    sys.path.append('tests')
    sys.path.append('models')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model definition file in models dir')
    parser.add_argument('--param', type=str,
                        help='trained parameters file in result dir')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datadir', type=str, default='data/FLIC-full')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'tile'],
                        help='test or create tiled image')
    parser.add_argument('--n_imgs', type=int, default=9,
                        help='how many images will be tiled')
    parser.add_argument('--resize', type=int, default=-1,
                        help='resize the results of tiling')
    parser.add_argument('--seed', type=int, default=9,
                        help='random seed to select images to be tiled')
    parser.add_argument('--draw_limb', type=bool, default=True,
                        help='whether draw limb line to visualize')
    parser.add_argument('--text_scale', type=float, default=1.0,
                        help='text scale when drawing indices of joints')
    args = parser.parse_args()

    result_dir = os.path.dirname(args.param)
    log_fn ='{}/log.txt'.format(args.param)
    args.joint_num = 16
    args.fname_index = 0
    args.joint_index = 1
    args.size = 220
    # for line in open(log_fn):
    #     if 'Namespace' in line:
    #         args.joint_num = int(
    #             re.search('joint_num=([0-9]+)', line).groups()[0])
    #         args.fname_index = int(
    #             re.search('fname_index=([0-9]+)', line).groups()[0])
    #         args.joint_index = int(
    #             re.search('joint_index=([0-9]+)', line).groups()[0])
    #         break

    if args.mode == 'test':
        test(args)
    elif args.mode == 'tile':
        np.random.seed(args.seed)
        tile(args)