from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
from ctypes import Array
from threading import Thread
from queue import Queue

import numpy as np
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.training import extensions
from transform import Transform
from chainer import dataset
from PIL import Image
import random

import chainer
import cmd_options
import dataset
import importlib #imp is deprecated, imp -> importlib
from importlib.machinery import SourceFileLoader
import logger
import logging
import loss
import os
import shutil
import sys
import tempfile
import time

def create_result_dir(model_path, resume_model):
    if not os.path.exists('results'):
        os.mkdir('results')
    if resume_model is None:
        prefix = '{}_{}'.format(
            os.path.splitext(os.path.basename(model_path))[0],
            time.strftime('%Y-%m-%d_%H-%M-%S'))
        result_dir = tempfile.mkdtemp(prefix=prefix, dir='results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        result_dir = os.path.dirname(resume_model)

    return result_dir

def create_logger(args, result_dir):
    logging.basicConfig(filename='{}/log.txt'.format(result_dir))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    msg_format = '%(asctime)s [%(levelname)s] %(message)s'
    formatter = logging.Formatter(msg_format)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    logging.info(sys.version_info)
    logging.info('chainer version: {}'.format(chainer.__version__))
    logging.info('cuda: {}, cudnn: {}'.format(
        chainer.cuda.available, chainer.cuda.cudnn_enabled))
    logging.info(args)


def get_model(model_path, n_joints, result_dir, resume_model):
    model_fn = os.path.basename(model_path)
    model_name = model_fn.split('.')[0]
    model = SourceFileLoader(model_name, model_path).load_module()#model 파일 소스 받아옴
    model = getattr(model, model_name) #해당 모델 객체를 가져옴

    # Initialize
    model = model(n_joints) #초기화

    # Copy files -> 아마 존재하는 Model이 있으면 복사해서 받아오는듯.
    dst = '{}/{}'.format(result_dir, model_fn)
    if not os.path.exists(dst): #해당 경로가 존재하면
        shutil.copy(model_path, dst) #파일을 복사

    # load model
    if resume_model is not None: #default: None
        serializers.load_npz(resume_model, model) #resume_model npz파일에서 해당 Model object 가져와서 model에 저장. (아마 checkpoint 역할..?)

    return model

def get_optimizer(model, opt, lr, adam_alpha=None, adam_beta1=None,
                  adam_beta2=None, adam_eps=None, weight_decay=None,
                  resume_opt=None):
    if opt == 'MomentumSGD': #optimizer 설정
        optimizer = optimizers.MomentumSGD(lr=lr, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optimizers.Adam(
            alpha=adam_alpha, beta1=adam_beta1,
            beta2=adam_beta2, eps=adam_eps)
    elif opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=lr)
    elif opt == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=lr)
    else:
        raise Exception('No optimizer is selected')

        # The first model as the master model
    optimizer.setup(model) #model에 대한 optimizer setup

    if opt == 'MomentumSGD':
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(weight_decay))

    if resume_opt is not None: #checkpoint
        serializers.load_npz(resume_opt, optimizer)

    return optimizer

def transform(args, x_queue, datadir, fname_index, joint_index, o_queue):
    trans = Transform(args) #Transform ?? -> 무슨 함수 쓰는건지 잘 모르겠는데 일단 PIL으로 import함
    while True:
        x=x_queue.get()
        if x is None:
            break
        x, t = trans.transform(x.split(','), datadir, fname_index, joint_index)
        o_queue.put((x.transpose((2,0,1)),t))

def load_data(args, input_q, minibatch_q):
    c = args.channel
    s = args.size
    d = args.join_num * 2

    input_data_base = Array(ctypes.c_float, args.batchsize * c * s * s)
    input_data = np.ctypeslib.as_array(input_data_base.get_obj())
    input_data = input_data.reshape((args.batchsize, c, s, s))

    label_base = Array(ctypes.c_float, args.batchsize*d)
    label = np.ctypeslib.as_array(label_base.get_obj())
    label = label.reshape((args.batchsize, d))

    x_queue, o_queue = Queue(), Queue()
    workers = [Thread(target=transform,
                       args = (args, x_queue, args.datadir, args.fname_index,
                               args.join_index, o_queue))
               for _ in range(args.batchsize)] #batchsize만큼 Process worker 생성
    for w in workers:
        w.start() #각 worker start
    while True:
        x_batch = input_q.get()
        if x_batch is None:
            break

        #data augumentation
        for x in x_batch:
            x_queue.put(x)
            #x_queue=[x,x,x...] (x in x_batch <- input_q)
        j = 0
        while j != len(x_batch):
            a,b = o_queue.get()
            input_data[j] = a
            label[j] = b
            j+=1
            # input_data=[a1,a2,a3..], label=[b1,b2,b3...] <- o_queue=[(a1,b1),(a2,b2)...]
        minibatch_q.put([input_data, label])

    for _ in range(args.batchsize):
        x_queue.put(None) #x_queue=[x,x,x....,None,None,None....]
    for w in workers:
        w.join() #join



if __name__=='__main__':
    args = cmd_options.get_arguments() #실행 때 입력한 arg값 받아옴
    result_dir = create_result_dir(args.model, args.resume_model) #결과값 저장할 directory 생성
    create_logger(args, result_dir) #log 파일 생성함 +최초 log 출력
    model = get_model(args.model, args.n_joints, result_dir, args.resume_model) #model 받아옴 (AlexNet.py...)
    model = loss.PoseEstimationError(model) #model에 대한 poseEstimationError 설정 (init)
    opt = get_optimizer(model, args.opt, args.lr, adam_alpha=args.adam_alpha,
                        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
                        adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                        resume_opt=args.resume_opt) #model에 대한 optimizer 설정
    train_dataset = dataset.PoseDataset(
        args.train_csv_fn, args.img_dir, args.im_size, args.fliplr,
        args.rotate, args.rotate_range, args.zoom, args.base_zoom,
        args.zoom_range, args.translate, args.translate_range, args.min_dim,
        args.coord_normalize, args.gcn, args.n_joints, args.fname_index,
        args.joint_index, args.symmetric_joints, args.ignore_label
    ) #train dataset에 대한 PoseDataset 설정(init) -> joint파일 받아오기, image load
    test_dataset=dataset.PoseDataset(
        args.test_csv_fn, args.img_dir, args.im_size, args.fliplr,
        args.rotate, args.rotate_range, args.zoom, args.base_zoom,
        args.zoom_range, args.translate, args.translate_range, args.min_dim,
        args.coord_normalize, args.gcn, args.n_joints, args.fname_index,
        args.joint_index, args.symmetric_joints, args.ignore_label
    ) #test dataset에 대한 PoseDataset 설정(init) -> joint파일 받아오기, image load

    train_iter = iterators.MultithreadIterator(train_dataset, args.batchsize, n_threads = 24) #example을 parallel하게 불러옴 repat=True, shuffle=None이 default.
    test_iter = iterators.MultithreadIterator(test_dataset, args.batchsize, n_threads = 24, repeat=False, shuffle=False)
    #The dataset is sent to the worker processes in the standard way using pickle.
    #병렬화 처리 초기화하는 코드인듯...

    gpus = [int(i) for i in args.gpus.split(',')] #gpus:0,1,2...
    devices = {'main':gpus[0]}
    if len(gpus)>2: #1개 이상일때
        for gid in gpus[1:]:
            devices.update({'gpu{}'.format(gid):gid}) #ex)gpu1:1
    updater = training.ParallelUpdater(train_iter, opt, devices=devices) #optimization
    #Implementation of a parallel GPU Updater. (만약에 CPU사용하면 그냥 device = -1 해주면 chainer에서 알아서 처리하는듯함)

    interval = (args.snapshot, 'epoch') #snapshot -> makes regular snapshots of the Trainer object during training.
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)
    #The standard training loop in Chainer.
    #Trainer is an implementation of a training loop. Users can invoke the training by calling the run() method.
    #(args.epoch, 'epoch') -> trigger that determines when to stop the training loop : 해당 epoch만큼만 돌고 나면 stop.
    trainer.extend(extensions.DumpGraph('main/loss')) #can register extensions to the trainer, where some configurations can be added.
    #3.0~버전에서는 dump_graph였는데 최신 버전에서는 DumpGraph인듯..? dump_graph도 오류는 안 뜨는데 일단 변경함
    trainer.extend(extensions.snapshot_object(
        model, 'epoch-{.updater.epoch}-model.npz', savefun=serializers.save_npz), trigger=interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'epoch-{.updater.epoch}-state.npz', savefun=serializers.save_npz), trigger=interval)
    trainer.extend(extensions.snapshot(), trigger=interval)
    # writer = extensions.snapshot_writers.ProcessWriter(savefun=serializers.save_npz)
    # trainer.extend(extensions.snapshot(writer=writer), trigger=interval)
    #graph, snapshot 추가함. trigger object that determines when to invoke the extension -> snapshot 횟수마다 해당 extension 수행되는 것 같음.

    if args.opt == 'MomentumSGD' or args.opt == 'AdaGrad':
        trainer.reporter.add_observer('lr', opt.lr)
        #During the training, it also creates a Reporter object to store observed values on each update.
        # For each iteration, it creates a fresh observation dictionary and stores it in the observation attribute.
        trainer.extend(extensions.StepShift(
            'lr', args.lr, args.lr_decay_freq, args.lr_decay_ratio
        )) #InteralShift라는 함수가 없음.... 아마 이것도 상속해서 만든 클래스같은데 파일이 사라짐

    trainer.extend(
        extensions.LogReport(trigger=(args.show_log_iter, 'iteration')))
    trainer.extend(logger.LogPrinter(
        ['epoch', 'main/loss', 'validation/main/loss', 'lr']))
    #extensions.PrintReport 상속해서 ? 새로 클래서 만듦
    #log 관련 설정

    eval_model = model.copy() #model을 copy해옴
    eval_model.predictor.train = False #loss -> predictor : train false
    trainer.extend(
        extensions.Evaluator(test_iter, eval_model, device=gpus[0]),
        trigger = (args.valid_freq, 'epoch')
    ) #Perform test every this epoch : valid_freq마다 test_iter에 대해서(dataset iterator) eval_model을 평가하는듯.
    trainer.run()



