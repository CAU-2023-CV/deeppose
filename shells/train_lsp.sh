#!/bin/bash
# Copyright (c) 2016 Shunta Saito

CHAINER_TYPE_CHECK=0 \
python scripts/train.py \
--model models/AlexNet.py \
--gpus 7 \
--epoch 100 \
--batchsize 128 \
--snapshot 10 \
--valid_freq 5 \
--train_csv_fn data/lspet_dataset/train_joints.csv \
--test_csv_fn data/lspet_dataset/test_joints.csv \
--img_dir data/lspet_dataset/images \
--test_freq 10 \
--seed 1701 \
--im_size 220 \
--fliplr \
--rotate \
--rotate_range 10 \
--zoom \
--zoom_range 0.2 \
--translate \
--translate_range 5 \
--coord_normalize \
--gcn \
--n_joints 14 \
--fname_index 0 \
--joint_index 1 \
--symmetric_joints "[[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]" \
--opt Adam
