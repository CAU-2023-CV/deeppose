@echo off

REM train_mpii.sh 와 같은 기능입니다. 사용방법 : 터미널에 .\shells\train_mpii 입력
set CHAINER_TYPE_CHECK=0

python scripts/train.py ^
--model models\AlexNet.py ^
--gpus 0 ^
--epoch 1 ^
--batchsize 128 ^
--snapshot 1 ^
--valid_freq 1 ^
--train_csv_fn data\mpii\train_joints.csv ^
--test_csv_fn data\mpii\test_joints.csv ^
--img_dir data\mpii\images ^
--test_freq 10 ^
--seed 1701 ^
--im_size 220 ^
--fliplr ^
--rotate ^
--rotate_range 10 ^
--zoom ^
--zoom_range 0.2 ^
--translate ^
--translate_range 5 ^
--coord_normalize ^
--gcn ^
--n_joints 16 ^
--fname_index 0 ^
--joint_index 1 ^
--symmetric_joints "[[12, 13], [11, 14], [10, 15], [2, 3], [1, 4], [0, 5]]" ^
--opt Adam