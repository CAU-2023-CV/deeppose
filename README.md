# Deeppose

- **[This is not an official code]** Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks.](https://arxiv.org/abs/1312.4659)
- Original code we referenced: [mitmul/Deeppose](https://github.com/mitmul/deeppose)
- We modified the code to match the versions of the libraries below. However, MPII is the only dataset we performed. other datasets have not been performed due to url error.

# Tested Environment
- Ubuntu 20.0.4 LTS
- Anaconda 22.9.0
- GPU : Nvidia RTX 3060
- GPU Setting
  - Nvidia-driver 525.105.17
  - CUDA 11.8
  - cuDNN 8.9.2
  - CuPy 7.8
- Python 3.7.5
- Chainer 7.8.0
- Numpy 1.21.5
- scikit-image 0.19.3
- OpenCV 3.4.2

# Dataset preparation
     bash datasets/download.sh
     python datasets/mpii_dataset.py

# Start Training
     bash shells/train_mpii.sh

# Make Output
- After train, you can make output image file(.png)
- this is by evaluate_flic.py, but this file has not been completely modified yet. However, it is possible to check the results through image file.
 
      python scripts/evaluate_flic.py
      --model {result-dir}/{model}.py
      --param {result-dir}/{model}.npz
      --batchsize {batchsize}
      --gpu {num} --datadir data/mpii
      --n_imgs {num} --resize {size} --seed {seed-num}
      --mode {test or tile}
  
- if you choose mode as test -> will make result image about test set (it has little bug until, we'll fix it)
- if you choose mode as tile -> will make result image about random image in test set.
- but, you have to execute test mode first and then execute tile.
